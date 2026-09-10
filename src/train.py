from __future__ import annotations
import os
import sys
import shutil
import math
import copy
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.amp
import torch.nn.functional as F
from types import SimpleNamespace
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

# Import YOLO
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xywh2xyxy

# Import RT-DETR components
from rtdetr.zoo.rtdetr.rtdetr_criterion import RTDETRCriterion

from rtdetr.zoo.rtdetr.matcher import HungarianMatcher
from models import build_rtdetr_model
from kd_loss import FeatureKD, match_boxes_to_gt
from training_utils import (
    normalize_accumulated_gradients,
    resolve_ultralytics_train_imgsz,
    step_optimizer_with_scaler,
)
import label_budget as label_budget_meta
from experiment_artifacts import (
    RUN_PROVENANCE_KEY,
    build_run_provenance,
    budget_run_tag,
    budget_named_path,
    content_addressed_path,
    file_sha256,
    manifest_summary,
    verify_run_provenance,
    write_ultralytics_budget_dataset,
)
from training_regime import ResolvedTrainingRegime, resolve_training_regime

# Import configurations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs.base_cfg as cfg_base
import configs.train_yolo_ultralytics_cfg as cfg_auto
import configs.train_cfg as cfg_manual

# Import dataloaders and evaluator
from dataloader import build_dataloaders
from evaluate import run_evaluation


# --- Helper Functions ---
def get_lrs(optimizer):
    """Return a representative (head, pretrained) learning rate for logging.

    Bias groups are excluded: they warm up from `warmup_bias_lr` downwards, so
    during warmup they read several times higher than the group they are meant
    to represent. Both selections fall back to including bias groups so a
    configuration made entirely of biases still reports something.
    """
    def pick(is_pretrained: bool) -> float:
        groups = [pg for pg in optimizer.param_groups
                  if pg.get('is_pretrained', False) == is_pretrained]
        non_bias = [pg for pg in groups if not pg.get('is_bias', False)]
        return max((pg['lr'] for pg in (non_bias or groups)), default=0.0)

    return pick(False), pick(True)


def detection_loss_items_for_logging(loss_items) -> dict[str, float]:
    """Normalize Ultralytics detection-loss metrics across API versions.

    Older releases returned a tensor ordered as ``box, cls, dfl``. Newer
    releases return a dictionary keyed by ``box_loss``, ``cls_loss`` and
    either ``dfl_loss`` or ``l1_loss`` (including from ``E2ELoss``). These
    values are diagnostics only, so unknown or non-scalar entries are ignored
    instead of interrupting training after a successful backward pass.
    """
    legacy_names = ("box_loss", "cls_loss", "dfl_loss")
    dict_names = (*legacy_names, "l1_loss")

    if isinstance(loss_items, dict):
        values = ((name, loss_items.get(name)) for name in dict_names)
    elif torch.is_tensor(loss_items):
        values = zip(legacy_names, loss_items.detach().reshape(-1))
    elif isinstance(loss_items, (list, tuple)):
        values = zip(legacy_names, loss_items)
    else:
        return {}

    normalized = {}
    for name, value in values:
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


class FocalLossWrapper(torch.nn.Module):
    """Wraps a BCE loss module to apply Focal Loss modulation.

    Mirrors `ultralytics.utils.loss.FocalLoss`, including its alpha factor.
    `alpha` is a foreground/background balance, not a per-class weight and not a
    global scale: it multiplies positives by `alpha` and negatives by
    `1 - alpha`, element-wise. Under the task-aligned assigner `label` is the
    soft alignment score rather than 0/1, so the factor interpolates between the
    two. Set `alpha=0.0` to disable it and keep the plain gamma modulation.

    Args:
        bce (torch.nn.Module): The BCE module to wrap. Must use reduction="none".
        gamma (float): Focusing parameter; higher values suppress easy examples.
        alpha (float): Positive/negative balance. 0.0 disables it.
    """
    def __init__(self, bce: torch.nn.Module, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.bce = bce
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = self.bce(pred, label)
        pred_prob = torch.sigmoid(pred)
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = loss * modulating_factor
        if self.alpha > 0:
            loss = loss * (label * self.alpha + (1 - label) * (1 - self.alpha))
        return loss


def apply_focal_loss(criterion, gamma: float, alpha: float = 0.25) -> list[str]:
    """Wrap every BCE module inside `criterion` with Focal Loss modulation.

    Ultralytics 8.4.x dropped `hyp.fl_gamma`, so focal modulation has to be
    installed by hand. `E2ELoss` holds no `.bce` of its own -- it delegates to
    two independent `v8DetectionLoss` instances (`one2many`/`one2one`), each
    with its own BCE -- so wrapping only the top-level object is a silent
    no-op on the end-to-end models this project trains.

    Args:
        criterion: An `E2ELoss` or `v8DetectionLoss` instance.
        gamma (float): Focal Loss gamma. Must be > 0.
        alpha (float): Focal Loss positive/negative balance; 0.0 disables it.

    Returns:
        list[str]: Names of the sub-criteria that were wrapped.
    """
    targets = [(name, getattr(criterion, name))
               for name in ("one2many", "one2one") if hasattr(criterion, name)]
    if not targets:
        targets = [(type(criterion).__name__, criterion)]

    wrapped = []
    for name, sub in targets:
        if not hasattr(sub, "bce"):
            raise RuntimeError(
                f"Cannot apply Focal Loss: {type(criterion).__name__}.{name} has no 'bce' "
                f"attribute. Ultralytics changed the loss internals; update apply_focal_loss()."
            )
        sub.bce = FocalLossWrapper(sub.bce, gamma, alpha)
        wrapped.append(name)
    return wrapped


def compute_class_weights(dataset, cls_pw: float, device: torch.device) -> torch.Tensor | None:
    """Inverse-frequency class weights for the classification loss.

    Ports `DetectionTrainer.compute_class_weights()`: `(1 / count) ** cls_pw`,
    normalised to mean 1.0 over classes. `cls_pw=0.0` disables weighting (every
    weight would be 1.0 anyway); `cls_pw=1.0` is full inverse frequency. Shared
    by both trainers so YOLO and RT-DETR balance identically -- they differ only
    in where the resulting tensor is plugged in.

    Normalising over classes does NOT keep the classification term's magnitude:
    the loss mass sits on the frequent classes, which get the small weights, so
    the total shrinks (measured ~0.54x at cls_pw=1.0). `preserve_magnitude`
    below rescales that away per batch, which is why it defaults on.

    Args:
        dataset: Training dataset exposing `class_instance_counts()`.
        cls_pw (float): Damping exponent in [0, 1].
        device (torch.device): Device to place the weight tensor on.

    Returns:
        torch.Tensor | None: Shape `(num_classes,)`, or None when disabled.
    """
    if not 0.0 <= cls_pw <= 1.0:
        raise ValueError(f"cls_pw must be in [0, 1], got {cls_pw}.")
    if cls_pw == 0.0:
        return None

    counts = dataset.class_instance_counts()
    if not counts.any():
        print("WARNING: no annotated instances counted; skipping class weighting.")
        return None

    weights = (1.0 / np.where(counts == 0, 1.0, counts)) ** cls_pw
    weights = weights / weights.mean()

    names = [cfg_manual.category_mapping[i]["name"] for i in sorted(cfg_manual.category_mapping)]
    print(f"Class weights (cls_pw={cls_pw}): "
          + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights)))

    return torch.from_numpy(weights).float().to(device)


class ClassWeightWrapper(torch.nn.Module):
    """Applies per-class weights to a classification loss map, magnitude-neutral.

    Wrapping the loss module rather than using Ultralytics' `model.class_weights`
    hook gives access to the unreduced `(batch, anchors, classes)` map, so the
    weighted total can be rescaled back to the unweighted total on every batch:

        scale = loss.sum() / (loss * w).sum()

    The scale is detached, so it is a constant factor in the backward pass and
    the gradient direction is exactly that of the reweighted loss. The effect is
    that `cls_pw` purely *redistributes* gradient between classes and never
    changes how classification trades off against the box and DFL terms -- so it
    can be ablated without also having to retune the `cls` gain.

    Args:
        inner (torch.nn.Module): Loss module returning an unreduced map.
        weights (torch.Tensor): Per-class weights, shape `(num_classes,)`.
        preserve_magnitude (bool): Rescale to hold the total constant. False
            reproduces Ultralytics' behaviour, where the total shrinks.
    """
    def __init__(self, inner: torch.nn.Module, weights: torch.Tensor, preserve_magnitude: bool = True):
        super().__init__()
        self.inner = inner
        self.register_buffer("weights", weights.detach().clone().view(1, 1, -1))
        self.preserve_magnitude = preserve_magnitude

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = self.inner(pred, label)
        weighted = loss * self.weights.to(dtype=loss.dtype)
        if self.preserve_magnitude:
            total = weighted.sum()
            if total.abs() > 0:
                weighted = weighted * (loss.sum() / total).detach()
        return weighted


def apply_class_weights(criterion, weights: torch.Tensor, preserve_magnitude: bool = True) -> list[str]:
    """Wrap every BCE module inside `criterion` with per-class weighting.

    Reaches the same per-branch modules as `apply_focal_loss()`, and composes
    with it: call focal first so the class weights end up outermost, applied to
    the focal-modulated map.

    Args:
        criterion: An `E2ELoss` or `v8DetectionLoss` instance.
        weights (torch.Tensor): Per-class weights, shape `(num_classes,)`.
        preserve_magnitude (bool): Passed to `ClassWeightWrapper`.

    Returns:
        list[str]: Names of the sub-criteria that were wrapped.
    """
    targets = [(name, getattr(criterion, name))
               for name in ("one2many", "one2one") if hasattr(criterion, name)]
    if not targets:
        targets = [(type(criterion).__name__, criterion)]

    wrapped = []
    for name, sub in targets:
        if not hasattr(sub, "bce"):
            raise RuntimeError(
                f"Cannot apply class weights: {type(criterion).__name__}.{name} has no 'bce' "
                f"attribute. Ultralytics changed the loss internals; update apply_class_weights()."
            )
        if getattr(sub, "class_weights", None) is not None:
            raise RuntimeError(
                f"{type(criterion).__name__}.{name} already carries native class_weights; "
                f"applying the wrapper too would weight the loss twice."
            )
        sub.bce = ClassWeightWrapper(sub.bce, weights, preserve_magnitude).to(weights.device)
        wrapped.append(name)
    return wrapped


def set_class_weights(model: torch.nn.Module, weights: torch.Tensor | None) -> None:
    """Attach class weights to `model` so Ultralytics' native hook picks them up.

    Must be called *before* the criterion is constructed -- `v8DetectionLoss`
    reads `model.class_weights` once, in its `__init__`, and multiplies its
    per-class BCE map by it. This is the magnitude-shrinking path; the
    magnitude-neutral one is `apply_class_weights()`, applied afterwards.

    Args:
        model (torch.nn.Module): Model the criterion will be built from.
        weights (torch.Tensor | None): Output of `compute_class_weights()`.
    """
    if weights is not None:
        model.class_weights = weights


def set_seed(seed: int = 42):
    """Sets the seed for reproducibility across all random number generators.

    Args:
        seed (int, optional): The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Enable deterministic cuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_amp_settings(precision: str | None = None) -> dict:
    """Helper to parse precision config into PyTorch AMP kwargs.
    
    Args:
        precision (str | None): Precision setting. One of: "fp16", "bf16", "fp32". If None, uses the default from config.py.

    Returns:
        dict: AMP settings for PyTorch.
    """
    
    # If precision is not provided, use the default from config.py
    if precision is None:
        precision = cfg_manual.shared_train_config["precision"].lower()
    
    # Return AMP settings based on precision
    if precision == "fp16":
        return {"enabled": True, "dtype": torch.float16, "scaler_enabled": True}
    elif precision == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16, "scaler_enabled": False}
    elif precision == "fp32":
        return {"enabled": False, "dtype": torch.float32, "scaler_enabled": False}
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use fp16, bf16, or fp32.")


def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    """Number of optimizer steps one epoch performs under gradient accumulation.

    Both loops also step on the last batch of an epoch, so a partial
    accumulation window still counts. Counters that advance once per *optimizer*
    step -- EMA decay and LinearWarmup -- must be restored in these units when
    resuming, not in batches, or they jump ahead by the accumulation factor.

    Args:
        num_batches (int): Batches in one epoch.
        accumulation_steps (int): Batches accumulated per optimizer step.

    Returns:
        int: Optimizer steps performed in one epoch.
    """
    return math.ceil(num_batches / max(1, int(accumulation_steps)))


def kd_ramp_factor(
    weight: float,
    epoch_progress: float,
    warmup_epochs: float,
    ramp_epochs: float,
    total_epochs: float = 0.0,
    fade_epochs: float = 0.0,
    cooldown_epochs: float = 0.0,
) -> float:
    """Weight of a KD term at a point in training, under warmup ramp and end-of-training fade out.

    All three KD methods share this schedule so their warmup and fade settings
    mean the same thing:

    * `epoch_progress < warmup_epochs`                                        -> 0.0 (term is off entirely)
    * `warmup_epochs <= epoch_progress < warmup_epochs + ramp_epochs`         -> linear 0 -> `weight`
    * plateau phase                                                           -> `weight`
    * `total_epochs - cooldown - fade <= epoch_progress < total_epochs - cooldown` -> linear `weight` -> 0
    * `epoch_progress >= total_epochs - cooldown`                             -> 0.0 (term held at zero)

    `ramp_epochs = 0` switches a term on at full strength the instant its
    warmup elapses. `fade_epochs = 0` drops it to zero the instant the cooldown
    window begins. A smooth ramp prevents discontinuities in total loss from
    disrupting validation mAP or prematurely tripping early stopping.

    Args:
        weight (float): Configured full-strength weight for the term.
        epoch_progress (float): Fractional epoch, i.e. `epoch + batch/len(loader)`.
        warmup_epochs (float): Epochs to hold the term at zero initially.
        ramp_epochs (float): Epochs to ramp up over once warmup elapses.
        total_epochs (float, optional): Total training epochs. Required for fading.
        fade_epochs (float, optional): Epochs to ramp down over at end of training.
        cooldown_epochs (float, optional): Final epochs to hold the term at zero.

    Returns:
        float: The weight to apply at this point in training.
    """
    if epoch_progress < warmup_epochs:
        warmup_factor = 0.0
    elif ramp_epochs <= 0:
        warmup_factor = 1.0
    else:
        warmup_factor = min(1.0, (epoch_progress - warmup_epochs) / ramp_epochs)

    fade_factor = 1.0
    if total_epochs > 0 and (fade_epochs > 0 or cooldown_epochs > 0):
        fade_end = total_epochs - cooldown_epochs
        if epoch_progress >= fade_end:
            fade_factor = 0.0
        elif fade_epochs > 0 and epoch_progress > (fade_end - fade_epochs):
            fade_factor = max(0.0, min(1.0, (fade_end - epoch_progress) / fade_epochs))

    return weight * min(warmup_factor, fade_factor)


def kd_gradient_probe(
    kd_term: torch.Tensor,
    supervised_term: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> tuple[float, float, float]:
    """Compare a KD term's gradient on the student against the supervised loss's.

    The scalar loss ratio is a poor proxy for how hard a KD term is actually
    pulling. A term worth 3% of the loss can carry a third of the gradient (or a
    thousandth of it), because the two objectives differ in curvature -- the
    feature terms pass through an L2 normalisation whose gradient scales with
    ``1 / ||f||``, which the loss value does not reveal. Direction matters as
    much as magnitude: a KD gradient that is large but roughly orthogonal to the
    supervised one is adding information, whereas one that is *anti*-aligned is
    spending the step undoing supervision, and both look identical in the loss.

    Both terms are differentiated with ``retain_graph=True`` so the caller's real
    backward pass still runs afterwards. ``torch.autograd.grad`` does not
    accumulate into ``.grad``, so gradient accumulation is unaffected.

    Args:
        kd_term: Scalar KD loss, as it enters the total (same normalisation).
        supervised_term: Scalar detection loss, same normalisation.
        params: Student parameters to measure over. The KD adapters are
            deliberately excluded -- the question is what reaches the *student*.

    Returns:
        tuple[float, float, float]: ``(kd_norm, ratio, cosine)`` where ``ratio``
        is ``||g_kd|| / ||g_sup||`` and ``cosine`` is their cosine similarity in
        the flattened parameter space. ``cosine`` is 0.0 when either gradient
        vanishes, which is the honest reading: no direction to compare.
    """
    grads_kd = torch.autograd.grad(
        kd_term, params, retain_graph=True, allow_unused=True
    )
    grads_sup = torch.autograd.grad(
        supervised_term, params, retain_graph=True, allow_unused=True
    )

    dot = torch.zeros((), device=params[0].device, dtype=torch.float32)
    kd_sq = torch.zeros_like(dot)
    sup_sq = torch.zeros_like(dot)
    for g_kd, g_sup in zip(grads_kd, grads_sup):
        # A parameter the KD term does not touch contributes nothing to the dot
        # product but still counts towards the supervised norm, so the two are
        # accumulated independently rather than skipping the pair outright.
        if g_kd is not None:
            g_kd = g_kd.float()
            kd_sq += g_kd.pow(2).sum()
        if g_sup is not None:
            g_sup = g_sup.float()
            sup_sq += g_sup.pow(2).sum()
        if g_kd is not None and g_sup is not None:
            dot += (g_kd * g_sup).sum()

    # One host sync for the whole probe rather than three.
    dot_v, kd_sq_v, sup_sq_v = torch.stack((dot, kd_sq, sup_sq)).tolist()
    kd_norm = math.sqrt(max(kd_sq_v, 0.0))
    sup_norm = math.sqrt(max(sup_sq_v, 0.0))
    ratio = kd_norm / sup_norm if sup_norm > 0.0 else 0.0
    cosine = dot_v / (kd_norm * sup_norm) if kd_norm > 0.0 and sup_norm > 0.0 else 0.0
    return kd_norm, ratio, cosine


def slice_parsed_preds(parsed, keep_idx: torch.Tensor):
    """Select a sub-batch from parsed detection-head outputs, preserving nesting.

    `v8DetectionLoss.parse_output` returns `{"boxes", "scores", "feats"}` and
    `E2ELoss`'s returns `{"one2many": {...}, "one2one": {...}}`; every leaf is a
    tensor (or list of tensors, for `feats`) whose first dimension is the batch.
    Recursing lets one helper handle both. The result can be handed straight
    back to `criterion(...)`, whose `parse_output` is the identity on dicts.

    Args:
        parsed: Output of `parse_output`, or any nesting of tensors within it.
        keep_idx (torch.Tensor): 1-D index tensor of images to retain.

    Returns:
        The same structure, indexed down to `keep_idx` along the batch dimension.
    """
    if isinstance(parsed, dict):
        return {k: slice_parsed_preds(v, keep_idx) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return type(parsed)(slice_parsed_preds(v, keep_idx) for v in parsed)
    return parsed[keep_idx]


def slice_ultralytics_batch(batch: dict, keep_idx: torch.Tensor) -> dict | None:
    """Select a sub-batch from an Ultralytics ground-truth batch dict."""
    if keep_idx.numel() == 0:
        return None
    device = keep_idx.device
    batch_idx = batch["batch_idx"]
    
    mask = torch.isin(batch_idx, keep_idx)
    mapping = torch.zeros(keep_idx.max().item() + 1, dtype=torch.long, device=device)
    mapping[keep_idx] = torch.arange(keep_idx.numel(), dtype=torch.long, device=device)
    new_batch_idx = mapping[batch_idx[mask]]
    
    sliced = {
        "img": batch["img"][keep_idx],
        "cls": batch["cls"][mask],
        "bboxes": batch["bboxes"][mask],
        "batch_idx": new_batch_idx,
    }
    for k, v in batch.items():
        if k not in sliced and isinstance(v, torch.Tensor) and v.shape[0] == batch["img"].shape[0]:
            sliced[k] = v[keep_idx]
    return sliced



def scale_teacher_pseudo_base_loss(
    loss: torch.Tensor,
    candidate_count: int,
    effective_count: int,
) -> torch.Tensor:
    """Restore the intended full-batch weight for a filtered teacher subset."""

    # Ultralytics returns a detection loss already multiplied by the number of
    # images it received. The caller divides by the original batch size, so a
    # filtered pseudo objective must scale by candidate_count / effective_count.
    if candidate_count < 0 or effective_count <= 0:
        raise ValueError("candidate_count must be non-negative and effective_count positive.")
    return loss * (candidate_count / effective_count)
def pack_cached_teacher_preds(targets: list[dict], num_classes: int, device) -> dict:
    """Pack variable-length cached teacher predictions into dense batch tensors.

    The dataloader emits a different number of surviving teacher boxes per image
    -- the mosaic crop and the visibility filter remove different numbers from
    each. Padding to a common width lets the response-KD paths keep the dense
    ``(B, Q, *)`` layout they used when the teacher ran live.

    Padding rows are given a strongly negative logit rather than zero: their
    confidence is then ~0, so the ``max_probs > threshold`` filter those paths
    already apply discards them without any additional masking.

    Args:
        targets: Per-image target dicts carrying ``kd_boxes`` and ``kd_logits``.
        num_classes: Width of the logit vector.
        device: Device to build the batch tensors on.

    Returns:
        dict: ``pred_logits`` (B, Q, nc) and ``pred_boxes`` (B, Q, 4) normalised
        ``cxcywh``, matching the live teacher's output contract.
    """
    counts = [int(t["kd_boxes"].shape[0]) for t in targets]
    width = max(counts) if counts else 0
    batch = len(targets)

    logits = torch.full((batch, max(width, 1), num_classes), -30.0, device=device)
    boxes = torch.zeros((batch, max(width, 1), 4), device=device)

    for i, target in enumerate(targets):
        n = counts[i]
        if n == 0:
            continue
        boxes[i, :n] = target["kd_boxes"].to(device=device, dtype=boxes.dtype)
        logits[i, :n] = target["kd_logits"].to(device=device, dtype=logits.dtype)

    return {"pred_logits": logits, "pred_boxes": boxes}


def kd_run_suffix(kd_config: dict) -> str:
    """Filename suffix identifying which KD methods a run used.

    Without it every KD variant would land on the same `yolo_manual_best.pth`,
    so one run would silently overwrite its predecessor and the checkpoint on
    disk would carry no record of what produced it -- recoverable afterwards
    only by matching file mtimes against TensorBoard runs.

    The suffix goes at the end of the stem, after ``best``/``last``, so the
    existing ``save_name.replace("best", "last")`` still resolves correctly.

    Args:
        kd_config: The `kd_config` mapping from the training config.

    Returns:
        str: ``""`` when KD is off or no method is enabled, otherwise
        ``"_kd-<methods>"``, e.g. ``"_kd-feat"`` or ``"_kd-hard-soft"``.
        Methods keep a fixed order so the same combination always produces the
        same filename.
    """
    if not kd_config.get("enabled", False):
        return ""

    methods = [
        ("pseudo_label", "hard"),
        ("pseudo_label_soft", "soft"),
        ("feature_based", "feat"),
    ]
    active = [
        tag for key, tag in methods
        if kd_config.get(key, {}).get("enabled", False)
    ]
    return f"_kd-{'-'.join(active)}" if active else ""


def training_regime_run_suffix(
    regime: ResolvedTrainingRegime, kd_config: dict
) -> str:
    """Name a run's checkpoints from its resolved supervision regime.

    Hybrid delegates to ``kd_run_suffix``; gt_only takes no suffix at all, so
    plain supervised runs keep unadorned filenames. Teacher-only and
    semi-supervised each get their own namespace and list only the scheduled
    auxiliaries -- their pseudo-detection base is implied by the regime name.
    """

    if regime.mode == "hybrid":
        return kd_run_suffix(kd_config)
    if regime.mode == "gt_only":
        return ""

    auxiliaries = []
    if regime.use_aux_kd:
        # Teacher-only already uses hard pseudo detections as its base
        # objective. In semi-supervised mode the GT-anchored hard auxiliary is
        # a distinct, scheduled loss on the labeled partition.
        if (regime.is_semi_supervised
                and kd_config.get("pseudo_label", {}).get("enabled", False)):
            auxiliaries.append("hard")
        if kd_config.get("pseudo_label_soft", {}).get("enabled", False):
            auxiliaries.append("soft")
        if kd_config.get("feature_based", {}).get("enabled", False):
            auxiliaries.append("feat")
    tail = f"-{'-'.join(auxiliaries)}" if auxiliaries else ""
    base = "_semi-supervised" if regime.is_semi_supervised else "_teacher-only"
    return f"{base}{tail}"


def training_manifest_for_regime(regime: ResolvedTrainingRegime, cfg) -> str:
    """Select the labelled subset or full unlabeled pool for one regime."""

    if regime.is_semi_supervised:
        return cfg.full_train_json

    if not regime.use_teacher_pseudo_base:
        return cfg.train_json
    if regime.teacher_only_train_json is None:
        return cfg.full_train_json
    return regime.teacher_only_train_json


def validation_manifest_for_regime(regime: ResolvedTrainingRegime, cfg) -> str:
    """Select the validation image pool for one regime.

    A budget restricts *labels*, not images. Teacher-scored validation spends no
    annotations -- it ranks epochs by agreement with predictions the teacher
    already made -- so confining it to the budgeted split would discard usable
    images for no saving, exactly as confining teacher-only *training* to them
    would. It therefore uses the whole validation pool.

    Ground-truth validation is the opposite case: it reads annotations, so it is
    part of what the budget pays for and follows it. That distinction is why the
    switch is `validation_targets` rather than `use_ground_truth` -- a
    teacher-only run may still select on labelled validation, and when it does,
    those labels count.

    A side effect worth having: teacher-scored validation is then the *same*
    image set at every budget, so selection noise stops varying between arms
    that are being compared. At the quarter budget that is 548 images instead
    of 136.
    """

    if regime.validation_targets == "teacher" or regime.validation_targets == "ground_truth_teacher":
        return cfg.full_val_json
    return cfg.val_json


def build_training_run_provenance(
    *,
    backend: str,
    budget: dict,
    train_manifest: str,
    validation: dict,
    regime: ResolvedTrainingRegime,
    kd_config: dict,
    train_subsample_interval: int,
    seed: int,
    teacher: dict | None = None,
    teacher_cache: dict | None = None,
) -> dict:
    """Record every data/supervision choice that makes a run distinct.

    The label-budget stamp intentionally describes which labels were available.
    This separate record describes what the backend actually consumed, so a
    different frame interval, unlabeled pool, teacher/cache, target policy, or
    active KD configuration cannot share artifacts or optimizer state.
    """

    train_interval = int(train_subsample_interval)
    validation_interval = int(validation.get("subsample_interval", 1))
    if train_interval <= 0 or validation_interval <= 0:
        raise ValueError("Training and validation subsample intervals must be positive.")

    active_kd = kd_config if regime.kd_pipeline_enabled else {"enabled": False}
    identity = {
        "backend": backend,
        "label_budget": budget,
        "sampling": {
            "algorithm": "file-name-sort-stride-v1",
            "seed": int(seed),
            "train_interval": train_interval,
            "validation_interval": validation_interval,
        },
        "train_pool": manifest_summary(train_manifest),
        "validation": validation,
        "training_regime": regime.as_dict(),
        "kd": active_kd,
        "teacher": teacher,
        "teacher_cache": teacher_cache,
    }
    return build_run_provenance(identity)


def build_teacher_pseudo_batch(
    batch: dict,
    keep_idx: torch.Tensor,
    teacher_boxes: torch.Tensor,
    teacher_classes: torch.Tensor,
    teacher_conf: torch.Tensor,
    conf_threshold: float,
    empty_target_policy: str = "skip",
) -> tuple[dict | None, torch.Tensor, int, int]:
    """Build a pure-teacher detection batch without consulting GT tensors.

    This is deliberately separate from ``build_gt_anchored_pseudo_batch`` so
    the hybrid target semantics and arithmetic remain untouched. The returned
    ``effective_keep_idx`` addresses the original batch; the target
    ``batch_idx`` is re-numbered for the correspondingly sliced predictions.

    ``skip`` drops teacher-silent images rather than asserting that they are
    background. ``background`` retains them intentionally with zero target
    rows. In either case, no value is read from ``batch['cls']``,
    ``batch['bboxes']`` or ``batch['batch_idx']``.
    """

    if empty_target_policy not in {"skip", "background"}:
        raise ValueError(
            "empty_target_policy must be 'skip' or 'background'; got "
            f"{empty_target_policy!r}."
        )

    device = teacher_boxes.device
    keep_idx = keep_idx.to(device=device, dtype=torch.long)
    box_list: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    idx_list: list[torch.Tensor] = []
    effective_indices: list[int] = []
    n_teacher_boxes = 0
    n_empty_skipped = 0

    for original_i in keep_idx.tolist():
        selected = teacher_conf[original_i] > conf_threshold
        selected_boxes = teacher_boxes[original_i][selected]
        selected_classes = teacher_classes[original_i][selected]
        if selected_boxes.shape[0] == 0 and empty_target_policy == "skip":
            n_empty_skipped += 1
            continue

        new_i = len(effective_indices)
        effective_indices.append(original_i)
        if selected_boxes.shape[0] == 0:
            continue

        n_teacher_boxes += int(selected_boxes.shape[0])
        box_list.append(selected_boxes)
        cls_list.append(selected_classes.reshape(-1, 1).long())
        idx_list.append(
            torch.full(
                (selected_boxes.shape[0],), new_i,
                dtype=torch.long, device=device,
            )
        )

    effective_keep_idx = torch.as_tensor(
        effective_indices, dtype=torch.long, device=device
    )
    if effective_keep_idx.numel() == 0:
        return None, effective_keep_idx, n_teacher_boxes, n_empty_skipped

    pseudo_batch = {
        "img": batch["img"][effective_keep_idx],
        "cls": (
            torch.cat(cls_list, dim=0)
            if cls_list
            else torch.zeros((0, 1), dtype=torch.long, device=device)
        ),
        "bboxes": (
            torch.cat(box_list, dim=0)
            if box_list
            else teacher_boxes.new_zeros((0, 4))
        ),
        "batch_idx": (
            torch.cat(idx_list, dim=0)
            if idx_list
            else torch.zeros((0,), dtype=torch.long, device=device)
        ),
    }
    return pseudo_batch, effective_keep_idx, n_teacher_boxes, n_empty_skipped


def teacher_feature_targets(
    pseudo_batch: dict | None, effective_keep_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map pure-teacher pseudo targets back to full-batch indices for masks."""

    if pseudo_batch is None or pseudo_batch["bboxes"].numel() == 0:
        device = effective_keep_idx.device
        return torch.zeros((0, 4), device=device), torch.zeros(
            (0,), dtype=torch.long, device=device
        )
    original_idx = effective_keep_idx[pseudo_batch["batch_idx"].long()]
    return pseudo_batch["bboxes"], original_idx


def teacher_feature_disable_mask(
    disable_kd_feature: torch.Tensor,
    effective_keep_idx: torch.Tensor,
) -> torch.Tensor:
    """Also mask images excluded by teacher-only's ``skip`` policy.

    FeatureKD has background, attention, and global terms, so merely omitting a
    silent image's boxes does not omit that image's gradient. This explicit
    image mask makes ``empty_target_policy='skip'`` apply to the whole feature
    auxiliary. Under ``background``, the effective set includes every image and
    the mask reduces to the ordinary augmentation policy.
    """

    eligible = torch.zeros_like(disable_kd_feature, dtype=torch.bool)
    if effective_keep_idx.numel():
        eligible[effective_keep_idx.long()] = True
    return disable_kd_feature.bool() | ~eligible



def semi_supervised_feature_targets(
    batch: dict,
    pseudo_batch: dict | None,
    effective_keep_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FeatureKD foreground targets for both supervision partitions."""

    teacher_boxes, teacher_batch_idx = teacher_feature_targets(
        pseudo_batch, effective_keep_idx
    )
    gt_boxes = batch["bboxes"]
    gt_batch_idx = batch["batch_idx"].long()

    if teacher_boxes.numel() == 0:
        return gt_boxes, gt_batch_idx
    if gt_boxes.numel() == 0:
        return teacher_boxes.to(dtype=gt_boxes.dtype), teacher_batch_idx
    return (
        torch.cat((gt_boxes, teacher_boxes.to(dtype=gt_boxes.dtype)), dim=0),
        torch.cat((gt_batch_idx, teacher_batch_idx), dim=0),
    )


def semi_supervised_feature_disable_mask(
    disable_kd_feature: torch.Tensor,
    has_gt: torch.Tensor,
    effective_keep_idx: torch.Tensor,
) -> torch.Tensor:
    """Apply ``skip`` only to silent unlabeled images, never labeled ones."""

    eligible = has_gt.to(
        device=disable_kd_feature.device, dtype=torch.bool
    ).clone()
    if effective_keep_idx.numel():
        eligible[effective_keep_idx.long()] = True
    return disable_kd_feature.bool() | ~eligible


def hard_kd_keep_indices(
    disable_kd_hard: torch.Tensor,
    has_gt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the 1-D image indices eligible for GT-anchored hard KD.

    Hard KD matches teacher boxes to authoritative annotations, so it runs only
    on labeled images in semi-supervised mode.
    """

    keep_mask = ~disable_kd_hard.bool()
    if has_gt is not None:
        keep_mask &= has_gt.to(device=keep_mask.device, dtype=torch.bool)
    return keep_mask.nonzero(as_tuple=True)[0]
def build_gt_anchored_pseudo_batch(
    batch: dict,
    keep_idx: torch.Tensor,
    teacher_boxes: torch.Tensor,
    teacher_classes: torch.Tensor,
    teacher_conf: torch.Tensor,
    conf_threshold: float,
    gt_match_iou: float,
    unmatched_mode: str,
) -> tuple[dict | None, int, int]:
    """Build hard pseudo-label targets that cannot contradict the annotations.

    Pass 2 of hard pseudo-label KD runs a full detection loss over the student's
    own predictions with a second target set. When that set was the teacher's
    detections alone, every annotated object the teacher missed became an
    explicit negative at a true-positive location -- on VisDrone, overwhelmingly
    the small and crowded objects. The target set is therefore anchored on the
    ground truth: every annotated object is present exactly once, and a teacher
    box only replaces an object's *coordinates* when the two agree on it.

    Boxes are normalised ``cxcywh`` throughout. The teacher's are normalised
    against the same image as the student's, so no rescaling is needed.

    Args:
        batch: Ultralytics-format batch with ``img``, ``cls``, ``bboxes`` and
            ``batch_idx``, covering the *full* (unsliced) batch.
        keep_idx: Indices of images KD is allowed to run on.
        teacher_boxes: Teacher boxes, ``(B, Q, 4)`` normalised ``cxcywh``.
        teacher_classes: Teacher class indices, ``(B, Q)``.
        teacher_conf: Teacher max class probability, ``(B, Q)``.
        conf_threshold: Minimum confidence for a teacher box to be considered.
        gt_match_iou: Minimum IoU for a teacher box to be taken as an
            annotation's detection. See `match_boxes_to_gt`.
        unmatched_mode: ``"drop"`` discards confident teacher boxes matching no
            annotation; ``"add"`` admits them as extra objects, which is only
            sound where the annotations are known to be incomplete.

    Returns:
        tuple: ``(pseudo_batch, n_matched, n_added)``. ``pseudo_batch`` is None
        when no image survives `keep_idx`; its ``batch_idx`` addresses the
        sliced batch, so it pairs with ``slice_parsed_preds(..., keep_idx)``.
    """
    if keep_idx.numel() == 0:
        return None, 0, 0

    device = batch["bboxes"].device
    box_list: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    idx_list: list[torch.Tensor] = []
    n_matched = 0
    n_added = 0

    for new_i, b_i in enumerate(keep_idx.tolist()):
        conf_mask = teacher_conf[b_i] > conf_threshold
        t_boxes = teacher_boxes[b_i][conf_mask].to(device=device, dtype=batch["bboxes"].dtype)
        t_cls = teacher_classes[b_i][conf_mask].to(device=device)

        gt_sel = batch["batch_idx"].reshape(-1) == b_i
        target_boxes = batch["bboxes"][gt_sel].clone()
        target_cls = batch["cls"].reshape(-1)[gt_sel].clone()

        matched_teacher = torch.zeros(t_boxes.shape[0], dtype=torch.bool, device=device)
        if t_boxes.numel() and target_boxes.numel():
            t_idx, g_idx = match_boxes_to_gt(
                t_boxes, t_cls, target_boxes, target_cls, gt_match_iou
            )
            # Matching is same-class, so only the coordinates move; the label
            # stays the annotation's.
            target_boxes[g_idx] = t_boxes[t_idx]
            matched_teacher[t_idx] = True
            n_matched += int(t_idx.numel())

        if unmatched_mode == "add":
            extra_boxes = t_boxes[~matched_teacher]
            if extra_boxes.numel():
                target_boxes = torch.cat((target_boxes, extra_boxes), dim=0)
                target_cls = torch.cat(
                    (target_cls, t_cls[~matched_teacher].to(target_cls.dtype)), dim=0
                )
                n_added += int(extra_boxes.shape[0])

        # An image with neither annotations nor admitted teacher boxes is
        # genuinely empty and is correctly scored as background; it simply
        # contributes no target rows.
        if target_boxes.numel():
            box_list.append(target_boxes)
            cls_list.append(target_cls.reshape(-1, 1).to(batch["cls"].dtype))
            idx_list.append(
                torch.full(
                    (target_boxes.shape[0],), new_i,
                    device=device, dtype=batch["batch_idx"].dtype,
                )
            )

    pseudo_batch = {
        "img": batch["img"][keep_idx],
        "cls": torch.cat(cls_list, dim=0) if cls_list else batch["cls"].new_zeros((0, 1)),
        "bboxes": torch.cat(box_list, dim=0) if box_list else batch["bboxes"].new_zeros((0, 4)),
        "batch_idx": torch.cat(idx_list, dim=0) if idx_list else batch["batch_idx"].new_zeros((0,)),
    }
    return pseudo_batch, n_matched, n_added


def soft_kd_branch_loss(
    branch_parsed: dict,
    branch_criterion,
    t_labels_pad: torch.Tensor,
    t_bboxes_pad: torch.Tensor,
    t_logits_pad: torch.Tensor,
    mask_gt: torch.Tensor,
    temperature: float,
    confidence_weight: float = 1.0,
    relation_weight: float = 1.0,
) -> torch.Tensor:
    """Response distillation for a single detection head, split into two terms.

    Factored out of the training loop so the same term can be applied to
    YOLO26's one2many and one2one branches, which have separate score tensors
    and separate assigners (``tal_topk`` 10 vs 7/1). Distilling one2many alone
    supervises a branch that `E2ELoss` decays from 0.8 to 0.1 across training
    and that never runs at inference.

    The single temperature-softened sigmoid BCE this replaces was measured to
    be actively harmful. Textbook KD softens a *softmax*, where temperature
    redistributes a fixed unit of probability mass between classes. A
    multi-label sigmoid head has no such constraint, so temperature inflates
    the mass instead: at T=2.5 the teacher's targets put >0.1 on 93% of all
    class slots and carried 2.0 units per query against the teacher's own
    0.57, pulling every background class up toward 0.5 while the supervised
    BCE drove the same slots to 0.

    Absolute confidence and relative class structure are therefore distilled
    separately, which is the only way to get temperature's benefit without its
    calibration damage:

    * **Confidence**, at T=1, on the teacher's selected class only. Nothing is
      asked of the other nine slots, so no background class is ever pulled up.
    * **Relation**, at T>1, as a KL between the two softmax distributions over
      all classes. Softmax mass is fixed at one unit, so here temperature
      genuinely redistributes and the conventional ``T**2`` gradient
      correction is legitimate. This is what carries the "dark knowledge" --
      on VisDrone, that a pedestrian is confusable with people, or a truck
      with a van.

    Args:
        branch_parsed: One branch's parsed head outputs (``boxes``, ``scores``,
            ``feats``).
        branch_criterion: The `v8DetectionLoss` owning that branch's assigner
            and strides.
        t_labels_pad: Padded teacher class indices, ``(B, M, 1)``.
        t_bboxes_pad: Padded teacher boxes, ``(B, M, 4)`` image-scale ``xyxy``.
        t_logits_pad: Padded raw teacher logits, ``(B, M, nc)``. Raw rather than
            pre-softened: the two terms need different temperatures.
        mask_gt: Validity mask over the padding, ``(B, M)``.
        temperature: Temperature for the relation term only. 1.0 reduces it to
            a plain softmax KL.
        confidence_weight: Gain on the confidence term.
        relation_weight: Gain on the relation term. 0.0 disables it.

    Returns:
        torch.Tensor: Scalar loss, normalised by the assigner's
        ``target_scores_sum`` exactly as the supervised BCE term is.
    """
    s_distri = branch_parsed["boxes"].permute(0, 2, 1).contiguous()
    s_scores = branch_parsed["scores"].permute(0, 2, 1).contiguous()

    anchor_points, stride_tensor = make_anchors(
        branch_parsed["feats"], branch_criterion.stride, 0.5
    )
    s_boxes = branch_criterion.bbox_decode(anchor_points, s_distri) * stride_tensor

    # Cast to float32: Ultralytics' IoU maths inside the assigner assumes it,
    # and AMP dtype promotion misbehaves there.
    _, _, target_scores, fg_mask, target_gt_idx = branch_criterion.assigner(
        s_scores.detach().sigmoid().float(),
        s_boxes.detach().float(),
        (anchor_points * stride_tensor).float(),
        t_labels_pad.float(),
        t_bboxes_pad.float(),
        mask_gt.unsqueeze(-1),
    )

    # Map each anchor to the raw logits of the teacher object assigned to it.
    b_dim, n_dim = target_gt_idx.shape
    batch_arange = torch.arange(b_dim, device=target_gt_idx.device).view(-1, 1).expand(b_dim, n_dim)
    t_logits = t_logits_pad[batch_arange, target_gt_idx].detach().float()  # (B, N, nc)
    student_logits = s_scores.float()

    # TAL box quality is applied as a loss weight on both terms, never as a
    # target modifier, so neither the probabilities nor the distributions are
    # mathematically distorted by it.
    align_scores, _ = target_scores.max(dim=-1)
    weight = (fg_mask * align_scores).unsqueeze(-1)

    total = None

    if confidence_weight != 0.0:
        # Teacher's selected class only. Gathering one slot per anchor leaves
        # the other nine untouched by KD, which is the whole point: the
        # supervised BCE owns them, and nothing pulls them off zero.
        selected = t_logits.argmax(dim=-1, keepdim=True)             # (B, N, 1)
        conf_target = t_logits.gather(-1, selected).sigmoid()
        conf_logit = student_logits.gather(-1, selected)
        conf_loss = F.binary_cross_entropy_with_logits(
            conf_logit, conf_target, reduction="none"
        )
        total = (conf_loss * weight).sum() * confidence_weight

    if relation_weight != 0.0:
        # Relative structure over all classes. Softmax normalises the mass, so
        # temperature redistributes rather than inflates and T**2 is the
        # correct gradient correction.
        t_log_probs = F.log_softmax(t_logits / temperature, dim=-1)
        s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        relation = F.kl_div(
            s_log_probs, t_log_probs, reduction="none", log_target=True
        ).sum(dim=-1, keepdim=True) * (temperature ** 2)
        relation_total = (relation * weight).sum() * relation_weight
        total = relation_total if total is None else total + relation_total

    if total is None:
        return student_logits.sum() * 0.0

    target_scores_sum = target_scores.sum().clamp(min=1.0)
    return total / target_scores_sum


def update_pretrained_lr_ratio(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    lr_warmup: 'LinearWarmup | None',
    epoch: int,
    total_epochs: int,
) -> None:
    """Dynamically update the learning rate of pretrained parameters,
    fading the reduction factor towards 1.0 (no reduction) if configured.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler | None): The learning rate scheduler.
        lr_warmup ('LinearWarmup | None'): The learning rate warmup scheduler (custom LinearWarmup wrapper).
        epoch (int): The current epoch.
        total_epochs (int): The total number of epochs.
    """
    initial_ratio = cfg_manual.shared_train_config.get("pretrained_lr_ratio", 0.1)
    fade = cfg_manual.shared_train_config.get("fade_pretrained_lr_ratio", False)
    
    if fade:
        # Faster, smooth cosine fade over the specified fraction of training
        fade_fraction = cfg_manual.shared_train_config.get("fade_pretrained_lr_ratio_fraction", 0.5)
        fade_epochs = max(1, int(total_epochs * fade_fraction))
        if epoch < fade_epochs:
            progress = epoch / fade_epochs
            # 0.5 * (1 - cos(pi * progress)) maps from 0.0 to 1.0 smoothly
            r = initial_ratio + (1.0 - initial_ratio) * 0.5 * (1 - math.cos(math.pi * progress))
        else:
            r = 1.0
    else:
        r = initial_ratio
        
    # Find the current active learning rate of the head groups (this includes any scheduler decays)
    current_head_lr = None
    for group in optimizer.param_groups:
        if not group.get("is_pretrained", False):
            current_head_lr = group["lr"]
            break
            
    if current_head_lr is None:
        return
        
    if fade:
        print(f"Epoch {epoch+1}: Pretrained LR ratio is {r:.4f} (Pretrained LR: {current_head_lr * r:.2e}, Head LR: {current_head_lr:.2e})")
        
    # Update pretrained groups' learning rates
    for idx, group in enumerate(optimizer.param_groups):
        if group.get("is_pretrained", False):
            # Update the active learning rate based on the head's decayed learning rate
            group["lr"] = current_head_lr * r
            
            # Update initial reference learning rates for components that recalculate from scratch
            base = group.get("unreduced_lr")
            if base is not None:
                # Update initial_lr (used by YOLO manual warmup)
                group["initial_lr"] = base * r
                
                # Update scheduler's internal base_lr (used by LambdaLR)
                if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
                    lr_scheduler.base_lrs[idx] = base * r
                    
                # Update target LRs inside LinearWarmup (used by RT-DETR warmup)
                if lr_warmup is not None and hasattr(lr_warmup, "_target_lrs"):
                    lr_warmup._target_lrs[idx] = base * r


def _validation_subsample_interval() -> int:
    """Return the validated frame interval used for model-selection validation."""

    interval = int(
        cfg_manual.dataloader_config.get("val_subsample_interval", 1)
    )
    if interval <= 0:
        raise ValueError(
            "dataloader_config['val_subsample_interval'] must be positive, "
            f"got {interval}."
        )
    return interval


def validate_and_early_stop(
    model: torch.nn.Module, 
    ema: ModelEMA, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    model_type: str, 
    device: torch.device, 
    writer: SummaryWriter, 
    best_map: float, 
    patience_counter: int, 
    patience: int, 
    save_name: str,
    checkpoint_extras: dict = None,
    val_json_path: str | None = None,
    val_targets: str = "ground_truth",
) -> tuple[bool, float, int]:
    """
    Run validation, log metrics, handle early stopping, and save best weights.
    
    Args:
        model (torch.nn.Module): The model to validate.
        ema (ModelEMA): The Exponential Moving Average model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        epoch (int): The current epoch.
        model_type (str): The type of model being validated.
        device (torch.device): The device to run the validation on.
        writer (SummaryWriter): The TensorBoard writer.
        best_map (float): The best mAP score seen so far.
        patience_counter (int): The patience counter.
        patience (int): The patience value.
        save_name (str): The name of the file to save the best weights to.
        checkpoint_extras (dict, optional): Extra dict to save to checkpoint.
        val_json_path (str, optional): Manifest to validate against. Defaults
            to the labelled `cfg_manual.val_json`.
        val_targets (str): "ground_truth" or "teacher". Only changes labelling
            and logging -- the selection logic is identical either way, because
            what is being ranked is still "which epoch scores best against the
            configured targets".

    Returns:
        stop_training (bool): True if early stopping triggered.
        best_map (float): Updated best mAP.
        patience_counter (int): Updated patience counter.
    """
    # Set the model to evaluation mode
    ema.module.eval()
    
    # Run evaluation. The validation loader built during training setup is not
    # consumed by this path, so the interval must be forwarded explicitly.
    validation_subsample_interval = _validation_subsample_interval()
    val_metrics = run_evaluation(
        model=ema.module,
        model_type=model_type,
        gt_json_path=val_json_path or cfg_manual.val_json,
        condition="clean",
        conf=0.001,
        device=device,
        subsample_interval=validation_subsample_interval,
    )
    
    # Log validation metrics. Scoring against teacher boxes measures agreement
    # with the teacher rather than accuracy, so it gets its own tag: the two
    # curves are on different scales and must never be read as one series.
    map_50_95 = val_metrics.get("mAP_50_95", 0.0)
    if val_targets == "teacher":
        prefix, label = "Validation/teacher_agreement_", "Teacher-agreement"
    else:
        prefix, label = "Validation/", "Val"
    writer.add_scalar(f'{prefix}mAP_50_95', map_50_95, epoch)
    writer.add_scalar(f'{prefix}mAP_50', val_metrics.get("mAP_50", 0.0), epoch)
    print(f"{label} mAP@0.5:0.95: {map_50_95:.4f}  |  {label} mAP@0.5: {val_metrics.get('mAP_50', 0.0):.4f}")
    
    # Early stopping logic
    stop_training = False
    os.makedirs(cfg_manual.weights_dir, exist_ok=True)
    
    improved = False
    if map_50_95 > best_map:
        best_map = map_50_95
        patience_counter = 0
        improved = True
    else:
        patience_counter += 1
        print(f"--> No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            stop_training = True
            
    # Prepare checkpoint dictionary
    ckpt_dict = {
        'model': model.state_dict(),
        'ema': {'module': ema.module.state_dict()},
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_map': best_map,
        'patience_counter': patience_counter,
    }
    if checkpoint_extras is not None:
        checkpoint_extras = dict(checkpoint_extras)
        validation_manifest = checkpoint_extras.get('validation_manifest')
        if isinstance(validation_manifest, dict):
            checkpoint_extras['validation_manifest'] = {
                **validation_manifest,
                'subsample_interval': validation_subsample_interval,
            }
        ckpt_dict.update(checkpoint_extras)
        # Ensure the updated patience_counter and best_map override any stale values passed in extras
        ckpt_dict['patience_counter'] = patience_counter
        ckpt_dict['best_map'] = best_map

    # Save the 'last' checkpoint
    last_save_name = save_name.replace("best", "last")
    torch.save(ckpt_dict, os.path.join(cfg_manual.weights_dir, last_save_name))

    # Save the 'best' checkpoint if there was improvement
    if improved:
        torch.save(ckpt_dict, os.path.join(cfg_manual.weights_dir, save_name))
        print(f"--> Saved new best checkpoint at epoch {epoch+1}")
            
    return stop_training, best_map, patience_counter


# --- Exponential Moving Average Model ---
class ModelEMA:
    """
    Exponential moving average of model parameters.
    Keeps a shadow copy of the model parameters and updates it each training step.

    Args:
        model (torch.nn.Module): The model to apply EMA to.
        decay (float): The decay rate (momentum) of the EMA.
        warmups (int): The number of warm-up iterations. It is the time constant in the exponential increase of EMA decay rate.
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, warmups: int = 2000):
        self.decay = decay
        self.warmups = warmups
        self._step = 0

        # Copy the model parameters as the initial EMA weights
        self.module = copy.deepcopy(model)
        self.module.eval()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self._step += 1

        # Ramp up the decay from 0 to target decay rate based on warmup iterations.
        d = self.decay * (1 - math.exp(-self._step / self.warmups))

        # Update EMA parameters
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1 - d)

        # Update buffers (EMA for floating-point, copy for integer/bool)
        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            if ema_b.dtype.is_floating_point:
                ema_b.data.mul_(d).add_(model_b.data, alpha=1 - d)
            else:
                ema_b.data.copy_(model_b.data)


# --- Automatic YOLO Training ---
def train_yolo_ultra():
    """
    Train YOLO model using Ultralytics API.
    """
    print("\nStarting YOLO Ultra Training...")

    regime = resolve_training_regime(
        getattr(cfg_manual, "training_regime", None), cfg_manual.kd_config
    )
    if regime.mode != "gt_only":
        raise ValueError(
            "Automatic Ultralytics training supports supervised gt_only runs. "
            f"The configured regime resolves to {regime.mode!r}; use "
            "model_format='yolo_manual' for teacher-only or hybrid KD training."
        )

    train_imgsz = resolve_ultralytics_train_imgsz(
        cfg_auto.input_height, cfg_auto.input_width
    )
    run_budget = label_budget_meta.budget_record(cfg_base)
    dataset = write_ultralytics_budget_dataset(
        train_manifest=cfg_auto.train_json,
        val_manifest=cfg_auto.val_json,
        output_dir=os.path.join(cfg_auto.cache_dir, "ultralytics"),
        budget_record=run_budget,
        category_mapping=cfg_auto.category_mapping,
    )
    validation_record = {
        **manifest_summary(cfg_auto.val_json),
        "targets": "ground_truth",
        # Ultralytics consumes the exact generated list; the manual-loader
        # val_subsample knob does not apply to this backend.
        "subsample_interval": 1,
    }
    run_provenance = build_training_run_provenance(
        backend="yolo_ultra",
        budget=run_budget,
        train_manifest=cfg_auto.train_json,
        validation=validation_record,
        regime=regime,
        kd_config=cfg_manual.kd_config,
        train_subsample_interval=1,
        seed=cfg_auto.shared_train_config.get("seed", 42),
    )
    run_tag = f"{dataset['run_tag']}_{run_provenance['tag']}"
    print(
        f"Annotation budget: {label_budget_meta.describe(run_budget)}; "
        f"Ultralytics lists contain {dataset['train_images']:,} train and "
        f"{dataset['val_images']:,} validation images."
    )
    
    # Initialize YOLO
    weights_path = cfg_auto.yolo_train_config["pretrained_model_path"]
    yaml_path = cfg_auto.yolo_train_config["model_yaml_path"]
    
    if os.path.exists(weights_path):
        print(f"Loaded pretrained weights locally from {weights_path}")
        model = YOLO(weights_path)
    else:
        # The file doesn't exist locally. Try auto-downloading by passing the
        # official release name (basename only, e.g. "yolov26n.pt") so that
        # Ultralytics can resolve it from its GitHub releases.
        model_name = os.path.basename(weights_path)
        print(f"WARNING: Pretrained weights not found locally at {weights_path}.")
        print(f"Attempting to let Ultralytics download '{model_name}'...")
        try:
            model = YOLO(model_name)
            print(f"Successfully downloaded and loaded pretrained weights for '{model_name}'!")
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Reverting to training from scratch using architecture blueprint: {yaml_path}")
            model = YOLO(yaml_path)
    
    # Train YOLO
    model.train(
        data=dataset["path"],
        epochs=cfg_auto.shared_train_config["epochs"],
        imgsz=train_imgsz,
        batch=cfg_auto.dataloader_config["batch_size"],
        device=0 if torch.cuda.is_available() else "cpu",
        project=os.path.abspath(os.path.join("runs", "yolo_ultra")),
        name=run_tag,
        close_mosaic=cfg_auto.augmentation.get("mosaic", {}).get("close_mosaic_epochs", 10),
        amp=cfg_auto.shared_train_config["precision"] != "fp32",
        patience=cfg_auto.shared_train_config.get("early_stopping_patience", 10),
        seed=cfg_auto.shared_train_config.get("seed", 42),
    )
    print("YOLO Ultra Training Complete. Weights saved via Ultralytics.")
    
    # Copy weights to custom directory
    save_dir = model.trainer.save_dir if hasattr(model, "trainer") and model.trainer else None
    if save_dir:
        weights_src_dir = os.path.join(save_dir, "weights")
        weights_dst_dir = cfg_base.weights_dir
        os.makedirs(weights_dst_dir, exist_ok=True)
        
        mapping = {
            "best.pt": budget_named_path("yolo_ultra_best.pt", run_budget),
            "last.pt": budget_named_path("yolo_ultra_last.pt", run_budget),
        }
        for src_name, dst_name in mapping.items():
            src = os.path.join(weights_src_dir, src_name)
            if os.path.exists(src):
                dst = os.path.join(weights_dst_dir, dst_name)
                shutil.copy2(src, dst)
                label_budget_meta.stamp_checkpoint(
                    dst,
                    run_budget,
                    extra_metadata={
                        "training_regime": regime.as_dict(),
                        "validation_manifest": validation_record,
                        RUN_PROVENANCE_KEY: run_provenance,
                    },
                )
                print(f"Copied {src_name} to {dst}")


# --- Manual YOLO Training ---
def collate_to_ultralytics_format(
    images: torch.Tensor,
    targets: list[dict],
    device: torch.device,
) -> dict:
    """
    Convert per-image target dicts into the flat-batch format that
    Ultralytics' v8DetectionLoss / E2ELoss expects.

    Dataloader returns:

        targets = [
            {'boxes': Tensor(N_i, 4), 'labels': Tensor(N_i,)},  # image 0
            {'boxes': Tensor(N_j, 4), 'labels': Tensor(N_j,)},  # image 1
            ...
        ]

    Ultralytics expects:

        batch = {
            'img':       Tensor(B, 3, H, W),
            'batch_idx': Tensor(N_total,),   # which image each box belongs to
            'cls':       Tensor(N_total, 1),
            'bboxes':    Tensor(N_total, 4),  # [cx, cy, w, h] normalised
        }
    
    Args:
        images (torch.Tensor): Batch of images.
        targets (list[dict]): List of target dicts.
        device (torch.device): Device to move tensors to.
    
    Returns:
        dict: Batch in Ultralytics format.
    """
    # Initialize lists to store batch information
    batch_idx_list = []
    cls_list = []
    bbox_list = []
    has_gt_list = []

    for i, t in enumerate(targets):
        # Extract has_gt if present, otherwise default to True
        has_gt = t.get('has_gt', torch.tensor([True], dtype=torch.bool))
        has_gt_list.append(has_gt)

        # Get the number of ground truth boxes in the current image
        n = t['labels'].shape[0]
        if n == 0:
            continue

        # Append batch information to lists
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        cls_list.append(t['labels'].unsqueeze(-1).long())   # (N, 1)
        bbox_list.append(t['boxes'])                         # (N, 4)

    # Concatenate all lists into single tensors
    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list, dim=0).to(device)
        cls = torch.cat(cls_list, dim=0).to(device)
        bboxes = torch.cat(bbox_list, dim=0).to(device)
    else:
        # Edge case: no ground-truth in this batch
        batch_idx = torch.zeros(0, dtype=torch.long, device=device)
        cls = torch.zeros(0, 1, dtype=torch.long, device=device)
        bboxes = torch.zeros(0, 4, dtype=torch.float32, device=device)

    # Return batch in Ultralytics format
    return {
        'img': images.to(device),
        'batch_idx': batch_idx,
        'cls': cls,
        'bboxes': bboxes,
        'has_gt': torch.cat(has_gt_list, dim=0).to(device) if has_gt_list else torch.ones(len(images), dtype=torch.bool, device=device),
    }


def train_yolo_manual():
    """
    Manual PyTorch training loop for YOLO model.
    """
    print("--- Starting YOLO Manual Training Loop ---")

    # Get device and hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyp = cfg_manual.yolo_train_config["hyp"]

    regime = resolve_training_regime(
        getattr(cfg_manual, "training_regime", None), cfg_manual.kd_config
    )
    if cfg_manual.model_format != "yolo_manual":
        raise ValueError(
            "train_yolo_manual requires model_format='yolo_manual'; got "
            f"{cfg_manual.model_format!r}."
        )

    kd_feature_based = cfg_manual.kd_config.get("feature_based", {})
    kd_pseudo = cfg_manual.kd_config.get("pseudo_label", {})
    kd_pseudo_soft = cfg_manual.kd_config.get("pseudo_label_soft", {})
    feature_kd_enabled = bool(
        regime.use_aux_kd and kd_feature_based.get("enabled", False)
    )
    # In teacher-only, hard pseudo detection is the base objective and must not
    # also run through the legacy GT-anchored auxiliary branch.
    # Teacher-only uses hard pseudo detection as its base objective. Semi-
    # supervised runs retain the GT-anchored auxiliary on their labeled split.
    hard_kd_enabled = bool(
        regime.mode in {"hybrid", "semi_supervised"}
        and regime.use_aux_kd
        and kd_pseudo.get("enabled", False)
    )
    soft_kd_enabled = bool(
        regime.use_aux_kd and kd_pseudo_soft.get("enabled", False)
    )
    kd_enabled = regime.kd_pipeline_enabled
    print(
        f"Training regime: {regime.mode} "
        f"(ground truth={'on' if regime.use_ground_truth else 'off'}, "
        f"teacher base={'on' if regime.use_teacher_pseudo_base else 'off'})."
    )

    # Any run that reads a teacher must agree with it about how many annotations
    # this experiment is allowed. Checked here, before the dataloader, the cache
    # or the model are built, so a mismatch costs seconds rather than an epoch.
    run_budget = label_budget_meta.budget_record(cfg_manual)
    regime_suffix = training_regime_run_suffix(regime, cfg_manual.kd_config)
    teacher_record = None
    if kd_enabled:
        teacher_weights = cfg_manual.teacher["weights"]
        teacher_record = {
            "path": os.path.abspath(teacher_weights),
            "sha256": file_sha256(teacher_weights),
            "variant": cfg_manual.teacher["variant"],
        }
        teacher_ckpt = torch.load(teacher_weights, map_location="cpu", weights_only=False)
        label_budget_meta.verify_teacher_budget(
            label_budget_meta.read_budget_record(teacher_ckpt),
            run_budget,
            teacher_weights,
        )
        del teacher_ckpt
        print(f"Annotation budget: {label_budget_meta.describe(run_budget)} (teacher agrees).")

        from kd_cache import ensure_teacher_caches

        ensure_teacher_caches(
            cache_dir=cfg_manual.teacher_cache_dir,
            weights_path=teacher_weights,
            train_needed=bool(regime.use_teacher_pseudo_base or hard_kd_enabled or soft_kd_enabled),
            val_needed=bool(regime.validation_targets in {"teacher", "ground_truth_teacher"}),
        )
    else:
        print(f"Annotation budget: {label_budget_meta.describe(run_budget)}.")

    # --- DataLoader Setup ---
    # Validation goes through run_evaluation(), which builds its own loader from
    # cfg_manual.val_json, so only the training loader is used here.
    train_json = training_manifest_for_regime(regime, cfg_manual)
    if regime.mode == "hybrid":
        # Keep the legacy call completely unchanged. Its None/default options
        # resolve through the same global helpers as before this knob existed.
        train_loader, _ = build_dataloaders(
            train_json,
            cfg_manual.val_json,
        )
    else:
        response_thresholds = [regime.teacher_conf_threshold]
        if soft_kd_enabled:
            response_thresholds.append(
                float(kd_pseudo_soft.get("teacher_conf_threshold", 0.0))
            )
        if hard_kd_enabled:
            response_thresholds.append(
                float(kd_pseudo.get("teacher_conf_threshold", 0.0))
            )
            
        train_labeled_image_ids = None
        if regime.is_semi_supervised:
            from pycocotools.coco import COCO
            coco_train = COCO(cfg_manual.train_json)
            # COCO.imgs holds the IDs of images in the budget
            train_labeled_image_ids = frozenset(coco_train.imgs.keys())
            
        train_loader, _ = build_dataloaders(
            train_json,
            cfg_manual.val_json,
            train_use_ground_truth=regime.use_ground_truth,
            train_labeled_image_ids=train_labeled_image_ids,
            train_enable_teacher_cache=regime.use_teacher_pseudo_base,
            train_emit_teacher_view=(
                feature_kd_enabled
                and kd_feature_based.get("teacher_clean_view", True)
            ),
            train_teacher_conf_threshold=(
                min(response_thresholds)
                if regime.use_teacher_pseudo_base
                else None
            ),
            require_empty_train_annotations=regime.require_empty_annotations,
        )
        if regime.is_semi_supervised:
            n_total = len(train_loader.dataset)
            n_labeled = len(train_labeled_image_ids.intersection(set(train_loader.dataset.image_ids)))
            print(f"Semi-supervised pool: {n_labeled:,} labeled images, {n_total - n_labeled:,} unlabeled images.")

    train_cache_record = None
    if getattr(train_loader.dataset, "teacher_cache", None) is not None:
        cache_meta = train_loader.dataset.teacher_cache.meta
        train_cache_record = {
            "path": cache_meta["artifact_path"],
            "sha256": cache_meta["artifact_sha256"],
            "schema_version": cache_meta.get("schema_version"),
            "state_source": cache_meta.get("state_source"),
            "input_hw": cache_meta.get("input_hw"),
            "num_classes": cache_meta.get("num_classes"),
            "top_k": cache_meta.get("top_k"),
            "min_confidence": cache_meta.get("min_confidence"),
        }

    # Resolve what validation scores against. Built once, before training, so a
    # cache or coverage problem fails immediately rather than at the first
    # epoch boundary -- and so the exact target set the run was selected
    # against survives on disk for inspection afterwards.
    validation_source = validation_manifest_for_regime(regime, cfg_manual)
    validation_json = validation_source
    validation_cache_record = None
    if regime.validation_targets in ("teacher", "ground_truth_teacher"):
        from kd_cache import default_cache_path, write_teacher_pseudo_annotations

        validation_labeled_image_ids = None
        if regime.validation_targets == "ground_truth_teacher":
            from pycocotools.coco import COCO
            coco_val = COCO(cfg_manual.val_json)
            validation_labeled_image_ids = frozenset(coco_val.imgs.keys())

        validation_cache_path = default_cache_path(
            cfg_manual.teacher_cache_dir, "val", cfg_manual.teacher["weights"]
        )
        validation_cache_sha = file_sha256(validation_cache_path)
        validation_cache_record = {
            "path": os.path.abspath(validation_cache_path),
            "sha256": validation_cache_sha,
        }
        validation_output = content_addressed_path(
            regime.validation_pseudo_json,
            run_budget,
            f"cache-{validation_cache_sha}",
            f"conf-{regime.validation_teacher_conf_threshold:g}",
        )
        summary = write_teacher_pseudo_annotations(
            cache_path=validation_cache_path,
            source_json=validation_source,
            output_json=validation_output,
            conf_threshold=regime.validation_teacher_conf_threshold,
            weights_path=cfg_manual.teacher["weights"],
            input_hw=(cfg_manual.input_height, cfg_manual.input_width),
            num_classes=cfg_manual.num_classes,
            expected_cache_sha256=validation_cache_sha,
            labeled_image_ids=validation_labeled_image_ids,
        )
        validation_json = summary["path"]
        _val_desc = (
            "mixed ground-truth / teacher predictions" 
            if regime.validation_targets == "ground_truth_teacher" 
            else "teacher predictions"
        )
        print(
            f"Validation targets: {_val_desc} -> {summary['path']} "
            f"({summary['annotations']:,} boxes over {summary['images']:,} images, "
            f"{summary['empty_images']:,} with none above conf "
            f"{regime.validation_teacher_conf_threshold})."
        )
        print(
            "  Reported as Validation/teacher_agreement_* : this ranks epochs by "
            "agreement with the teacher, not by accuracy."
        )
    else:
        print(f"Validation targets: ground truth ({validation_json}).")
    validation_record = {
        **manifest_summary(validation_json),
        "targets": regime.validation_targets,
        "subsample_interval": _validation_subsample_interval(),
    }
    if validation_cache_record is not None:
        validation_record["teacher_cache"] = validation_cache_record

    run_provenance = build_training_run_provenance(
        backend="yolo_manual",
        budget=run_budget,
        train_manifest=train_json,
        validation=validation_record,
        regime=regime,
        kd_config=cfg_manual.kd_config,
        train_subsample_interval=cfg_manual.dataloader_config.get(
            "frame_subsample_interval", 1
        ),
        seed=cfg_manual.shared_train_config.get("seed", 42),
        teacher=teacher_record,
        teacher_cache=train_cache_record,
    )
    tensorboard_dir = os.path.join(
        "runs",
        "yolo_manual",
        f"{regime.mode}{regime_suffix}",
        budget_run_tag(run_budget),
        run_provenance["tag"],
    )
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # --- Model Setup ---
    # Load YOLO model
    model = DetectionModel(cfg_manual.yolo_train_config["model_yaml_path"], ch=3, nc=cfg_manual.num_classes)

    loaded_keys = set()
    # Load pretrained weights. Mismatching pretrained layers will be ignored.
    if os.path.exists(cfg_manual.yolo_train_config["pretrained_model_path"]):
        ckpt = torch.load(cfg_manual.yolo_train_config["pretrained_model_path"], map_location="cpu", weights_only=False)
        ckpt_model = ckpt.get("model", ckpt)
        if hasattr(ckpt_model, 'state_dict'):
            csd = ckpt_model.float().state_dict()
        else:
            csd = ckpt_model
        updated = intersect_dicts(csd, model.state_dict())
        model.load_state_dict(updated, strict=False)
        loaded_keys = set(updated.keys())
        print(f"Loaded {len(updated)}/{len(model.state_dict())} layers from {cfg_manual.yolo_train_config['pretrained_model_path']}")
    else:
        print(f"WARNING: {cfg_manual.yolo_train_config['pretrained_model_path']} not found, training from scratch.")

    # Send model to device
    model.to(device)

    # --- KD Setup ---
    feature_kd_module = None
    # Bound unconditionally: the probe's guard is evaluated on every batch, long
    # before the branch that would otherwise set it.
    feature_grad_probe_interval = 0
    if kd_enabled:
        print("Knowledge Distillation Enabled.")
        # Only feature KD needs the teacher resident on the GPU. The response
        # methods read their predictions from the offline cache, so building a
        # ~686 MB RT-DETR they will never call wastes both the load time and the
        # VRAM -- which is the memory the cache exists to give back.
        live_teacher_needed = feature_kd_enabled
        teacher_type = cfg_manual.teacher["model"]
        if teacher_type != "rtdetr":
            # Everything below -- the encoder feature hook, the teacher's
            # ImageNet re-normalization, the pred_logits/pred_boxes contract --
            # is RT-DETR specific, so anything else would fail later with a
            # NameError instead of saying what is unsupported.
            raise ValueError(
                f"Unsupported teacher model '{teacher_type}'; only 'rtdetr' is implemented."
            )
        teacher_variant = cfg_manual.teacher["variant"]
        teacher_model = None
        if not live_teacher_needed:
            print(
                "KD: feature KD is off, so no live teacher is built -- the response "
                f"methods read cached predictions from {cfg_manual.teacher_cache_dir}/."
            )
        else:
            teacher_model = build_rtdetr_model(variant=teacher_variant, pretrained_backbone=False, num_classes=cfg_manual.num_classes)
        if teacher_model is not None:
            if os.path.exists(cfg_manual.teacher["weights"]):
                ckpt = torch.load(cfg_manual.teacher["weights"], map_location="cpu", weights_only=False)
                t_state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))

                # Filter mismatched and build dictionaries
                model_state = teacher_model.state_dict()
                filtered = {k: v for k, v in t_state.items() if k in model_state and v.shape == model_state[k].shape}

                # Ensure 100% of the teacher's weights are loaded
                missing_keys = set(model_state.keys()) - set(filtered.keys())
                if missing_keys:
                    raise RuntimeError(
                        f"Teacher weight mismatch: {len(missing_keys)} keys missing or mismatched in shape. "
                        f"In Knowledge Distillation, the teacher must be fully trained on the target dataset, "
                        f"so its checkpoint must exactly match the model architecture. "
                        f"Missing/mismatched examples: {list(missing_keys)[:5]}"
                    )

                teacher_model.load_state_dict(filtered, strict=True)
                print(f"Loaded KD Teacher weights from {cfg_manual.teacher['weights']}")
            else:
                raise RuntimeError(f"Teacher weights not found at {cfg_manual.teacher['weights']}. KD cannot proceed with a random teacher.")
            teacher_model.to(device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

        if feature_kd_enabled:
            # Extract student channels from the model's detection head
            try:
                # model.model[-1] is typically the Detection/Segment head in YOLO26
                student_head = model.model[-1]
                student_channels = [student_head.cv2[i][0].conv.in_channels for i in range(student_head.nl)]
                print(f"Found student channels from model head: {student_channels}")
            except Exception as e:
                raise RuntimeError(f"Failed to extract student channels from detection head. "
                                   f"Ensure the model architecture exposes 'model.model[-1].cv2' or adjust channel extraction. Error: {e}")
            # Extract teacher channels dynamically (RT-DETR encoder outputs a list of channel dimensions)
            teacher_channels = teacher_model.encoder.out_channels
                
            feature_kd_module = FeatureKD(
                student_channels=student_channels,
                teacher_channels=teacher_channels,
                scale_weights=kd_feature_based.get("scale_weights"),
                adapter=kd_feature_based.get("adapter", "conv1x1"),
                adapter_norm=kd_feature_based.get("adapter_norm", "group"),
                attention_temperature=kd_feature_based.get("attention_temperature", 0.5),
                fg_weight=kd_feature_based.get("fg_weight", 1.0),
                bg_weight=kd_feature_based.get("bg_weight", 0.5),
                attention_weight=kd_feature_based.get("attention_weight", 1.0),
                global_weight=kd_feature_based.get("global_weight", 0.5),
            ).to(device)
            kd_feature_weight = kd_feature_based.get("kd_weight", 1.0)
            feature_warmup_epochs = kd_feature_based.get("kd_warmup_epochs", 5)
            feature_ramp_epochs = kd_feature_based.get("kd_ramp_epochs", 0)
            feature_fade_epochs = kd_feature_based.get("kd_fade_epochs", 0)
            feature_cooldown_epochs = kd_feature_based.get("kd_cooldown_epochs", 0)
            # Two extra backward passes, so this is sampled rather than run
            # every step. 0 disables it.
            feature_grad_probe_interval = int(
                kd_feature_based.get("grad_probe_interval", 0)
            )
            if not regime.use_ground_truth:
                # The existing probe compares KD against a GT gradient. There
                # is no honest supervised reference in teacher-only training.
                feature_grad_probe_interval = 0
            
            # Register a hook on the teacher's encoder to intercept multi-scale features
            teacher_feats_cache = []
            def teacher_encoder_hook(module, input, output):
                teacher_feats_cache.clear()
                # RT-DETR encoder outputs a list of 3 feature maps
                teacher_feats_cache.extend(output)
            
            teacher_model.encoder.register_forward_hook(teacher_encoder_hook)
            
            print(f"Feature-Based KD Initialized (weight: {kd_feature_weight}, "
                  f"warmup: {feature_warmup_epochs} epochs, ramp: {feature_ramp_epochs} epochs, "
                  f"fade: {feature_fade_epochs} epochs, cooldown: {feature_cooldown_epochs} epochs). "
                  "Projection adapters created.")

        if hard_kd_enabled:
            pseudo_conf_thresh = kd_pseudo.get("teacher_conf_threshold", 0.5)
            pseudo_weight = kd_pseudo.get("kd_weight", 0.5)
            pseudo_warmup_epochs = kd_pseudo.get("kd_warmup_epochs", 5)
            pseudo_ramp_epochs = kd_pseudo.get("kd_ramp_epochs", 0)
            pseudo_fade_epochs = kd_pseudo.get("kd_fade_epochs", 0)
            pseudo_cooldown_epochs = kd_pseudo.get("kd_cooldown_epochs", 0)
            pseudo_gt_match_iou = kd_pseudo.get("gt_match_iou", 0.5)
            pseudo_unmatched = kd_pseudo.get("unmatched_teacher_boxes", "drop")
            if pseudo_unmatched not in {"drop", "add"}:
                raise ValueError(
                    "kd_config['pseudo_label']['unmatched_teacher_boxes'] must be "
                    f"'drop' or 'add'; got {pseudo_unmatched!r}."
                )
            print(f"Two-Pass Hard Pseudo-Labeling KD Initialized (weight: {pseudo_weight}, "
                  f"conf_thresh: {pseudo_conf_thresh}, warmup: {pseudo_warmup_epochs} epochs, "
                  f"ramp: {pseudo_ramp_epochs} epochs, fade: {pseudo_fade_epochs} epochs, "
                  f"cooldown: {pseudo_cooldown_epochs} epochs, gt_match_iou: {pseudo_gt_match_iou}, "
                  f"unmatched teacher boxes: {pseudo_unmatched}).")

        if soft_kd_enabled:
            pseudo_soft_conf_thresh = kd_pseudo_soft.get("teacher_conf_threshold", 0.5)
            pseudo_soft_temperature = kd_pseudo_soft.get("temperature", 2.5)
            pseudo_soft_weight = kd_pseudo_soft.get("kd_weight", 0.5)
            pseudo_soft_warmup_epochs = kd_pseudo_soft.get("kd_warmup_epochs", 5)
            pseudo_soft_ramp_epochs = kd_pseudo_soft.get("kd_ramp_epochs", 0)
            pseudo_soft_fade_epochs = kd_pseudo_soft.get("kd_fade_epochs", 0)
            pseudo_soft_cooldown_epochs = kd_pseudo_soft.get("kd_cooldown_epochs", 0)
            pseudo_soft_conf_gain = kd_pseudo_soft.get("confidence_weight", 1.0)
            pseudo_soft_relation_gain = kd_pseudo_soft.get("relation_weight", 1.0)
            if pseudo_soft_conf_gain == 0.0 and pseudo_soft_relation_gain == 0.0:
                raise ValueError(
                    "kd_config['pseudo_label_soft'] has both confidence_weight and "
                    "relation_weight at 0.0, which disables the term entirely. "
                    "Set `enabled: False` instead."
                )
            pseudo_soft_heads = kd_pseudo_soft.get("distill_heads", "both")
            if pseudo_soft_heads not in {"both", "one2one", "one2many"}:
                raise ValueError(
                    "kd_config['pseudo_label_soft']['distill_heads'] must be "
                    f"'both', 'one2one' or 'one2many'; got {pseudo_soft_heads!r}."
                )
            print(f"Two-Pass Soft Pseudo-Labeling KD Initialized (weight: {pseudo_soft_weight}, "
                  f"conf_thresh: {pseudo_soft_conf_thresh}, temp: {pseudo_soft_temperature}, "
                  f"warmup: {pseudo_soft_warmup_epochs} epochs, ramp: {pseudo_soft_ramp_epochs} epochs, "
                  f"fade: {pseudo_soft_fade_epochs} epochs, cooldown: {pseudo_soft_cooldown_epochs} epochs, "
                  f"heads: {pseudo_soft_heads}, conf gain: {pseudo_soft_conf_gain}, "
                  f"relation gain: {pseudo_soft_relation_gain} @ T={pseudo_soft_temperature}).")

        # Pre-compute teacher normalization tensors (convert YOLO [0,1] → ImageNet convention)
        # The dataloader normalizes images for the student (YOLO: divide-by-255 only).
        # The teacher (RT-DETR) expects ImageNet mean/std, so re-normalize at forward time.
        teacher_norm = cfg_manual.augmentation["normalize"]["rtdetr"]
        student_norm = cfg_manual.augmentation["normalize"][cfg_manual.model_format]
        teacher_mean = torch.tensor(teacher_norm["mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        teacher_std  = torch.tensor(teacher_norm["std"],  device=device, dtype=torch.float32).view(1, 3, 1, 1)
        student_mean = torch.tensor(student_norm["mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        student_std  = torch.tensor(student_norm["std"],  device=device, dtype=torch.float32).view(1, 3, 1, 1)

    # --- Loss Setup ---
    # Set loss gains and epochs (number of epochs is needed for the o2m weight decay schedule)
    model.args = SimpleNamespace(
        box=hyp["box"],
        cls=hyp["cls"],
        dfl=hyp["dfl"],
        fl_gamma=hyp.get("fl_gamma", 0.0),
        epochs=cfg_manual.shared_train_config["epochs"],
    )

    # Inverse-frequency class weighting. The magnitude-preserving path needs the
    # unreduced loss map, so it wraps the BCE modules after the criterion exists;
    # the Ultralytics-parity path goes through model.class_weights, which
    # v8DetectionLoss reads in __init__ and so must be set before it is built.
    class_weights = (
        compute_class_weights(
            train_loader.dataset, float(hyp.get("cls_pw", 0.0)), device
        )
        if regime.use_ground_truth
        else None
    )
    preserve_cls_magnitude = bool(hyp.get("cls_pw_preserve_magnitude", True))
    if class_weights is not None and not preserve_cls_magnitude:
        set_class_weights(model, class_weights)

    # Initialize loss function
    if getattr(model, 'end2end', False):
        criterion = E2ELoss(model)
        print("Loss: E2ELoss (one-to-many + one-to-one with decay)")
    else:
        criterion = v8DetectionLoss(model)
        print("Loss: v8DetectionLoss (standard)")

    # Ultralytics 8.4.x ignores hyp.fl_gamma, so install Focal Loss ourselves.
    # This has to reach the per-branch BCE modules, not just the top-level
    # criterion -- see apply_focal_loss().
    fl_gamma = float(hyp.get("fl_gamma", 0.0))
    fl_alpha = float(hyp.get("fl_alpha", 0.0))
    if fl_gamma > 0.0:
        wrapped = apply_focal_loss(criterion, fl_gamma, fl_alpha)
        print(f"  Focal Loss gamma={fl_gamma} alpha={fl_alpha} applied to: {', '.join(wrapped)}")

    # Outermost, so the weights scale the focal-modulated map
    if class_weights is not None and preserve_cls_magnitude:
        wrapped = apply_class_weights(criterion, class_weights, preserve_magnitude=True)
        print(f"  Class weights (magnitude-preserving) applied to: {', '.join(wrapped)}")

    # --- Optimizer Setup ---
    # Collect parameter groups:
    # 1. Pretrained parameters (backbone, neck, and bbox regression heads) - non-norm: lr=pretrained_lr, wd=weight_decay
    # 2. Pretrained parameters (backbone, neck, and bbox regression heads) - norm/bias: lr=pretrained_lr, wd=0.0
    # 3. Newly initialized parameters (e.g. classification heads) - non-norm: lr=base_lr, wd=weight_decay
    # 4. Newly initialized parameters (e.g. classification heads) - norm/bias: lr=base_lr, wd=0.0

    base_lr = hyp["lr0"]
    weight_decay = hyp["weight_decay"]
    pretrained_lr_ratio = cfg_manual.shared_train_config.get("pretrained_lr_ratio", 0.1)
    pretrained_lr = base_lr * pretrained_lr_ratio

    groups = {
        "pretrained_decay": {"params": [], "lr": pretrained_lr, "weight_decay": weight_decay, "is_pretrained": True, "unreduced_lr": base_lr, "is_bias": False},
        "pretrained_norm":  {"params": [], "lr": pretrained_lr, "weight_decay": 0.0, "is_pretrained": True, "unreduced_lr": base_lr, "is_bias": False},
        "pretrained_bias":  {"params": [], "lr": pretrained_lr, "weight_decay": 0.0, "is_pretrained": True, "unreduced_lr": base_lr, "is_bias": True},
        "head_decay":       {"params": [], "lr": base_lr, "weight_decay": weight_decay, "is_pretrained": False, "unreduced_lr": base_lr, "is_bias": False},
        "head_norm":        {"params": [], "lr": base_lr, "weight_decay": 0.0, "is_pretrained": False, "unreduced_lr": base_lr, "is_bias": False},
        "head_bias":        {"params": [], "lr": base_lr, "weight_decay": 0.0, "is_pretrained": False, "unreduced_lr": base_lr, "is_bias": True},
    }

    norm_param_ids = set()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            for p in m.parameters(recurse=False):
                norm_param_ids.add(id(p))
                
    if feature_kd_enabled and feature_kd_module is not None:
        for m in feature_kd_module.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm)):
                for p in m.parameters(recurse=False):
                    norm_param_ids.add(id(p))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = name.endswith('.bias')
        is_norm = id(param) in norm_param_ids
        is_head = name not in loaded_keys
        
        if is_head:
            if is_bias:
                groups["head_bias"]["params"].append(param)
            elif is_norm:
                groups["head_norm"]["params"].append(param)
            else:
                groups["head_decay"]["params"].append(param)
        else:
            if is_bias:
                groups["pretrained_bias"]["params"].append(param)
            elif is_norm:
                groups["pretrained_norm"]["params"].append(param)
            else:
                groups["pretrained_decay"]["params"].append(param)

    # Add FeatureKD parameters to the newly initialized (head) parameter groups
    if feature_kd_enabled:
        for name, param in feature_kd_module.named_parameters():
            if not param.requires_grad:
                continue
            is_bias = name.endswith('.bias')
            is_norm = id(param) in norm_param_ids
            if is_bias:
                groups["head_bias"]["params"].append(param)
            elif is_norm:
                groups["head_norm"]["params"].append(param)
            else:
                groups["head_decay"]["params"].append(param)

    # Filter out empty groups
    param_groups = [v for v in groups.values() if len(v["params"]) > 0]

    opt_type = hyp.get("optimizer", "SGD")
    if opt_type == "AdamW":
        optimizer = AdamW(
            param_groups,
            lr=base_lr,
            betas=(hyp["momentum"], 0.999),
            weight_decay=weight_decay,
        )
    else:
        optimizer = SGD(
            param_groups,
            lr=base_lr,
            momentum=hyp["momentum"],
            nesterov=True,
            weight_decay=weight_decay,
        )

    # --- LR Schedule ---
    # Cosine annealing from lr0 -> lr0 * lrf
    lf = lambda x: ((1 - math.cos(x * math.pi / cfg_manual.shared_train_config["epochs"])) / 2) * (hyp["lrf"] - 1) + 1
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

    # Warmup config
    warmup_iters = max(round(hyp["warmup_epochs"] * len(train_loader)), 100)
    print(f"Warmup: {warmup_iters} iterations ({hyp['warmup_epochs']} epochs)")

    # --- EMA + AMP ---
    amp_cfg = get_amp_settings()
    ema = ModelEMA(model, decay=cfg_manual.shared_train_config["ema_decay"], warmups=cfg_manual.shared_train_config["ema_warmups"])
    scaler = GradScaler(enabled=amp_cfg["scaler_enabled"] and torch.cuda.is_available()) # Explicitly disable scaler when CPU is used

    # --- Training Loop ---

    # Initialize tracking variables 
    global_step = 0
    best_map = 0.0
    start_epoch = 0
    patience = cfg_manual.shared_train_config.get("early_stopping_patience", 10)
    patience_counter = 0
    # Clamped to at least 1: the step condition below divides by it, and
    # optimizer_steps_per_epoch() already treats anything smaller as "every batch".
    ACCUMULATION_STEPS = max(1, int(cfg_manual.shared_train_config.get("gradient_accumulation_steps", 1)))
    consecutive_inf = 0          # track consecutive inf batches
    MAX_CONSECUTIVE_INF = 10     # exit only after this many in a row

    # Student parameters the KD gradient probe measures over. The projection
    # adapters are deliberately left out: the question the probe answers is how
    # hard the KD term pulls on the *student*, and an adapter with enough
    # capacity to absorb the objective internally would otherwise hide exactly
    # that by satisfying the loss without the student's features moving.
    probe_params = [p for p in model.parameters() if p.requires_grad]
    grad_probe_warned = False

    # Checkpoints are tagged with the KD methods that produced them, so runs of
    # different variants cannot overwrite one another and a resume can only
    # pick up a run of the same kind.
    yolo_best_name = budget_named_path(
        f"yolo_manual_best{regime_suffix}.pth", run_budget
    )
    print(f"Checkpoints for this run: {yolo_best_name} / {yolo_best_name.replace('best', 'last')}")

    if cfg_manual.shared_train_config.get("resume_training", False):
        last_ckpt = os.path.join(cfg_manual.weights_dir, yolo_best_name.replace("best", "last"))
        best_ckpt = os.path.join(cfg_manual.weights_dir, yolo_best_name)
        resume_ckpt = last_ckpt if os.path.exists(last_ckpt) else best_ckpt
        
        if os.path.exists(resume_ckpt):
            print(f"Resuming YOLO manual training from {resume_ckpt}...")
            ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            label_budget_meta.verify_budget_match(
                label_budget_meta.read_budget_record(ckpt),
                run_budget,
                resume_ckpt,
                artifact_label="Resume checkpoint",
            )
            verify_run_provenance(
                ckpt.get(RUN_PROVENANCE_KEY), run_provenance, resume_ckpt
            )
            saved_regime = ckpt.get('training_regime')
            if saved_regime is not None and saved_regime.get('mode') != regime.mode:
                # gt_only and hybrid-with-KD-disabled are the same numerical
                # regime and deliberately share the no-suffix checkpoint name.
                # Other cross-regime resumes are unsafe and must fail before
                # any optimizer state is loaded.
                same_effective_supervision = (
                    bool(saved_regime.get('use_ground_truth', True))
                    == regime.use_ground_truth
                    and bool(saved_regime.get('kd_pipeline_enabled', False))
                    == regime.kd_pipeline_enabled
                )
                if not same_effective_supervision:
                    raise RuntimeError(
                        f"Checkpoint regime {saved_regime.get('mode')!r} does not "
                        f"match requested regime {regime.mode!r}."
                    )
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            if 'ema' in ckpt and 'module' in ckpt['ema']:
                ema.module.load_state_dict(ckpt['ema']['module'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
            if 'best_map' in ckpt:
                best_map = ckpt['best_map']
            if 'patience_counter' in ckpt:
                patience_counter = ckpt['patience_counter']
            if 'scaler' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                except Exception as e:
                    print(f"Failed to load scaler state: {e}")
            if 'kd_feature_state' in ckpt and feature_kd_enabled:
                feature_kd_module.load_state_dict(ckpt['kd_feature_state'])
                print("Loaded FeatureKD state.")
                
            if 'lr_scheduler' in ckpt:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                for _ in range(start_epoch):
                    if hasattr(criterion, 'update'):
                        criterion.update()
            else:
                # Backward compatibility
                for _ in range(start_epoch):
                    lr_scheduler.step()
                    if hasattr(criterion, 'update'):
                        criterion.update()

            # New checkpoints carry the exact count because teacher-silent
            # accumulation windows deliberately do not step. The formula is
            # retained only for checkpoints written before this metadata.
            ema._step = int(
                ckpt.get(
                    'ema_step',
                    start_epoch
                    * optimizer_steps_per_epoch(len(train_loader), ACCUMULATION_STEPS),
                )
            )
            global_step = start_epoch * len(train_loader)
        else:
            print(f"WARNING: Resume checkpoint not found at {resume_ckpt}.")

    # Training loop
    for epoch in range(start_epoch, cfg_manual.shared_train_config["epochs"]):
        model.train()
        if feature_kd_module is not None:
            # The projection adapters carry BatchNorm, so they need the same
            # mode as the student they are trained beside.
            feature_kd_module.train()
        optimizer.zero_grad()

        # Update pretrained LR ratio and apply fading if configured
        update_pretrained_lr_ratio(optimizer, lr_scheduler, None, epoch, cfg_manual.shared_train_config["epochs"])

        # Update augmentation schedule to disable mosaic and mixup at late epochs
        train_loader.dataset.update_aug_schedule(epoch, cfg_manual.shared_train_config["epochs"])
        
        epoch_total_loss = 0.0
        finite_batches = 0
        valid_batches_in_window = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_manual.shared_train_config['epochs']}", leave=False)
        for batch_idx, (images, targets) in enumerate(pbar):
            # Linear warmup (lr, momentum)
            ni = epoch * len(train_loader) + batch_idx  # total iterations since start of training
            if ni <= warmup_iters:
                alpha = ni / warmup_iters
                for pg in optimizer.param_groups:
                    # LR warmup: bias starts at warmup_bias_lr, others start at 0
                    lr_start = hyp["warmup_bias_lr"] if pg.get("is_bias", False) else 0.0
                    lr_end = pg['initial_lr'] * lf(epoch)
                    pg['lr'] = lr_start + (lr_end - lr_start) * alpha
                    
                    # Momentum warmup. SGD keeps it in 'momentum'; AdamW's
                    # equivalent is beta1, so warming only 'momentum' would
                    # silently do nothing under the configured AdamW.
                    warm_momentum = (hyp["warmup_momentum"]
                                     + (hyp["momentum"] - hyp["warmup_momentum"]) * alpha)
                    if 'momentum' in pg:
                        pg['momentum'] = warm_momentum
                    elif 'betas' in pg:
                        pg['betas'] = (warm_momentum, pg['betas'][1])

            # Collate to Ultralytics batch format
            batch = collate_to_ultralytics_format(images, targets, device)

            # Forward pass + compute loss
            with torch.autocast(device_type=device.type, dtype=amp_cfg["dtype"], enabled=amp_cfg["enabled"], cache_enabled=True):
                preds = model(batch["img"])
                
                batch_size = images.shape[0]
                parsed_full = None
                parsed = None

                if regime.use_ground_truth:
                    if regime.is_semi_supervised:
                        # Slice batch to only images with GT for supervised loss calculation
                        gt_idx = batch["has_gt"].nonzero(as_tuple=True)[0]
                        if gt_idx.numel() > 0:
                            if parsed_full is None:
                                parse_owner = criterion.one2many if isinstance(criterion, E2ELoss) else criterion
                                parsed_full = parse_owner.parse_output(preds)
                            
                            gt_batch = slice_ultralytics_batch(batch, gt_idx)
                            gt_preds = slice_parsed_preds(parsed_full, gt_idx)
                            
                            loss, loss_items = criterion(gt_preds, gt_batch)
                            
                        else:
                            loss = None
                            loss_items = {}
                    else:
                        # gt_only and hybrid supervise the whole batch, so the
                        # criterion takes the predictions and targets as-is.
                        loss, loss_items = criterion(preds, batch)
                else:
                    # Do not even invoke the criterion with the GT-shaped batch
                    # in teacher-only mode. Multiplying a GT loss by zero would
                    # still be a weaker and less auditable isolation boundary.
                    loss = None
                    loss_items = {}

                # Initialize KD loss tracking variables
                kd_feature_loss_val = 0.0
                current_feature_weight = 0.0
                # `kd_feature_total` is kept addressable so the gradient probe
                # below can differentiate the feature term on its own. The other
                # three track what the term actually contributed, as opposed to
                # what it was worth per eligible image.
                kd_feature_total = None
                kd_feature_exposure = 0.0
                kd_feature_effective = 0.0
                kd_feature_terms = None
                kd_soft_loss_val = 0.0
                current_soft_weight = 0.0
                kd_hard_loss_val = 0.0
                # Diagnostics for GT-anchored pseudo-labelling: how much of the
                # teacher's confident output the annotations actually corroborate.
                n_pseudo_matched = 0
                n_pseudo_added = 0
                current_hard_weight = 0.0
                kd_total = 0.0
                teacher_base_loss_val = 0.0
                teacher_base_target_count = 0
                teacher_base_empty_skipped = 0
                teacher_pseudo_batch = None
                teacher_effective_keep_idx = torch.zeros(
                    0, dtype=torch.long, device=device
                )
                # A semi-supervised batch may contain no labeled samples. A
                # teacher-silent `skip` batch therefore has no backwardable
                # objective even though the regime permits GT supervision.
                batch_has_supervision = loss is not None

                if kd_enabled:
                    # Check which KD methods are active for this batch. The
                    # per-method dicts are bound in the `kd_enabled` setup above,
                    # so they always exist on this branch; their weight and
                    # schedule variables are bound only when that method is
                    # enabled, hence the `enabled` guard on each factor.
                    #
                    # All three terms share `kd_ramp_factor`, so a method is
                    # active exactly when its scheduled weight is non-zero.
                    # Resolving the weights here rather than at each use site
                    # lets the teacher forward below be skipped entirely while
                    # every term is still held at zero.
                    epoch_progress = epoch + (batch_idx / len(train_loader))
                    total_train_epochs = cfg_manual.shared_train_config["epochs"]

                    if feature_kd_enabled:
                        current_feature_weight = kd_ramp_factor(
                            kd_feature_weight,
                            epoch_progress,
                            feature_warmup_epochs,
                            feature_ramp_epochs,
                            total_epochs=total_train_epochs,
                            fade_epochs=feature_fade_epochs,
                            cooldown_epochs=feature_cooldown_epochs,
                        )
                    kd_feature_based_active = current_feature_weight > 0.0

                    if hard_kd_enabled:
                        current_hard_weight = kd_ramp_factor(
                            pseudo_weight,
                            epoch_progress,
                            pseudo_warmup_epochs,
                            pseudo_ramp_epochs,
                            total_epochs=total_train_epochs,
                            fade_epochs=pseudo_fade_epochs,
                            cooldown_epochs=pseudo_cooldown_epochs,
                        )
                    kd_pseudo_active = current_hard_weight > 0.0

                    if soft_kd_enabled:
                        current_soft_weight = kd_ramp_factor(
                            pseudo_soft_weight,
                            epoch_progress,
                            pseudo_soft_warmup_epochs,
                            pseudo_soft_ramp_epochs,
                            total_epochs=total_train_epochs,
                            fade_epochs=pseudo_soft_fade_epochs,
                            cooldown_epochs=pseudo_soft_cooldown_epochs,
                        )
                    kd_pseudo_soft_active = current_soft_weight > 0.0

                    teacher_run_needed = (
                        regime.use_teacher_pseudo_base
                        or kd_feature_based_active
                        or kd_pseudo_active
                        or kd_pseudo_soft_active
                    )
                    teacher_preds = None

                    # Extract disable_kd mask (boolean tensor shape B). The
                    # dataloader's targets are on CPU, so the fallback has to be
                    # too -- torch.cat cannot mix CPU and CUDA tensors.
                    def _kd_mask(method: str) -> torch.Tensor:
                        # Per-method, because the three tolerate different
                        # augmentations: hard KD is GT-anchored and immune to
                        # mixup, feature KD distils the composite the student
                        # sees, and only soft KD carries an absolute-confidence
                        # target the blend invalidates. Falls back to the union
                        # flag for any target dict predating the split.
                        return torch.cat([
                            t.get(f"disable_kd_{method}", t.get("disable_kd", torch.tensor([False])))
                            for t in targets
                        ]).to(device)

                    disable_kd_feature = _kd_mask("feature")
                    disable_kd_hard = _kd_mask("hard")
                    disable_kd_soft = _kd_mask("soft")

                    # Response KD reads the teacher from the offline cache
                    # (`src/kd_cache.py`), already warped into this image's frame
                    # by the dataloader. The live teacher forward is therefore
                    # needed only by feature KD, which distils the composite the
                    # student actually sees.
                    cached_kd = all("kd_boxes" in t for t in targets)
                    if cached_kd:
                        teacher_run_needed = kd_feature_based_active

                    if teacher_run_needed:
                        # --- Teacher forward (frozen, no grad) ---
                        # Run teacher ONCE per batch and reuse across KD methods
                        with torch.no_grad():
                            # Prefer the dataloader's weak view: the student's
                            # exact geometry without the appearance corruption
                            # the teacher was never trained on. Falls back to the
                            # student's own image when the view is switched off.
                            if all("teacher_view" in t for t in targets):
                                teacher_source = torch.stack(
                                    [t["teacher_view"] for t in targets]
                                ).to(device, non_blocking=True)
                            else:
                                teacher_source = batch["img"]
                            # Re-normalize: undo student normalization, then apply teacher normalization
                            raw_input = teacher_source * student_std + student_mean
                            teacher_input = (raw_input - teacher_mean) / teacher_std
                            # This forward pass will also populate teacher_feats_cache via the hook if feature KD is enabled
                            teacher_preds = teacher_model(teacher_input)

                    if cached_kd:
                        # Pack the per-image cached predictions into the dense
                        # (B, Q, *) layout the response paths below already
                        # expect, so nothing downstream needs to know where the
                        # teacher came from. Padding rows carry a strongly
                        # negative logit, so their confidence is ~0 and the
                        # existing threshold filter drops them for free.
                        teacher_preds = pack_cached_teacher_preds(
                            targets, cfg_manual.num_classes, device
                        )

                    # Teacher-only base detection objective. It is deliberately
                    # outside every KD schedule: box/class/DFL supervision is at
                    # full configured strength from the first through last step.
                    if regime.use_teacher_pseudo_base:
                        if regime.is_semi_supervised:
                            # has_gt is a (B,) bool tensor from the collated targets.
                            unlabeled_idx = (~batch["has_gt"]).nonzero(as_tuple=True)[0]
                        else:
                            unlabeled_idx = torch.arange(batch_size, device=device)

                        if unlabeled_idx.numel() > 0:
                            with torch.no_grad():
                                teacher_logits = teacher_preds["pred_logits"]
                                teacher_boxes_cxcywh = teacher_preds["pred_boxes"]
                                teacher_probs = teacher_logits.sigmoid()
                                teacher_conf, teacher_classes = teacher_probs.max(dim=-1)
                                (
                                    teacher_pseudo_batch,
                                    teacher_effective_keep_idx,
                                    teacher_base_target_count,
                                    teacher_base_empty_skipped,
                                ) = build_teacher_pseudo_batch(
                                    batch,
                                    unlabeled_idx,
                                    teacher_boxes_cxcywh,
                                    teacher_classes,
                                    teacher_conf,
                                    regime.teacher_conf_threshold,
                                    regime.empty_target_policy,
                                )
    
                            if teacher_pseudo_batch is not None:
                                if parsed_full is None:
                                    parse_owner = (
                                        criterion.one2many
                                        if isinstance(criterion, E2ELoss)
                                        else criterion
                                    )
                                    parsed_full = parse_owner.parse_output(preds)
                                teacher_base_loss, loss_items = criterion(
                                    slice_parsed_preds(
                                        parsed_full, teacher_effective_keep_idx
                                    ),
                                    teacher_pseudo_batch,
                                )
                                if teacher_base_loss.dim() > 0:
                                    teacher_base_loss = teacher_base_loss.sum()
    
                                effective_count = teacher_effective_keep_idx.numel()
                                # The outer loop divides by B. `criterion` has
                                # already multiplied its mean by K effective
                                # images, so restore only the requested
                                # unlabeled fraction U/B. For teacher-only,
                                # U == B; for semi-supervised U may be smaller.
                                teacher_base_total = scale_teacher_pseudo_base_loss(
                                    teacher_base_loss,
                                    int(unlabeled_idx.numel()),
                                    effective_count,
                                )
                                kd_total = kd_total + teacher_base_total
                                teacher_base_loss_val = (
                                    teacher_base_loss / effective_count
                                ).item()
                                batch_has_supervision = True

                    # Feature-based KD
                    if kd_feature_based_active:
                        # Ensure student features are parsed
                        if parsed is None:
                            if isinstance(criterion, E2ELoss):
                                parsed_full = criterion.one2many.parse_output(preds)
                                parsed = parsed_full["one2many"]
                            else:
                                parsed_full = criterion.parse_output(preds)
                                parsed = parsed_full

                        student_feats = parsed["feats"]
                        teacher_feats = teacher_feats_cache
                        
                        if regime.is_semi_supervised:
                            feature_boxes, feature_batch_idx = (
                                semi_supervised_feature_targets(
                                    batch,
                                    teacher_pseudo_batch,
                                    teacher_effective_keep_idx,
                                )
                            )
                            feature_disable_mask = semi_supervised_feature_disable_mask(
                                disable_kd_feature,
                                batch["has_gt"],
                                teacher_effective_keep_idx,
                            )
                        elif regime.use_teacher_pseudo_base:
                            feature_boxes, feature_batch_idx = teacher_feature_targets(
                                teacher_pseudo_batch, teacher_effective_keep_idx
                            )
                            feature_disable_mask = teacher_feature_disable_mask(
                                disable_kd_feature, teacher_effective_keep_idx
                            )
                        else:
                            # Original hybrid feature masks remain GT-derived.
                            feature_boxes = batch["bboxes"]
                            feature_batch_idx = batch["batch_idx"]
                            feature_disable_mask = disable_kd_feature

                        feat_loss = feature_kd_module(
                            student_feats,
                            teacher_feats,
                            feature_boxes,
                            feature_batch_idx,
                            feature_disable_mask,
                        )

                        # Weight (including its warmup/ramp schedule) is resolved above.
                        # Preserve zero contribution from augmentation-masked
                        # images after the common division by `batch_size`.
                        if regime.use_teacher_pseudo_base:
                            feature_valid_count = (
                                ~feature_disable_mask
                            ).sum().to(feat_loss.dtype)
                        else:
                            feature_valid_count = (~disable_kd_feature).sum().to(feat_loss.dtype)
                        kd_feature_total = (
                            feat_loss * current_feature_weight * feature_valid_count
                        )
                        kd_total = kd_total + kd_feature_total
                        if feature_valid_count.item() > 0:
                            batch_has_supervision = True
                        kd_feature_loss_val = feat_loss.item()
                        # What the term was worth per *eligible* image is not
                        # what it added: `disable_on_augs` images contribute
                        # nothing, so the amount that actually reached the total
                        # is smaller by this fraction. Logging only the former
                        # (the historical behaviour) overstates the term's share
                        # of `Total_Loss` by 1 / exposure -- at mosaic p=0.5 and
                        # mixup p=0.15 that is a factor of 2.35.
                        kd_feature_exposure = (feature_valid_count / batch_size).item()
                        kd_feature_effective = (kd_feature_total / batch_size).item()
                        if feature_kd_module.last_terms is not None:
                            kd_feature_terms = feature_kd_module.last_terms.tolist()

                    # Two-Pass Hard Pseudo-Labeling KD
                    if kd_pseudo_active:
                        with torch.no_grad():

                            t_logits = teacher_preds["pred_logits"]  # (B, 300, nc)
                            t_boxes_cxcywh = teacher_preds["pred_boxes"]  # (B, 300, 4) normalised
                            
                            # Filter by confidence
                            t_probs = t_logits.sigmoid()
                            max_probs, class_idx = t_probs.max(dim=-1) # (B, 300)
                            
                            # A disable_kd image must be dropped from the second
                            # pass entirely, not merely left out of the
                            # pseudo-batch. v8DetectionLoss computes its
                            # classification BCE over every anchor of every image
                            # in the tensor it is handed, and images absent from
                            # `batch_idx` get `mask_gt = 0` -> `target_scores = 0`,
                            # so they are trained as pure background. Worse, its
                            # `target_scores_sum` normaliser counts only the
                            # images that do carry pseudo-boxes, so the fewer of
                            # those there are, the larger the background penalty
                            # on all the rest becomes.
                            #
                            if regime.is_semi_supervised:
                                # This auxiliary is GT-anchored; unlabeled
                                # images are supervised only by the pure-teacher
                                # base, not a second contradictory pass.
                                keep_idx = hard_kd_keep_indices(
                                    disable_kd_hard, batch["has_gt"]
                                )
                            else:
                                keep_idx = hard_kd_keep_indices(disable_kd_hard)

                            # Ground truth is authoritative on object presence,
                            # so an image the teacher had nothing confident to
                            # say about is never scored as empty: its
                            # annotations still stand, and only the boxes the
                            # teacher agrees on are replaced by its own.
                            pseudo_batch, n_pseudo_matched, n_pseudo_added = build_gt_anchored_pseudo_batch(
                                batch,
                                keep_idx,
                                t_boxes_cxcywh,
                                class_idx,
                                max_probs,
                                pseudo_conf_thresh,
                                pseudo_gt_match_iou,
                                pseudo_unmatched,
                            )

                        # Pass 2: compute loss with pseudo-batch, over the kept
                        # images only. Slicing the parsed head outputs keeps the
                        # loss identical in form to Pass 1 while confining it to
                        # images the teacher was allowed to label. The KD gradient
                        # therefore scales with how many of those there are, which
                        # is the intended behaviour.
                        if pseudo_batch is not None:
                            if parsed_full is None:
                                parse_owner = criterion.one2many if isinstance(criterion, E2ELoss) else criterion
                                parsed_full = parse_owner.parse_output(preds)
                            pseudo_loss, _ = criterion(slice_parsed_preds(parsed_full, keep_idx), pseudo_batch)
                            if pseudo_loss.dim() > 0:
                                pseudo_loss = pseudo_loss.sum()
                            kd_total = kd_total + (pseudo_loss * current_hard_weight)
                            batch_has_supervision = True
                            # Normalise by the kept count, not the full batch, so
                            # the logged value stays a per-image loss and does not
                            # drift with the mosaic/mixup rate.
                            kd_hard_loss_val = (pseudo_loss / keep_idx.numel()).item()

                     # Two-Pass Soft Pseudo-Labeling KD (Implementation Plan 2)
                    if kd_pseudo_soft_active:
                        
                        # --- Resolve the student head outputs ---
                        # E2ELoss.parse_output returns {"one2many": {...}, "one2one": {...}}
                        # v8DetectionLoss.parse_output returns {"boxes": ..., "scores": ..., "feats": ...}
                        if parsed is None:
                            if isinstance(criterion, E2ELoss):
                                parsed_full = criterion.one2many.parse_output(preds)
                                parsed = parsed_full["one2many"]
                            else:
                                parsed_full = criterion.parse_output(preds)
                                parsed = parsed_full

                        # --- Teacher forward + target prep (no grad needed) ---
                        with torch.no_grad():
                            t_logits = teacher_preds["pred_logits"]  # (B, 300, nc)
                            t_boxes_cxcywh = teacher_preds["pred_boxes"]  # (B, 300, 4) normalised
                            
                            H, W = images.shape[2], images.shape[3]
                            scale = torch.tensor([W, H, W, H], device=device, dtype=t_boxes_cxcywh.dtype)
                            t_boxes_xyxy = xywh2xyxy(t_boxes_cxcywh) * scale
                            
                            t_probs = t_logits.sigmoid()
                            max_probs, class_idx = t_probs.max(dim=-1) # (B, 300)
                            
                            
                            # Pad targets for the assigner
                            B_dim = images.shape[0]
                            nc = t_probs.shape[2]
                            
                            # Suppress targets for disable_kd images here rather
                            # than zeroing their loss after the fact. Masking the
                            # BCE alone left their teacher boxes in the assigner,
                            # so they still contributed to `target_scores_sum`
                            # below -- inflating the denominator of a numerator
                            # they had been removed from, and shrinking the soft
                            # KD term in proportion to the mosaic/mixup rate.
                            # Dropping them up front keeps both sides consistent
                            # and shrinks the target padding as a side effect.
                            kd_ok = (~disable_kd_soft).tolist()
                            b_objs = [
                                int((max_probs[b_i] > pseudo_soft_conf_thresh).sum().item()) if kd_ok[b_i] else 0
                                for b_i in range(B_dim)
                            ]
                            max_objs = max(b_objs) if b_objs else 0
                            
                            soft_kd_has_targets = max_objs > 0
                            if soft_kd_has_targets:
                                # fp32 pads: every consumer below casts to fp32
                                # anyway (the assigner's IoU maths and the BCE
                                # both require it), and building them at the
                                # student's AMP dtype only risks a mismatch.
                                t_labels_pad = torch.zeros((B_dim, max_objs, 1), device=device, dtype=torch.float32)
                                t_bboxes_pad = torch.zeros((B_dim, max_objs, 4), device=device, dtype=torch.float32)
                                # Raw logits, not softened probabilities: the
                                # confidence and relation terms need different
                                # temperatures applied to the same logits.
                                t_logits_pad = torch.zeros((B_dim, max_objs, nc), device=device, dtype=torch.float32)
                                mask_gt = torch.zeros((B_dim, max_objs), device=device, dtype=torch.bool)
                                
                                for b_i in range(B_dim):
                                    mask = max_probs[b_i] > pseudo_soft_conf_thresh
                                    num_obj = b_objs[b_i]
                                    if num_obj > 0:
                                        t_labels_pad[b_i, :num_obj, 0] = class_idx[b_i][mask].float()
                                        t_bboxes_pad[b_i, :num_obj] = t_boxes_xyxy[b_i][mask]
                                        t_logits_pad[b_i, :num_obj] = t_logits[b_i][mask].float()
                                        mask_gt[b_i, :num_obj] = True

                        # --- Assigner + loss computation (grad enabled for student) ---
                        if soft_kd_has_targets:
                            # Select the head(s) to distil. YOLO26 infers through
                            # one2one; one2many is auxiliary and E2ELoss decays
                            # it from 0.8 to 0.1 across training, so distilling
                            # it alone supervises a branch that is progressively
                            # switched off. When both are
                            # distilled they are combined with E2ELoss's own
                            # live gains, so the KD term follows exactly the
                            # same decay as the supervised loss it accompanies.
                            if isinstance(criterion, E2ELoss):
                                if pseudo_soft_heads == "both":
                                    soft_branches = [
                                        (parsed_full["one2many"], criterion.one2many, criterion.o2m),
                                        (parsed_full["one2one"], criterion.one2one, criterion.o2o),
                                    ]
                                elif pseudo_soft_heads == "one2one":
                                    soft_branches = [(parsed_full["one2one"], criterion.one2one, 1.0)]
                                else:
                                    soft_branches = [(parsed_full["one2many"], criterion.one2many, 1.0)]
                            else:
                                # Single-head model: `distill_heads` does not apply.
                                soft_branches = [(parsed_full, criterion, 1.0)]

                            soft_raw = None
                            for branch_parsed, branch_criterion, branch_gain in soft_branches:
                                branch_loss = soft_kd_branch_loss(
                                    branch_parsed,
                                    branch_criterion,
                                    t_labels_pad,
                                    t_bboxes_pad,
                                    t_logits_pad,
                                    mask_gt,
                                    pseudo_soft_temperature,
                                    pseudo_soft_conf_gain,
                                    pseudo_soft_relation_gain,
                                ) * branch_gain
                                soft_raw = branch_loss if soft_raw is None else soft_raw + branch_loss

                            # Ultralytics' native loss returns a value multiplied
                            # by batch size, and the caller divides the total by
                            # `batch_size` again -- so this term must be scaled
                            # the same way to survive that division at the right
                            # magnitude. Scale only by images that supplied at
                            # least one selected teacher target. A KD-eligible
                            # image with no selected target has an empty
                            # foreground mask and contributes to neither the
                            # numerator nor `target_scores_sum`; counting it
                            # here would transfer its missing weight onto the
                            # images that did contribute.
                            kd_soft_target_count = sum(count > 0 for count in b_objs)
                            kd_soft_total = (
                                soft_raw * current_soft_weight * kd_soft_target_count
                            )
                            kd_total = kd_total + kd_soft_total
                            kd_soft_loss_val = soft_raw.item()
                            batch_has_supervision = True

            if regime.use_ground_truth:
                # Preserve the original supervised/hybrid reduction and
                # addition order exactly.
                if loss is not None:
                    if loss.dim() > 0:
                        loss = loss.sum()
                    supervised_loss = loss
                else:
                    supervised_loss = None

                # Add total KD loss to scalar loss
                if kd_enabled:
                    if supervised_loss is not None:
                        loss = supervised_loss + kd_total
                    else:
                        loss = (
                            kd_total
                            if isinstance(kd_total, torch.Tensor)
                            else batch["img"].new_zeros(())
                        )
            else:
                supervised_loss = None
                # A teacher-silent `skip` batch has no legitimate objective.
                # Give the bookkeeping a scalar, but it is explicitly excluded
                # from backward/AdamW/EMA below.
                loss = (
                    kd_total
                    if isinstance(kd_total, torch.Tensor)
                    else batch["img"].new_zeros(())
                )

            # --- KD gradient probe -------------------------------------------
            # Sampled, because it costs two extra backward passes. Both terms are
            # divided by `batch_size` so the logged norms are per-image and stay
            # comparable across batch sizes; the ratio and cosine are invariant
            # to that anyway.
            kd_grad_norm = kd_grad_ratio = kd_grad_cosine = None
            if (
                feature_grad_probe_interval > 0
                and kd_feature_total is not None
                and kd_feature_total.requires_grad
                # A fully masked batch carries a graph-connected zero, so the
                # probe would spend two backward passes to report a ratio of 0
                # and drag the logged series down with a number that describes
                # the mask rather than the term.
                and kd_feature_exposure > 0.0
                and supervised_loss is not None
                and global_step % feature_grad_probe_interval == 0
            ):
                if scaler.is_enabled():
                    # Under fp16 the unscaled backward this probe performs can
                    # underflow in the activations, which would report a
                    # plausible-looking but wrong ratio. bf16 and fp32 need no
                    # scaler and are unaffected.
                    if not grad_probe_warned:
                        print("[WARNING] KD gradient probe skipped: it is unreliable "
                              "under fp16 loss scaling. Use bf16 or fp32 to enable it.")
                        grad_probe_warned = True
                else:
                    kd_grad_norm, kd_grad_ratio, kd_grad_cosine = kd_gradient_probe(
                        kd_feature_total / batch_size,
                        supervised_loss / batch_size,
                        probe_params,
                    )

            # Ultralytics YOLO loss is typically returned as a sum over the batch.
            # Convert it to a mean over the mini-batch, then scale by accumulation steps
            # to make it a true mean over the nominal batch size.
            loss_value = (loss / batch_size).item()

            # Check for inf/nan before backward() to avoid propagating corrupted
            # gradients. Only this batch is dropped: the check runs before
            # backward(), so nothing corrupt has entered the accumulation
            # window, and the gradients banked from its earlier batches are
            # still valid. The step below therefore still runs on schedule --
            # skipping it would carry those gradients into the next window and
            # double the effective step size.
            batch_is_finite = math.isfinite(loss_value)
            if not batch_has_supervision:
                # Expected when `empty_target_policy='skip'` and every image is
                # teacher-silent. Do not let weight decay or EMA turn a zero
                # target batch into a parameter update.
                consecutive_inf = 0
            elif batch_is_finite:
                consecutive_inf = 0
                scaled_loss = (loss / batch_size) / ACCUMULATION_STEPS
                scaler.scale(scaled_loss).backward()
                valid_batches_in_window += 1
            else:
                consecutive_inf += 1
                print(f"\n[WARNING] Inf/NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}. "
                      f"Skipping batch. ({consecutive_inf}/{MAX_CONSECUTIVE_INF})")
                if consecutive_inf >= MAX_CONSECUTIVE_INF:
                    print(f"Loss is not recoverable after {MAX_CONSECUTIVE_INF} consecutive inf batches. Stopping.")
                    sys.exit(1)

            # Perform optimizer step only after accumulating gradients for ACCUMULATION_STEPS batches
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                if valid_batches_in_window > 0:
                    # Unscale and renormalize a partial/filtered accumulation window.
                    scaler.unscale_(optimizer)
                    if feature_kd_enabled and feature_kd_module is not None:
                        params_to_clip = list(model.parameters()) + list(feature_kd_module.parameters())
                    else:
                        params_to_clip = list(model.parameters())
                    normalize_accumulated_gradients(
                        params_to_clip, ACCUMULATION_STEPS, valid_batches_in_window
                    )
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=hyp.get("max_norm", 10.0))

                    optimizer_stepped = step_optimizer_with_scaler(scaler, optimizer)
                    # EMA advances once per actual optimizer step. GradScaler
                    # may skip that step when fp16 gradients overflow.
                    if optimizer_stepped:
                        ema.update(model)

                optimizer.zero_grad()
                valid_batches_in_window = 0

            if not batch_has_supervision:
                global_step += 1
                continue

            # A skipped batch has no loss worth logging.
            if not batch_is_finite:
                continue

            epoch_total_loss += loss_value
            finite_batches += 1

            writer.add_scalar('Training/Total_Loss', loss_value, global_step)
            for loss_name, loss_item in detection_loss_items_for_logging(loss_items).items():
                writer.add_scalar(f'Training/{loss_name}', loss_item, global_step)
            if feature_kd_enabled:
                # Reading these scalars: `kd_feature_loss_weighted` is the
                # value per *eligible* image, NOT the term's share of
                # `Total_Loss`. Images excluded by `disable_on_augs` contribute
                # nothing, so what actually reached the total is smaller by
                # `kd_feature_exposure`. Divide `kd_feature_loss_effective` by
                # `Total_Loss` for the real share; dividing the weighted value
                # instead overstates it by 1 / exposure.
                writer.add_scalar('Training/kd_feature_loss_weighted', kd_feature_loss_val * current_feature_weight, global_step)
                writer.add_scalar('Training/kd_feature_weight', current_feature_weight, global_step)
                # Fraction of the batch feature KD was allowed to see. Falls to
                # zero on a fully masked batch, and steps up to 1.0 when mosaic
                # and mixup close for the final epochs -- a jump in KD exposure
                # that is invisible in every other scalar here.
                writer.add_scalar('Training/kd_feature_exposure', kd_feature_exposure, global_step)
                # The term as it enters `Total_Loss`, in the same per-image units.
                writer.add_scalar('Training/kd_feature_loss_effective', kd_feature_effective, global_step)
                if kd_feature_terms is not None:
                    # Rows are scales (P3..P5), columns are the four objectives.
                    # Both marginals sum to the unweighted feature loss, so the
                    # split can be read either way without renormalising.
                    for term_i, term_name in enumerate(feature_kd_module.term_names):
                        writer.add_scalar(
                            f'Training/kd_feat_term_{term_name}',
                            sum(row[term_i] for row in kd_feature_terms),
                            global_step,
                        )
                    for scale_i, row in enumerate(kd_feature_terms):
                        writer.add_scalar(
                            f'Training/kd_feat_scale_p{scale_i + 3}', sum(row), global_step
                        )
                if kd_grad_ratio is not None:
                    # The quantity to select `kd_weight` on. The loss share says
                    # nothing about how hard the term pulls: the feature terms
                    # differentiate through an L2 normalisation whose gradient
                    # scales with 1/||f||, so a 3%-of-loss term can carry a third
                    # of the gradient or a thousandth of it.
                    writer.add_scalar('Training/kd_grad_norm', kd_grad_norm, global_step)
                    writer.add_scalar('Training/kd_grad_ratio', kd_grad_ratio, global_step)
                    # Sign is what matters here. Near zero means KD is adding
                    # information orthogonal to supervision; persistently
                    # negative means it is spending the step undoing it, which no
                    # loss-magnitude scalar can distinguish from the former.
                    writer.add_scalar('Training/kd_grad_cosine', kd_grad_cosine, global_step)
            if soft_kd_enabled:
                writer.add_scalar('Training/kd_soft_loss_weighted', kd_soft_loss_val * current_soft_weight, global_step)
                writer.add_scalar('Training/kd_soft_weight', current_soft_weight, global_step)
            if hard_kd_enabled:
                writer.add_scalar('Training/kd_hard_loss_weighted', kd_hard_loss_val * current_hard_weight, global_step)
                # Teacher boxes the annotations corroborate, and (in "add" mode)
                # the ones they do not. A matched count near zero means the
                # teacher and the labels disagree about what is in the image,
                # which caps what hard pseudo-labelling can contribute.
                writer.add_scalar('Training/kd_hard_teacher_matched', n_pseudo_matched, global_step)
                writer.add_scalar('Training/kd_hard_teacher_added', n_pseudo_added, global_step)
                writer.add_scalar('Training/kd_hard_weight', current_hard_weight, global_step)
            if regime.use_teacher_pseudo_base:
                writer.add_scalar(
                    'Training/teacher_base_loss', teacher_base_loss_val, global_step
                )
                writer.add_scalar(
                    'Training/teacher_base_targets', teacher_base_target_count, global_step
                )
                writer.add_scalar(
                    'Training/teacher_base_empty_skipped',
                    teacher_base_empty_skipped,
                    global_step,
                )
            
            lr_head, lr_pre = get_lrs(optimizer)
            writer.add_scalar('Training/LR_head', lr_head, global_step)
            writer.add_scalar('Training/LR_pretrained', lr_pre, global_step)
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr_head": f"{lr_head:.6f}",
                "lr_pre": f"{lr_pre:.6f}"
            })
            global_step += 1

        # Update LR schedule at the end of epoch
        lr_scheduler.step()

        # Decay the o2m/o2o weights in E2ELoss
        if hasattr(criterion, 'update'):
            criterion.update()

        # Print end-of-epoch summary
        avg_loss = epoch_total_loss / max(1, finite_batches)
        lr_head, lr_pre = get_lrs(optimizer)
        print(f"Epoch {epoch+1}/{cfg_manual.shared_train_config['epochs']}  "
              f"LR_head={lr_head:.6f}  "
              f"LR_pre={lr_pre:.6f}  "
              f"Avg Loss={avg_loss:.4f}")
              
        checkpoint_extras = {
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'patience_counter': patience_counter,
            'training_regime': regime.as_dict(),
            'validation_manifest': validation_record,
            'ema_step': ema._step,
            label_budget_meta.BUDGET_RECORD_KEY: run_budget,
            RUN_PROVENANCE_KEY: run_provenance,
        }
        if feature_kd_enabled:
            checkpoint_extras['kd_feature_state'] = feature_kd_module.state_dict()
            
        # --- Validation and Early Stopping ---
        stop_training, best_map, patience_counter = validate_and_early_stop(
            model=model,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            model_type="yolo_manual",
            device=device,
            writer=writer,
            best_map=best_map,
            patience_counter=patience_counter,
            patience=patience,
            save_name=yolo_best_name,
            checkpoint_extras=checkpoint_extras,
            val_json_path=validation_json,
            val_targets=regime.validation_targets,
        )
        if stop_training:
            break

    writer.close()
    print("YOLO Manual Training Complete.")


# --- RT-DETR Training ---
class LinearWarmup:
    """
    Linear warmup wrapper that overrides the optimizer LR for the first
    warmup_steps iterations, then hands control back to the base scheduler.

    Args:
        optimizer: Base optimizer.
        warmup_steps: Number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self._step = 0

        # Store per-group target LR (the value the param group was initialised with)
        self._target_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        # Step once to initialize the LR for the first batch
        self.step()

    def step(self):
        """Update the learning rate for the next step."""
        self._step += 1
        if self._step <= self.warmup_steps:
            # Linearly increase LR from 0 to target_lr over warmup_steps iterations.
            alpha = self._step / self.warmup_steps
            for pg, target in zip(self.optimizer.param_groups, self._target_lrs):
                pg['lr'] = target * alpha

    @property
    def finished(self):
        """Check if warmup is finished."""
        return self._step >= self.warmup_steps


def train_rtdetr():
    """
    Custom PyTorch training loop for RT-DETR using the official GitHub source.
    Mirrors the official det_engine.py / optimizer.yml / rtdetr_r50vd.yml config
    so that loss values match the reference implementation exactly.
    """
    print("--- Starting RT-DETR Training (Custom Loop) ---")

    regime = resolve_training_regime(
        getattr(cfg_manual, "training_regime", None), cfg_manual.kd_config
    )
    if regime.mode != "gt_only":
        raise ValueError(
            "RT-DETR training currently supports supervised gt_only runs. "
            f"The configured regime resolves to {regime.mode!r}; use "
            "model_format='yolo_manual' for teacher-only or hybrid KD training."
        )

    run_budget = label_budget_meta.budget_record(cfg_manual)
    validation_record = {
        **manifest_summary(cfg_manual.val_json),
        "targets": "ground_truth",
        "subsample_interval": _validation_subsample_interval(),
    }
    run_provenance = build_training_run_provenance(
        backend="rtdetr",
        budget=run_budget,
        train_manifest=cfg_manual.train_json,
        validation=validation_record,
        regime=regime,
        kd_config=cfg_manual.kd_config,
        train_subsample_interval=cfg_manual.dataloader_config.get(
            "frame_subsample_interval", 1
        ),
        seed=cfg_manual.shared_train_config.get("seed", 42),
    )
    rtdetr_best_name = budget_named_path("rtdetr_best.pth", run_budget)
    print(f"Annotation budget: {label_budget_meta.describe(run_budget)}.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create summary writer
    writer = SummaryWriter(
        log_dir=os.path.join(
            "runs", "rtdetr", budget_run_tag(run_budget), run_provenance["tag"]
        )
    )

    # --- Dataloader Setup ---
    # Validation goes through run_evaluation(), which builds its own loader from
    # cfg_manual.val_json, so only the training loader is used here.
    train_loader, _ = build_dataloaders(
        cfg_manual.train_json,
        cfg_manual.val_json,
    )

    # Construct the model.
    # Only download ImageNet backbone weights if not loading full offline weights.
    has_offline_weights = os.path.exists(cfg_manual.rtdetr_train_config["weights"])
    variant = cfg_manual.rtdetr_train_config.get("variant", "small")
    model = build_rtdetr_model(variant=variant, pretrained_backbone=not has_offline_weights)

    loaded_keys = set()
    # Load official pretrained weights (skip the 80-class head via strict=False)
    if os.path.exists(cfg_manual.rtdetr_train_config["weights"]):
        checkpoint = torch.load(cfg_manual.rtdetr_train_config["weights"], map_location="cpu", weights_only=False)
        # Official checkpoints store the EMA model under 'ema' → 'module'
        if 'ema' in checkpoint and 'module' in checkpoint['ema']:
            state_dict = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        # Filter out keys with shape mismatches (e.g., classification heads)
        model_state = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                print(f"  Skipping {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
            elif k in model_state:
                filtered_state_dict[k] = v
                loaded_keys.add(k)

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained weights from {cfg_manual.rtdetr_train_config['weights']}")
        print(f"  Missing keys   : {len(missing)} (expected due to mismatched heads)")
        print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: Pretrained weights not found at {cfg_manual.rtdetr_train_config['weights']}, training from scratch.")

    model.to(device)

    # --- Loss Engine ---
    # Reference: configs/rtdetr/include/rtdetr_r50vd.yml

    # Hungarian Matcher (runs inside criterion.forward)
    matcher = HungarianMatcher(
        weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        use_focal_loss=True,     # RT-DETR uses sigmoid focal cost
        alpha=0.25,
        gamma=2.0,
    )

    # RTDETR Criterion — uses VFL (Varifocal Loss) for classification.
    # The loss-term gains are the RT-DETR counterpart of the YOLO box/cls/dfl
    # gains, so they live in the config rather than being hardcoded here.
    criterion = RTDETRCriterion(
        matcher=matcher,
        weight_dict=dict(cfg_manual.rtdetr_train_config.get(
            "loss_weights", {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}
        )),
        losses=['vfl', 'boxes'],
        alpha=cfg_manual.rtdetr_train_config.get("varifocal_alpha", 0.75),
        gamma=cfg_manual.rtdetr_train_config.get("varifocal_gamma", 2.0),
        num_classes=cfg_manual.num_classes,
        # Same inverse-frequency weighting the YOLO path gets, multiplied into
        # the per-class VFL map instead of the per-class BCE map.
        class_weights=compute_class_weights(
            train_loader.dataset,
            float(cfg_manual.rtdetr_train_config.get("cls_pw", 0.0)),
            device,
        ),
        class_weights_preserve_magnitude=bool(
            cfg_manual.rtdetr_train_config.get("cls_pw_preserve_magnitude", True)
        ),
    )
    criterion.to(device)

    # --- Optimizer ---
    # Parameters are divided into dynamic groups based on loaded checkpoint weights:
    # 1. Pretrained parameters (backbone, encoder, decoder) - decay: lr = base_lr * pretrained_lr_ratio, wd = weight_decay
    # 2. Pretrained parameters (backbone, encoder, decoder) - no decay (norm/bias): lr = base_lr * pretrained_lr_ratio, wd = 0.0
    # 3. Newly initialized head parameters (e.g. class_embed) - decay: lr = base_lr, wd = weight_decay
    # 4. Newly initialized head parameters (e.g. class_embed) - no decay (norm/bias): lr = base_lr, wd = 0.0
    base_lr = cfg_manual.rtdetr_train_config.get("learning_rate", 2e-4)
    weight_decay = cfg_manual.rtdetr_train_config.get("weight_decay", 1e-4)
    pretrained_lr_ratio = cfg_manual.shared_train_config.get("pretrained_lr_ratio", 0.1)
    pretrained_lr = base_lr * pretrained_lr_ratio

    groups = {
        "pretrained_decay": {"params": [], "lr": pretrained_lr, "weight_decay": weight_decay, "is_pretrained": True, "unreduced_lr": base_lr},
        "pretrained_no_decay": {"params": [], "lr": pretrained_lr, "weight_decay": 0.0, "is_pretrained": True, "unreduced_lr": base_lr},
        "head_decay": {"params": [], "lr": base_lr, "weight_decay": weight_decay, "is_pretrained": False, "unreduced_lr": base_lr},
        "head_no_decay": {"params": [], "lr": base_lr, "weight_decay": 0.0, "is_pretrained": False, "unreduced_lr": base_lr},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_norm_or_bias = ('norm' in name or 'bn' in name or name.endswith('.bias'))
        is_head = name not in loaded_keys
        
        if is_head:
            if is_norm_or_bias:
                groups["head_no_decay"]["params"].append(param)
            else:
                groups["head_decay"]["params"].append(param)
        else:
            if is_norm_or_bias:
                groups["pretrained_no_decay"]["params"].append(param)
            else:
                groups["pretrained_decay"]["params"].append(param)

    # Filter out empty groups
    param_groups = [v for v in groups.values() if len(v["params"]) > 0]

    optimizer = AdamW(param_groups, lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    # LR Scheduler
    milestones = cfg_manual.rtdetr_train_config.get("lr_milestones", [1000])
    gamma = cfg_manual.rtdetr_train_config.get("lr_gamma", 0.1)
    lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Linear Warmup
    warmup_steps = cfg_manual.rtdetr_train_config.get("lr_warmup_steps", 2000)
    lr_warmup = LinearWarmup(optimizer, warmup_steps=warmup_steps)

    # --- EMA + AMP Scaler ---
    amp_cfg = get_amp_settings()
    ema = ModelEMA(model, decay=cfg_manual.shared_train_config["ema_decay"], warmups=cfg_manual.shared_train_config["ema_warmups"])
    scaler = GradScaler(enabled=amp_cfg["scaler_enabled"] and torch.cuda.is_available())

    # --- Training Loop ---

    # Initialize tracking variables
    # Clamped to at least 1: the step condition below divides by it, and
    # optimizer_steps_per_epoch() already treats anything smaller as "every batch".
    ACCUMULATION_STEPS = max(1, int(cfg_manual.shared_train_config["gradient_accumulation_steps"]))
    CLIP_MAX_NORM = 0.1   # official: clip_max_norm: 0.1
    global_step = 0
    best_map = 0.0
    start_epoch = 0
    patience = cfg_manual.shared_train_config.get("early_stopping_patience", 10)
    patience_counter = 0
    consecutive_inf = 0          # track consecutive inf batches
    MAX_CONSECUTIVE_INF = 10     # exit only after this many in a row
    
    if cfg_manual.shared_train_config.get("resume_training", False):
        last_ckpt = os.path.join(
            cfg_manual.weights_dir, rtdetr_best_name.replace("best", "last")
        )
        best_ckpt = os.path.join(cfg_manual.weights_dir, rtdetr_best_name)
        resume_ckpt = last_ckpt if os.path.exists(last_ckpt) else best_ckpt
        
        if os.path.exists(resume_ckpt):
            print(f"Resuming RT-DETR training from {resume_ckpt}...")
            ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            label_budget_meta.verify_budget_match(
                label_budget_meta.read_budget_record(ckpt),
                run_budget,
                resume_ckpt,
                artifact_label="Resume checkpoint",
            )
            verify_run_provenance(
                ckpt.get(RUN_PROVENANCE_KEY), run_provenance, resume_ckpt
            )
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            if 'ema' in ckpt and 'module' in ckpt['ema']:
                ema.module.load_state_dict(ckpt['ema']['module'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
            if 'best_map' in ckpt:
                best_map = ckpt['best_map']
            if 'patience_counter' in ckpt:
                patience_counter = ckpt['patience_counter']
            if 'scaler' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                except Exception as e:
                    print(f"Failed to load scaler state: {e}")
                    
            if 'lr_scheduler' in ckpt:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            else:
                for _ in range(start_epoch):
                    lr_scheduler.step()
                
            # LinearWarmup.step() and ema.update() both run once per optimizer
            # step, so their counters are restored in optimizer steps -- in
            # batches they would skip ahead by the accumulation factor and end
            # the LR warmup the moment training resumes. global_step is a
            # per-batch counter used only for logging.
            steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), ACCUMULATION_STEPS)
            lr_warmup._step = start_epoch * steps_per_epoch
            ema._step = start_epoch * steps_per_epoch
            global_step = start_epoch * len(train_loader)
        else:
            print(f"WARNING: Resume checkpoint not found at {resume_ckpt}.")
            
    # Training loop
    for epoch in range(start_epoch, cfg_manual.shared_train_config["epochs"]):
        # Update dataloader epoch to handle Mosaic/MixUp closing
        train_loader.dataset.update_aug_schedule(epoch, cfg_manual.shared_train_config["epochs"])

        # Update pretrained LR ratio and apply fading if configured
        update_pretrained_lr_ratio(optimizer, lr_scheduler, lr_warmup, epoch, cfg_manual.shared_train_config["epochs"])

        model.train()
        criterion.train()
        optimizer.zero_grad()
        
        epoch_total_loss = 0.0
        finite_batches = 0
        valid_batches_in_window = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_manual.shared_train_config['epochs']}", leave=False)
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            
            # Send target labels and boxes to device
            formatted_targets = [
                {'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)}
                for t in targets
            ]

            # Forward pass (model under autocast, criterion in fp32)
            with torch.autocast(device_type=device.type, dtype=amp_cfg["dtype"], enabled=amp_cfg["enabled"], cache_enabled=True):
                # Passing targets to model enables CDN (contrastive denoising)
                outputs = model(images, targets=formatted_targets)

            # Criterion outside autocast to keep Hungarian matching in fp32
            with torch.autocast(device_type=device.type, enabled=False):
                loss_dict = criterion(outputs, formatted_targets)

            # Get total loss
            total_loss = sum(loss_dict.values())
            loss_value = total_loss.item()

            # Check for inf/nan before backward() to avoid propagating corrupted
            # gradients. Only this batch is dropped: the check runs before
            # backward(), so nothing corrupt has entered the accumulation
            # window, and the gradients banked from its earlier batches are
            # still valid. The step below therefore still runs on schedule --
            # skipping it would carry those gradients into the next window and
            # double the effective step size.
            batch_is_finite = math.isfinite(loss_value)
            if batch_is_finite:
                consecutive_inf = 0
                # Scale by accumulation steps and backward
                scaled_loss = total_loss / ACCUMULATION_STEPS
                scaler.scale(scaled_loss).backward()
                valid_batches_in_window += 1
            else:
                consecutive_inf += 1
                print(f"\n[WARNING] Inf/NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}. "
                      f"Skipping batch. ({consecutive_inf}/{MAX_CONSECUTIVE_INF})")
                print(loss_dict)
                if consecutive_inf >= MAX_CONSECUTIVE_INF:
                    print(f"Loss is not recoverable after {MAX_CONSECUTIVE_INF} consecutive inf batches. Stopping.")
                    sys.exit(1)

            # Gradient accumulation step
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                if valid_batches_in_window > 0:
                    # Unscale and renormalize a partial/filtered accumulation window.
                    scaler.unscale_(optimizer)
                    model_parameters = list(model.parameters())
                    normalize_accumulated_gradients(
                        model_parameters, ACCUMULATION_STEPS, valid_batches_in_window
                    )
                    if CLIP_MAX_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=CLIP_MAX_NORM)

                    optimizer_stepped = step_optimizer_with_scaler(scaler, optimizer)
                    # EMA and warmup advance once per actual optimizer step.
                    # GradScaler may skip that step when fp16 gradients overflow.
                    if optimizer_stepped:
                        ema.update(model)
                        lr_warmup.step()

                optimizer.zero_grad()
                valid_batches_in_window = 0

            # A skipped batch has no loss worth logging.
            if not batch_is_finite:
                continue

            epoch_total_loss += loss_value
            finite_batches += 1

            writer.add_scalar('Training/Total_Loss', loss_value, global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Training/{k}', v.item(), global_step)
            
            lr_head, lr_pre = get_lrs(optimizer)
            writer.add_scalar('Training/LR_head', lr_head, global_step)
            writer.add_scalar('Training/LR_pretrained', lr_pre, global_step)
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr_head": f"{lr_head:.6f}",
                "lr_pre": f"{lr_pre:.6f}"
            })
            global_step += 1

        # Step the LR scheduler once per epoch (after warmup finishes)
        lr_scheduler.step()

        avg_loss = epoch_total_loss / max(1, finite_batches)
        lr_head, lr_pre = get_lrs(optimizer)
        print(f"Epoch {epoch+1} complete. "
              f"LR_head={lr_head:.6f}  "
              f"LR_pre={lr_pre:.6f}  "
              f"Avg Loss={avg_loss:.4f}")
        
        checkpoint_extras = {
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'patience_counter': patience_counter,
            'training_regime': regime.as_dict(),
            # A teacher is the thing a later student checks itself against, so
            # this is the checkpoint where the budget matters most.
            label_budget_meta.BUDGET_RECORD_KEY: run_budget,
            RUN_PROVENANCE_KEY: run_provenance,
            'validation_manifest': validation_record,
        }
        
        # --- Validation & Early Stopping ---
        stop_training, best_map, patience_counter = validate_and_early_stop(
            model=model,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            model_type="rtdetr",
            device=device,
            writer=writer,
            best_map=best_map,
            patience_counter=patience_counter,
            patience=patience,
            save_name=rtdetr_best_name,
            checkpoint_extras=checkpoint_extras
        )
        if stop_training:
            break

    writer.close()
    print("RT-DETR Training Complete.")


# --- Main ---
if __name__ == "__main__":
    seed = cfg_manual.shared_train_config.get("seed", 42)
    print(f"Edge Object Detection | Active Model: {cfg_manual.model_format} | Global Seed: {seed}")
    set_seed(seed)
    
    if cfg_manual.model_format == "yolo_ultra":
        train_yolo_ultra()
    elif cfg_manual.model_format == "yolo_manual":
        train_yolo_manual()
    elif cfg_manual.model_format == "rtdetr":
        train_rtdetr()
    else:
        raise ValueError(f"Unsupported MODEL_FORMAT: {cfg_manual.model_format}")
