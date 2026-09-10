"""Training-loop and distillation-loss unit tests.

`train.py` imports Ultralytics at module scope, so the tests that need it skip
where it is unavailable. `kd_loss.py` only needs torch and always runs.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import math
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kd_loss import (  # noqa: E402
    FeatureKD,
    boxes_to_instance_masks,
    boxes_to_spatial_mask,
    match_boxes_to_gt,
)


# ── Gradient accumulation bookkeeping ────────────────────────────────────────
@pytest.mark.parametrize(
    ("num_batches", "accumulation_steps", "expected"),
    [
        (100, 1, 100),
        (100, 8, 13),   # 12 full windows plus the tail step on the last batch
        (96, 8, 12),    # divides evenly, no tail step
        (5, 8, 1),      # a single short window still steps
        (100, 0, 100),  # a nonsensical setting must not divide by zero
    ],
)
def test_optimizer_steps_per_epoch(num_batches, accumulation_steps, expected):
    """EMA and LR-warmup counters advance per optimizer step, not per batch.

    Restoring them in batches on resume skips them ahead by the accumulation
    factor, which ends the LR warmup the moment training resumes.
    """
    pytest.importorskip("ultralytics")
    from train import optimizer_steps_per_epoch

    assert optimizer_steps_per_epoch(num_batches, accumulation_steps) == expected


def test_resume_counters_stay_behind_the_batch_count():
    """Counters restored in batch units rather than optimizer steps run 8x fast."""
    pytest.importorskip("ultralytics")
    from train import optimizer_steps_per_epoch

    batches_per_epoch, accumulation_steps, epoch = 1000, 8, 3
    restored = epoch * optimizer_steps_per_epoch(batches_per_epoch, accumulation_steps)
    assert restored == 375
    assert restored < epoch * batches_per_epoch


def test_validation_forwards_and_checkpoints_the_configured_subsample_interval(
    tmp_path, monkeypatch
):
    pytest.importorskip("ultralytics")
    import train

    calls = {}
    saved = []

    def fake_evaluation(**kwargs):
        calls.update(kwargs)
        return {"mAP_50_95": 0.2, "mAP_50": 0.3}

    class StateHolder:
        def eval(self):
            return self

        def state_dict(self):
            return {}

    monkeypatch.setitem(
        train.cfg_manual.dataloader_config, "val_subsample_interval", 3
    )
    monkeypatch.setattr(train.cfg_manual, "weights_dir", str(tmp_path))
    monkeypatch.setattr(train, "run_evaluation", fake_evaluation)
    monkeypatch.setattr(
        train.torch, "save", lambda payload, path: saved.append((payload, path))
    )

    train.validate_and_early_stop(
        model=StateHolder(),
        ema=SimpleNamespace(module=StateHolder()),
        optimizer=SimpleNamespace(state_dict=lambda: {}),
        epoch=0,
        model_type="rtdetr",
        device=torch.device("cpu"),
        writer=SimpleNamespace(add_scalar=lambda *_args: None),
        best_map=0.0,
        patience_counter=0,
        patience=5,
        save_name="model_best.pth",
        checkpoint_extras={
            "validation_manifest": {
                "targets": "ground_truth",
                "manifest": "val.json",
                "fingerprint": "abc",
            }
        },
        val_json_path="val.json",
    )

    assert calls["subsample_interval"] == 3
    assert len(saved) == 2  # last and improved best
    for checkpoint, _path in saved:
        assert checkpoint["validation_manifest"]["subsample_interval"] == 3


# ── Checkpoint naming ────────────────────────────────────────────────────────
def test_kd_run_suffix_distinguishes_variants():
    """Without a suffix every KD variant writes the same yolo_manual_best.pth."""
    pytest.importorskip("ultralytics")
    from train import kd_run_suffix

    off = {"enabled": False, "feature_based": {"enabled": True}}
    assert kd_run_suffix(off) == ""
    # KD on but nothing selected is still an ordinary run.
    assert kd_run_suffix({"enabled": True}) == ""

    def cfg(**flags):
        base = {
            "enabled": True,
            "pseudo_label": {"enabled": False},
            "pseudo_label_soft": {"enabled": False},
            "feature_based": {"enabled": False},
        }
        for key, value in flags.items():
            base[key] = {"enabled": value}
        return base

    assert kd_run_suffix(cfg(feature_based=True)) == "_kd-feat"
    assert kd_run_suffix(cfg(pseudo_label=True)) == "_kd-hard"
    assert kd_run_suffix(cfg(pseudo_label_soft=True)) == "_kd-soft"
    # Order is fixed, so a combination always names the same file.
    assert kd_run_suffix(
        cfg(pseudo_label=True, pseudo_label_soft=True, feature_based=True)
    ) == "_kd-hard-soft-feat"

    # The suffix must sit after "best" so the last-checkpoint rewrite still works.
    name = f"yolo_manual_best{kd_run_suffix(cfg(feature_based=True))}.pth"
    assert name.replace("best", "last") == "yolo_manual_last_kd-feat.pth"


# ── Teacher/ground-truth matching ────────────────────────────────────────────
def test_match_boxes_to_gt_requires_class_and_iou_agreement():
    gt_boxes = torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.2, 0.2]])
    gt_cls = torch.tensor([0, 1])

    # Near-identical box, right class -> matches.
    pred = torch.tensor([[0.205, 0.205, 0.1, 0.1]])
    idx_p, idx_g = match_boxes_to_gt(pred, torch.tensor([0]), gt_boxes, gt_cls, 0.5)
    assert idx_p.tolist() == [0] and idx_g.tolist() == [0]

    # Same geometry, wrong class -> no match.
    idx_p, _ = match_boxes_to_gt(pred, torch.tensor([7]), gt_boxes, gt_cls, 0.5)
    assert idx_p.numel() == 0

    # Right class, too far away -> no match.
    far = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    idx_p, _ = match_boxes_to_gt(far, torch.tensor([0]), gt_boxes, gt_cls, 0.5)
    assert idx_p.numel() == 0

    # Empty on either side is not an error.
    empty = torch.zeros(0, 4)
    assert match_boxes_to_gt(empty, torch.zeros(0), gt_boxes, gt_cls, 0.5)[0].numel() == 0
    assert match_boxes_to_gt(pred, torch.tensor([0]), empty, torch.zeros(0), 0.5)[0].numel() == 0


def test_match_boxes_to_gt_allocates_each_box_once():
    """Two annotations competing for one detection must not both claim it."""
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.52, 0.52, 0.2, 0.2]])
    gt_cls = torch.tensor([0, 0])
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

    idx_p, idx_g = match_boxes_to_gt(pred, torch.tensor([0]), gt_boxes, gt_cls, 0.5)
    assert idx_p.numel() == 1 and idx_g.numel() == 1
    assert len(set(idx_p.tolist())) == idx_p.numel()


# ── Ultralytics loss-item compatibility ──────────────────────────────────────
def test_detection_loss_items_for_logging_accepts_current_dict_api():
    pytest.importorskip("ultralytics")
    from train import detection_loss_items_for_logging

    loss_items = {
        "box_loss": torch.tensor(1.25),
        "cls_loss": torch.tensor(2.5),
        "l1_loss": torch.tensor(3.75),
    }

    assert detection_loss_items_for_logging(loss_items) == {
        "box_loss": 1.25,
        "cls_loss": 2.5,
        "l1_loss": 3.75,
    }


def test_detection_loss_items_for_logging_accepts_legacy_tensor_api():
    pytest.importorskip("ultralytics")
    from train import detection_loss_items_for_logging

    result = detection_loss_items_for_logging(torch.tensor([1.25, 2.5, 3.75]))

    assert result == {"box_loss": 1.25, "cls_loss": 2.5, "dfl_loss": 3.75}


# ── Feature-based knowledge distillation ─────────────────────────────────────
def _feature_kd_inputs(batch=2, size=4, student_ch=4, teacher_ch=8, scales=2):
    torch.manual_seed(0)
    student = [torch.randn(batch, student_ch, size, size) for _ in range(scales)]
    teacher = [torch.randn(batch, teacher_ch, size, size) for _ in range(scales)]
    return student, teacher


def test_feature_kd_averages_over_contributing_scales():
    """The loss must not depend on how many scales the batch happened to fill."""
    student, teacher = _feature_kd_inputs(scales=2)
    module = FeatureKD(student_channels=[4, 4], teacher_channels=[8, 8]).eval()

    # One centred box per image, so every scale has foreground.
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    batch_idx = torch.tensor([0, 1])

    loss = module(student, teacher, boxes, batch_idx)
    assert loss.ndim == 0
    assert loss.requires_grad
    assert torch.isfinite(loss) and loss.detach().item() > 0.0


def test_feature_kd_weights_instances_not_area():
    """A large box must not outvote a small one just by covering more pixels.

    A binary union mask makes an object's contribution proportional to its
    area, which on VisDrone lets a handful of vehicles drown out the pedestrians
    the model actually struggles with.
    """
    torch.manual_seed(0)
    batch, student_ch, teacher_ch, size = 1, 4, 8, 32
    student = [torch.randn(batch, student_ch, size, size)]
    teacher = [torch.randn(batch, teacher_ch, size, size)]
    module = FeatureKD(student_channels=[4], teacher_channels=[8]).eval()

    tiny = torch.tensor([[0.2, 0.2, 0.08, 0.08]])
    huge = torch.tensor([[0.7, 0.7, 0.5, 0.5]])
    idx = torch.tensor([0])

    scale_tiny, fg_tiny, n_tiny = boxes_to_instance_masks(
        tiny, idx, batch, size, size, torch.tensor([True])
    )
    scale_huge, _, _ = boxes_to_instance_masks(
        huge, idx, batch, size, size, torch.tensor([True])
    )
    assert n_tiny == 1
    # Each instance's weights sum to ~1 whatever its area, so neither dominates.
    assert scale_tiny.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert scale_huge.sum().item() == pytest.approx(1.0, abs=1e-5)
    # ...while the binary mask they replaced differs by the full area ratio.
    assert fg_tiny.sum().item() < 0.2 * boxes_to_spatial_mask(
        huge, idx, batch, size, size, torch.tensor([True])
    ).sum().item()

    both = module(student, teacher, torch.cat([tiny, huge]), torch.tensor([0, 0]))
    assert torch.isfinite(both)


def test_instance_masks_match_half_open_bounds_and_preserve_overlap_mass():
    """Guards the vectorized rasterizer that replaced the per-box Python loop."""
    one = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    idx = torch.tensor([0])
    valid = torch.tensor([True])

    scale, fg, n = boxes_to_instance_masks(one, idx, 1, 4, 4, valid)
    # Same footprint as the binary rasterizer it shares bounds with.
    assert torch.equal(fg, boxes_to_spatial_mask(one, idx, 1, 4, 4, valid))
    assert n == 1
    expected = torch.zeros(1, 1, 4, 4)
    expected[0, 0, 1:3, 1:3] = 0.25          # 1 / (2*2)
    assert torch.allclose(scale, expected)

    # Overlapping boxes: weights add, preserving one unit of total mass per
    # instance instead of losing mass at overlaps while retaining both in the
    # instance-count denominator.
    two = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    scale2, _, n2 = boxes_to_instance_masks(
        two, torch.tensor([0, 0]), 1, 4, 4, valid
    )
    assert n2 == 2
    assert scale2[0, 0, 2, 2].item() == pytest.approx(0.25 + 1 / 16)
    assert scale2[0, 0, 0, 0].item() == pytest.approx(1 / 16)  # large box only
    assert scale2.sum().item() == pytest.approx(2.0)

    # A box too thin to cover a pixel is not an instance.
    degenerate = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    _, _, n3 = boxes_to_instance_masks(degenerate, idx, 1, 4, 4, valid)
    assert n3 == 0

    # Boxes are chunked internally; crossing the chunk boundary must be seamless.
    many = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).repeat(300, 1)
    scale4, _, n4 = boxes_to_instance_masks(
        many, torch.zeros(300, dtype=torch.long), 1, 4, 4, valid
    )
    assert n4 == 300
    assert scale4.sum().item() == pytest.approx(300.0)
    assert torch.allclose(scale4, expected * 300)


def test_feature_kd_scale_weights_emphasise_p3():
    """`scale_weights` must actually change the mix, and must be validated."""
    student, teacher = _feature_kd_inputs(scales=2)
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    batch_idx = torch.tensor([0, 1])

    torch.manual_seed(1)
    even = FeatureKD([4, 4], [8, 8], scale_weights=[1.0, 1.0]).eval()
    torch.manual_seed(1)
    p3_heavy = FeatureKD([4, 4], [8, 8], scale_weights=[8.0, 1.0]).eval()
    assert not torch.isclose(
        even(student, teacher, boxes, batch_idx),
        p3_heavy(student, teacher, boxes, batch_idx),
    )

    with pytest.raises(ValueError, match="one entry per scale"):
        FeatureKD([4, 4], [8, 8], scale_weights=[1.0])
    with pytest.raises(ValueError, match="sum to zero"):
        FeatureKD([4, 4], [8, 8], scale_weights=[0.0, 0.0])


def test_feature_kd_adapter_is_shallow_by_default():
    """The default adapter must not be deep enough to absorb the alignment."""
    default = FeatureKD([4], [8])
    deep = FeatureKD([4], [8], adapter="dwsep")
    assert sum(p.numel() for p in default.adapters.parameters()) < sum(
        p.numel() for p in deep.adapters.parameters()
    )
    # No BatchNorm anywhere by default: KD trains at a physical batch of 4.
    assert not any(
        isinstance(m, torch.nn.BatchNorm2d) for m in default.adapters.modules()
    )
    assert any(isinstance(m, torch.nn.GroupNorm) for m in default.adapters.modules())

    with pytest.raises(ValueError, match="Unknown adapter"):
        FeatureKD([4], [8], adapter="mlp")


def test_feature_kd_global_term_is_optional():
    """`global_weight=0` must build no context blocks at all."""
    assert FeatureKD([4], [8], global_weight=0.0).context_blocks is None
    assert FeatureKD([4], [8], global_weight=0.5).context_blocks is not None


def test_feature_kd_breakdown_reconciles_with_the_returned_loss():
    """The logged per-term split must add up to the scalar that is optimised.

    The breakdown exists to tell fg/bg/attn/global apart in TensorBoard. A split
    that does not sum to the loss is worse than no split at all: it would send a
    `kd_weight` sweep after a term that is not actually the one dominating.
    """
    student, teacher = _feature_kd_inputs(scales=2)
    module = FeatureKD([4, 4], [8, 8], scale_weights=[6.0, 1.0]).eval()
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    batch_idx = torch.tensor([0, 1])

    loss = module(student, teacher, boxes, batch_idx)

    assert module.last_terms.shape == (2, len(module.term_names))
    assert torch.allclose(module.last_terms.sum(), loss.detach(), rtol=1e-5)
    assert not module.last_terms.requires_grad

    # Both marginals are usable on their own: scale gains are already folded in
    # and normalised, so P3's share reflects `scale_weights`.
    per_scale = module.last_terms.sum(dim=1)
    assert per_scale[0] > per_scale[1]


def test_feature_kd_breakdown_stays_shaped_when_nothing_is_distilled():
    """A fully masked batch must still leave a zero breakdown, not a stale one.

    The mask fires on whole batches often enough to matter -- at mosaic p=0.5 and
    a physical batch of 4, roughly one step in ten. If `last_terms` kept the
    previous step's values there, the logged split would silently over-report.
    """
    student, teacher = _feature_kd_inputs(scales=2)
    module = FeatureKD([4, 4], [8, 8]).eval()
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    batch_idx = torch.tensor([0, 1])

    module(student, teacher, boxes, batch_idx)
    assert module.last_terms.sum() > 0.0

    loss = module(student, teacher, boxes, batch_idx, torch.ones(2, dtype=torch.bool))
    assert loss.detach().item() == 0.0
    assert module.last_terms.shape == (2, len(module.term_names))
    assert module.last_terms.sum().item() == 0.0


def test_kd_gradient_probe_reports_direction_and_magnitude():
    """Ratio must scale with the KD weight; cosine must recover +1 and -1.

    This is the statistic `kd_weight` is selected on, so a probe that silently
    returned zero -- or that mixed up the two gradients -- would be worse than
    not measuring at all.
    """
    pytest.importorskip("ultralytics")
    from train import kd_gradient_probe

    student, teacher = _feature_kd_inputs(scales=2)
    student = [s.clone().requires_grad_(True) for s in student]
    module = FeatureKD([4, 4], [8, 8]).eval()
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    batch_idx = torch.tensor([0, 1])
    kd = module(student, teacher, boxes, batch_idx)

    _, ratio_self, cos_self = kd_gradient_probe(kd, kd, student)
    assert ratio_self == pytest.approx(1.0, rel=1e-5)
    assert cos_self == pytest.approx(1.0, abs=1e-5)

    _, _, cos_opposed = kd_gradient_probe(kd, -kd, student)
    assert cos_opposed == pytest.approx(-1.0, abs=1e-5)

    # A supervised stand-in whose gradient is not collinear with the KD term.
    heads = [torch.randn(1, 4, 1, 1) for _ in student]
    supervised = sum(((s * h).sum(1) - 1.0).pow(2).mean() for s, h in zip(student, heads))
    norm_a, ratio_a, _ = kd_gradient_probe(kd * 0.09, supervised, student)
    norm_b, ratio_b, _ = kd_gradient_probe(kd * 0.18, supervised, student)
    assert ratio_b == pytest.approx(2.0 * ratio_a, rel=1e-4)
    assert norm_b == pytest.approx(2.0 * norm_a, rel=1e-4)


def test_kd_gradient_probe_tolerates_parameters_the_kd_term_never_touches():
    """Most of the student gets no KD gradient at all; that is not an error.

    `torch.autograd.grad` returns None for those, and the supervised norm must
    still count them -- otherwise the ratio is measured over a subspace chosen by
    the KD term itself and always looks larger than it is.
    """
    pytest.importorskip("ultralytics")
    from train import kd_gradient_probe

    student, teacher = _feature_kd_inputs(scales=1)
    student = [s.clone().requires_grad_(True) for s in student]
    module = FeatureKD([4], [8]).eval()
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    kd = module(student, teacher, boxes, torch.tensor([0]))

    untouched = torch.randn(5, requires_grad=True)
    supervised = sum((s**2).mean() for s in student) + untouched.sum()

    _, ratio_with, _ = kd_gradient_probe(kd, supervised, student + [untouched])
    _, ratio_without, _ = kd_gradient_probe(kd, supervised, student)

    assert math.isfinite(ratio_with) and ratio_with > 0.0
    # The extra parameter enlarges the supervised norm only, so the ratio falls.
    assert ratio_with < ratio_without


def test_feature_kd_is_zero_without_ground_truth():
    """No boxes means no foreground to distil, at any scale."""
    student, teacher = _feature_kd_inputs(scales=2)
    module = FeatureKD(student_channels=[4, 4], teacher_channels=[8, 8]).eval()

    loss = module(student, teacher, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))
    assert loss.detach().item() == 0.0


def test_feature_mask_uses_half_open_pixel_bounds():
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    mask = boxes_to_spatial_mask(
        boxes,
        batch_idx=torch.tensor([0]),
        batch_size=1,
        height=4,
        width=4,
        valid_mask=torch.tensor([True]),
    )

    expected = torch.zeros(1, 1, 4, 4)
    expected[0, 0, 1:3, 1:3] = 1
    assert torch.equal(mask, expected)


def test_feature_kd_validates_disable_mask_length():
    student, teacher = _feature_kd_inputs(batch=2, scales=1)
    module = FeatureKD(student_channels=[4], teacher_channels=[8]).eval()

    with pytest.raises(ValueError, match="one value per image"):
        module(
            student,
            teacher,
            torch.zeros(0, 4),
            torch.zeros(0, dtype=torch.long),
            disable_kd_mask=torch.tensor([False]),
        )


def test_feature_kd_excludes_masked_images():
    """Images whose augmentation invalidated the teacher must not contribute."""
    student, teacher = _feature_kd_inputs(scales=1)
    module = FeatureKD(student_channels=[4], teacher_channels=[8]).eval()
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    batch_idx = torch.tensor([0, 1])

    both = module(student, teacher, boxes, batch_idx)
    first_only = module(
        student, teacher, boxes, batch_idx,
        disable_kd_mask=torch.tensor([False, True]),
    )
    assert not torch.isclose(both, first_only)

    none = module(
        student, teacher, boxes, batch_idx,
        disable_kd_mask=torch.tensor([True, True]),
    )
    assert none.detach().item() == 0.0


# ── Focal Loss wiring ────────────────────────────────────────────────────────
def _detection_model(nc=10):
    from ultralytics.nn.tasks import DetectionModel
    import configs.train_cfg as cfg
    from types import SimpleNamespace

    model = DetectionModel(cfg.yolo_train_config["model_yaml_path"], ch=3, nc=nc, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, fl_gamma=1.5, epochs=50)
    return model


def test_focal_loss_reaches_every_e2e_sub_criterion():
    """Wrapping only the top-level criterion is a silent no-op.

    YOLO26 is `end2end`, so the criterion is `E2ELoss`, which owns no `.bce` of
    its own -- it delegates to two independent `v8DetectionLoss` instances. A
    wrapper installed on the `E2ELoss` object itself is never called, and focal
    modulation silently never happens.
    """
    pytest.importorskip("ultralytics")
    from ultralytics.utils.loss import E2ELoss
    from train import FocalLossWrapper, apply_focal_loss

    criterion = E2ELoss(_detection_model())
    assert not hasattr(criterion, "bce"), "E2ELoss gained a .bce; revisit apply_focal_loss()"

    assert apply_focal_loss(criterion, 1.5) == ["one2many", "one2one"]
    assert isinstance(criterion.one2many.bce, FocalLossWrapper)
    assert isinstance(criterion.one2one.bce, FocalLossWrapper)


def test_focal_loss_wraps_a_plain_detection_criterion():
    """The non-end2end branch keeps working through the same entry point."""
    pytest.importorskip("ultralytics")
    from ultralytics.utils.loss import v8DetectionLoss
    from train import FocalLossWrapper, apply_focal_loss

    criterion = v8DetectionLoss(_detection_model())
    assert apply_focal_loss(criterion, 1.5) == ["v8DetectionLoss"]
    assert isinstance(criterion.bce, FocalLossWrapper)


def test_focal_modulation_downweights_easy_examples():
    """Gamma must scale the loss down, most strongly where the model is right."""
    pytest.importorskip("ultralytics")
    from train import FocalLossWrapper

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    focal = FocalLossWrapper(bce, gamma=1.5)
    pred = torch.tensor([[5.0, 0.0]])       # confidently correct, then undecided
    label = torch.tensor([[1.0, 1.0]])

    raw, modulated = bce(pred, label), focal(pred, label)
    assert modulated.shape == raw.shape
    assert (modulated <= raw).all()
    # The confident anchor is suppressed far harder than the ambiguous one.
    assert (modulated[0, 0] / raw[0, 0]) < (modulated[0, 1] / raw[0, 1])


# ── Learning-rate reporting ──────────────────────────────────────────────────
def test_get_lrs_ignores_bias_groups():
    """Bias groups warm up from `warmup_bias_lr`, so they misreport the group.

    During warmup a bias group's LR is several times the LR of the weights it
    sits beside; reporting the maximum picks the bias group and overstates both
    numbers.
    """
    pytest.importorskip("ultralytics")
    from train import get_lrs

    # A parameter may only belong to one group, so each group needs its own.
    optimizer = torch.optim.SGD([
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-4,
         "is_pretrained": True, "is_bias": False},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3,
         "is_pretrained": True, "is_bias": True},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3,
         "is_pretrained": False, "is_bias": False},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-2,
         "is_pretrained": False, "is_bias": True},
    ])

    assert get_lrs(optimizer) == (1e-3, 1e-4)


def test_get_lrs_falls_back_when_every_group_is_bias():
    """A configuration made only of bias groups still reports something."""
    pytest.importorskip("ultralytics")
    from train import get_lrs

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([
        {"params": [parameter], "lr": 5e-4, "is_pretrained": False, "is_bias": True},
    ])

    assert get_lrs(optimizer) == (5e-4, 0.0)


# ── Knowledge Distillation Config Schema ──────────────────────────────────────
def test_kd_config_schema_consistency():
    """All KD methods should use kd_weight consistently for their loss weighting."""
    import configs.train_cfg as cfg

    kd_cfg = cfg.kd_config
    for kd_method in ("pseudo_label", "pseudo_label_soft", "feature_based"):
        assert kd_method in kd_cfg
        assert "kd_weight" in kd_cfg[kd_method]
        assert "kd_warmup_epochs" in kd_cfg[kd_method]
        assert "kd_ramp_epochs" in kd_cfg[kd_method]
        assert "kd_fade_epochs" in kd_cfg[kd_method]
        assert "kd_cooldown_epochs" in kd_cfg[kd_method]
        assert "weight" not in kd_cfg[kd_method]


def test_kd_ramp_factor_warmup_and_plateau():
    pytest.importorskip("ultralytics")
    from train import kd_ramp_factor

    weight = 0.18
    # Warmup hold at 0
    assert kd_ramp_factor(weight, epoch_progress=0.0, warmup_epochs=4, ramp_epochs=4) == 0.0
    assert kd_ramp_factor(weight, epoch_progress=3.99, warmup_epochs=4, ramp_epochs=4) == 0.0
    # Ramp up: epoch 4 -> 8 (midpoint at 6.0 is 0.5 * weight)
    assert math.isclose(kd_ramp_factor(weight, epoch_progress=6.0, warmup_epochs=4, ramp_epochs=4), 0.09)
    # Plateau at full weight
    assert math.isclose(kd_ramp_factor(weight, epoch_progress=8.0, warmup_epochs=4, ramp_epochs=4), 0.18)
    assert math.isclose(kd_ramp_factor(weight, epoch_progress=25.0, warmup_epochs=4, ramp_epochs=4), 0.18)


def test_kd_ramp_factor_fade_out():
    pytest.importorskip("ultralytics")
    from train import kd_ramp_factor

    weight = 0.20
    total_epochs = 50
    fade_epochs = 4
    warmup_epochs = 4
    ramp_epochs = 4

    # Before fade starts (epoch 45) -> full weight
    assert math.isclose(
        kd_ramp_factor(weight, 45.0, warmup_epochs, ramp_epochs, total_epochs, fade_epochs), 0.20
    )
    # Start of fade (epoch 46.0) -> full weight
    assert math.isclose(
        kd_ramp_factor(weight, 46.0, warmup_epochs, ramp_epochs, total_epochs, fade_epochs), 0.20
    )
    # Mid-fade (epoch 48.0) -> half weight
    assert math.isclose(
        kd_ramp_factor(weight, 48.0, warmup_epochs, ramp_epochs, total_epochs, fade_epochs), 0.10
    )
    # 3/4 through fade (epoch 49.0) -> quarter weight
    assert math.isclose(
        kd_ramp_factor(weight, 49.0, warmup_epochs, ramp_epochs, total_epochs, fade_epochs), 0.05
    )
    # End of training (epoch 50.0) -> 0.0
    assert math.isclose(
        kd_ramp_factor(weight, 50.0, warmup_epochs, ramp_epochs, total_epochs, fade_epochs), 0.0
    )


def test_kd_ramp_factor_fade_and_cooldown_combined():
    pytest.importorskip("ultralytics")
    from train import kd_ramp_factor

    weight = 0.10
    total_epochs = 50
    fade_epochs = 4
    cooldown_epochs = 2  # Fade from 44->48, hold at 0 for 48->50

    # Epoch 44: full weight
    assert math.isclose(
        kd_ramp_factor(weight, 44.0, 4, 4, total_epochs, fade_epochs, cooldown_epochs), 0.10
    )
    # Epoch 46: half weight
    assert math.isclose(
        kd_ramp_factor(weight, 46.0, 4, 4, total_epochs, fade_epochs, cooldown_epochs), 0.05
    )
    # Epoch 48: reaches 0
    assert math.isclose(
        kd_ramp_factor(weight, 48.0, 4, 4, total_epochs, fade_epochs, cooldown_epochs), 0.0
    )
    # Epoch 49 and 50 (cooldown): held at 0
    assert kd_ramp_factor(weight, 49.0, 4, 4, total_epochs, fade_epochs, cooldown_epochs) == 0.0
    assert kd_ramp_factor(weight, 50.0, 4, 4, total_epochs, fade_epochs, cooldown_epochs) == 0.0


def test_kd_ramp_factor_backwards_compatible():
    pytest.importorskip("ultralytics")
    from train import kd_ramp_factor

    # Without total_epochs / fade / cooldown, behaves identically to legacy warmup ramp
    assert kd_ramp_factor(0.5, 2.0, 4, 4) == 0.0
    assert math.isclose(kd_ramp_factor(0.5, 6.0, 4, 4), 0.25)
    assert math.isclose(kd_ramp_factor(0.5, 10.0, 4, 4), 0.5)
    assert math.isclose(kd_ramp_factor(0.5, 50.0, 4, 4), 0.5)



# ── GT-anchored hard pseudo-labels ───────────────────────────────────────────
def _pseudo_batch_fixture(n_gt=5, batch=4, queries=8, device="cpu"):
    """A batch whose teacher boxes deliberately overlap only some annotations."""
    torch.manual_seed(0)
    batch_dict = {
        "img": torch.zeros(batch, 3, 32, 32, device=device),
        "cls": torch.zeros(batch * n_gt, 1, device=device),
        "bboxes": torch.rand(batch * n_gt, 4, device=device) * 0.2 + 0.4,
        "batch_idx": torch.arange(batch, device=device).repeat_interleave(n_gt).float(),
    }
    boxes = torch.rand(batch, queries, 4, device=device) * 0.2 + 0.4
    classes = torch.zeros(batch, queries, dtype=torch.long, device=device)
    conf = torch.rand(batch, queries, device=device)
    return batch_dict, boxes, classes, conf


def test_pseudo_batch_never_drops_an_annotation():
    """A teacher miss must not turn an annotated object into a background label."""
    pytest.importorskip("ultralytics")
    from train import build_gt_anchored_pseudo_batch

    batch, boxes, classes, conf = _pseudo_batch_fixture()
    keep = torch.tensor([0, 1, 2])
    n_gt_kept = int((batch["batch_idx"] < 3).sum())

    # Teacher totally silent: every annotation must still survive.
    silent, matched, added = build_gt_anchored_pseudo_batch(
        batch, keep, boxes, classes, torch.zeros_like(conf), 0.5, 0.5, "drop"
    )
    assert silent["bboxes"].shape[0] == n_gt_kept
    assert matched == 0 and added == 0

    # And with a confident teacher, presence is still exactly the annotations'.
    loud, _, added_loud = build_gt_anchored_pseudo_batch(
        batch, keep, boxes, classes, torch.ones_like(conf), 0.5, 0.5, "drop"
    )
    assert loud["bboxes"].shape[0] == n_gt_kept and added_loud == 0


def test_pseudo_batch_add_mode_admits_unmatched_teacher_boxes():
    pytest.importorskip("ultralytics")
    from train import build_gt_anchored_pseudo_batch

    batch, boxes, classes, conf = _pseudo_batch_fixture()
    keep = torch.tensor([0, 1, 2])
    n_gt_kept = int((batch["batch_idx"] < 3).sum())

    added_batch, matched, added = build_gt_anchored_pseudo_batch(
        batch, keep, boxes, classes, torch.ones_like(conf), 0.5, 0.5, "add"
    )
    assert added > 0
    assert added_batch["bboxes"].shape[0] == n_gt_kept + added
    # "add" only ever grows the target set relative to "drop".
    dropped, _, _ = build_gt_anchored_pseudo_batch(
        batch, keep, boxes, classes, torch.ones_like(conf), 0.5, 0.5, "add"
    )
    assert dropped["bboxes"].shape[0] >= n_gt_kept
    assert added_batch["cls"].shape[0] == added_batch["bboxes"].shape[0]


def test_pseudo_batch_indexes_the_sliced_batch():
    """batch_idx must address the sliced predictions, not the original batch."""
    pytest.importorskip("ultralytics")
    from train import build_gt_anchored_pseudo_batch

    batch, boxes, classes, conf = _pseudo_batch_fixture()
    keep = torch.tensor([1, 3])            # non-contiguous, skipping image 0
    sliced, _, _ = build_gt_anchored_pseudo_batch(
        batch, keep, boxes, classes, conf, 0.5, 0.5, "drop"
    )
    assert sliced["img"].shape[0] == 2
    assert set(sliced["batch_idx"].unique().tolist()) <= {0.0, 1.0}

    # No kept images at all is a no-op, not a crash.
    none, _, _ = build_gt_anchored_pseudo_batch(
        batch, torch.zeros(0, dtype=torch.long), boxes, classes, conf, 0.5, 0.5, "drop"
    )
    assert none is None


# ── Soft KD across both YOLO26 heads ─────────────────────────────────────────
def test_soft_kd_branch_loss_covers_the_inference_head():
    """one2many alone is auxiliary; YOLO26 infers through one2one."""
    pytest.importorskip("ultralytics")
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss
    from train import soft_kd_branch_loss

    torch.manual_seed(0)
    nc, batch = 10, 2
    model = DetectionModel("yolo26n.yaml", ch=3, nc=nc)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    model.train()
    criterion = E2ELoss(model)

    parsed = criterion.one2many.parse_output(model(torch.rand(batch, 3, 128, 160)))
    assert set(parsed) == {"one2many", "one2one"}

    n_obj = 4
    t_labels = torch.randint(0, nc, (batch, n_obj, 1)).float()
    t_boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0]]).repeat(batch, n_obj, 1)
    t_boxes = t_boxes + torch.rand(batch, n_obj, 4) * 5
    t_logits = torch.randn(batch, n_obj, nc) * 3
    mask_gt = torch.ones(batch, n_obj, dtype=torch.bool)

    # The two branches use different assigners, so they are not the same term.
    assert criterion.one2many.assigner.topk != criterion.one2one.assigner.topk

    losses = {}
    for name in ("one2many", "one2one"):
        loss = soft_kd_branch_loss(
            parsed[name], getattr(criterion, name),
            t_labels, t_boxes, t_logits, mask_gt, 2.0,
        )
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.requires_grad
        losses[name] = loss

    # Gradient must actually reach the inference head's scores.
    grad = torch.autograd.grad(
        losses["one2one"], parsed["one2one"]["scores"], retain_graph=True
    )[0]
    assert grad.abs().sum() > 0


def test_soft_kd_excluded_images_contribute_nothing():
    """An image with no teacher targets must not raise the others' KD weight."""
    pytest.importorskip("ultralytics")
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss
    from train import soft_kd_branch_loss

    torch.manual_seed(0)
    nc, batch, n_obj = 10, 3, 4
    model = DetectionModel("yolo26n.yaml", ch=3, nc=nc)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    model.train()
    criterion = E2ELoss(model)
    parsed = criterion.one2many.parse_output(model(torch.rand(batch, 3, 128, 160)))

    t_labels = torch.randint(0, nc, (batch, n_obj, 1)).float()
    t_boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0]]).repeat(batch, n_obj, 1)
    t_logits = torch.randn(batch, n_obj, nc) * 3

    full = torch.ones(batch, n_obj, dtype=torch.bool)
    partial = full.clone()
    partial[2] = False                      # image 2 excluded, as disable_kd would

    args = (parsed["one2many"], criterion.one2many, t_labels, t_boxes, t_logits)
    assert not torch.isclose(
        soft_kd_branch_loss(*args, full, 2.0),
        soft_kd_branch_loss(*args, partial, 2.0),
    )
    # With nothing assigned anywhere the term collapses to zero, not to a NaN.
    empty = torch.zeros(batch, n_obj, dtype=torch.bool)
    assert torch.isfinite(soft_kd_branch_loss(*args, empty, 2.0))


def test_instance_mask_mass_survives_overlap():
    """Total mask mass must equal the instance count the loss divides by."""
    valid = torch.tensor([True])
    idx = torch.tensor([0, 0])
    # Two identical boxes: mass 2 for n_instances 2, not mass 1.
    twins = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    scale, _, n = boxes_to_instance_masks(twins, idx, 1, 8, 8, valid)
    assert n == 2
    assert scale.sum().item() == pytest.approx(2.0, abs=1e-4)

    # Partial overlap of differently sized objects: still one unit each.
    mixed = torch.tensor([[0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.2, 0.2]])
    scale2, _, n2 = boxes_to_instance_masks(mixed, idx, 1, 16, 16, valid)
    assert n2 == 2
    assert scale2.sum().item() == pytest.approx(2.0, abs=1e-4)


def test_soft_kd_leaves_non_selected_classes_alone():
    """The confidence term must touch only the teacher's chosen class.

    The objective this replaced softened every sigmoid slot toward 0.5, so it
    taught the student to fire on nine classes it should have left at zero.
    """
    pytest.importorskip("ultralytics")
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss
    from train import soft_kd_branch_loss

    torch.manual_seed(0)
    nc, batch, n_obj = 10, 2, 4
    model = DetectionModel("yolo26n.yaml", ch=3, nc=nc)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    model.train()
    criterion = E2ELoss(model)
    parsed = criterion.one2many.parse_output(model(torch.rand(batch, 3, 128, 160)))
    branch = parsed["one2many"]

    t_labels = torch.zeros(batch, n_obj, 1)
    t_boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0]]).repeat(batch, n_obj, 1)
    t_logits = torch.full((batch, n_obj, nc), -6.0)
    t_logits[..., 0] = 4.0                      # class 0 confident, rest strongly negative
    mask_gt = torch.ones(batch, n_obj, dtype=torch.bool)
    args = (branch, criterion.one2many, t_labels, t_boxes, t_logits, mask_gt)

    # Confidence-only: gradient may reach the selected class alone.
    conf_only = soft_kd_branch_loss(*args, 1.0, confidence_weight=1.0, relation_weight=0.0)
    grad = torch.autograd.grad(conf_only, branch["scores"], retain_graph=True)[0]
    # scores is (B, nc, N): class 0 gets gradient, the other nine do not.
    assert grad[:, 0].abs().sum() > 0
    assert grad[:, 1:].abs().sum() == 0

    # Relation-only is a proper divergence: identical distributions give zero.
    same = soft_kd_branch_loss(
        branch, criterion.one2many, t_labels, t_boxes,
        branch["scores"].permute(0, 2, 1).contiguous().detach()[:, :n_obj].clone(),
        mask_gt, 2.0, confidence_weight=0.0, relation_weight=1.0,
    )
    assert torch.isfinite(same) and same.item() >= 0.0

    # Both terms together exceed either alone (all terms are non-negative).
    rel_only = soft_kd_branch_loss(*args, 2.0, confidence_weight=0.0, relation_weight=1.0)
    both = soft_kd_branch_loss(*args, 2.0, confidence_weight=1.0, relation_weight=1.0)
    assert both.item() > max(conf_only.item(), rel_only.item())


# ── Cached teacher predictions ───────────────────────────────────────────────
def test_pack_cached_teacher_preds_pads_without_inventing_detections():
    """Padding rows must be filtered out by the same threshold as real ones."""
    pytest.importorskip("ultralytics")
    from train import pack_cached_teacher_preds

    nc = 10
    targets = [
        {"kd_boxes": torch.rand(3, 4), "kd_logits": torch.randn(3, nc)},
        {"kd_boxes": torch.zeros(0, 4), "kd_logits": torch.zeros(0, nc)},
        {"kd_boxes": torch.rand(1, 4), "kd_logits": torch.randn(1, nc)},
    ]
    packed = pack_cached_teacher_preds(targets, nc, torch.device("cpu"))

    assert packed["pred_logits"].shape == (3, 3, nc)
    assert packed["pred_boxes"].shape == (3, 3, 4)

    # Every padded slot sits below any usable confidence threshold.
    conf = packed["pred_logits"].sigmoid().max(-1).values
    assert conf[0, :3].shape == (3,)
    assert (conf[1] < 0.01).all()          # image with no teacher boxes at all
    assert (conf[2, 1:] < 0.01).all()      # padding after its single real box

    # An all-empty batch must not crash or produce a zero-width tensor.
    empty = pack_cached_teacher_preds(
        [{"kd_boxes": torch.zeros(0, 4), "kd_logits": torch.zeros(0, nc)}],
        nc, torch.device("cpu"),
    )
    assert empty["pred_boxes"].shape == (1, 1, 4)
    assert (empty["pred_logits"].sigmoid().max(-1).values < 0.01).all()


# ── Teacher weak view (geometry shared, appearance not) ──────────────────────
def _reload_dataloader(monkeypatch_cfg):
    """Reload dataloader.py so module-level config reads take effect."""
    import importlib
    import dataloader
    return importlib.reload(dataloader)


def _write_tiny_coco(tmp_path: Path, num_images: int = 1) -> Path:
    """Create a small labelled COCO dataset for transform integration tests."""
    images, annotations, categories = [], [], []
    for i in range(num_images):
        image_id = i + 1
        # A non-uniform image makes appearance changes observable, while the
        # large box reliably survives the mosaic window crop.
        yy, xx = np.mgrid[:32, :48]
        rgb = np.stack(
            ((xx * 5 + i * 17) % 256, (yy * 7 + i * 29) % 256, (xx + yy * 3) % 256),
            axis=-1,
        ).astype(np.uint8)
        image_path = tmp_path / f"frame_{image_id}.png"
        assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        images.append(
            {"id": image_id, "file_name": str(image_path), "width": 48, "height": 32}
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": i,
                "bbox": [8.0, 4.0, 32.0, 24.0],
                "area": 768.0,
                "iscrowd": 0,
            }
        )
        categories.append({"id": i, "name": f"class_{i}"})

    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        )
    )
    return annotation_path


def _configure_small_aug_pipeline(monkeypatch, dl, *, mosaic_p=0.0, mixup_p=0.0):
    """Make the real dataset pipeline small, deterministic and inexpensive."""
    monkeypatch.setattr(dl.cfg, "input_height", 32)
    monkeypatch.setattr(dl.cfg, "input_width", 48)
    monkeypatch.setitem(dl.cfg.kd_config, "enabled", True)
    monkeypatch.setitem(dl.cfg.kd_config["feature_based"], "enabled", True)
    monkeypatch.setitem(dl.cfg.kd_config["feature_based"], "teacher_clean_view", True)
    monkeypatch.setitem(dl.cfg.kd_config, "disable_on_augs", [])

    aug = dl.cfg.augmentation
    monkeypatch.setitem(aug["mosaic"], "enabled", mosaic_p > 0)
    monkeypatch.setitem(aug["mosaic"], "p", mosaic_p)
    monkeypatch.setitem(aug["mosaic"], "scale_limit", [0.0, 0.0])
    monkeypatch.setitem(aug["mixup"], "enabled", mixup_p > 0)
    monkeypatch.setitem(aug["mixup"], "p", mixup_p)
    monkeypatch.setitem(aug["lsj"], "enabled", False)
    monkeypatch.setitem(aug["lsj"], "min_visibility", 0.0)
    monkeypatch.setitem(aug["lsj"], "min_visibility_candidates", [0.0])
    monkeypatch.setitem(aug["lsj"], "max_crop_trials", 50)
    monkeypatch.setitem(aug["horizontal_flip"], "enabled", True)
    monkeypatch.setitem(aug["horizontal_flip"], "p", 1.0)
    for name in ("photometric_distort", "cutout", "hard_negative"):
        monkeypatch.setitem(aug[name], "enabled", False)


def test_teacher_weak_view_shares_geometry_with_the_student():
    """The two views must be pixel-identical once appearance augs are off.

    That is the property feature KD depends on: if a flip or crop reached one
    view and not the other, the student would be matched against mirrored or
    offset teacher features and the loss would be quietly meaningless.
    """
    pytest.importorskip("ultralytics")
    pytest.importorskip("albumentations")
    import configs.train_cfg as cfg

    original = {
        k: cfg.augmentation[k]["enabled"]
        for k in ("photometric_distort", "cutout", "hard_negative")
    }
    clean_view = cfg.kd_config["feature_based"].get("teacher_clean_view", True)
    # Drive every flag the behaviour depends on, including the master switch.
    # Reading whatever the shipped config happens to have made this test pass or
    # fail on an unrelated edit -- it broke the moment KD was switched off to set
    # up the no-KD control run.
    kd_on = cfg.kd_config.get("enabled", False)
    feature_on = cfg.kd_config["feature_based"].get("enabled", False)
    try:
        for k in original:
            cfg.augmentation[k]["enabled"] = False
        cfg.kd_config["enabled"] = True
        cfg.kd_config["feature_based"]["enabled"] = True
        cfg.kd_config["feature_based"]["teacher_clean_view"] = True
        dl = _reload_dataloader(cfg)

        assert dl._teacher_weak_view_wanted() is True
        geom = dl.get_train_transforms(mosaic=True, seed=0, stage="geometric")
        appear = dl.get_train_transforms(mosaic=True, seed=0, stage="appearance")
        names = lambda c: {type(t).__name__ for t in c.transforms}

        # The teacher's stage must contain no appearance corruption at all...
        assert not (names(geom) & {"ColorJitter", "CoarseDropout", "OneOf"})
        # ...and must not finish the sample, since appearance follows it.
        assert not (names(geom) & {"Normalize", "ToTensorV2"})
        # ...while the appearance stage must move no pixels between coordinates.
        assert not (names(appear) & {"HorizontalFlip", "RandomCrop", "RandomScale"})
        # Boxes are resolved by the geometric stage alone.
        assert appear.processors.get("bboxes") is None
        assert geom.processors.get("bboxes") is not None
    finally:
        for k, v in original.items():
            cfg.augmentation[k]["enabled"] = v
        cfg.kd_config["enabled"] = kd_on
        cfg.kd_config["feature_based"]["enabled"] = feature_on
        cfg.kd_config["feature_based"]["teacher_clean_view"] = clean_view
        _reload_dataloader(cfg)


@pytest.mark.parametrize("appearance_enabled", [False, True])
def test_teacher_weak_view_matches_pixels_before_appearance(
    tmp_path, monkeypatch, appearance_enabled
):
    """Exercise the actual dataset fork, not only its transform declarations."""
    pytest.importorskip("ultralytics")
    import dataloader as dl

    _configure_small_aug_pipeline(monkeypatch, dl)
    pd = dl.cfg.augmentation["photometric_distort"]
    monkeypatch.setitem(pd, "enabled", appearance_enabled)
    if appearance_enabled:
        # Fixed brightness makes divergence deterministic instead of relying on
        # a random ColorJitter draw happening not to be the identity.
        monkeypatch.setitem(pd, "brightness", (0.5, 0.5))
        monkeypatch.setitem(pd, "contrast", 0.0)
        monkeypatch.setitem(pd, "saturation", 0.0)
        monkeypatch.setitem(pd, "hue", 0.0)

    annotation_path = _write_tiny_coco(tmp_path)
    dataset = dl.ObjectDetectionDataset(
        str(annotation_path),
        transforms=dl.get_train_transforms(seed=7),
        is_training=True,
        seed=7,
        # This test exercises the live feature-KD view, not response KD. Keep
        # it independent of whatever full-resolution cache exists on disk.
        enable_teacher_cache=False,
    )
    student_view, target = dataset[0]
    teacher_view = target["teacher_view"]

    assert student_view.shape == teacher_view.shape == (3, 32, 48)
    max_delta = (student_view - teacher_view).abs().max().item()
    if appearance_enabled:
        assert max_delta > 1e-3
    else:
        assert max_delta < 1e-5


def test_teacher_weak_view_geometry_fallback_remains_hwc_until_fork(monkeypatch):
    """A failed crop retry must not normalize or tensorize before the fork."""
    pytest.importorskip("ultralytics")
    import dataloader as dl

    _configure_small_aug_pipeline(monkeypatch, dl)

    class AlwaysDropsBoxes:
        def __init__(self):
            self.params = SimpleNamespace(min_visibility=0.1)
            self.processors = {"bboxes": SimpleNamespace(params=self.params)}

        def __call__(self, **kwargs):
            return {
                "image": kwargs["image"],
                "bboxes": [],
                "class_labels": [],
            }

    image = np.full((20, 30, 3), 127, dtype=np.uint8)
    g_image, boxes, labels = dl.apply_transforms_with_retry(
        transforms=AlwaysDropsBoxes(),
        fallback_transforms=dl.get_geometric_fallback_transforms(seed=3),
        image=image,
        boxes=[[2.0, 3.0, 12.0, 10.0]],
        labels=[1],
        min_visibility=0.9,
        max_trials=2,
    )

    assert isinstance(g_image, np.ndarray)
    assert g_image.shape == (32, 48, 3)
    assert boxes and labels == [1]
    teacher = dl.get_output_transforms()(image=g_image)["image"]
    student = dl.get_train_transforms(stage="appearance", seed=3)(image=g_image)["image"]
    assert torch.allclose(student, teacher, atol=1e-5, rtol=0.0)


def test_worker_init_reseeds_every_staged_pipeline(monkeypatch):
    """Forked workers must not share geometry or appearance RNG streams."""
    pytest.importorskip("ultralytics")
    import dataloader as dl

    class SeedRecorder:
        def __init__(self):
            self.seed = None

        def set_random_seed(self, seed):
            self.seed = seed

    names = (
        "transforms",
        "mosaic_transforms",
        "fallback_transforms",
        "geometric_transforms",
        "mosaic_geometric_transforms",
        "geometric_fallback_transforms",
        "appearance_transforms",
        "output_transforms",
    )
    dataset = SimpleNamespace(**{name: SeedRecorder() for name in names})
    monkeypatch.setattr(torch, "initial_seed", lambda: 1234)
    monkeypatch.setattr(
        torch.utils.data, "get_worker_info", lambda: SimpleNamespace(dataset=dataset)
    )

    dl._worker_init_fn(0)
    seeds = [getattr(dataset, name).seed for name in names]
    assert seeds == list(range(1234, 1234 + len(names)))


def test_weak_view_is_off_when_feature_kd_is_off():
    """No feature KD means no second view to pay for."""
    pytest.importorskip("ultralytics")
    import configs.train_cfg as cfg

    enabled = cfg.kd_config["feature_based"]["enabled"]
    kd_on = cfg.kd_config.get("enabled", False)
    try:
        # Feature KD off, master switch on.
        cfg.kd_config["enabled"] = True
        cfg.kd_config["feature_based"]["enabled"] = False
        dl = _reload_dataloader(cfg)
        assert dl._teacher_weak_view_wanted() is False

        # Feature KD on, master switch off -- the no-KD control's setting.
        cfg.kd_config["enabled"] = False
        cfg.kd_config["feature_based"]["enabled"] = True
        dl = _reload_dataloader(cfg)
        assert dl._teacher_weak_view_wanted() is False
        assert dl._teacher_cache_wanted() is False
    finally:
        cfg.kd_config["enabled"] = kd_on
        cfg.kd_config["feature_based"]["enabled"] = enabled
        _reload_dataloader(cfg)


def test_teacher_cache_rejects_a_mismatched_teacher(tmp_path):
    """A stale cache must fail loudly, not distil predictions from another model."""
    pytest.importorskip("ultralytics")
    import numpy as np
    from kd_cache import (
        CACHE_SCHEMA_VERSION,
        CACHE_STATE_POLICY,
        TeacherCache,
        _image_identity,
        _select_teacher_state,
    )

    weights = tmp_path / "teacher.pth"
    weights.write_bytes(b"\x01" * 4096)
    cache_file = tmp_path / "train.pt"

    valid_meta = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "state_policy": CACHE_STATE_POLICY,
        "state_source": "ema.module",
        "weights_fingerprint": "deadbeefdeadbeef",
        "input_hw": (736, 1280),
        "num_classes": 10,
        "min_confidence": 0.05,
        "image_identities": {
            1: _image_identity(
                {"id": 1, "file_name": "frame.png", "width": 1280, "height": 736}
            )
        },
    }
    payload = {
        "boxes": {1: np.zeros((2, 4), dtype=np.float16)},
        "logits": {1: np.zeros((2, 10), dtype=np.float16)},
        "meta": valid_meta,
    }

    # A cache made before the raw-model -> EMA correction has the same weights
    # file fingerprint. Schema validation is what forces that stale cache out.
    legacy = {**payload, "meta": {"weights_fingerprint": "deadbeefdeadbeef"}}
    torch.save(legacy, cache_file)
    with pytest.raises(ValueError, match="schema version"):
        TeacherCache.load(
            str(cache_file), str(weights), (736, 1280),
            expected_num_classes=10, requested_min_confidence=0.5,
        )

    torch.save(payload, cache_file)

    with pytest.raises(ValueError, match="different checkpoint"):
        TeacherCache.load(
            str(cache_file), str(weights), (736, 1280),
            expected_num_classes=10, requested_min_confidence=0.5,
        )

    # Right teacher, wrong resolution is equally unusable.
    from kd_cache import _weights_fingerprint
    payload["meta"]["weights_fingerprint"] = _weights_fingerprint(str(weights))
    torch.save(payload, cache_file)
    with pytest.raises(ValueError, match="built at"):
        TeacherCache.load(
            str(cache_file), str(weights), (640, 640),
            expected_num_classes=10, requested_min_confidence=0.5,
        )

    # A cache for another class mapping can have the right teacher file and
    # resolution while still assigning every logit column the wrong meaning.
    payload["meta"]["num_classes"] = 9
    torch.save(payload, cache_file)
    with pytest.raises(ValueError, match="num_classes=9.*expects 10"):
        TeacherCache.load(
            str(cache_file), str(weights), (736, 1280),
            expected_num_classes=10, requested_min_confidence=0.5,
        )
    payload["meta"]["num_classes"] = 10

    # Threshold sweeps must not silently request predictions the cache discarded
    # when it was built.
    payload["meta"]["min_confidence"] = 0.05
    torch.save(payload, cache_file)
    with pytest.raises(ValueError, match="stores predictions only.*requests 0.01"):
        TeacherCache.load(
            str(cache_file), str(weights), (736, 1280),
            expected_num_classes=10, requested_min_confidence=0.01,
        )

    # The exact cache floor is valid; the comparison must not be off by one.
    exact_floor_cache = TeacherCache.load(
        str(cache_file), str(weights), (736, 1280),
        expected_num_classes=10, requested_min_confidence=0.05,
    )
    assert len(exact_floor_cache) == 1

    # A stricter requested threshold is also fully covered by the cache.
    cache = TeacherCache.load(
        str(cache_file), str(weights), (736, 1280),
        expected_num_classes=10, requested_min_confidence=0.5,
    )
    assert len(cache) == 1
    boxes, logits = cache.get(1)
    assert boxes.shape == (2, 4) and logits.shape == (2, 10)
    # An image absent from the cache yields nothing rather than raising.
    assert cache.get(999)[0].shape == (0, 4)

    with pytest.raises(FileNotFoundError, match="Build it first"):
        TeacherCache.load(
            str(tmp_path / "missing.pt"), str(weights), (736, 1280),
            expected_num_classes=10, requested_min_confidence=0.5,
        )

    ema_state, model_state = {"x": torch.ones(1)}, {"x": torch.zeros(1)}
    selected, source = _select_teacher_state(
        {"ema": {"module": ema_state}, "model": model_state}
    )
    assert selected is ema_state and source == "ema.module"


@pytest.mark.parametrize(("mosaic_p", "mixup_p"), [(1.0, 0.0), (0.0, 1.0)])
def test_cached_teacher_boxes_and_logits_stay_paired_through_composites(
    tmp_path, monkeypatch, mosaic_p, mixup_p
):
    """Cached predictions follow GT through mosaic/crop/flip and MixUp."""
    pytest.importorskip("ultralytics")
    import dataloader as dl

    _configure_small_aug_pipeline(
        monkeypatch, dl, mosaic_p=mosaic_p, mixup_p=mixup_p
    )
    annotation_path = _write_tiny_coco(tmp_path, num_images=4)
    dataset = dl.ObjectDetectionDataset(
        str(annotation_path),
        transforms=dl.get_train_transforms(seed=11),
        is_training=True,
        seed=11,
        # A synthetic cache is attached below; do not load the real cache at
        # the tiny test resolution first.
        enable_teacher_cache=False,
    )

    class SyntheticTeacherCache:
        def get(self, image_id):
            boxes = np.array([[0.5, 0.5, 2.0 / 3.0, 0.75]], dtype=np.float32)
            logits = np.full((1, dl.cfg.num_classes), -20.0, dtype=np.float32)
            logits[0, int(image_id) - 1] = 20.0
            return boxes, logits

    dataset.teacher_cache = SyntheticTeacherCache()
    dl.random.seed(11)
    _, target = dataset[0]

    gt_boxes = target["boxes"]
    gt_classes = target["labels"]
    kd_boxes = target["kd_boxes"]
    kd_classes = target["kd_logits"].argmax(dim=-1)
    assert len(kd_boxes) > 0
    assert len(kd_boxes) == len(gt_boxes)

    # Every cached box began as an exact duplicate of one GT box. Geometric
    # transforms may reorder or discard pairs, but the class encoded in its
    # logits must still identify an equal transformed GT box.
    unmatched = set(range(len(gt_boxes)))
    for box, cls in zip(kd_boxes, kd_classes):
        matches = [
            i for i in unmatched
            if int(gt_classes[i]) == int(cls)
            and torch.allclose(gt_boxes[i], box, atol=1e-5, rtol=0.0)
        ]
        assert matches, f"cached class {int(cls)} lost its matching transformed GT box"
        unmatched.remove(matches[0])
    assert not unmatched


# ── Per-method KD augmentation policy ────────────────────────────────────────
def test_kd_disable_flags_are_resolved_per_method():
    """One global switch could not express what the three methods each need."""
    pytest.importorskip("ultralytics")
    import configs.train_cfg as cfg
    from dataloader import ObjectDetectionDataset

    resolve = ObjectDetectionDataset._kd_disable_flags
    saved = {k: cfg.kd_config[k].get("disable_on_augs") for k in
             ("feature_based", "pseudo_label", "pseudo_label_soft")}
    saved_global = cfg.kd_config.get("disable_on_augs")
    try:
        cfg.kd_config["feature_based"]["disable_on_augs"] = ["mixup"]
        cfg.kd_config["pseudo_label"]["disable_on_augs"] = []
        cfg.kd_config["pseudo_label_soft"]["disable_on_augs"] = ["mixup"]

        # A mixup sample: hard KD keeps it, the other two skip it.
        flags = resolve({"mixup"})
        assert flags == {"feature": True, "hard": False, "soft": True}

        # A plain mosaic sample: nobody excludes mosaic, so all three run.
        assert resolve({"mosaic"}) == {"feature": False, "hard": False, "soft": False}
        assert resolve(set()) == {"feature": False, "hard": False, "soft": False}

        # Mosaic + mixup together still only trips the mixup-sensitive methods.
        assert resolve({"mosaic", "mixup"})["hard"] is False

        # A method without its own key inherits the global fallback.
        del cfg.kd_config["pseudo_label"]["disable_on_augs"]
        cfg.kd_config["disable_on_augs"] = ["mosaic"]
        assert resolve({"mosaic"})["hard"] is True
    finally:
        for k, v in saved.items():
            if v is None:
                cfg.kd_config[k].pop("disable_on_augs", None)
            else:
                cfg.kd_config[k]["disable_on_augs"] = v
        cfg.kd_config["disable_on_augs"] = saved_global
