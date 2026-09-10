#!/usr/bin/env python3
"""
Unified evaluation pipeline for all trained object detection models.

Supports:
  - YOLO26 via Ultralytics API  (--model yolo_ultra)
  - YOLO26 manual checkpoint    (--model yolo_manual)
  - RT-DETR custom checkpoint (--model rtdetr)

Test sets  : test_baseline, test_occlusion  (from data/processed_annotations/)
Conditions : clean, rain, night, motion_blur

Metrics (pycocotools), all scored at the EVAL_MAX_DETS detection budgets:
  - mAP@0.5, mAP@0.5:0.95, mAP@0.75
  - AP_small, AP_medium, AP_large
  - AR@10, AR@100, AR@200
  - Per-class AP@0.5
  - Inference latency (ms / image)
  - Peak main-process RAM and CUDA memory during evaluation

Usage:
    python src/evaluate.py --model yolo_ultra  --weights weights/yolo_ultra_best.pt
    python src/evaluate.py --model yolo_manual --weights weights/yolo_manual_best.pth
    python src/evaluate.py --model rtdetr      --weights weights/rtdetr_best.pth
"""

import argparse
from collections.abc import Mapping
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime

from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    YOLO = None
    DetectionModel = None
    non_max_suppression = None

from rtdetr.zoo.rtdetr.rtdetr import RTDETR
from models import build_rtdetr_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import configs.eval_cfg as cfg
import configs.train_cfg as cfg_train
import label_budget as label_budget_meta
from experiment_artifacts import RUN_PROVENANCE_KEY, subsample_image_ids


# --- Helper Functions ---
def get_amp_settings() -> dict:
    """Helper to parse precision config into PyTorch AMP kwargs."""
    p = cfg_train.shared_train_config["precision"].lower()
    if p == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    elif p == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    elif p == "fp32":
        return {"enabled": False, "dtype": torch.float32}
    else:
        raise ValueError(f"Unsupported precision: {p}")


def _get_process_rss_mb() -> float | None:
    """Return the current resident memory of the evaluation process in MiB.

    ``/proc/self/statm`` is available on the Linux targets used for this
    project and reports resident (rather than virtual) memory. Returning
    ``None`` makes the benchmark gracefully unavailable on platforms without
    procfs instead of reporting a misleading zero.
    """
    try:
        with open("/proc/self/statm", "r") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
    except (OSError, IndexError, ValueError):
        return None


def _get_cuda_device_memory_mb(device: torch.device) -> float | None:
    """Return total used memory on ``device`` in MiB, when CUDA exposes it.

    Unlike PyTorch allocator statistics, this includes allocations made by
    CUDA-backed ONNX Runtime and TensorRT. It is device-wide, so other CUDA
    processes can affect this value.
    """
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return (total_bytes - free_bytes) / (1024 ** 2)
    except (RuntimeError, ValueError):
        return None


class _EvaluationMemoryBenchmark:
    """Collect peak memory measurements for one ``run_evaluation`` call."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda_enabled = device.type == "cuda" and torch.cuda.is_available()
        self.process_start_mb: float | None = None
        self.process_peak_mb: float | None = None
        self.cuda_device_start_mb: float | None = None
        self.cuda_device_peak_mb: float | None = None

    def start(self) -> None:
        """Reset CUDA peak statistics and capture the evaluation baseline."""
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        self.process_start_mb = self.process_peak_mb
        self.cuda_device_start_mb = self.cuda_device_peak_mb

    def sample(self) -> None:
        """Record current memory, retaining only the high-water marks."""
        process_mb = _get_process_rss_mb()
        if process_mb is not None:
            self.process_peak_mb = max(self.process_peak_mb or process_mb, process_mb)

        if self.cuda_enabled:
            cuda_device_mb = _get_cuda_device_memory_mb(self.device)
            if cuda_device_mb is not None:
                self.cuda_device_peak_mb = max(
                    self.cuda_device_peak_mb or cuda_device_mb, cuda_device_mb
                )

    def metrics(self) -> dict:
        """Return JSON-safe memory metrics, omitting unavailable measurements."""
        result = {}
        if self.process_start_mb is not None:
            result["process_rss_start_MB"] = float(self.process_start_mb)
        if self.process_peak_mb is not None:
            result["process_rss_peak_MB"] = float(self.process_peak_mb)
        if self.process_start_mb is not None and self.process_peak_mb is not None:
            result["process_rss_delta_MB"] = float(
                self.process_peak_mb - self.process_start_mb
            )

        if self.cuda_device_start_mb is not None:
            result["cuda_device_memory_start_MB"] = float(self.cuda_device_start_mb)
        if self.cuda_device_peak_mb is not None:
            result["cuda_device_memory_peak_MB"] = float(self.cuda_device_peak_mb)
        if self.cuda_device_start_mb is not None and self.cuda_device_peak_mb is not None:
            result["cuda_device_memory_delta_MB"] = float(
                self.cuda_device_peak_mb - self.cuda_device_start_mb
            )
        if self.cuda_enabled:
            result["cuda_torch_peak_allocated_MB"] = float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            )
            result["cuda_torch_peak_reserved_MB"] = float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            )
        return result


class _ComputeTimer:
    """Accumulate pure device-compute time for the batch currently in flight.

    ``run_evaluation`` reports two different latencies and they answer two
    different questions:

    * ``latency_ms_*``          everything inside ``infer_batch`` -- the host to
                                device copy, the forward pass and the decode of
                                raw tensors into COCO dicts. This is the number
                                a deployed system actually experiences, so it
                                stays the headline.
    * ``compute_latency_ms_*``  the forward pass alone, measured with CUDA
                                events on the device. This is the number that
                                changes when the *model* changes.

    The distinction is not cosmetic. On YOLO26n at 736x1280 the end-to-end
    figure is roughly three times the compute figure (7.4 ms vs 2.3 ms), and
    the host-side remainder is noisy enough that two builds of one unchanged
    configuration differ by ~3%. Anything smaller than that -- a coverage
    change worth a few percent of engine time, say -- is invisible end to end
    and has to be read off the compute column.

    Backends opt in by wrapping their forward pass and nothing else. Where a
    runtime cannot isolate its own device time the metric is simply absent
    rather than silently measuring a different span; see ``infer_onnx``.

    What this number is not
    -----------------------
    It is the forward pass *as the evaluation pipeline runs it*, not the
    engine's synthetic ceiling. Measured on the INT8 YOLO26n engine, the same
    forward costs progressively more as the surrounding context is added back:

        2.34 ms   execute alone, tight loop, no copies      (ceiling)
        2.74 ms   + H2D/D2H enqueued on the same stream
        3.14 ms   + a stream synchronise per call, as TRTWrapper does
        3.54 ms   + the live eval loop's dataloader contention  (reported)

    CUDA events themselves are trustworthy -- in a tight loop they agree with
    ``perf_counter`` to within 0.3% -- and idle host gaps between inferences do
    not inflate them, so this is not a clock or downclocking artifact. The
    growth is real serialisation: ``TRTWrapper`` stages its input through
    *pageable* host memory, so the copies do not overlap the compute the way
    pinned memory would, and some of that lands inside the event window.

    The practical consequence is that this column compares artifacts measured
    the same way, and must not be compared against a standalone benchmark of
    the same engine -- that will always look faster. Giving the input its own
    pagelocked staging buffer would narrow the gap and speed up the end-to-end
    path too; that is an inference-path change, not a timing one.
    """

    def __init__(self):
        self.enabled = False
        self.batch_ms: float | None = None

    def reset(self) -> None:
        self.batch_ms = None

    def record(self, ms: float) -> None:
        """Add one device-side window to the batch currently being timed."""
        self.batch_ms = (self.batch_ms or 0.0) + float(ms)


# Module-level so a backend deep in the call stack can report without every
# intermediate function having to thread a timer argument through.
_COMPUTE_TIMER = _ComputeTimer()


@contextlib.contextmanager
def _compute_window(device: torch.device | None = None):
    """Time the enclosed block on the device with CUDA events.

    A no-op unless a caller has enabled ``_COMPUTE_TIMER`` and CUDA is the
    device in use, so the inference functions can wrap their forward pass
    unconditionally.
    """
    use_cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")
    if not _COMPUTE_TIMER.enabled or not use_cuda:
        yield
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        end.synchronize()
        _COMPUTE_TIMER.record(start.elapsed_time(end))


# --- Hard-Negative Condition Transforms ---
def build_degradation_transform(condition: str) -> A.Compose | None:
    """Return an Albumentations transform that applies a visual degradation.
    These only modify pixel values, not geometry, so ground-truth boxes
    remain valid.

    Args:
        condition (str): The type of degradation to apply. One of:
            "clean", "rain", "night", "motion_blur"

    Returns:
        A.Compose transform, or None for "clean".
    """
    valid_conditions = {"clean", "rain", "night", "motion_blur"}
    if condition not in valid_conditions:
        raise ValueError(
            f"Unknown evaluation condition {condition!r}. "
            f"Expected one of: {', '.join(sorted(valid_conditions))}."
        )

    # Get hard-negative augmentation settings from config
    hn = cfg_train.augmentation.get("hard_negative", {})

    # Apply the degradation based on the condition
    if condition == "rain":
        rr = hn.get("random_rain", {})
        return A.Compose([A.RandomRain(
            brightness_coefficient=rr.get("brightness_coefficient", 0.9),
            drop_width=rr.get("drop_width", 1),
            blur_value=rr.get("blur_value", 3),
            p=1.0,
        )])
    elif condition == "night":
        cj = hn.get("color_jitter_night", {})
        return A.Compose([A.ColorJitter(
            brightness=tuple(cj.get("brightness", [0.2, 0.5])),
            contrast=tuple(cj.get("contrast", [0.5, 0.8])),
            p=1.0,
        )])
    elif condition == "motion_blur":
        mb = hn.get("motion_blur", {})
        return A.Compose([A.MotionBlur(
            blur_limit=mb.get("blur_limit", 7), 
            p=1.0,
        )])
    return None  # clean


def build_preprocess_transform(model_format: str) -> A.Compose:
    """Resize + normalise for model input (no bbox params needed).
    
    Args:
        model_format (str): The model format to use. One of:
            "rtdetr", "yolo", "yolo_ultra", "yolo_manual"

    Returns:
        A.Compose transform.
    """
    # Get augmentation settings from config
    aug = cfg_train.augmentation

    # Get normalization settings from config based on model format
    if model_format not in aug["normalize"]:
        raise ValueError(
            f"Unknown model format {model_format!r}. "
            f"Expected one of: {', '.join(sorted(aug['normalize']))}."
        )
    norm = aug["normalize"][model_format]

    # Return the preprocessing transform
    return A.Compose([
        A.Resize(height=cfg.input_height, width=cfg.input_width),
        A.Normalize(mean=norm["mean"], std=norm["std"], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def resolve_image_path(file_name: str, annotation_path: str | None = None) -> str | None:
    """Resolve COCO image paths after a repository is moved to another machine."""
    raw = Path(file_name)
    project_root = Path(__file__).resolve().parent.parent
    candidates = [raw]
    if annotation_path:
        candidates.append(Path(annotation_path).resolve().parent / raw)
    if not raw.is_absolute():
        candidates.extend((project_root / raw, project_root / "data" / raw))

    parts = raw.parts
    if "data" in parts:
        candidates.append(project_root.joinpath(*parts[parts.index("data"):]))
    if "VisDrone" in parts:
        candidates.append(project_root / "data" / Path(*parts[parts.index("VisDrone"):]))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(str(candidate))
        if normalized not in seen and os.path.isfile(normalized):
            return normalized
        seen.add(normalized)
    return None


def _tensor_shape(value_info) -> list[int] | None:
    """Return a concrete ONNX tensor shape, or ``None`` when it is dynamic."""
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            try:
                shape.append(int(dim.dim_param))
            except (TypeError, ValueError):
                return None
        else:
            return None
    return shape


def profile_onnx_model(model_path: str) -> dict:
    """Count deployed parameters and MACs from a static ONNX graph."""
    import onnx

    model = onnx.load(model_path)
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = SymbolicShapeInference.infer_shapes(
            model, auto_merge=True, guess_output_rank=True
        )
    except Exception:
        model = onnx.shape_inference.infer_shapes(model)

    shapes = {
        value.name: _tensor_shape(value)
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    shapes.update({name: list(tensor.dims) for name, tensor in initializers.items()})

    parameter_count = sum(math.prod(tensor.dims) for tensor in initializers.values())
    macs = 0
    counted_nodes = 0
    uncounted_nodes = []

    for node in model.graph.node:
        if node.op_type not in ("Conv", "Gemm", "MatMul"):
            continue

        input_a = shapes.get(node.input[0]) if node.input else None
        input_b = shapes.get(node.input[1]) if len(node.input) > 1 else None
        output = shapes.get(node.output[0]) if node.output else None
        node_macs = 0

        if node.op_type == "Conv" and input_b and output:
            node_macs = math.prod(output) * math.prod(input_b[1:])
        elif node.op_type in ("Gemm", "MatMul") and input_a and input_b and output:
            reduction_dim = input_a[-1]
            if node.op_type == "Gemm":
                attributes = {
                    attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
                }
                if attributes.get("transA", 0):
                    reduction_dim = input_a[-2]
            node_macs = math.prod(output) * reduction_dim

        if node_macs:
            macs += node_macs
            counted_nodes += 1
        else:
            uncounted_nodes.append(node.name or node.op_type)

    if uncounted_nodes:
        raise ValueError(
            f"Could not infer shapes for {len(uncounted_nodes)} MAC-bearing ONNX nodes "
            f"(first: {uncounted_nodes[0]})."
        )

    return {
        "params_M": parameter_count / 1e6,
        "macs_G": macs / 1e9,
        "profile_source": str(Path(model_path).resolve()),
        "profile_basis": "deployed ONNX graph",
        "mac_nodes": counted_nodes,
    }


def _resolve_artifact_reference(value: str, metadata_path: Path) -> Path | None:
    """Resolve an artifact path stored as either absolute or project-relative."""
    path = Path(value)
    project_root = Path(__file__).resolve().parent.parent
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, project_root / path]
    candidates.append(metadata_path.parent / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_profile_onnx(weights_path: str) -> Path | None:
    """Find the clean ONNX graph that defines an ONNX/TRT artifact."""
    artifact = Path(weights_path)
    if artifact.suffix.lower() == ".onnx" and artifact.is_file():
        return artifact.resolve()

    metadata_path = Path(str(artifact) + ".metadata.json")
    if metadata_path.is_file():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            # Prefer FP32 so Q/DQ scale tensors are not counted as parameters.
            for key in ("fp32_onnx", "engine_source_onnx", "onnx", "model"):
                value = metadata.get(key)
                if isinstance(value, str) and value.lower().endswith(".onnx"):
                    resolved = _resolve_artifact_reference(value, metadata_path)
                    if resolved is not None:
                        return resolved
        except (OSError, ValueError, TypeError):
            pass

    sibling = artifact.with_suffix(".onnx")
    return sibling.resolve() if sibling.is_file() else None


def profile_deployment_artifact(weights_path: str) -> dict:
    """Profile an ONNX or TensorRT artifact through its source ONNX graph."""
    source = resolve_profile_onnx(weights_path)
    if source is None:
        raise FileNotFoundError(
            "No source ONNX graph was found. Keep the engine metadata sidecar and source ONNX "
            "next to the TensorRT engine to report Parameters and MACs."
        )
    return profile_onnx_model(str(source))



# --- Model Loaders ---
def load_yolo_ultralytics(weights_path: str) -> YOLO:
    """Load a YOLO model via the Ultralytics high-level API.

    Args:
        weights_path (str): Path to the YOLO weights file.

    Returns:
        YOLO: Loaded YOLO model.
    """

    if YOLO is None:
        raise ImportError("Ultralytics is required to load a yolo_ultra model.")

    # Load the YOLO model
    model = YOLO(weights_path)
    enforce_fp32_anchor_cache(model)
    return model

def load_compatible_state_dict(model: torch.nn.Module, state_dict: dict, checkpoint_name: str) -> None:
    """Load an evaluation checkpoint only when every tensor is compatible."""
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_name} does not contain a state dictionary.")

    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    mismatched = sorted(
        key
        for key in set(model_state) & set(state_dict)
        if getattr(state_dict[key], "shape", None) != model_state[key].shape
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"{len(missing)} missing (e.g. {missing[:3]})")
        if unexpected:
            details.append(f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
        if mismatched:
            details.append(f"{len(mismatched)} shape-mismatched (e.g. {mismatched[:3]})")
        raise RuntimeError(
            f"Checkpoint {checkpoint_name} is incompatible with the configured model: "
            + "; ".join(details)
        )
    model.load_state_dict(state_dict, strict=True)



def load_yolo_manual(weights_path: str, device: torch.device, eval_coco: bool = False) -> DetectionModel:
    """Load a YOLO DetectionModel from a manual-training checkpoint.
    
    Args:
        weights_path (str): Path to the YOLO weights file.
        device (torch.device): Device to load the model onto.
        eval_coco (bool): Whether to load the model for 80-class COCO evaluation.

    Returns:
        DetectionModel: Loaded YOLO model.
    """
    if DetectionModel is None:
        raise ImportError("Ultralytics is required to load a yolo_manual model.")

    # Load YOLO model
    nc = 80 if eval_coco else cfg.num_classes
    model = DetectionModel(cfg_train.yolo_train_config["model_yaml_path"], ch=3, nc=nc)

    # Load checkpoint
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Prefer EMA weights
    ema = ckpt.get("ema")
    if ema is not None and hasattr(ema, "__contains__") and "module" in ema:
        state_dict = ema["module"]
    elif ema is not None and hasattr(ema, "state_dict"):
        state_dict = ema.state_dict()
    else:
        model_obj = ckpt.get("model", ckpt)
        state_dict = model_obj.state_dict() if hasattr(model_obj, "state_dict") else model_obj

    load_compatible_state_dict(model, state_dict, weights_path)
    model.to(device).eval()
    enforce_fp32_anchor_cache(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Detect anchor-cache precision
# ─────────────────────────────────────────────────────────────────────────────
# Ultralytics' Detect head builds its anchor grid lazily, on the first inference
# forward for a given feature shape, and then caches it:
#
#     if self.dynamic or self.shape != shape:
#         self.anchors, self.strides = make_anchors(x["feats"], self.stride, 0.5)
#         self.shape = shape
#
# `make_anchors` takes its dtype from the feature maps it is handed. Inference
# here runs under `torch.autocast(bfloat16)`, so if that forward is the one that
# misses the cache, the anchors are built *and cached* in bf16 and every later
# batch decodes against them.
#
# That is not a rounding nicety. The grid is `arange(w) + 0.5`, and bf16 carries
# 8 mantissa bits, so above 128 its spacing is 1.0 and the half-pixel offset is
# lost outright. At 736x1280 the P3 grid is 160 wide, which corrupts 32 of its
# 160 columns by a full 0.5 px -- the rightmost fifth of every frame, and 91.9%
# of VisDrone's objects sit on P3. Measured on the quarter-budget semi-supervised
# student, that is 18.14 -> 17.17 mAP.
#
# Nothing reports it. The run completes, the prediction count is unchanged, and
# the boxes are all plausible. `main()` happens to be safe only because it
# profiles MACs in fp32 before evaluating, which populates the cache as a side
# effect -- and only at the batch it profiles with (1). Any other entry point,
# and any batch > 1, loses it again. Hence an explicit priming step and a
# fail-closed check, rather than a comment asking future callers to be careful.
def detect_head(model):
    """Return the Ultralytics Detect head of a model, or None if it has none."""
    inner = getattr(model, "model", None)
    # `yolo_ultra` wraps DetectionModel one level deeper than `yolo_manual`.
    for candidate in (inner, getattr(inner, "model", None)):
        if isinstance(candidate, torch.nn.Sequential) and len(candidate):
            head = candidate[-1]
            if hasattr(head, "stride") and hasattr(head, "shape"):
                return head
    return None


def enforce_fp32_anchor_cache(model) -> None:
    """Make the Detect head build its anchor grid in fp32 whatever the autocast context.

    Priming the cache with a warm-up forward does not work: the cache holds
    exactly one shape, so a run that ends on a short final batch -- 1610 images
    at batch 8 leaves 2 -- misses again on that batch and rebuilds the grid in
    whatever dtype the enclosing autocast is using. Any shape change reopens the
    hole, so the fix has to be at the point of construction.

    This wraps the head's decode so the cache is populated from fp32 stand-ins
    before the head's own code looks at it. The original method then sees a cache
    hit and decodes against fp32 anchors, so none of Ultralytics' decode
    arithmetic is reimplemented here -- only the dtype the grid is built with.
    The coupling is to two attribute names and the miss condition; if a future
    Ultralytics changes those, `assert_fp32_anchor_cache` fails loudly rather
    than letting the precision loss return silently.

    Idempotent, and a no-op for models without a Detect head.
    """
    head = detect_head(model)
    if head is None or getattr(head, "_fp32_anchors_enforced", False):
        return
    if not hasattr(head, "_get_decode_boxes"):
        raise RuntimeError(
            "Detect head has no '_get_decode_boxes'; this Ultralytics version moved the "
            "anchor cache, so the fp32 enforcement below no longer applies. Re-check "
            "where make_anchors is called before trusting any reported mAP."
        )

    from ultralytics.utils.tal import make_anchors

    original = head._get_decode_boxes

    def _decode_with_fp32_anchors(x):
        feats = x["feats"]
        if head.dynamic or head.shape != feats[0].shape:
            # make_anchors takes its dtype from the features it is handed, so
            # hand it fp32 ones. Only runs on a cache miss.
            fp32_feats = [f.float() for f in feats]
            head.anchors, head.strides = (
                a.transpose(0, 1) for a in make_anchors(fp32_feats, head.stride, 0.5)
            )
            head.shape = feats[0].shape
        # The cache is now correct and current; stop the original from rebuilding
        # it in the ambient dtype, then restore the caller's setting.
        was_dynamic = head.dynamic
        head.dynamic = False
        try:
            return original(x)
        finally:
            head.dynamic = was_dynamic

    head._get_decode_boxes = _decode_with_fp32_anchors
    head._fp32_anchors_enforced = True


def assert_fp32_anchor_cache(model, where: str) -> None:
    """Raise if the Detect head is holding a reduced-precision anchor cache.

    Cheap enough to call after an evaluation as well as before one: a shape the
    caller did not anticipate would rebuild the cache mid-run under autocast,
    and this is what turns that into a failure instead of a quiet ~1 mAP loss.
    """
    head = detect_head(model)
    if head is None:
        return
    for name in ("anchors", "strides"):
        tensor = getattr(head, name, None)
        if tensor is None:
            raise RuntimeError(
                f"Detect head has no cached {name!r} {where}. The anchor cache must be "
                f"built in fp32 before inference; call enforce_fp32_anchor_cache() on the model."
            )
        if tensor.dtype != torch.float32:
            raise RuntimeError(
                f"Detect head {name!r} is {tensor.dtype} {where}, not float32. The anchor "
                f"grid was built under autocast, which drops the half-pixel offset above "
                f"x=128 and silently costs roughly 1 mAP at 736x1280. Call "
                f"enforce_fp32_anchor_cache() on the model before running inference."
            )


def load_rtdetr(weights_path: str, device: torch.device, eval_coco: bool = False) -> RTDETR:
    """Load an RT-DETR model from a custom-training checkpoint.
    
    Args:
        weights_path (str): Path to the RT-DETR weights file.
        device (torch.device): Device to load the model onto.
        eval_coco (bool): Whether to load the model for 80-class COCO evaluation.

    Returns:
        RTDETR: Loaded RT-DETR model.
    """
    # Initialize RT-DETR model
    nc = 80 if eval_coco else cfg.num_classes
    variant = cfg_train.rtdetr_train_config.get("variant", "small")
    model = build_rtdetr_model(variant=variant, pretrained_backbone=False, num_classes=nc)

    # Load checkpoint
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Prefer EMA weights
    if "ema" in ckpt and "module" in ckpt["ema"]:
        state_dict = ckpt["ema"]["module"]
    else:
        state_dict = ckpt.get("model", ckpt)

    import re
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        new_k = re.sub(r'encoder\.input_proj\.(\d+)\.0\.', r'encoder.input_proj.\g<1>.conv.', new_k)
        new_k = re.sub(r'encoder\.input_proj\.(\d+)\.1\.', r'encoder.input_proj.\g<1>.norm.', new_k)
        new_k = re.sub(r'decoder\.enc_output\.0\.', r'decoder.enc_output.proj.', new_k)
        new_k = re.sub(r'decoder\.enc_output\.1\.', r'decoder.enc_output.norm.', new_k)
        new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        print(f"Warning: {len(missing)} missing and {len(unexpected)} unexpected keys when loading RT-DETR weights.")
    model.to(device).eval()
    return model


# --- Inference Functions ---
def load_onnx(weights_path: str):
    """Load an ONNX model for evaluation using ONNX Runtime.
    
    Args:
        weights_path (str): Path to the .onnx file.
        
    Returns:
        onnxruntime.InferenceSession
    """
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"Loading ONNX model from {weights_path} with providers {providers}")
    session = ort.InferenceSession(weights_path, providers=providers)
    return session

def infer_onnx(session, img_tensors, orig_hws, conf, onnx_arch: str, eval_coco: bool = False) -> list[dict]:
    """Run ONNX inference and return COCO-format predictions on a batch.

    Deliberately reports no ``compute_latency_ms``. ``session.run`` takes host
    numpy in and hands host numpy back, so the CUDA execution provider does its
    own H2D and D2H inside that call and there is no span here that means the
    same thing as the TensorRT and PyTorch compute windows. Timing ``run``
    anyway would put a copy-inclusive number in a copy-exclusive column and
    make the ONNX row quietly incomparable with every other row.

    Closing the gap needs ``run_with_iobinding`` against device-resident
    buffers, which is a real change to this function's data flow rather than a
    timing tweak; until then the ONNX row shows end-to-end latency only.
    """
    input_name = session.get_inputs()[0].name
    # Convert tensor to numpy for ORT
    img_np = img_tensors.cpu().numpy()
    
    outputs = session.run(None, {input_name: img_np})
    
    output_preds = []
    
    if onnx_arch == "rtdetr":
        # Assuming exported as [pred_logits, pred_boxes]
        scores = torch.from_numpy(outputs[0]).sigmoid()
        boxes = torch.from_numpy(outputs[1])
            
        max_scores, cls_indices = scores.max(dim=-1)
        keep = max_scores >= conf

        # Bounded by orig_hws, not by the output batch: a static-batch graph may
        # have been fed padding rows past the end of the dataset.
        for b in range(len(orig_hws)):
            orig_h, orig_w, image_id = orig_hws[b]
            b_keep = keep[b]
            b_boxes = boxes[b][b_keep]
            b_scores = max_scores[b][b_keep]
            b_cls = cls_indices[b][b_keep]

            for i in range(len(b_boxes)):
                cx, cy, w, h = b_boxes[i].tolist()
                bx = (cx - w / 2) * orig_w
                by = (cy - h / 2) * orig_h
                bw = w * orig_w
                bh = h * orig_h
                
                c_idx = int(b_cls[i])
                if eval_coco:
                    if c_idx not in COCO_TO_VISDRONE:
                        continue
                    cat_id = COCO_TO_VISDRONE[c_idx]
                else:
                    cat_id = _yolo_idx_to_coco_id(c_idx)
                    
                output_preds.append({
                    "image_id": int(image_id),
                    "category_id": cat_id,
                    "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
                    "score": round(float(b_scores[i]), 5),
                })
    else:
        output0 = outputs[0]
        preds_tensor = torch.from_numpy(output0) if not isinstance(output0, torch.Tensor) else output0
        if preds_tensor.ndim == 3 and preds_tensor.shape[-1] == 6:
            # End-to-end model (YOLO26): output is (batch, max_det, 6)
            output_preds.extend(
                decode_yolo_e2e_output(output0, orig_hws, conf, eval_coco)
            )
        else:
            # NMS-based model (YOLOv8, YOLO11, etc.): output is (batch, 4+nc, anchors)
            output_preds.extend(
                decode_yolo_nms_output(preds_tensor, orig_hws, conf, eval_coco)
            )

    return output_preds


def decode_yolo_e2e_output(output0, orig_hws, conf, eval_coco: bool = False) -> list[dict]:
    """Decode an end-to-end YOLO detection head into COCO predictions.

    YOLO26 is NMS-free (`end2end: True`), so the exported ONNX/TensorRT graph has
    a single output of shape (batch, max_det, 6) holding
    [x1, y1, x2, y2, score, class] already sorted by score, in *input-pixel*
    coordinates. Boxes are rescaled to the original image because the
    preprocessing is a plain resize, not a letterbox.

    ``orig_hws`` bounds the decode, so a static-batch graph fed padding rows
    past the end of the dataset contributes nothing for them.
    """
    preds = torch.as_tensor(output0)
    if preds.ndim != 3 or preds.shape[-1] != 6:
        raise ValueError(
            f"Expected an end-to-end YOLO output of shape (batch, max_det, 6), got {tuple(preds.shape)}. "
            "Re-export with src/export.py; anchor-style (batch, 4+nc, anchors) outputs are not supported."
        )

    inp_h = cfg.input_height
    inp_w = cfg.input_width
    output_preds = []

    for b in range(len(orig_hws)):
        orig_h, orig_w, image_id = orig_hws[b]
        scale_w = orig_w / inp_w
        scale_h = orig_h / inp_h

        det = preds[b]
        det = det[det[:, 4] >= conf]
        for x1, y1, x2, y2, score, cls_idx in det.tolist():
            c_idx = int(cls_idx)
            if eval_coco:
                if c_idx not in COCO_TO_VISDRONE:
                    continue
                cat_id = COCO_TO_VISDRONE[c_idx]
            else:
                cat_id = _yolo_idx_to_coco_id(c_idx)

            bx = x1 * scale_w
            by = y1 * scale_h
            bw = (x2 - x1) * scale_w
            bh = (y2 - y1) * scale_h
            output_preds.append({
                "image_id": int(image_id),
                "category_id": cat_id,
                "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
                "score": round(float(score), 5),
            })

    return output_preds


def decode_yolo_nms_output(preds_tensor, orig_hws, conf, eval_coco: bool = False) -> list[dict]:
    """Decode an NMS-based YOLO output (e.g. YOLOv8, YOLO11) into COCO predictions.

    Args:
        preds_tensor: Tensor of shape (batch, 4+nc, anchors)
        orig_hws: List of (orig_h, orig_w, image_id)
        conf: Confidence threshold
        eval_coco: Whether to evaluate in COCO-to-VisDrone mapping mode

    Returns:
        List of COCO prediction dicts.
    """
    results = non_max_suppression(
        preds_tensor, conf_thres=conf,
        iou_thres=cfg.nms_iou_threshold or 0.65,
        max_det=200,
    )
    inp_h = cfg.input_height
    inp_w = cfg.input_width
    output_preds = []
    for b, det in enumerate(results):
        if b >= len(orig_hws):
            break
        if det is None or len(det) == 0:
            continue
        orig_h, orig_w, image_id = orig_hws[b]
        det = det.cpu().numpy()
        for *xyxy, score, cls_idx in det:
            c_idx = int(cls_idx)
            if eval_coco:
                if c_idx not in COCO_TO_VISDRONE:
                    continue
                cat_id = COCO_TO_VISDRONE[c_idx]
            else:
                cat_id = _yolo_idx_to_coco_id(c_idx)
            x1, y1, x2, y2 = xyxy
            x1 = x1 / inp_w * orig_w
            y1 = y1 / inp_h * orig_h
            x2 = x2 / inp_w * orig_w
            y2 = y2 / inp_h * orig_h
            output_preds.append({
                "image_id": int(image_id),
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": round(float(score), 5),
            })
    return output_preds

# Each returns list of COCO prediction dicts
# Prediction dict: {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}
def _coco_cat_ids() -> list[int]:
    """Return sorted list of COCO category IDs from config.

    Returns:
        list[int]: Sorted list of COCO category IDs.
    """
    return sorted(cfg.category_mapping.keys())


# Mapping of 80 COCO classes (0-indexed) to the 10 VisDrone classes (0-indexed)
COCO_TO_VISDRONE = {
    0: 0,  # person -> pedestrian
    1: 2,  # bicycle -> bicycle
    2: 3,  # car -> car
    3: 9,  # motorcycle -> motor
    5: 8,  # bus -> bus
    7: 5,  # truck -> truck
}


def _yolo_idx_to_coco_id(cls_idx: int) -> int:
    """Map a 0-based model class index to the annotation's category_id.

    YOLO outputs a contiguous model class index. Dataset preparation orders
    configured COCO categories by ID, so indexing the same order restores the
    original annotation ``category_id``, including non-contiguous mappings.

    Args:
        cls_idx (int): 0-based model class index.

    Returns:
        int: COCO category ID as written in the annotation file.
    """
    class_ids = _coco_cat_ids()
    index = int(cls_idx)
    if index < 0 or index >= len(class_ids):
        raise ValueError(
            f"Model produced class index {index}, but the configured dataset has "
            f"{len(class_ids)} classes."
        )
    return class_ids[index]


def infer_yolo_ultralytics(model, batch_images_np, image_ids, conf, eval_coco: bool = False) -> list[dict]:
    """Run Ultralytics YOLO prediction on a batch of numpy images.

    Args:
        model: YOLO model.
        batch_images_np: List of Numpy images (H, W, 3 RGB).
        image_ids: List of Image IDs.
        conf: Confidence threshold.
        eval_coco (bool): Whether to evaluate in COCO-to-VisDrone mapping mode.

    Returns:
        list[dict]: List of COCO prediction dicts.
    """
    predict_kwargs = dict(
        conf=conf,
        verbose=False,
        batch=len(batch_images_np),
        imgsz=(cfg.input_height, cfg.input_width),
        # Match the fixed rectangular preprocessing used by ONNX/TensorRT.
        rect=False,
    )
    if not getattr(model.model, 'end2end', False) and cfg.nms_iou_threshold is not None:
        predict_kwargs["iou"] = cfg.nms_iou_threshold
    
    # YOLO predict natively supports list of numpy arrays
    results = model.predict(batch_images_np, **predict_kwargs)
    
    preds = []
    for b, res in enumerate(results):
        image_id = image_ids[b]
        if res and len(res.boxes):
            boxes = res.boxes
            for xyxy, c, s in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = xyxy

                # Convert COCO to VisDrone classes
                c_idx = int(c)
                if eval_coco:
                    if c_idx not in COCO_TO_VISDRONE:
                        continue
                    cat_id = COCO_TO_VISDRONE[c_idx]
                else:
                    cat_id = _yolo_idx_to_coco_id(c_idx)

                preds.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                })
    return preds


def infer_yolo_manual(model, img_tensors, orig_hws, conf, device, eval_coco: bool = False) -> list[dict]:
    """Run manual YOLO inference and return COCO-format predictions on a batch.

       Supports both end-to-end models (e.g., YOLO26) whose output is already
       ch, N, 6) in [x1, y1, x2, y2, conf, cls] format, and traditional
       NMS-based models (e.g., YOLO26) whose raw (batch, 4+nc, anchors) output
       is normalized to the same (N, 6) format by non_max_suppression.

    Args:
        model: Ultralytics DetectionModel (e2e or NMS-based).
        img_tensors: Preprocessed image tensor (B, 3, H, W).
        orig_hws: List of (orig_h, orig_w, image_id).
        conf: Confidence threshold.
        device: Device to run inference on.
        eval_coco (bool): Whether to evaluate in COCO-to-VisDrone mapping mode.

    Returns:
        list[dict]: List of COCO prediction dicts.
    """
    # Get model predictions
    amp_cfg = get_amp_settings()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_cfg["dtype"], enabled=amp_cfg["enabled"]):
            with _compute_window(device):
                preds = model(img_tensors.to(device))

    # Handle tuple output (predictions, feature_maps)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if getattr(model, 'end2end', False):
        # End-to-end model (e.g., YOLO26): output is already (batch, N, 6)
        return decode_yolo_e2e_output(preds, orig_hws, conf, eval_coco)
    else:
        # NMS-based model (e.g., YOLOv8, YOLO11): raw output is (batch, 4+nc, anchors)
        return decode_yolo_nms_output(preds, orig_hws, conf, eval_coco)


def infer_rtdetr(model, img_tensors, orig_hws, conf, device, eval_coco: bool = False) -> list[dict]:
    """Run RT-DETR inference and return COCO-format predictions on a batch.

    Args:
        model: RT-DETR model.
        img_tensors: Tensor (B, 3, H, W).
        orig_hws: List of (orig_h, orig_w, image_id).
        conf: Confidence threshold.
        device: Device to load the model onto.
        eval_coco (bool): Whether to evaluate in COCO-to-VisDrone mapping mode.

    Returns:
        list[dict]: List of COCO prediction dicts.
    """
    # Get model predictions
    amp_cfg = get_amp_settings()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_cfg["dtype"], enabled=amp_cfg["enabled"]):
            with _compute_window(device):
                outputs = model(img_tensors.to(device))

    scores = outputs["pred_logits"].sigmoid()        # (B, Q, nc)
    boxes = outputs["pred_boxes"]                    # (B, Q, 4) normalised cxcywh

    # Find the max score and class index for all queries
    max_scores, cls_indices = scores.max(dim=-1)     # (B, Q), (B, Q)
    
    # Filter by confidence threshold
    keep = max_scores >= conf                        # (B, Q) bool mask
    
    # Copy relevant tensors to CPU
    keep = keep.cpu()
    max_scores = max_scores.cpu()
    cls_indices = cls_indices.cpu()
    boxes = boxes.cpu()

    # Convert predictions to COCO format
    cat_ids = _coco_cat_ids()
    output = []
    
    for b in range(scores.shape[0]):
        orig_h, orig_w, image_id = orig_hws[b]
        
        # Get indices of queries that passed the threshold
        q_indices = torch.where(keep[b])[0]
        for q in q_indices:
            q = q.item()
            score = max_scores[b, q].item()
            cls_idx = cls_indices[b, q].item()
            c_idx = int(cls_idx)
            
            # Convert COCO to VisDrone classes
            if eval_coco:
                if c_idx not in COCO_TO_VISDRONE:
                    continue
                cat_id = COCO_TO_VISDRONE[c_idx]
            else:
                cat_id = cat_ids[c_idx]

            # Convert normalised cxcywh to pixel xywh in original image coords
            cx, cy, w, h = boxes[b, q].tolist()
            px = (cx - w / 2) * orig_w
            py = (cy - h / 2) * orig_h
            pw = w * orig_w
            ph = h * orig_h

            output.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(px), float(py), float(pw), float(ph)],
                "score": float(score),
            })
    return output


# --- Evaluation Engine ---
class EvalDataset(torch.utils.data.Dataset):
    """Evaluation dataset for image object detection.

    Every image in ``image_ids`` must be readable. An image that is skipped here
    is still scored by COCOeval, which counts its ground truth as missed and
    silently deflates mAP, so an unreadable image is a hard error instead.

    Args:
        coco_gt (COCO): COCO ground truth object.
        image_ids (list): List of image IDs.
        degradation (callable): Degradation function to apply to images.
        preprocess (callable): Preprocessing function to apply to images.
        model_type (str): The type of model being evaluated.
        annotation_path (str | None): Path to the annotation file the ids came
            from. Lets relative ``file_name`` entries resolve against it after
            the dataset or repository has been moved.
    """
    def __init__(self, coco_gt, image_ids, degradation, preprocess, model_type,
                 annotation_path: str | None = None):
        self.coco_gt = coco_gt
        self.image_ids = image_ids
        self.degradation = degradation
        self.preprocess = preprocess
        self.model_type = model_type
        self.annotation_path = annotation_path

    def __len__(self) -> int:
        """Return the number of images in the dataset.
        
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple:
        """Return the image at the given index.

        Args:
            idx (int): The index of the image to return.

        Returns:
            tuple: Tuple of processed image, original image size, and image ID.

        Raises:
            FileNotFoundError: The image cannot be located or decoded. Dropping
                it would leave its ground truth in the COCOeval set with no
                detections, understating mAP without any warning.
        """
        # Get image from COCO dataset
        img_id = self.image_ids[idx]
        img_info = self.coco_gt.loadImgs(img_id)[0]
        img_path = resolve_image_path(img_info["file_name"], self.annotation_path)
        if img_path is None:
            raise FileNotFoundError(
                f"Evaluation image {img_id} not found: {img_info['file_name']} "
                f"(referenced by {self.annotation_path}). Re-run src/prepare_dataset.py "
                "if the dataset moved."
            )

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Evaluation image {img_id} could not be decoded: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Apply degradation
        if self.degradation is not None:
            image = self.degradation(image=image)["image"]

        # Preprocess image
        if self.model_type == "yolo_ultra":
            # Ultralytics natively handles BGR NumPy arrays
            processed_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            processed_img = self.preprocess(image=image)["image"].float()

        return processed_img, orig_h, orig_w, img_id


def eval_collate_fn(batch: list) -> tuple:
    """Collate function for evaluation dataset.

    Every item is kept. ``EvalDataset`` raises rather than returning ``None``
    for an unreadable image, so a short batch here means the end of the dataset
    and nothing else.

    Args:
        batch (list): Batch of images.

    Returns:
        tuple: Tuple of processed images, original image sizes, and image IDs.
    """
    if not batch:
        return None, []

    # Unzip batch and create list of original image sizes and ids
    processed_imgs, orig_hs, orig_ws, img_ids = zip(*batch)
    
    # Stack processed images into a single tensor if they are torch tensors
    if isinstance(processed_imgs[0], torch.Tensor):
        processed_imgs = torch.stack(processed_imgs, dim=0)
    else:
        processed_imgs = list(processed_imgs)
    
    # Zip original image sizes and ids
    orig_hws = list(zip(orig_hs, orig_ws, img_ids))
    
    return processed_imgs, orig_hws


class TRTWrapper:
    """Minimal fixed-shape TensorRT runner with strict I/O validation."""

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit as _pycuda_autoinit
            _ = _pycuda_autoinit  # import initializes the CUDA context
        except ImportError as exc:
            raise ImportError("tensorrt and pycuda are required for TRT evaluation.") from exc

        self.trt = trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            serialized = f.read()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create a TensorRT execution context.")

        self.inputs = []
        self.outputs = []
        self.bindings = {}
        self.stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise ValueError(
                    f"Tensor {name} has dynamic shape {shape}; this wrapper requires fixed shapes."
                )
            size = int(trt.volume(shape))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            device_mem = cuda.mem_alloc(size * dtype.itemsize)
            self.bindings[name] = int(device_mem)
            tensor = {
                "name": name,
                "shape": shape,
                "size": size,
                "dtype": dtype,
                "device": device_mem,
            }
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
            else:
                tensor["host"] = cuda.pagelocked_empty(size, dtype)
                self.outputs.append(tensor)

        if len(self.inputs) != 1:
            raise ValueError(f"RT-DETR TensorRT wrapper expects one input, found {len(self.inputs)}.")
        self.input_shape = self.inputs[0]["shape"]

        # Reused across calls so timing costs no per-inference allocation. They
        # bracket execute_async_v3 on the same stream, which excludes both the
        # host-side numpy conversion and the H2D/D2H copies around it.
        self._start_event = cuda.Event()
        self._end_event = cuda.Event()

    def infer(self, img_tensor):
        input_info = self.inputs[0]
        img_np = np.asarray(img_tensor.detach().cpu().numpy(), dtype=input_info["dtype"])
        img_np = np.ascontiguousarray(img_np)
        if tuple(img_np.shape) != input_info["shape"]:
            raise ValueError(
                f"TensorRT input shape mismatch: got {img_np.shape}, expected {input_info['shape']}"
            )
        if img_np.nbytes != input_info["size"] * input_info["dtype"].itemsize:
            raise ValueError("TensorRT input byte size does not match its allocated device buffer.")

        self.cuda.memcpy_htod_async(input_info["device"], img_np, self.stream)
        for name, ptr in self.bindings.items():
            if not self.context.set_tensor_address(name, ptr):
                raise RuntimeError(f"TensorRT rejected the device address for tensor {name}.")

        # Enqueued on the same stream as the execute, so reading them after the
        # existing synchronize below adds no extra host synchronisation.
        timed = _COMPUTE_TIMER.enabled
        if timed:
            self._start_event.record(self.stream)
        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v3 returned failure.")
        if timed:
            self._end_event.record(self.stream)

        for output in self.outputs:
            self.cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)
        self.stream.synchronize()
        if timed:
            _COMPUTE_TIMER.record(self._end_event.time_since(self._start_event))
        return {
            output["name"]: np.asarray(output["host"]).reshape(output["shape"]).copy()
            for output in self.outputs
        }


def load_trt(weights_path: str):
    """Load a TRT engine for evaluation.

    Both architectures use the same wrapper: engines built by src/trt_export.py
    carry no Ultralytics metadata, so the YOLO path decodes the raw end-to-end
    output tensor rather than going through the Ultralytics runtime.
    """
    return TRTWrapper(weights_path)

def infer_trt(model, img_tensors, orig_hws, conf, onnx_arch: str, eval_coco: bool = False) -> list[dict]:
    """Run TRT inference and return COCO-format predictions on a batch."""
    outputs = model.infer(img_tensors)

    if "yolo" in onnx_arch:
        name = "output0" if "output0" in outputs else next(iter(outputs))
        output0 = outputs[name]
        preds_tensor = torch.from_numpy(output0) if not isinstance(output0, torch.Tensor) else output0
        if preds_tensor.ndim == 3 and preds_tensor.shape[-1] == 6:
            # End-to-end model (YOLO26): output is (batch, max_det, 6)
            return decode_yolo_e2e_output(output0, orig_hws, conf, eval_coco)
        else:
            # NMS-based model (YOLOv8, YOLO11, etc.): output is (batch, 4+nc, anchors)
            return decode_yolo_nms_output(preds_tensor, orig_hws, conf, eval_coco)

    else:
        # RT-DETR emits raw per-query logits and normalized cxcywh boxes.
        logits = outputs['pred_logits']
        boxes = outputs['pred_boxes']

        scores = torch.from_numpy(logits).sigmoid()
        boxes = torch.from_numpy(boxes)

        output_preds = []

        max_scores, cls_indices = scores.max(dim=-1)
        keep = max_scores >= conf

        # Bounded by orig_hws: the engine's batch is fixed, so the tail of the
        # dataset is padded and those rows must not produce predictions.
        for b in range(len(orig_hws)):
            orig_h, orig_w, image_id = orig_hws[b]
            b_keep = keep[b]
            b_boxes = boxes[b][b_keep]
            b_scores = max_scores[b][b_keep]
            b_cls = cls_indices[b][b_keep]

            for box, score, cls_idx in zip(b_boxes, b_scores, b_cls):
                score = score.item()
                cls_idx = cls_idx.item()
                
                cx, cy, w, h = box.tolist()
                cx_abs, cy_abs = cx * orig_w, cy * orig_h
                w_abs, h_abs = w * orig_w, h * orig_h
                x1 = cx_abs - w_abs / 2
                y1 = cy_abs - h_abs / 2
                
                if eval_coco:
                    if cls_idx not in COCO_TO_VISDRONE:
                        continue
                    coco_cat_id = COCO_TO_VISDRONE[cls_idx]
                else:
                    coco_cat_id = _yolo_idx_to_coco_id(cls_idx)
                    
                output_preds.append({
                    "image_id": image_id,
                    "category_id": coco_cat_id,
                    "bbox": [x1, y1, w_abs, h_abs],
                    "score": score,
                })
        return output_preds


def _resolve_evaluation_batch_size(model, model_type: str, requested: int | None) -> int:
    """Resolve a fair evaluation batch and validate static runtime artifacts."""
    # Falls back to the evaluation config, not the training one: a training
    # batch is chosen for throughput, while every latency figure this harness
    # reports is defined at batch 1.
    batch_size = (
        int(cfg.eval_config["batch_size"]) if requested is None else int(requested)
    )
    if batch_size <= 0:
        raise ValueError(f"Evaluation batch size must be positive, got {batch_size}.")

    fixed_batch = None
    if model_type == "onnx":
        input_shape = model.get_inputs()[0].shape
        first_dim = input_shape[0]
        if isinstance(first_dim, (int, np.integer)):
            fixed_batch = int(first_dim)
    elif model_type == "trt":
        fixed_batch = int(model.input_shape[0])

    if fixed_batch is not None and fixed_batch != batch_size:
        raise ValueError(
            f"Requested evaluation batch {batch_size}, but the {model_type.upper()} artifact "
            f"has fixed batch {fixed_batch}. Re-export it with "
            f"export_config['batch_size']={batch_size} for a fair comparison."
        )
    return batch_size


def pad_batch_to_static_size(processed_imgs, orig_hws, model_type: str, batch_size: int):
    """Pad a short final batch up to a static graph's baked-in batch size.

    ONNX and TensorRT artifacts have the batch baked into the graph, so the last
    batch of a dataset that does not divide evenly has to be padded before the
    runtime will accept it. Every decoder walks ``orig_hws`` rather than the
    output batch, so the padded rows contribute no predictions.

    Args:
        processed_imgs: Stacked image tensor, or a list for runtimes that take
            raw arrays (returned unchanged).
        orig_hws: The batch's real (orig_h, orig_w, image_id) entries.
        model_type (str): The type of model being run.
        batch_size (int): The batch size baked into the artifact.

    Returns:
        The image batch, padded when the runtime requires a fixed batch.
    """
    if model_type not in ("onnx", "trt"):
        return processed_imgs
    if not isinstance(processed_imgs, torch.Tensor) or len(orig_hws) >= batch_size:
        return processed_imgs

    padding = processed_imgs[-1:].expand(batch_size - len(orig_hws), -1, -1, -1)
    return torch.cat([processed_imgs, padding], dim=0)


def run_evaluation(
    model,
    model_type: str,
    gt_json_path: str,
    condition: str,
    conf: float,
    device: torch.device,
    eval_coco: bool = False,
    subsample_interval: int = 1,
    onnx_arch: str = "yolo",
    num_workers: int | None = None,
    batch_size: int | None = None,
    warmup_batches: int | None = None,
) -> dict:
    """
    Run inference on every image in a COCO JSON test set under a given
    condition and return pycocotools metrics, latency, and peak memory stats.

    Args:
        model: The model to evaluate.
        model_type (str): The type of model being evaluated.
        gt_json_path (str): The path to the ground truth JSON file.
        condition (str): The condition to evaluate under. 
            Valid options are: "clean", "night", "rain", "motion_blur".
        conf (float): The confidence threshold.
        device (torch.device): The device to run the evaluation on.
        eval_coco (bool): Whether to evaluate in COCO-to-VisDrone mapping mode.
        subsample_interval (int): Subsample interval for test set images.
        onnx_arch (str): Architecture of the ONNX model ("yolo" or "rtdetr").
        num_workers (int | None): DataLoader workers. Uses the configured default when omitted.
        batch_size (int | None): Requested evaluation batch. Static ONNX/TRT
            artifacts must have the same baked batch size. Uses the dataloader
            configuration when omitted.
        warmup_batches (int | None): Untimed inference passes run before the
            timed loop, so runtime start-up cost is not charged to the first
            latency sample. 0 disables it. Uses the evaluation configuration
            when omitted.

    Returns:
        dict: Dictionary containing evaluation results.
    """
    valid_model_types = {"yolo_ultra", "yolo_manual", "rtdetr", "onnx", "trt"}
    if model_type not in valid_model_types:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            f"Expected one of: {', '.join(sorted(valid_model_types))}."
        )
    if not 0.0 <= float(conf) <= 1.0:
        raise ValueError(f"Confidence threshold must be in [0, 1], got {conf}.")
    subsample_interval = int(subsample_interval)
    if subsample_interval <= 0:
        raise ValueError(
            f"Subsample interval must be a positive integer, got {subsample_interval}."
        )
    if num_workers is not None and int(num_workers) < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
    if warmup_batches is None:
        warmup_batches = getattr(cfg, "eval_config", {}).get("warmup_batches", 3)
    warmup_batches = int(warmup_batches)
    if warmup_batches < 0:
        raise ValueError(f"warmup_batches must be non-negative, got {warmup_batches}.")
    if model_type in {"onnx", "trt"} and onnx_arch not in {"yolo", "yolo_ultra", "yolo_manual", "rtdetr"}:
        raise ValueError(
            f"Unknown deployment architecture {onnx_arch!r}; expected 'yolo' or 'rtdetr'."
        )
    degradation = build_degradation_transform(condition)

    # Fingerprint the exact labels before loading them. The path alone is not
    # enough lineage: a manifest can be regenerated in place under the same
    # name with different images, boxes, or categories.
    evaluated_ground_truth = ground_truth_lineage(gt_json_path)

    # Load COCO ground truth
    coco_gt = COCO(gt_json_path)

    # In zero-shot COCO evaluation, map fine-grained VisDrone classes to their 
    # coarser COCO equivalents to prevent misleading false negative penalties.
    if eval_coco:
        # 1: people -> 0: pedestrian
        # 4: van -> 3: car
        # 6: tricycle -> 2: bicycle
        # 7: awning-tricycle -> 2: bicycle
        gt_mapping = {1: 0, 4: 3, 6: 2, 7: 2}
        for ann in coco_gt.dataset.get('annotations', []):
            if ann['category_id'] in gt_mapping:
                ann['category_id'] = gt_mapping[ann['category_id']]
        coco_gt.createIndex()

    image_ids = sorted(
        coco_gt.imgs, key=lambda image_id: coco_gt.imgs[image_id].get("file_name", "")
    )

    if subsample_interval > 1:
        image_ids = subsample_image_ids(image_ids, subsample_interval)
        print(f"Subsampling evaluation dataset: using 1/{subsample_interval} frames. Total frames: {len(image_ids)}")

    # Build the model-specific preprocessing transform.
    if model_type in ["onnx", "trt"]:
        model_fmt = onnx_arch
    else:
        model_fmt = "yolo" if model_type in ("yolo_ultra", "yolo_manual") else "rtdetr"
    preprocess = build_preprocess_transform(model_fmt)

    # Resolve the requested batch and enforce static deployment contracts.
    batch_size = _resolve_evaluation_batch_size(model, model_type, batch_size)

    # Create DataLoader
    dataset = EvalDataset(
        coco_gt, image_ids, degradation, preprocess, model_type, annotation_path=gt_json_path
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=(
            cfg_train.dataloader_config["num_workers"] if num_workers is None else int(num_workers)
        ),
        collate_fn=eval_collate_fn,
        pin_memory=cfg_train.dataloader_config["pin_memory"] if model_type not in ("yolo_ultra", "onnx") else False
    )

    cuda_timed = torch.cuda.is_available() and device.type == "cuda"

    # The anchor grid is forced to fp32 by enforce_fp32_anchor_cache() at load
    # time; the check after the loop confirms it survived the run.
    if model_type in ("yolo_manual", "yolo_ultra"):
        enforce_fp32_anchor_cache(model)

    def infer_batch(processed_imgs, orig_hws) -> list[dict]:
        """Run one batch through the loaded runtime and return COCO predictions."""
        if model_type == "yolo_ultra":
            return infer_yolo_ultralytics(
                model, processed_imgs, [hw[2] for hw in orig_hws], conf, eval_coco
            )
        if model_type == "onnx":
            return infer_onnx(model, processed_imgs, orig_hws, conf, onnx_arch, eval_coco)
        if model_type == "trt":
            return infer_trt(model, processed_imgs, orig_hws, conf, onnx_arch, eval_coco)

        img_tensors = processed_imgs.to(device)
        if model_type == "yolo_manual":
            return infer_yolo_manual(model, img_tensors, orig_hws, conf, device, eval_coco)
        return infer_rtdetr(model, img_tensors, orig_hws, conf, device, eval_coco)

    all_predictions = []
    latencies = []
    compute_latencies = []
    warmed_up = warmup_batches <= 0
    memory_benchmark = _EvaluationMemoryBenchmark(device)
    memory_benchmark.start()
    _COMPUTE_TIMER.enabled = cuda_timed

    # Process images in batches
    for processed_imgs, orig_hws in tqdm(dataloader, desc=f"Eval ({condition})", leave=False):
        if not orig_hws:
            continue

        processed_imgs = pad_batch_to_static_size(
            processed_imgs, orig_hws, model_type, batch_size
        )
        executed_images = (
            processed_imgs.shape[0] if isinstance(processed_imgs, torch.Tensor) else len(orig_hws)
        )

        # Warm up on the first batch and discard the result. cuDNN autotuning,
        # lazy kernel loading and the ONNX Runtime / TensorRT first-inference
        # setup all land in the first call, and would otherwise be charged to the
        # first timed sample -- which dominates the mean on the small subsets the
        # INT8 accuracy gate and the exported engine metadata are measured on.
        if not warmed_up:
            for _ in range(warmup_batches):
                infer_batch(processed_imgs, orig_hws)
            if cuda_timed:
                torch.cuda.synchronize(device)
            warmed_up = True

        # Timed inference
        if cuda_timed:
            torch.cuda.synchronize(device)
        _COMPUTE_TIMER.reset()
        t0 = time.perf_counter()

        preds = infer_batch(processed_imgs, orig_hws)

        if cuda_timed:
            torch.cuda.synchronize(device)
        # Per-image latency counts the padding the runtime actually executed,
        # but only the real images contribute a sample.
        batch_latency = (time.perf_counter() - t0) * 1000 / executed_images  # ms per image
        latencies.extend([batch_latency] * len(orig_hws))
        # None whenever the active backend cannot isolate its device time, so
        # the metric is absent for that run rather than partly populated.
        if _COMPUTE_TIMER.batch_ms is not None:
            compute_latencies.extend(
                [_COMPUTE_TIMER.batch_ms / executed_images] * len(orig_hws)
            )
        all_predictions.extend(preds)
        memory_benchmark.sample()

    _COMPUTE_TIMER.enabled = False

    # A shape this function did not anticipate would have rebuilt the anchor
    # cache mid-run, under autocast. Fail rather than report the result.
    if model_type in ("yolo_manual", "yolo_ultra"):
        assert_fp32_anchor_cache(model, "after evaluation")

    # Compute COCO metrics
    metrics = _compute_coco_metrics(coco_gt, all_predictions, image_ids)
    metrics["latency_ms_mean"] = float(np.mean(latencies)) if latencies else 0.0
    metrics["latency_ms_p95"] = float(np.percentile(latencies, 95)) if latencies else 0.0
    metrics["fps"] = 1000.0 / metrics["latency_ms_mean"] if metrics["latency_ms_mean"] > 0 else 0.0
    # Device-only timings, present only for backends that can isolate their
    # forward pass. `compute_fps` is the artifact's throughput ceiling: what the
    # model would sustain if the surrounding pipeline were free. The gap between
    # it and `fps` is the pipeline's share, which is what to attack once the
    # model is fast enough.
    if compute_latencies:
        compute_mean = float(np.mean(compute_latencies))
        metrics["compute_latency_ms_mean"] = compute_mean
        metrics["compute_latency_ms_p95"] = float(np.percentile(compute_latencies, 95))
        metrics["compute_fps"] = 1000.0 / compute_mean if compute_mean > 0 else 0.0
        metrics["pipeline_overhead_ms_mean"] = float(
            metrics["latency_ms_mean"] - compute_mean
        )
    metrics["num_images"] = len(latencies)
    metrics["num_predictions"] = len(all_predictions)
    metrics["batch_size"] = int(batch_size)
    metrics["warmup_batches"] = int(warmup_batches)
    metrics["dataset"] = str(gt_json_path)
    metrics["ground_truth"] = evaluated_ground_truth
    metrics["condition"] = condition
    metrics["confidence_threshold"] = float(conf)
    metrics["subsample_interval"] = int(subsample_interval)
    metrics.update(memory_benchmark.metrics())

    return metrics


# Detection budgets handed to COCOeval, ascending. VisDrone frames hold hundreds
# of small objects, so the widest budget is well above the COCO default of 100.
# Every AP metric is reported at the widest one; the three AR metrics are
# reported at one budget each, and EVAL_AR_KEYS names them from the same tuple so
# the metric keys can never drift from the budgets they were measured at.
EVAL_MAX_DETS = (10, 100, 200)
EVAL_AR_KEYS = tuple(f"AR_{max_dets}" for max_dets in EVAL_MAX_DETS)


def _mean_valid(values: np.ndarray) -> float:
    """Mean of a COCOeval slice, ignoring its -1 'not evaluated' sentinel."""
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else 0.0


def _compute_coco_metrics(coco_gt, predictions, image_ids) -> dict:
    """Run pycocotools COCOeval and extract all standard metrics.

    Args:
        coco_gt: COCO ground truth object.
        predictions: List of COCO-format predictions.
        image_ids: List of image IDs.

    Returns:
        dict: Dictionary containing evaluation results.
    """
    # If no predictions, return empty metrics
    if not predictions:
        return _empty_metrics()

    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.params.maxDets = list(EVAL_MAX_DETS)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # precision shape: (T, R, K, A, M) = (iou_thrs, recall_thrs, cats, areas, max_dets)
    precision = coco_eval.eval["precision"]
    top = len(EVAL_MAX_DETS) - 1

    # Get 12 standard COCO stats.
    #
    # stats[0] is recomputed rather than read: pycocotools builds it with
    # `_summarize(1)`, whose maxDets defaults to 100, so it is scored at whichever
    # budget happens to equal 100 while stats[1..5] use maxDets[-1]. Deriving it
    # from the precision array keeps every AP metric on one detection budget.
    stats = coco_eval.stats
    map_50_95 = _mean_valid(precision[:, :, :, 0, top])
    if abs(map_50_95 - float(stats[0])) > 1e-9:
        # summarize() has just printed stats[0] at its own budget; say which
        # number this pipeline reports so the two cannot be confused.
        print(
            f" Reported mAP@0.50:0.95 at maxDets={EVAL_MAX_DETS[top]} = {map_50_95:.3f} "
            f"(the line above is pycocotools' default maxDets=100 summary)"
        )

    result = {
        "mAP_50_95": map_50_95,
        "mAP_50":    float(stats[1]),
        "mAP_75":    float(stats[2]),
        "AP_small":  float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large":  float(stats[5]),
        # stats[6:9] are AR at maxDets[0], maxDets[1] and maxDets[2] -- not at the
        # 1/10/100 of a default-parameter COCOeval.
        EVAL_AR_KEYS[0]: float(stats[6]),
        EVAL_AR_KEYS[1]: float(stats[7]),
        EVAL_AR_KEYS[2]: float(stats[8]),
        "AR_small":  float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large":  float(stats[11]),
    }

    # Calculate per-class AP@0.5
    cat_ids = coco_eval.params.catIds
    cat_names = {c["id"]: c["name"] for c in cfg.category_mapping.values()}
    per_class = {}

    for k_idx, cat_id in enumerate(cat_ids):
        # iou_idx=0 → IoU=0.5, area_idx=0 → all
        ap_50 = _mean_valid(precision[0, :, k_idx, 0, top])
        name = cat_names.get(cat_id, f"class_{cat_id}")
        per_class[name] = round(ap_50 * 100, 1)

    result["per_class_AP50"] = per_class
    return result


def _empty_metrics() -> dict:
    """Return a zeroed-out metrics dict when there are no predictions.

    Returns:
        dict: Dictionary containing zeroed-out metrics.
    """
    # Get keys for metrics
    keys = [
        "mAP_50_95", "mAP_50", "mAP_75",
        "AP_small", "AP_medium", "AP_large",
        *EVAL_AR_KEYS,
        "AR_small", "AR_medium", "AR_large",
    ]

    # Create dictionary of metrics with zero values
    result = {k: 0.0 for k in keys}
    result["per_class_AP50"] = {
        c["name"]: 0.0 for c in cfg.category_mapping.values()
    }
    return result


# --- Report Generation ---
EVALUATION_LINEAGE_VERSION = 1


def _file_sha256(path: str) -> str:
    """Return the full SHA-256 of an evaluated model artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ground_truth_lineage(annotation_path: str) -> dict:
    """Identify the exact COCO manifest used as evaluation ground truth."""
    return {
        "manifest": os.path.abspath(annotation_path),
        "fingerprint_schema": label_budget_meta.MANIFEST_FINGERPRINT_SCHEMA,
        "fingerprint": label_budget_meta.manifest_fingerprint(annotation_path),
    }


def _checkpoint_training_metadata(checkpoint_path: str) -> dict:
    """Read training metadata without pretending an unstamped artifact is known."""
    result = {
        "status": "not_a_pytorch_checkpoint",
        "training_regime": None,
        "label_budget": None,
        "run_provenance": None,
    }
    if Path(checkpoint_path).suffix.lower() not in {".pt", ".pth"}:
        return result

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        result["status"] = "unreadable"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if not isinstance(checkpoint, Mapping):
        result["status"] = "not_a_mapping"
        return result

    result["training_regime"] = checkpoint.get("training_regime")
    result["label_budget"] = label_budget_meta.read_budget_record(checkpoint)
    result["run_provenance"] = checkpoint.get(RUN_PROVENANCE_KEY)
    result["status"] = (
        "recorded"
        if result["training_regime"] is not None
        or result["label_budget"] is not None
        or result["run_provenance"] is not None
        else "not_recorded"
    )
    return result


def _evaluated_ground_truth(all_results: Mapping) -> list[dict]:
    """Collect unique ground-truth identities and the result cells using them."""
    manifests = {}
    for combo, metrics in all_results.items():
        if str(combo).startswith("_") or not isinstance(metrics, Mapping):
            continue
        record = metrics.get("ground_truth")
        if not isinstance(record, Mapping):
            continue
        record = dict(record)
        identity = (record.get("manifest"), record.get("fingerprint"))
        if identity not in manifests:
            record["evaluations"] = []
            manifests[identity] = record
        manifests[identity]["evaluations"].append(str(combo))
    return list(manifests.values())


def build_evaluation_lineage(
    checkpoint_path: str,
    model_type: str,
    all_results: Mapping,
) -> dict:
    """Build the immutable model/training/data identity stored with a report."""
    absolute_path = os.path.abspath(checkpoint_path)
    training = _checkpoint_training_metadata(absolute_path)
    return {
        "version": EVALUATION_LINEAGE_VERSION,
        "model_type": model_type,
        "checkpoint": {
            "path": absolute_path,
            "sha256": _file_sha256(absolute_path),
            "size_bytes": os.path.getsize(absolute_path),
            "metadata_status": training["status"],
            **({"metadata_error": training["error"]} if "error" in training else {}),
        },
        "training_regime": training["training_regime"],
        "label_budget": training["label_budget"],
        "run_provenance": training["run_provenance"],
        "ground_truth": _evaluated_ground_truth(all_results),
    }


def generate_reports(
    all_results: dict,
    model_type: str,
    output_dir: str,
    efficiency_stats: dict = None,
    lineage: dict | None = None,
):
    """Write both a JSON and a Markdown report from evaluation results.

    Args:
        all_results (dict): Dictionary containing evaluation results.
        model_type (str): The type of model being evaluated.
        output_dir (str): The directory to save the reports to.
        efficiency_stats (dict): Dictionary containing efficiency statistics.
        lineage (dict | None): Model, training, and ground-truth provenance.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add efficiency stats to results if provided
    if efficiency_stats:
        all_results["_efficiency"] = efficiency_stats
    if lineage is not None:
        all_results["_lineage"] = lineage
    
    # Get current timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Save JSON report ---
    json_path = os.path.join(output_dir, f"eval_{model_type}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON report saved to {json_path}")

    # --- Create Markdown report ---
    # Initialize list of strings for Markdown report
    md_path = os.path.join(output_dir, f"eval_{model_type}_{timestamp}.md")
    lines = [
        f"# Evaluation Report: `{model_type}`",
        f"**Generated:** {timestamp}",
        "",
    ]

    if lineage is not None:
        checkpoint = lineage.get("checkpoint", {})
        lines.extend([
            "## Evaluation Lineage",
            "",
            f"- **Checkpoint:** `{checkpoint.get('path', 'unavailable')}`",
            f"- **Checkpoint SHA-256:** `{checkpoint.get('sha256', 'unavailable')}`",
            f"- **Checkpoint metadata:** {checkpoint.get('metadata_status', 'unknown')}",
        ])
        training_regime = lineage.get("training_regime")
        if training_regime is None:
            lines.append("- **Training regime:** unavailable (not recorded in checkpoint)")
        else:
            lines.append(
                "- **Training regime:** `"
                + json.dumps(training_regime, sort_keys=True, default=str)
                + "`"
            )
        budget = lineage.get("label_budget")
        lines.append(
            "- **Label budget:** "
            + (label_budget_meta.describe(budget) if budget is not None else
               "unavailable (not stamped in checkpoint)")
        )
        run_provenance = lineage.get("run_provenance")
        lines.append(
            "- **Run provenance:** "
            + (
                f"`{run_provenance.get('tag')}`"
                if isinstance(run_provenance, Mapping)
                else "unavailable (not recorded in checkpoint)"
            )
        )
        ground_truth_records = lineage.get("ground_truth", [])
        if ground_truth_records:
            lines.append("- **Evaluated ground truth:**")
            for record in ground_truth_records:
                evaluations = ", ".join(record.get("evaluations", []))
                lines.append(
                    f"  - `{record.get('manifest')}` — fingerprint "
                    f"`{record.get('fingerprint')}` ({evaluations})"
                )
        else:
            lines.append("- **Evaluated ground truth:** unavailable")
        lines.append("")

    # Collect all (test_set, condition) keys
    combos = [c for c in all_results.keys() if not c.startswith("_")]

    # Summary table
    lines.append("## Summary Metrics\n")
    header = "| Metric | " + " | ".join(combos) + " |"
    sep = "|--------|" + "|".join(["-------:" for _ in combos]) + "|"
    lines.extend([header, sep])

    summary_keys = [
        ("mAP@0.5",      "mAP_50"),
        ("mAP@0.5:0.95", "mAP_50_95"),
        ("mAP@0.75",     "mAP_75"),
        ("AP_small",     "AP_small"),
        ("AP_medium",    "AP_medium"),
        ("AP_large",     "AP_large"),
        (f"AR@{EVAL_MAX_DETS[-1]}", EVAL_AR_KEYS[-1]),
    ]
    for label, key in summary_keys:
        vals = [f"{all_results[c].get(key, 0) * 100:.1f}" for c in combos]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    # Per-class AP@0.5 table
    lines.append("## Per-Class AP@0.5 (%)\n")
    header = "| Class | " + " | ".join(combos) + " |"
    sep = "|-------|" + "|".join(["-------:" for _ in combos]) + "|"
    lines.extend([header, sep])

    class_names = [c["name"] for c in sorted(cfg.category_mapping.values(), key=lambda x: x["id"])]
    for cls_name in class_names:
        vals = []
        for c in combos:
            pc = all_results[c].get("per_class_AP50", {})
            vals.append(f"{pc.get(cls_name, 0):.1f}")
        lines.append(f"| {cls_name} | " + " | ".join(vals) + " |")
    lines.append("")

    # Inference performance
    lines.append("## Inference Performance\n")
    header = "| Metric | " + " | ".join(combos) + " |"
    sep = "|--------|" + "|".join(["-------:" for _ in combos]) + "|"
    lines.extend([header, sep])

    for label, key, fmt in [
        ("Latency (ms)", "latency_ms_mean", ".1f"),
        ("P95 Latency (ms)", "latency_ms_p95", ".1f"),
        ("FPS", "fps", ".1f"),
        ("Compute latency (ms)", "compute_latency_ms_mean", ".2f"),
        ("Compute P95 latency (ms)", "compute_latency_ms_p95", ".2f"),
        ("Compute FPS", "compute_fps", ".1f"),
        ("Pipeline overhead (ms)", "pipeline_overhead_ms_mean", ".2f"),
        ("Peak process RSS (MB)", "process_rss_peak_MB", ".1f"),
        ("Process RSS delta (MB)", "process_rss_delta_MB", ".1f"),
        ("Peak CUDA device memory (MB)", "cuda_device_memory_peak_MB", ".1f"),
        ("CUDA device memory delta (MB)", "cuda_device_memory_delta_MB", ".1f"),
        ("Peak PyTorch CUDA allocated (MB)", "cuda_torch_peak_allocated_MB", ".1f"),
        ("Peak PyTorch CUDA reserved (MB)", "cuda_torch_peak_reserved_MB", ".1f"),
        ("Batch Size", "batch_size", ".0f"),
        ("Warmup Batches", "warmup_batches", ".0f"),
        ("Subsample Interval", "subsample_interval", ".0f"),
        ("# Images", "num_images", ".0f"),
        ("# Predictions", "num_predictions", ".0f"),
    ]:
        vals = []
        for c in combos:
            value = all_results[c].get(key)
            vals.append(f"{value:{fmt}}" if value is not None else "N/A")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    if any("compute_latency_ms_mean" in all_results[c] for c in combos):
        lines.append("")
        lines.append(
            "Latency is end-to-end per image: host-to-device copy, forward pass and "
            "decode to COCO records. Compute latency is the forward pass alone, timed "
            "with CUDA events on the device, and is the figure to compare when the "
            "model or its precision changes -- the end-to-end number carries enough "
            "host-side variance to hide a change of a few percent. Their difference "
            "is reported as pipeline overhead. ONNX Runtime reports no compute "
            "latency because `session.run` performs its own transfers internally."
        )
    if any("cuda_device_memory_peak_MB" in all_results[c] for c in combos):
        lines.append("")
        lines.append(
            "CUDA device memory includes CUDA runtime allocations (such as ONNX Runtime "
            "or TensorRT) but is device-wide and can include other processes."
        )
    lines.append("")

    # Degradation impact
    lines.append("## Degradation Impact (mAP@0.5 drop vs clean)\n")
    # Find clean baselines for each test set
    test_sets_seen = set()
    for c in combos:
        ts = c.split("/")[0] if "/" in c else c.split("_clean")[0]
        test_sets_seen.add(ts)

    # For each test set, calculate and report degradation
    for ts in sorted(test_sets_seen):
        clean_key = f"{ts}/clean"
        if clean_key not in all_results:
            continue
        baseline = all_results[clean_key].get("mAP_50", 0) * 100
        lines.append(f"**{ts}** (clean baseline: {baseline:.1f}%)\n")
        for c in combos:
            if c.startswith(ts + "/") and c != clean_key:
                cond = c.split("/")[1]
                val = all_results[c].get("mAP_50", 0) * 100
                drop = val - baseline
                lines.append(f"- {cond}: {val:.1f}% ({drop:+.1f}%)")
        lines.append("")

    # Model efficiency
    if efficiency_stats:
        lines.append("## Model Efficiency\n")
        if "params_M" in efficiency_stats:
            lines.append(f"- **Parameters:** {efficiency_stats['params_M']:.2f} M")
        if "macs_G" in efficiency_stats:
            lines.append(f"- **MACs:** {efficiency_stats['macs_G']:.2f} G")
        if efficiency_stats.get("profile_basis"):
            lines.append(f"- **Profiling Basis:** {efficiency_stats['profile_basis']}")
        if efficiency_stats.get("profile_source"):
            lines.append(f"- **Profiling Source:** `{efficiency_stats['profile_source']}`")
        lines.append("")

    # Save Markdown report
    report_text = "\n".join(lines)
    with open(md_path, "w") as f:
        f.write(report_text)
    print(f"Markdown report saved to {md_path}\n")
    print("="*80)
    print("EVALUATION REPORT")
    print("="*80)
    print(report_text)
    print("="*80 + "\n")


# --- Main CLI ---
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate trained detection models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=cfg.model_format, choices=["yolo_ultra", "yolo_manual", "rtdetr", "onnx", "trt"],
        help=f"Type of model to evaluate: 'yolo_ultra', 'yolo_manual', 'rtdetr', 'onnx', 'trt'. (default: {cfg.model_format})",
    )
    parser.add_argument(
        "--onnx-arch", type=str, default="yolo", choices=["yolo", "rtdetr"],
        help="Architecture of the ONNX/TRT model ('yolo' or 'rtdetr'). Only used when --model is 'onnx' or 'trt'.",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to the model weights file (.pt, .pth, or .onnx).",
    )
    
    # Safely get evaluation config defaults
    eval_cfg = getattr(cfg, "eval_config", {})
    default_test_sets = ",".join(eval_cfg.get("test_sets", ["val"]))
    default_conditions = ",".join(eval_cfg.get("conditions", ["clean"]))
    default_conf = eval_cfg.get("conf_threshold", 0.001)
    default_output = eval_cfg.get("output_dir", "runs/eval")
    default_batch_size = int(eval_cfg.get("batch_size", 1))

    parser.add_argument(
        "--test-sets", default=default_test_sets,
        help=f"Comma-separated list of test sets to evaluate (default: {default_test_sets}).",
    )
    parser.add_argument(
        "--conditions", default=default_conditions,
        help=f"Comma-separated list of conditions (default: {default_conditions}).",
    )
    parser.add_argument(
        "--conf", type=float, default=default_conf,
        help=f"Confidence threshold (default: {default_conf} for mAP evaluation).",
    )
    parser.add_argument(
        "--output-dir", default=default_output,
        help=f"Directory for evaluation reports (default: {default_output}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=default_batch_size,
        help=f"Evaluation batch size (default: {default_batch_size} for real-time latency).",
    )
    parser.add_argument(
        "--eval-coco", action="store_true",
        help="Evaluate pretrained COCO model by mapping its 80 classes to VisDrone classes.",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Subsample interval for evaluation (default: read test_subsample_interval from config).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Capture the artifact identity at load time, rather than after a long
    # evaluation when another job may already have published different bytes
    # at the same path.
    evaluation_lineage = build_evaluation_lineage(
        args.weights, args.model, all_results={}
    )

    # Load model
    print(f"\nLoading model: {args.model} from {args.weights}")
    if args.model == "yolo_ultra":
        model = load_yolo_ultralytics(args.weights)
        pytorch_model = model.model.to(device)
    elif args.model == "yolo_manual":
        model = load_yolo_manual(args.weights, device, eval_coco=args.eval_coco)
        pytorch_model = model
    elif args.model == "onnx":
        model = load_onnx(args.weights)
        pytorch_model = None # Profiling won't work easily for ONNX session
    elif args.model == "trt":
        model = load_trt(args.weights)
        pytorch_model = None
    else:
        model = load_rtdetr(args.weights, device, eval_coco=args.eval_coco)
        pytorch_model = model

    # --- Profile Model Efficiency ---
    if pytorch_model is not None:
        print("\nProfiling model efficiency (Parameters and MACs)...")
        
        # Create dummy input for profiling
        dummy_input = torch.randn(1, 3, cfg.input_height, cfg.input_width).to(device)
        
        # Calculate number of parameters
        params_m = sum(p.numel() for p in pytorch_model.parameters()) / 1e6

        # Calculate MACs
        pytorch_model.eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as f:
                pytorch_model(dummy_input)
        
        macs_g = (f.get_total_flops() / 2) / 1e9
        
        # Store efficiency stats
        efficiency_stats = {
            "params_M": params_m,
            "macs_G": macs_g,
        }
        
        # Print efficiency stats
        print(f"  Parameters : {params_m:.2f} M")
        print(f"  MACs       : {macs_g:.2f} G")
    else:
        print("\nProfiling deployment graph (Parameters and MACs)...")
        try:
            efficiency_stats = profile_deployment_artifact(args.weights)
            print(f"  Parameters : {efficiency_stats['params_M']:.2f} M")
            print(f"  MACs       : {efficiency_stats['macs_G']:.2f} G")
            print(f"  Source     : {efficiency_stats['profile_source']}")
        except Exception as exc:
            print(f"  WARNING: deployment profiling unavailable: {exc}")
            efficiency_stats = {}
        
    # --- Run Evaluations ---
    # Extract test sets and conditions
    test_sets = [s.strip() for s in args.test_sets.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    # Initialize results dictionary
    all_results = {}

    # Calculate total number of combinations and initialize counter
    total_combos = len(test_sets) * len(conditions)
    combo_idx = 0

    # Resolve subsampling interval
    subsample_interval = args.subsample if args.subsample is not None else cfg_train.dataloader_config.get("test_subsample_interval", 1)

    # Loop through test sets
    for ts in test_sets:
        # Get ground truth JSON path
        gt_json = os.path.join(cfg.processed_annotations_dir, f"{ts}.json")
        if not os.path.exists(gt_json):
            print(f"\nWARNING: {gt_json} not found, skipping {ts}.")
            continue

        # Loop through conditions
        for cond in conditions:
            # Increment combo index and create key
            combo_idx += 1
            key = f"{ts}/{cond}"
            print(f"\n{'='*60}")
            print(f"[{combo_idx}/{total_combos}] Evaluating: {key}")
            print(f"{'='*60}")

            # Run evaluation
            metrics = run_evaluation(
                model=model,
                model_type=args.model,
                gt_json_path=gt_json,
                condition=cond,
                conf=args.conf,
                device=device,
                eval_coco=args.eval_coco,
                subsample_interval=subsample_interval,
                onnx_arch=args.onnx_arch,
                batch_size=args.batch_size,
            )
            # Store results
            all_results[key] = metrics

            print(f"  mAP@0.5     = {metrics['mAP_50']*100:.1f}%")
            print(f"  mAP@0.5:0.95= {metrics['mAP_50_95']*100:.1f}%")
            print(
                f"  Average inference time (bs={metrics['batch_size']}) = "
                f"{metrics['latency_ms_mean']:.1f} ms per image ({metrics['fps']:.1f} FPS)"
            )
            if "compute_latency_ms_mean" in metrics:
                print(
                    f"  Compute only              = "
                    f"{metrics['compute_latency_ms_mean']:.2f} ms per image "
                    f"({metrics['compute_fps']:.1f} FPS), "
                    f"{metrics['pipeline_overhead_ms_mean']:.2f} ms pipeline overhead"
                )
            if "process_rss_peak_MB" in metrics:
                print(
                    f"  Peak process RSS = {metrics['process_rss_peak_MB']:.1f} MB "
                    f"({metrics['process_rss_delta_MB']:+.1f} MB during evaluation)"
                )
            if "cuda_device_memory_peak_MB" in metrics:
                print(
                    f"  Peak CUDA device memory = {metrics['cuda_device_memory_peak_MB']:.1f} MB "
                    f"({metrics['cuda_device_memory_delta_MB']:+.1f} MB during evaluation)"
                )
            if "cuda_torch_peak_allocated_MB" in metrics:
                print(
                    f"  Peak PyTorch CUDA allocated/reserved = "
                    f"{metrics['cuda_torch_peak_allocated_MB']:.1f}/"
                    f"{metrics['cuda_torch_peak_reserved_MB']:.1f} MB"
                )
            print(
                f"  Scope: {metrics['num_images']} images from {gt_json}, "
                f"condition={cond}, conf={args.conf}, subsample={subsample_interval}"
            )

    # --- Generate Reports ---
    # Generate reports
    print(f"\n{'='*60}")
    print("Generating reports...")
    evaluation_lineage["ground_truth"] = _evaluated_ground_truth(all_results)
    generate_reports(
        all_results,
        args.model,
        args.output_dir,
        efficiency_stats,
        lineage=evaluation_lineage,
    )
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
