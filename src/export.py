#!/usr/bin/env python3
"""Export trained detectors to ONNX and TensorRT, with optional INT8 PTQ.

Design
------
Every precision is expressed *in the ONNX graph*, never with TensorRT builder
flags.  TensorRT 11 is strongly-typed only: it removed ``BuilderFlag.FP16``,
``BuilderFlag.INT8`` and the whole ``IInt8Calibrator`` interface, so an engine
simply inherits the types of the network it parsed.  That makes ONNX the single
source of truth for both runtimes and keeps ONNX Runtime and TensorRT numerics
comparable.

The pipeline is fail-closed.  An artifact is only written after it passes:
  1. a PyTorch/ONNX parity check on a real validation image (FP32),
  2. a structural Q/DQ check for TensorRT compatibility (INT8),
  3. an mAP regression gate against the FP32 baseline (INT8).

Usage
-----
    # YOLO26n to FP32 ONNX
    python src/export.py --model yolo_manual --weights weights/yolo_manual_best.pth

    # YOLO26n to INT8 ONNX (PTQ) and a TensorRT INT8 engine
    python src/export.py --model yolo_manual --weights weights/yolo_manual_best.pth \
        --precision int8 --format onnx engine

    # RT-DETR to FP32 ONNX plus an FP16 engine
    python src/export.py --model rtdetr --weights weights/rtdetr_best.pth \
        --precision fp16 --format engine
"""

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import albumentations as A
import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from experiment_artifacts import RUN_PROVENANCE_KEY

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import configs.export_cfg as cfg
import configs.eval_cfg as cfg_eval
import configs.train_cfg as cfg_train
from evaluate import (
    build_preprocess_transform,
    load_rtdetr,
    resolve_image_path,
    run_evaluation,
)

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        create_calibrator,
        quantize_static,
        shape_inference,
    )
    from onnxruntime.quantization.calibrate import (
        HistogramCalibrater,
        HistogramCollector,
        TensorData,
        TensorsData,
        save_tensors_data,
    )
except ImportError:  # pragma: no cover - environment guard
    print("Please install onnx and onnxruntime: pip install onnx onnxruntime")
    sys.exit(1)

PRECISIONS = ("fp32", "fp16", "int8")
FORMATS = ("onnx", "engine")
YOLO_INT8_PROFILES = ("head_fp32", "all_conv")

# ─────────────────────────────────────────────────────────────────────────────
# INT8 coverage scopes
#
# A coverage scope is a named region of the graph.  `config.quantize_ops` maps
# each ONNX op type to the scope it should be quantized over, and
# `_resolve_quantize_ops` below turns that into {op_type: predicate(node_name)}.
# The RT-DETR export wrapper names the graph "/model/<module>/...".
# ─────────────────────────────────────────────────────────────────────────────
def _in_backbone(name: str) -> bool:
    return "/backbone/" in name


def _in_encoder(name: str) -> bool:
    return "/encoder/" in name


def _is_ffn_linear(name: str) -> bool:
    """The position-wise feed-forward Linears of a transformer block.

    Deliberately *not* matched: the attention projections, `sampling_offsets`
    and `attention_weights` (they steer the deformable-attention GridSample, so
    INT8 rounding moves where the model reads), and the bbox/score heads (logit
    dynamic range, and the bbox head compounds over six refinement steps).
    """
    return "/linear1/" in name or "/linear2/" in name


RTDETR_SCOPES = {
    "all": lambda name: True,
    "backbone": _in_backbone,
    "encoder": _in_encoder,
    "decoder": lambda name: "/decoder/" in name,
    "backbone_encoder": lambda name: _in_backbone(name) or _in_encoder(name),
    "backbone_late": lambda name: (
        "/backbone/res_layers.2/" in name or "/backbone/res_layers.3/" in name
    ),
    "backbone_last_stage": lambda name: "/backbone/res_layers.3/" in name,
    "ffn": _is_ffn_linear,
}

def _yolo26_get_layer(name: str) -> int:
    match = re.search(r"/model\.(\d+)/", name)
    return int(match.group(1)) if match else -1

def _yolo26_in_backbone(name: str) -> bool:
    return 0 <= _yolo26_get_layer(name) <= 9

def _yolo26_in_encoder(name: str) -> bool:
    # Neck/Encoder layers
    return 10 <= _yolo26_get_layer(name) <= 22

def _yolo26_in_decoder(name: str) -> bool:
    # Detect head is 23
    return _yolo26_get_layer(name) >= 23

# YOLO26 has backbone (0-9), neck/encoder (10-22), and head/decoder (23+).
# These scopes are specific to the YOLO26 ONNX graph topology and should not
# be used for other YOLO variants (e.g. YOLOv8, YOLO11) without re-profiling.
YOLO26_SCOPES = {
    "all": lambda name: True,
    "backbone": _yolo26_in_backbone,
    "encoder": _yolo26_in_encoder,
    "decoder": _yolo26_in_decoder,
    "backbone_encoder": lambda name: _yolo26_in_backbone(name) or _yolo26_in_encoder(name),
    "backbone_late": lambda name: 6 <= _yolo26_get_layer(name) <= 9,
    "backbone_last_stage": lambda name: _yolo26_get_layer(name) == 9,
}

# Scope names carried over from the previous per-architecture profile ladder,
# which named the op type in the scope.  Kept so older configs and the profile
# strings recorded in published metadata sidecars still resolve.
_SCOPE_ALIASES = {
    "all_conv": "all",
    "backbone_conv": "backbone",
    "backbone_encoder_conv": "backbone_encoder",
    "backbone_late_conv": "backbone_late",
    "backbone_last_stage_conv": "backbone_last_stage",
}

# ONNX Runtime 1.27 registers no QDQ quantizer for Add or Concat, so they fall
# through to the generic `QDQOperatorBase`, which wraps every input and output
# in Q/DQ with independent scales and no op-specific validation.  Functional but
# unpolished, hence their position at the front of the relaxation order.
GENERIC_QDQ_OPS = ("Add", "Concat")

# Conv scopes ordered widest to narrowest.  Relaxation only ever moves right.
_CONV_NARROWING = ("all", "backbone_encoder", "backbone", "backbone_late", "backbone_last_stage")

# What to give up first when the accuracy gate fails, least defensible first:
# the two ops without a purpose-built quantizer, then the FFN Gemms (the decoder
# is a low single-digit percentage of total MACs, so they buy little), then
# progressively narrower Conv coverage.
RTDETR_RELAXATION_ORDER = (
    ("Gemm", False),
    ("Add", False),
    ("Concat", False),
    ("Conv", "backbone_encoder"),
    ("Conv", "backbone"),
    ("Conv", "backbone_late"),
)


def _canonical_scope(scope) -> str | bool:
    if scope is True:
        return "all"
    if scope is None:
        return False
    if scope is False:
        return False
    return _SCOPE_ALIASES.get(str(scope), str(scope))


def _relaxation_hint(quantize_ops: dict) -> str:
    """Name the next coverage reduction to try, given the current settings.

    Only suggests strict reductions, so a caller that has already narrowed past
    a rung is never told to widen back to it.
    """
    for op_type, target in RTDETR_RELAXATION_ORDER:
        current = _canonical_scope(quantize_ops.get(op_type, False))
        if current is False:
            continue  # already off
        if target is False:
            return f"Set rtdetr_ptq_config['quantize_ops']['{op_type}'] = False"
        if (
            current in _CONV_NARROWING
            and target in _CONV_NARROWING
            and _CONV_NARROWING.index(target) > _CONV_NARROWING.index(current)
        ):
            return f"Set rtdetr_ptq_config['quantize_ops']['{op_type}'] = \"{target}\""
    return "Move to QAT"


def ptq_config(arch: str) -> dict:
    """Return the PTQ settings for an architecture.

    RT-DETR and YOLO have independent settings: they differ in what is worth
    quantizing and in how much memory calibration needs.
    """
    return cfg.rtdetr_ptq_config if arch == "rtdetr" else cfg.yolo_ptq_config


def _with_exclusions(predicate, patterns: list[str]):
    """Veto any node whose name contains one of ``patterns``."""
    return lambda name: predicate(name) and not any(p in name for p in patterns)


def _resolve_quantize_ops(
    quantize_ops: dict, scopes: dict, exclude_patterns: list[str] | None = None
) -> dict:
    """Turn a config ``quantize_ops`` mapping into {op_type: predicate}.

    Values are a scope name, ``True`` (the "all" scope) or ``False`` (skip).
    """
    spec = {}
    for op_type, scope in quantize_ops.items():
        if scope is False or scope is None:
            continue
        if scope is True:
            predicate = scopes["all"]
        else:
            key = _SCOPE_ALIASES.get(str(scope), str(scope))
            if key not in scopes:
                supported = ", ".join(scopes)
                raise ValueError(
                    f"Unsupported scope '{scope}' for op type '{op_type}'; "
                    f"choose one of: {supported}, or True/False."
                )
            predicate = scopes[key]
        spec[op_type] = _with_exclusions(predicate, exclude_patterns) if exclude_patterns else predicate

    if not spec:
        raise ValueError("quantize_ops enables no op types; nothing would be quantized.")
    return spec


def coverage_scope_id(quantize_ops: dict, exclude_patterns: list[str] | None = None) -> str:
    """Stable identifier for a coverage setting, used to key the calibration cache."""
    enabled = [
        f"{op}={'all' if scope is True else scope}"
        for op, scope in sorted(quantize_ops.items())
        if scope is not False and scope is not None
    ]
    if exclude_patterns:
        enabled.append("excl=" + ",".join(sorted(exclude_patterns)))
    return "_".join(enabled) or "none"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def input_size() -> tuple[int, int]:
    """Return the configured (height, width) network input size."""
    return cfg.input_height, cfg.input_width


def artifact_path(weights_path: str, precision: str, suffix: str) -> Path:
    """Return the canonical artifact path for a set of weights.

    Every artifact lives next to its weights and is named
    ``<stem>_<precision>.<suffix>`` so ONNX, engine and metadata files stay
    grouped and a re-export overwrites its own predecessor.
    """
    weights = Path(weights_path)
    return weights.with_name(f"{weights.stem}_{precision}{suffix}")


def staging_dir(weights_path: str):
    """Return a temporary directory alongside the weights.

    Staging on the destination filesystem keeps ``os.replace`` atomic and keeps
    multi-hundred-megabyte intermediates off ``/tmp``, which is frequently a
    RAM-backed tmpfs.
    """
    parent = Path(weights_path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=".export_", dir=parent)


def publish(staged: str | Path, final: str | Path) -> str:
    """Atomically move a staged artifact into place and return its path."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staged, final)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        # Different filesystems: copy into place, then swap atomically.
        same_fs_staged = final.with_suffix(final.suffix + ".tmp")
        shutil.copy2(staged, same_fs_staged)
        os.replace(same_fs_staged, final)
        os.remove(staged)
    return str(final)


def write_metadata(artifact: str | Path, payload: dict) -> str:
    """Write a provenance sidecar next to an artifact."""
    path = Path(str(artifact) + ".metadata.json")
    
    # Merge with existing metadata if present
    if path.exists():
        try:
            with open(path, "r") as f:
                existing_payload = json.load(f)
            existing_payload.update(payload)
            payload = existing_payload
        except Exception as e:
            print(f"Warning: Could not read existing metadata at {path}: {e}")
            
    staged = path.with_suffix(path.suffix + ".tmp")
    with open(staged, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return publish(staged, path)


@contextlib.contextmanager
def preserved_cuda_visibility():
    """Restore ``CUDA_VISIBLE_DEVICES`` after Ultralytics has clobbered it.

    ``ultralytics.utils.torch_utils.select_device("cpu")`` sets
    ``CUDA_VISIBLE_DEVICES=""`` process-wide and never puts it back, which
    silently hides the GPU from everything that runs afterwards -- including the
    TensorRT builder, which then fails with "CUDA initialization failure with
    error: 100".
    """
    sentinel = object()
    previous = os.environ.get("CUDA_VISIBLE_DEVICES", sentinel)
    try:
        yield
    finally:
        if previous is sentinel:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _create_ort_session(model_path: str, providers: list[str] | None = None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        model_path, sess_options=options, providers=providers or ["CPUExecutionProvider"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
class CocoCalibrationDataReader(CalibrationDataReader):
    """Deterministic calibration batches drawn from a COCO annotation file.

    Calibration reuses the exact preprocessing the evaluator applies, so the
    collected activation ranges describe the distribution the deployed model
    actually sees.  (Ultralytics' own calibration loader letterboxes instead,
    which does not match this project's plain-resize pipeline.)
    """

    def __init__(
        self,
        annotation_path: str,
        model_format: str,
        input_name: str,
        batch_size: int = 1,
        num_samples: int = 300,
        seed: int = 42,
        transform: A.Compose | None = None,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("Calibration batch size must be positive.")

        self.annotation_path = os.path.abspath(annotation_path)
        with open(self.annotation_path, "r") as f:
            images = list(json.load(f)["images"])

        order = np.random.default_rng(seed).permutation(len(images))
        self.images = [images[int(i)] for i in order[:num_samples]]
        self.transform = transform if transform is not None else build_preprocess_transform(model_format)
        self.input_name = input_name
        self.batch_size = int(batch_size)
        self.idx = 0
        self.pbar = None
        print(f"Calibration reader: {len(self.images)} images from {annotation_path}.")

    def __len__(self) -> int:
        return len(self.images)

    def rewind(self):
        self.idx = 0
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = None

    def get_next(self):
        if self.idx >= len(self.images):
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            return None

        if self.pbar is None:
            self.pbar = tqdm(total=len(self.images), desc="Calibrating", leave=False)

        batch = []
        while self.idx < len(self.images) and len(batch) < self.batch_size:
            info = self.images[self.idx]
            self.idx += 1
            self.pbar.update(1)

            image_path = resolve_image_path(info["file_name"], self.annotation_path)
            if image_path is None:
                print(f"WARNING: calibration image not found: {info['file_name']}")
                continue
            image = cv2.imread(image_path)
            if image is None:
                print(f"WARNING: could not read calibration image: {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(image=image)["image"]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor).permute(2, 0, 1)
            batch.append(tensor.float())

        # A static-batch graph cannot consume a short final batch.
        if len(batch) != self.batch_size:
            return None

        stacked = torch.stack(batch).cpu().numpy().astype(np.float32, copy=False)
        return {self.input_name: np.ascontiguousarray(stacked)}


class _ChunkedCalibrationDataReader(CalibrationDataReader):
    """Expose a bounded slice of another reader.

    The histogram calibrators -- entropy, percentile and MSE -- buffer every
    intermediate activation for every image before building histograms.  Feeding them the
    whole calibration set at once needs hundreds of gigabytes on a
    high-resolution backbone, so callers drive them one chunk at a time and let
    the collector merge histograms in between.

    This reader must be driven by ``_collect_calibration_ranges``, which calls
    ``start_chunk`` before each ``collect_data`` pass.  Handing it straight to
    ``quantize_static`` silently truncates calibration to a single chunk,
    because that calls ``collect_data`` exactly once.
    """

    def __init__(self, source: CalibrationDataReader, chunk_size: int):
        super().__init__()
        self.source = source
        self.chunk_size = int(chunk_size)
        self.served = 0
        self.exhausted = False
        self._pending = None

    def start_chunk(self) -> bool:
        """Reset the per-chunk budget; return False once the source is empty.

        Looks one item ahead so a source that ends exactly on a chunk boundary
        reports empty here rather than handing the calibrator a chunk with no
        data, which it rejects with "No data is collected".
        """
        self.served = 0
        if self.exhausted:
            return False
        if self._pending is None:
            self._pending = self.source.get_next()
            if self._pending is None:
                self.exhausted = True
                return False
        return True

    def get_next(self):
        if self.served >= self.chunk_size or self.exhausted:
            return None
        if self._pending is not None:
            item, self._pending = self._pending, None
        else:
            item = self.source.get_next()
        if item is None:
            self.exhausted = True
            return None
        self.served += 1
        return item

    def rewind(self):
        self.served = 0


# Calibration method name -> the CalibrationMethod stamped into the ONNX
# Runtime cache and passed to quantize_static.  "mse" has no member of its own
# (ONNX Runtime 1.27 defines min-max, entropy, percentile and distribution), so
# it borrows Percentile's: both derive a symmetric clipping range from an
# absolute-value histogram, and the label is only used to check that a cache
# matches the current run.  `calibration_cache_path` hashes the real method
# name, so an MSE cache and a percentile cache never alias.
_CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
    "mse": CalibrationMethod.Percentile,
}

# Histogram resolution for MSE, matching ONNX Runtime's percentile default.  It
# also sets the search resolution, since candidate thresholds are bin edges,
# and caps how many of those the threshold search evaluates per tensor.
_MSE_NUM_BINS = 2048
_MSE_MAX_CANDIDATES = _MSE_NUM_BINS
# Ceiling on the bins the error sum reads, generous enough that a histogram of
# the nominal resolution is never touched.  Both caps exist because ONNX
# Runtime grows the absolute-value histogram whenever a later chunk exceeds the
# current maximum, and the search is (candidates x bins).
_MSE_MAX_ERROR_BINS = 4 * _MSE_NUM_BINS
# Candidates evaluated per vectorised pass; bounds the search's peak memory to
# roughly this many rows of the histogram.
_MSE_CANDIDATE_BLOCK = 128


class _MSEHistogramCollector(HistogramCollector):
    """Choose each tensor's clipping threshold by minimising quantization MSE.

    ONNX Runtime ships min-max, entropy, percentile and distribution
    collectors; MSE -- the ``mse`` histogram calibrator of NVIDIA's
    pytorch-quantization -- is not among them, so it is implemented here on top
    of ONNX Runtime's absolute-value histogram.

    For every candidate threshold ``t`` the bin centres are put through the
    symmetric INT8 grid that ``t`` implies, and the squared error is summed
    over the histogram::

        scale  = 2t / (qmax - qmin)
        err(t) = sum_b count_b * (centre_b - clip(round(centre_b/scale), 0, qmax)*scale)^2

    The threshold with the smallest error wins.  The grid is the one ONNX
    Runtime will actually build: ``compute_scale_zp`` symmetrizes the range to
    [-t, t] and spreads it over the full [-128, 127], with zero point 0.

    Percentile clips a fixed fraction of the mass whatever the distribution
    looks like; MSE instead weighs clipping error against rounding error using
    the observed shape, so a tail is kept only while the resolution it costs is
    worth less than the outliers it saves.  Because the error is squared, MSE
    is the more conservative of the two -- a few far outliers hold the
    threshold out where percentile would cut them.
    """

    # The quantized range ONNX Runtime targets for QInt8 activations.
    _QMIN, _QMAX = -128, 127

    def __init__(self, symmetric: bool, num_bins: int):
        if not symmetric:
            # The collector histograms |x|, which cannot express an asymmetric
            # range.  The pipeline quantizes activations symmetrically anyway.
            raise ValueError("MSE calibration is implemented for symmetric ranges only.")
        super().__init__(
            method="mse",
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=None,
            percentile=None,
            scenario=None,
        )

    def collect(self, name_to_arr):
        # The base class dispatches on `method` and rejects "mse".  Absolute
        # value is also the representation whose histogram ONNX Runtime can
        # extend when a later chunk exceeds the current maximum, which is what
        # makes chunked collection exact.
        print("Collecting tensor data and making histogram ...")
        return self.collect_absolute_value(name_to_arr)

    def compute_collection_result(self):
        if not self.histogram_dict:
            raise ValueError("Histogram has not been collected. Please run collect() first.")
        print("Finding optimal threshold for each tensor using 'mse' algorithm ...")
        print(f"Number of tensors : {len(self.histogram_dict)}")
        print(f"Number of histogram bins : {self.num_bins}")

        thresholds_dict = {}
        for tensor, histogram in self.histogram_dict.items():
            hist, hist_edges = histogram[0], histogram[1]
            threshold = self._mse_threshold(hist, hist_edges)
            thresholds_dict[tensor] = TensorData(lowest=-threshold, highest=threshold)
        return thresholds_dict

    @classmethod
    def _mse_threshold(cls, hist, hist_edges):
        """Return the absolute threshold minimising quantization error.

        Candidates are the histogram's upper bin edges, evenly strided down to
        ``_MSE_MAX_CANDIDATES``, and the error sum reads at most
        ``_MSE_MAX_ERROR_BINS`` bins.  Both caps matter because ONNX Runtime
        grows the absolute-value histogram whenever a later chunk exceeds the
        current maximum: a tensor whose first chunk understated its range can
        end up with tens of thousands of bins, and an uncapped search over
        those costs quadratically in the overshoot rather than in the
        resolution that was actually asked for.

        Empty bins are dropped first -- activation histograms are mostly empty,
        and the error sum only reads populated bins.
        """
        counts = np.asarray(hist, dtype=np.float64)
        centres = (np.asarray(hist_edges[:-1], dtype=np.float64) + hist_edges[1:]) * 0.5

        populated = counts > 0
        counts, centres = counts[populated], centres[populated]

        candidates = np.asarray(hist_edges[1:], dtype=np.float64)
        candidates = candidates[candidates > 0]
        if counts.size == 0 or candidates.size == 0:
            # Nothing positive was ever observed, so there is no range to clip.
            return np.array(0, dtype=hist_edges.dtype)

        if counts.size > _MSE_MAX_ERROR_BINS:
            # Merge runs of adjacent bins, keeping their mass and its centre of
            # gravity.  Equivalent to having histogrammed more coarsely, which
            # is what an overgrown histogram effectively deserves.
            starts = np.arange(0, counts.size, int(np.ceil(counts.size / _MSE_MAX_ERROR_BINS)))
            merged = np.add.reduceat(counts, starts)
            centres = np.add.reduceat(counts * centres, starts) / merged
            counts = merged

        stride = max(1, int(np.ceil(candidates.size / _MSE_MAX_CANDIDATES)))
        # Keep the largest candidate: without it the search cannot return the
        # unclipped range, which is the right answer for a bounded activation.
        candidates = candidates[::-1][::stride][::-1]

        best_threshold, best_error = candidates[-1], np.inf
        for start in range(0, candidates.size, _MSE_CANDIDATE_BLOCK):
            # Blocked rather than one candidate at a time: the inner vector is
            # short, so per-call numpy overhead would otherwise dominate.
            block = candidates[start : start + _MSE_CANDIDATE_BLOCK, None]
            scale = 2.0 * block / (cls._QMAX - cls._QMIN)
            quantized = np.clip(np.rint(centres / scale), 0, cls._QMAX) * scale
            errors = ((centres - quantized) ** 2) @ counts
            best = int(np.argmin(errors))
            if errors[best] < best_error:
                best_threshold, best_error = float(block[best, 0]), float(errors[best])
        return np.array(best_threshold, dtype=hist_edges.dtype)


class MSECalibrater(HistogramCalibrater):
    """Histogram calibrater driven by :class:`_MSEHistogramCollector`."""

    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: list[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        symmetric: bool = True,
        num_bins: int = _MSE_NUM_BINS,
    ):
        super().__init__(
            model_path,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method="mse",
            symmetric=symmetric,
            num_bins=num_bins,
        )
        # collect_data() builds a stock HistogramCollector only when it has
        # none, so installing ours up front is all the injection needed.
        self.collector = _MSEHistogramCollector(symmetric=symmetric, num_bins=num_bins)

    def compute_data(self) -> TensorsData:
        # The base implementation identifies the method by isinstance and
        # raises on subclasses it does not know.  See _CALIBRATION_METHODS for
        # why the result is labelled Percentile.
        return TensorsData(
            _CALIBRATION_METHODS["mse"], self.collector.compute_collection_result()
        )


# Part of the calibration cache key.  Bump it whenever the meaning of a cache
# entry changes -- a different chunking strategy or a different rule for which
# tensors are collected -- so entries written under the previous definition are
# never silently reused.
_CALIBRATION_REVISION = 2


def calibration_cache_path(model_path: str, arch: str, scope: str) -> str:
    """Return a cache path keyed by the model, the data and the PTQ settings."""
    settings = ptq_config(arch)
    digest = hashlib.sha256()
    digest.update(Path(model_path).name.encode())
    digest.update(str(Path(model_path).stat().st_size).encode())
    with open(settings["calibration_annotations"], "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(
        json.dumps(
            {
                "input_size": input_size(),
                "normalization": cfg_train.augmentation["normalize"],
                "arch": arch,
                "revision": _CALIBRATION_REVISION,
                "method": settings["calibration_method"],
                "num_images": settings["num_calibration_images"],
                "seed": settings["calibration_seed"],
                "scope": scope,
                "onnxruntime": ort.__version__,
            },
            sort_keys=True,
        ).encode()
    )
    cache_dir = Path(settings["calibration_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{digest.hexdigest()[:20]}.json")


def _tensors_of_nodes(model_path: str, node_names: set[str]) -> set[str]:
    """Return the calibratable activation tensors touched by ``node_names``.

    Mirrors ONNX Runtime's own selection rule -- the inputs and outputs of a
    node, excluding initializers -- but restricted to the nodes actually being
    quantized.
    """
    model = onnx.load(model_path, load_external_data=False)
    known = {v.name for v in model.graph.value_info}
    known |= {o.name for o in model.graph.output}
    known |= {i.name for i in model.graph.input}
    initializers = {i.name for i in model.graph.initializer}

    tensors = set()
    for node in model.graph.node:
        if node.name not in node_names:
            continue
        for tensor in list(node.input) + list(node.output):
            if tensor in known and tensor not in initializers:
                tensors.add(tensor)
    return tensors


def _create_calibrator(
    method_name: str,
    model_path: str,
    op_types: list[str],
    augmented_model_path: str,
):
    """Build the calibrator for ``method_name``.

    ONNX Runtime's ``create_calibrator`` covers everything it has a
    ``CalibrationMethod`` for; MSE is this project's own calibrater and is
    constructed the same way -- augment the graph, then open a session on it.
    """
    if method_name != "mse":
        return create_calibrator(
            Path(model_path),
            op_types,
            augmented_model_path=augmented_model_path,
            calibrate_method=_CALIBRATION_METHODS[method_name],
            use_external_data_format=True,
            extra_options={"symmetric": True},
        )

    calibrator = MSECalibrater(
        Path(model_path),
        op_types,
        augmented_model_path=augmented_model_path,
        use_external_data_format=True,
        symmetric=True,
        num_bins=_MSE_NUM_BINS,
    )
    calibrator.augment_graph()
    calibrator.create_inference_session()
    return calibrator


def _collect_calibration_ranges(
    model_path: str,
    arch: str,
    op_types: list[str],
    keep_tensors: set[str],
    cache_path: str,
    method_name: str,
) -> int:
    """Calibrate only the tensors that will actually be quantized, then cache.

    ``quantize_static`` drives calibration itself, but two of its behaviours are
    wrong for this pipeline:

    1. It picks calibration tensors by *op type* alone -- ``nodes_to_quantize``
       is never consulted -- so quantizing 125 RT-DETR nodes would histogram 614
       tensors, including the whole transformer.  One of those, the anchor
       tensor from ``_generate_anchors``, is deliberately +inf (invalid anchor
       positions are masked with it), and ``np.histogram`` rejects a non-finite
       range outright.
    2. It calls ``collect_data`` once, so a chunked reader silently truncates
       calibration to a single chunk.

    Driving the calibrator directly fixes both: the tensor set is narrowed to
    the selected nodes, and each chunk is collected in turn while the histogram
    collector merges across calls.  The result is written to the cache that
    ``quantize_static`` loads, so it skips its own calibration entirely.

    Returns the number of tensors calibrated.
    """
    settings = ptq_config(arch)
    with tempfile.TemporaryDirectory(prefix="ort.calib.") as tmp_dir:
        calibrator = _create_calibrator(
            method_name,
            model_path,
            op_types,
            augmented_model_path=os.path.join(tmp_dir, "augmented.onnx"),
        )

        selected = calibrator.tensors_to_calibrate & keep_tensors
        if not selected:
            raise RuntimeError(
                f"No calibratable tensors for {arch}; the graph and the coverage "
                "settings disagree about which nodes exist."
            )
        dropped = len(calibrator.tensors_to_calibrate) - len(selected)
        calibrator.tensors_to_calibrate = selected
        print(
            f"Calibrating {len(selected)} tensors "
            f"({dropped} skipped that ONNX Runtime would have collected by op type)."
        )

        input_name = onnx.load(model_path, load_external_data=False).graph.input[0].name
        reader = CocoCalibrationDataReader(
            annotation_path=settings["calibration_annotations"],
            model_format=arch,
            input_name=input_name,
            batch_size=int(settings["calibration_batch_size"]),
            num_samples=int(settings["num_calibration_images"]),
            seed=int(settings["calibration_seed"]),
        )
        if method_name == "minmax":
            # Bounded by construction: folds ReduceMin/ReduceMax into the graph.
            calibrator.collect_data(reader)
        else:
            chunked = _ChunkedCalibrationDataReader(reader, settings["calibration_chunk_size"])
            while chunked.start_chunk():
                calibrator.collect_data(chunked)

        save_tensors_data(calibrator.compute_data(), cache_path)
        del calibrator
    return len(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Gates
# ─────────────────────────────────────────────────────────────────────────────
def validate_qdq_for_tensorrt(model_path: str) -> dict:
    """Validate an explicit INT8 Q/DQ graph without rewriting its arithmetic.

    TensorRT reads Q/DQ nodes as the quantization specification.  Silently
    removing or rewiring them would change the model, so an incompatibility is a
    hard error to be fixed where the Q/DQ graph is produced.
    """
    from onnx import TensorProto, numpy_helper

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    init_map = {init.name: init for init in model.graph.initializer}
    qdq_nodes = [n for n in model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    if not qdq_nodes:
        raise ValueError(f"{model_path} contains no Q/DQ nodes; it is not an explicit INT8 model.")

    errors = []
    q_count = sum(node.op_type == "QuantizeLinear" for node in qdq_nodes)
    for node in qdq_nodes:
        if len(node.input) < 2 or node.input[1] not in init_map:
            errors.append(f"{node.name}: scale is not a constant initializer")
            continue

        scale = np.asarray(numpy_helper.to_array(init_map[node.input[1]]))
        if scale.size == 0 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            errors.append(f"{node.name}: scale must be finite and positive")

        if len(node.input) >= 3 and node.input[2] in init_map:
            zero_point = np.asarray(numpy_helper.to_array(init_map[node.input[2]]))
            if np.any(zero_point != 0):
                errors.append(f"{node.name}: TensorRT INT8 requires symmetric zero-point 0")

        data_init = init_map.get(node.input[0])
        if data_init is not None:
            if len(data_init.dims) == 0:
                errors.append(f"{node.name}: Q/DQ on a scalar initializer is unsupported")
            if node.op_type == "DequantizeLinear" and data_init.data_type != TensorProto.INT8:
                errors.append(
                    f"{node.name}: pre-quantized input must be INT8, got ONNX dtype {data_init.data_type}"
                )
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), None)
            if axis is not None and not (-len(data_init.dims) <= axis < len(data_init.dims)):
                errors.append(f"{node.name}: axis {axis} is invalid for rank {len(data_init.dims)}")

    if errors:
        preview = "\n  - ".join(errors[:12])
        suffix = f"\n  ... and {len(errors) - 12} more" if len(errors) > 12 else ""
        raise ValueError(f"TensorRT-incompatible Q/DQ graph:\n  - {preview}{suffix}")

    return {"quantize_nodes": q_count, "dequantize_nodes": len(qdq_nodes) - q_count}


def first_preprocessed_image(annotation_path: str, model_format: str) -> torch.Tensor:
    """Return the first readable annotated image, preprocessed for the model."""
    with open(annotation_path, "r") as f:
        images = json.load(f)["images"]
    transform = build_preprocess_transform(model_format)
    for info in images:
        image_path = resolve_image_path(info["file_name"], annotation_path)
        image = cv2.imread(image_path) if image_path else None
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return transform(image=image)["image"].unsqueeze(0).float()
    raise FileNotFoundError(f"No readable images found in {annotation_path}")


def validate_onnx_parity(
    torch_model: torch.nn.Module,
    onnx_path: str,
    output_names: list[str],
    limits: dict[str, float],
    model_format: str,
    parity_manifest: str | None = None,
) -> dict:
    """Fail the export when the ONNX graph does not reproduce PyTorch."""
    onnx.checker.check_model(onnx.load(onnx_path))
    x = first_preprocessed_image(parity_manifest or cfg.val_json, model_format)

    with torch.inference_mode():
        reference = torch_model(x)
    if isinstance(reference, torch.Tensor):
        reference = (reference,)
    expected = {name: t.detach().cpu().numpy() for name, t in zip(output_names, reference)}

    session = _create_ort_session(onnx_path)
    actual = dict(
        zip(
            [o.name for o in session.get_outputs()],
            session.run(None, {session.get_inputs()[0].name: x.numpy()}),
        )
    )
    if set(actual) != set(expected):
        raise ValueError(f"Unexpected ONNX outputs: {sorted(actual)}, expected {sorted(expected)}")

    report = {}
    for name, ref in expected.items():
        got = actual[name]
        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            raise ValueError(f"Invalid {name} output: shape={got.shape}, expected={ref.shape}")
        max_abs = float(np.max(np.abs(got - ref)))
        report[name] = {
            "max_abs_error": max_abs,
            "mean_abs_error": float(np.mean(np.abs(got - ref))),
        }
        if max_abs > limits[name]:
            raise ValueError(
                f"FP32 ONNX parity failed for {name}: max abs error {max_abs:.6g} > {limits[name]:.6g}"
            )
    return report


def gate_validation_manifest(weights_path: str | None) -> tuple[str, str, str]:
    """Resolve what the accuracy gate should score against for one checkpoint.

    A budget restricts labels, so the gate must not quietly spend more of them
    than the run that produced the checkpoint was allowed. `cfg.val_json` already
    follows `label_budget`, which is right for a supervised model -- but a
    teacher-only model was selected on a *different* set: the full image pool
    scored against teacher predictions, because that costs no annotations. Gating
    such a model on budgeted ground truth would both contradict its own
    selection set and read labels its scenario says were never bought.

    Rather than reconstruct that set -- which would need the teacher and its
    cache present at export time, long after and possibly elsewhere -- this reads
    the manifest the training run already recorded in the checkpoint. It is
    exactly what the checkpoint was selected against, so FP32 and INT8 are
    compared on the run's own terms.

    Args:
        weights_path: Checkpoint being exported, or None to skip the lookup.

    Returns:
        tuple[str, str, str]: ``(manifest, targets, fallback_reason)``. The
        reason is empty when the checkpoint's own recorded manifest was used,
        and otherwise says why the configured budgeted split was substituted --
        a substitution that changes what is measured, so callers report it.
    """
    def fallback(reason: str) -> tuple[str, str, str]:
        return cfg.val_json, "ground_truth", reason

    if not weights_path or not os.path.isfile(weights_path):
        return fallback("no checkpoint to read")
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return fallback(f"checkpoint unreadable ({type(exc).__name__})")
    if not isinstance(checkpoint, dict):
        return fallback("checkpoint is not a state dictionary")

    provenance = checkpoint.get(RUN_PROVENANCE_KEY)
    if not isinstance(provenance, dict):
        return fallback("checkpoint records no run provenance")
    validation = provenance.get("identity", {}).get("validation")
    if not isinstance(validation, dict):
        return fallback("run provenance records no validation set")

    recorded = validation.get("manifest")
    if not recorded:
        return fallback("run provenance names no validation manifest")

    # Recorded paths are absolute. Retry under the current project root so a
    # moved or cloned checkout still finds its own manifests.
    candidates = [str(recorded)]
    if not os.path.isabs(recorded):
        candidates.append(os.path.join(PROJECT_ROOT, str(recorded)))
    else:
        candidates.append(os.path.join(PROJECT_ROOT, os.path.basename(str(recorded))))
        candidates.append(
            os.path.join(PROJECT_ROOT, cfg.processed_annotations_dir,
                         os.path.basename(str(recorded)))
        )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate, str(validation.get("targets", "ground_truth")), ""

    # A teacher-scored target file is content-addressed and may have been pruned.
    # Falling back changes *what is measured* -- teacher agreement becomes
    # labelled accuracy, on a different image set -- so it is named, not assumed.
    return fallback(
        f"recorded validation manifest is missing ({os.path.basename(str(recorded))}); "
        f"the gate would have scored {validation.get('targets', '?')} targets on "
        f"{validation.get('images', '?')} images"
    )


def _validation_subsample_interval(max_images: int, manifest: str | None = None) -> int:
    """Return an even validation sampling interval; non-positive means all images."""
    manifest = cfg.val_json if manifest is None else manifest
    with open(manifest, "r") as f:
        num_images = len(json.load(f)["images"])
    if num_images <= 0:
        raise ValueError(f"Validation annotation file contains no images: {manifest}")
    return 1 if max_images <= 0 else max(1, math.ceil(num_images / max_images))


def evaluate_onnx(model_path: str, arch: str, max_images: int,
                  gt_json_path: str | None = None) -> dict:
    """Evaluate an ONNX model on all or an evenly sampled validation subset."""
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    return run_evaluation(
        model=_create_ort_session(model_path, providers),
        model_type="onnx",
        gt_json_path=cfg.val_json if gt_json_path is None else gt_json_path,
        condition="clean",
        conf=float(cfg_eval.eval_config["conf_threshold"]),
        device=torch.device("cpu"),
        subsample_interval=_validation_subsample_interval(max_images, gt_json_path),
        onnx_arch=arch,
        num_workers=int(ptq_config(arch)["accuracy_num_workers"]),
        batch_size=int(cfg_eval.eval_config.get("batch_size", 1)),
    )


def evaluate_trt(engine_path: str, arch: str, max_images: int,
                 gt_json_path: str | None = None) -> dict:
    """Evaluate a TensorRT engine on all or an evenly sampled validation subset."""
    from evaluate import TRTWrapper

    return run_evaluation(
        model=TRTWrapper(engine_path),
        model_type="trt",
        gt_json_path=cfg.val_json if gt_json_path is None else gt_json_path,
        condition="clean",
        conf=float(cfg_eval.eval_config["conf_threshold"]),
        device=torch.device("cuda"),
        subsample_interval=_validation_subsample_interval(max_images, gt_json_path),
        onnx_arch=arch,
        num_workers=int(ptq_config(arch)["accuracy_num_workers"]),
        batch_size=int(cfg_eval.eval_config.get("batch_size", 1)),
    )


def check_accuracy_gate(baseline: dict, candidate: dict, label: str, arch: str) -> dict:
    """Compare two metric dicts against the architecture's mAP drop budgets."""
    settings = ptq_config(arch)
    map50_drop = baseline["mAP_50"] - candidate["mAP_50"]
    coco_drop = baseline["mAP_50_95"] - candidate["mAP_50_95"]
    max_map50 = float(settings["max_map50_drop"])
    max_coco = float(settings["max_map50_95_drop"])
    passed = bool(
        np.isfinite(map50_drop)
        and np.isfinite(coco_drop)
        and map50_drop <= max_map50
        and coco_drop <= max_coco
    )
    print(
        f"  [{label}] mAP@0.5:      {baseline['mAP_50']:.4f} -> {candidate['mAP_50']:.4f} "
        f"(drop {map50_drop:+.4f}, limit {max_map50:.4f})\n"
        f"  [{label}] mAP@0.5:0.95: {baseline['mAP_50_95']:.4f} -> {candidate['mAP_50_95']:.4f} "
        f"(drop {coco_drop:+.4f}, limit {max_coco:.4f})"
    )
    return {
        "metrics": candidate,
        "map50_drop": float(map50_drop),
        "map50_95_drop": float(coco_drop),
        "passed": passed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONNX precision conversion
# ─────────────────────────────────────────────────────────────────────────────
def convert_onnx_to_fp16(fp32_path: str, output_path: str) -> str:
    """Produce an FP16 ONNX graph with FP32 inputs and outputs.

    TensorRT 11 has no FP16 builder flag, so a half-precision engine has to come
    from a half-precision graph.  I/O stays FP32 so the runtime contract, the
    evaluator and the calibration preprocessing are unchanged.
    """
    from onnxruntime.transformers import float16

    model = float16.convert_float_to_float16(
        onnx.load(fp32_path),
        keep_io_types=True,
        op_block_list=(
            float16.DEFAULT_OP_BLOCK_LIST + list(cfg.export_config.get("fp16_op_block_list", []))
        ),
    )
    staged = str(output_path) + ".tmp"
    onnx.save(model, staged)
    return publish(staged, output_path)


def _selected_nodes(model_path: str, spec: dict) -> tuple[list[str], list[str]]:
    """Resolve a coverage profile against a graph.

    Returns the node names to quantize and the op types they span.  ONNX Runtime
    ANDs ``nodes_to_quantize`` with ``op_types_to_quantize``, so both have to be
    derived from the same selection or the extra nodes are silently dropped.
    """
    model = onnx.load(model_path, load_external_data=False)
    names, op_types = [], []
    for node in model.graph.node:
        predicate = spec.get(node.op_type)
        if predicate is None or not predicate(node.name):
            continue
        if not node.name:
            raise ValueError(f"Unnamed {node.op_type} node selected in {model_path}.")
        names.append(node.name)
        if node.op_type not in op_types:
            op_types.append(node.op_type)

    if not names:
        raise ValueError(f"Profile selected no nodes in {model_path}.")
    missing = sorted(set(spec) - set(op_types))
    if missing:
        # A profile op type that matches nothing means the graph changed shape
        # (different backbone, different opset fusions) and coverage is not what
        # the profile name claims.
        print(f"WARNING: profile op types matched no nodes: {', '.join(missing)}.")
    return names, op_types


def _excluded_nodes(model_path: str, patterns: list[str]) -> list[str]:
    """Return the names of nodes matching any exclusion substring."""
    if not patterns:
        return []
    model = onnx.load(model_path, load_external_data=False)
    return [
        node.name
        for node in model.graph.node
        if node.name and any(pattern in node.name for pattern in patterns)
    ]


def _resolve_yolo_int8_profile(profile: str | None = None) -> tuple[str, list[str]]:
    """Return the normalized YOLO INT8 coverage profile and exclusions.

    ``all_conv`` (also exposed as the ``e2e`` CLI alias) removes the configured
    Detect-head exclusion. It does not remove explicit Q/DQ nodes: those carry
    the INT8 scales required by strongly typed TensorRT. Non-Conv operations and
    the floating-point model I/O can still introduce precision boundaries.
    """
    settings = ptq_config("yolo")
    requested = profile if profile is not None else settings.get("int8_profile", "head_fp32")
    normalized = {"e2e": "all_conv"}.get(str(requested).lower(), str(requested).lower())
    if normalized not in YOLO_INT8_PROFILES:
        supported = ", ".join(YOLO_INT8_PROFILES)
        raise ValueError(f"Unsupported YOLO INT8 profile '{requested}'; choose one of: {supported}.")

    patterns = (
        list(settings.get("exclude_patterns", [])) if normalized == "head_fp32" else []
    )
    return normalized, patterns


def quantize_onnx_int8(
    fp32_path: str,
    output_path: str,
    model_format: str,
    op_types: list[str],
    nodes_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
    scope: str = "default",
) -> str:
    """Write a TensorRT-compatible explicit-QDQ INT8 model.

    ``model_format`` selects the architecture's PTQ settings as well as the
    calibration preprocessing.

    Symmetric activations, symmetric per-channel weights and zero-point 0 are
    TensorRT's requirements for INT8; they also keep the ONNX Runtime and
    TensorRT results comparable, so the ORT accuracy gate is predictive of the
    engine.
    """
    settings = ptq_config(model_format)
    method_name = str(settings["calibration_method"]).lower()
    if method_name not in _CALIBRATION_METHODS:
        supported = ", ".join(_CALIBRATION_METHODS)
        raise ValueError(
            f"Unsupported calibration method: {method_name}; choose one of: {supported}."
        )

    batch_size = int(settings["calibration_batch_size"])
    graph_batch = int(cfg.export_config["batch_size"])
    if batch_size != graph_batch:
        raise ValueError(
            f"calibration_batch_size ({batch_size}) must match the static ONNX batch ({graph_batch})."
        )

    # Calibrate ahead of quantize_static and hand it the result through the
    # cache, so the tensor set matches the nodes actually being quantized.  See
    # _collect_calibration_ranges for why quantize_static cannot do this itself.
    cache_path = calibration_cache_path(fp32_path, model_format, scope)
    if not Path(cache_path).is_file():
        selected_nodes = set(nodes_to_quantize or []) - set(nodes_to_exclude or [])
        if not selected_nodes:
            raise ValueError("quantize_onnx_int8 requires an explicit nodes_to_quantize selection.")
        _collect_calibration_ranges(
            model_path=fp32_path,
            arch=model_format,
            op_types=list(op_types),
            keep_tensors=_tensors_of_nodes(fp32_path, selected_nodes),
            cache_path=cache_path,
            method_name=method_name,
        )
    else:
        print(f"Reusing cached calibration ranges: {cache_path}")

    staged = str(output_path) + ".tmp"
    quantize_static(
        model_input=fp32_path,
        model_output=staged,
        calibration_data_reader=None,
        calibration_cache_path=cache_path,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=_CALIBRATION_METHODS[method_name],
        op_types_to_quantize=list(op_types),
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=True,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibTensorRangeSymmetric": True,
            "QuantizeBias": False,
        },
    )
    _clean_scalar_qdq_nodes(staged, fp32_path)
    return publish(staged, output_path)


def _clean_scalar_qdq_nodes(model_path: str, fp32_path: str) -> int:
    """Restore FP32 scalars that ORT serialized as pre-quantized DQ inputs.

    ONNX Runtime can replace a rank-0 FP32 initializer with an INT8 initializer
    named ``<original>_quantized`` followed by ``DequantizeLinear``. TensorRT
    11.1 cannot build that pattern because the DQ data tensor has rank zero.

    This is deliberately a narrow, fail-closed rewrite. It does not touch
    ``QuantizeLinear`` nodes or dynamic Q->DQ pairs, and it only removes a DQ
    after proving that its exact scalar FP32 source exists in ``fp32_path``.
    Returns the number of restored scalar initializers.
    """
    from onnx import TensorProto, numpy_helper

    model = onnx.load(model_path)
    init_map = {init.name: init for init in model.graph.initializer}
    fp32_model = onnx.load(fp32_path)
    fp32_init_map = {init.name: init for init in fp32_model.graph.initializer}
    graph_outputs = {output.name for output in model.graph.output}

    replacements: dict[str, str] = {}
    originals_to_add = {}
    removable_initializers = set()

    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        quantized = init_map.get(node.input[0])
        if quantized is None or len(quantized.dims) != 0:
            continue

        label = node.name or (node.output[0] if node.output else "<unnamed DequantizeLinear>")
        if quantized.data_type != TensorProto.INT8:
            raise ValueError(
                f"Cannot restore scalar DQ {label}: expected an INT8 initializer, "
                f"got ONNX dtype {quantized.data_type}."
            )
        if len(node.output) != 1 or node.output[0] in graph_outputs:
            raise ValueError(
                f"Cannot restore scalar DQ {label}: its output must be one internal tensor."
            )
        if not quantized.name.endswith("_quantized"):
            raise ValueError(
                f"Cannot restore scalar DQ {label}: initializer {quantized.name!r} does not "
                "have ONNX Runtime's '_quantized' suffix."
            )

        original_name = quantized.name.removesuffix("_quantized")
        original = fp32_init_map.get(original_name)
        if original is None:
            raise ValueError(
                f"Cannot restore scalar DQ {label}: FP32 initializer {original_name!r} "
                f"is absent from {fp32_path}."
            )
        if len(original.dims) != 0 or original.data_type != TensorProto.FLOAT:
            raise ValueError(
                f"Cannot restore scalar DQ {label}: {original_name!r} is not a scalar FP32 "
                "initializer in the source graph."
            )

        existing = init_map.get(original_name)
        if existing is not None:
            same_value = (
                existing.data_type == original.data_type
                and list(existing.dims) == list(original.dims)
                and np.array_equal(
                    numpy_helper.to_array(existing), numpy_helper.to_array(original)
                )
            )
            if not same_value:
                raise ValueError(
                    f"Cannot restore scalar DQ {label}: initializer {original_name!r} "
                    "already exists with a different value."
                )
        else:
            originals_to_add[original_name] = original

        replacements[node.output[0]] = original_name
        removable_initializers.update(node.input)

    if not replacements:
        return 0

    for original in originals_to_add.values():
        model.graph.initializer.append(original)

    kept_nodes = []
    for node in model.graph.node:
        if any(output in replacements for output in node.output):
            continue
        rewritten_inputs = [replacements.get(input_name, input_name) for input_name in node.input]
        if rewritten_inputs != list(node.input):
            node.ClearField("input")
            node.input.extend(rewritten_inputs)
        kept_nodes.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(kept_nodes)

    used_names = {input_name for node in model.graph.node for input_name in node.input}
    used_names.update(output.name for output in model.graph.output)
    kept_initializers = [
        init
        for init in model.graph.initializer
        if init.name not in removable_initializers or init.name in used_names
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept_initializers)

    # Validate before replacing the input file so a failed rewrite cannot
    # destroy the original quantized graph.
    onnx.checker.check_model(model)
    staged = f"{model_path}.scalar-clean.tmp"
    onnx.save(model, staged)
    os.replace(staged, model_path)
    return len(replacements)


def preprocess_for_quantization(fp32_path: str, output_path: str) -> str:
    """Run ONNX Runtime's shape inference and pre-optimization pass for PTQ."""
    staged = str(output_path) + ".tmp"
    shape_inference.quant_pre_process(
        input_model=fp32_path, output_model_path=staged, skip_optimization=False
    )
    return publish(staged, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────────────────────────────────────
def load_yolo(weights_path: str):
    """Return an Ultralytics ``YOLO`` wrapper for native or manual checkpoints.

    ``YOLO(path)`` only accepts ``.pt``; a manual-training ``.pth`` is accepted
    silently and then fails inside ``export()``.  Manual checkpoints are
    therefore reconstituted into a ``DetectionModel`` from the configured YAML
    and wrapped explicitly.
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path}")
    if path.suffix == ".pt":
        return YOLO(str(path))

    model_yaml = cfg_train.yolo_train_config["model_yaml_path"]
    print(f"Rebuilding {model_yaml} (nc={cfg.num_classes}) from manual checkpoint {path.name}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("ema", {}).get("module") or ckpt.get("model") or ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} does not contain a recognizable state dict.")

    model = DetectionModel(model_yaml, ch=3, nc=cfg.num_classes, verbose=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"{path} does not match {model_yaml}: {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys (first missing: {missing[:3]})."
        )

    model.names = {i: entry["name"] for i, entry in cfg.category_mapping.items()}
    model.task = "detect"
    # Ultralytics reads `model.args` as a mapping during export.
    model.args = {**DEFAULT_CFG_DICT, "model": model_yaml, "task": "detect", "imgsz": max(input_size())}

    wrapper = YOLO(model_yaml, task="detect")
    wrapper.model = model.eval()
    return wrapper


def export_yolo_onnx(
    weights_path: str, precision: str, int8_profile: str | None = None
) -> dict:
    """Export YOLO to ONNX at the requested precision and gate the result.

    Supports two YOLO output formats:

    - **End-to-end** (YOLO26): output shape ``(batch, max_det, 6)`` holding
      ``[x1, y1, x2, y2, score, class]`` in input-pixel coordinates.
    - **NMS-based** (YOLOv8, YOLO11, etc.): output shape ``(batch, 4+nc, anchors)``
      holding raw box distributions and class logits.

    FP32 and FP16 exports work for both formats. INT8 PTQ is only supported for 
    YOLO26 because the quantization scope definitions and exclude patterns are 
    specific to the YOLO26 ONNX graph topology.
    """
    gate_manifest, gate_targets, gate_fallback = gate_validation_manifest(weights_path)
    print(f"Accuracy gate scores against {gate_manifest} ({gate_targets} targets)."
          + (f" NOTE: not the set this checkpoint was selected on -- {gate_fallback}."
             if gate_fallback else ""))
    model = load_yolo(weights_path)
    height, width = input_size()
    batch = int(cfg.export_config["batch_size"])

    with staging_dir(weights_path) as temp_dir:
        with preserved_cuda_visibility():
            exported = model.export(
                format="onnx",
                imgsz=(height, width),
                batch=batch,
                dynamic=bool(cfg.export_config["dynamic_axes"]),
                opset=int(cfg.export_config["opset_version"]),
                simplify=bool(cfg.export_config["simplify"]),
                device="cpu",
                verbose=False,
            )
        fp32_staged = os.path.join(temp_dir, "fp32.onnx")
        shutil.move(str(exported), fp32_staged)

        fp32_path = publish(fp32_staged, artifact_path(weights_path, "fp32", ".onnx"))
        print(f"FP32 ONNX: {fp32_path}")

        # Detect whether this is an end-to-end model (YOLO26) or NMS-based (YOLOv8/v11).
        session = _create_ort_session(fp32_path)
        output_shape = session.get_outputs()[0].shape
        del session
        is_e2e = len(output_shape) == 3 and output_shape[-1] == 6
        if is_e2e:
            print(f"Detected end-to-end YOLO output: {output_shape}")
        else:
            print(f"Detected NMS-based YOLO output: {output_shape}")

        result = {"format": "onnx", "precision": precision, "fp32_onnx": fp32_path}
        if precision == "fp32":
            write_metadata(fp32_path, result)
            return result

        if precision == "fp16":
            fp16_path = convert_onnx_to_fp16(fp32_path, artifact_path(weights_path, "fp16", ".onnx"))
            result["model"] = fp16_path
            print(f"FP16 ONNX: {fp16_path}")
            write_metadata(fp16_path, result)
            return result

        if not is_e2e:
            raise ValueError(
                "INT8 PTQ is only supported for end-to-end YOLO models (e.g. YOLO26). "
                f"This model has output shape {output_shape}, which indicates an NMS-based "
                "architecture (e.g. YOLOv8, YOLO11). The quantization scope definitions "
                "and exclude patterns in yolo_ptq_config are specific to the YOLO26 ONNX "
                "graph topology and would produce incorrect results on other architectures."
            )

        settings = ptq_config("yolo")
        validation_images = int(settings["accuracy_validation_images"])
        print("\n--- Original Model (FP32 ONNX) Evaluation ---")
        print(f"Evaluating FP32 baseline on up to {validation_images} validation images...")
        baseline = evaluate_onnx(fp32_path, "yolo", validation_images, gate_manifest)
        if baseline["mAP_50"] <= 0:
            raise RuntimeError("FP32 accuracy gate failed: baseline mAP@0.5 is zero.")

        prepared = preprocess_for_quantization(fp32_path, os.path.join(temp_dir, "prepared.onnx"))
        candidate = os.path.join(temp_dir, "int8.onnx")
        profile, patterns = _resolve_yolo_int8_profile(int8_profile)
        quantize_ops = settings["quantize_ops"]
        spec = _resolve_quantize_ops(quantize_ops, YOLO26_SCOPES)
        nodes, op_types = _selected_nodes(prepared, spec)
        excluded = _excluded_nodes(prepared, patterns)
        print(
            f"Quantizing YOLO to INT8 (profile: {profile}; {len(nodes)} nodes across "
            f"{', '.join(op_types)}; {len(excluded)} kept in FP32 via {patterns})..."
        )
        quantize_onnx_int8(
            prepared,
            candidate,
            "yolo",
            op_types=op_types,
            nodes_to_quantize=nodes,
            nodes_to_exclude=excluded,
            scope=f"{profile}_{coverage_scope_id(quantize_ops)}_excl{len(excluded)}",
        )

        qdq = validate_qdq_for_tensorrt(candidate)
        print(f"Q/DQ graph accepted: {qdq['quantize_nodes']} Q / {qdq['dequantize_nodes']} DQ nodes.")
        print("\n--- INT8 ONNX Accuracy Gate Evaluation ---")
        gate = check_accuracy_gate(
            baseline, evaluate_onnx(candidate, "yolo", validation_images, gate_manifest),
            "INT8 ONNX", "yolo"
        )
        if not gate["passed"]:
            raise RuntimeError(
                f"INT8 accuracy gate failed (mAP@0.5 drop {gate['map50_drop']:.4f}, "
                f"mAP@0.5:0.95 drop {gate['map50_95_drop']:.4f}); no INT8 artifact was written."
            )

        int8_path = publish(candidate, artifact_path(weights_path, "int8", ".onnx"))
        result.update(
            {
                "model": int8_path,
                "int8_profile": profile,
                "quantize_ops": dict(quantize_ops),
                "quantized_op_types": op_types,
                "quantized_nodes": len(nodes),
                "excluded_patterns": patterns,
                "excluded_nodes": len(excluded),
                "qdq": qdq,
                "calibration_images": int(settings["num_calibration_images"]),
                "calibration_method": settings["calibration_method"],
                "fp32_metrics": baseline,
                "int8_gate": gate,
            }
        )
        print(f"INT8 ONNX: {int8_path}")
        write_metadata(int8_path, result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# RT-DETR
# ─────────────────────────────────────────────────────────────────────────────
class RTDETRExportWrapper(torch.nn.Module):
    """Give ONNX a stable tensor-only RT-DETR output contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        return outputs["pred_logits"], outputs["pred_boxes"]


def export_rtdetr_onnx(weights_path: str, precision: str) -> dict:
    """Export RT-DETR to ONNX at the requested precision and gate the result."""
    gate_manifest, gate_targets, gate_fallback = gate_validation_manifest(weights_path)
    print(f"Accuracy gate scores against {gate_manifest} ({gate_targets} targets)."
          + (f" NOTE: not the set this checkpoint was selected on -- {gate_fallback}."
             if gate_fallback else ""))
    print(f"Loading RT-DETR from {weights_path}...")
    model = load_rtdetr(weights_path, device=torch.device("cpu")).eval()
    model.deploy()
    export_model = RTDETRExportWrapper(model).eval()

    height, width = input_size()
    dummy = torch.randn(int(cfg.export_config["batch_size"]), 3, height, width, dtype=torch.float32)
    output_names = ["pred_logits", "pred_boxes"]
    dynamic_axes = None
    if cfg.export_config["dynamic_axes"]:
        dynamic_axes = {name: {0: "batch_size"} for name in ["images", *output_names]}

    with staging_dir(weights_path) as temp_dir:
        staged = os.path.join(temp_dir, "fp32.onnx")
        torch.onnx.export(
            export_model,
            dummy,
            staged,
            input_names=["images"],
            output_names=output_names,
            opset_version=int(cfg.export_config["opset_version"]),
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        parity = validate_onnx_parity(
            export_model,
            staged,
            output_names,
            {
                "pred_logits": float(ptq_config("rtdetr")["fp32_logits_max_abs_error"]),
                "pred_boxes": float(ptq_config("rtdetr")["fp32_boxes_max_abs_error"]),
            },
            "rtdetr",
        )
        fp32_path = publish(staged, artifact_path(weights_path, "fp32", ".onnx"))
        print(f"FP32 ONNX passed PyTorch parity: {fp32_path}")

        result = {"format": "onnx", "precision": precision, "fp32_onnx": fp32_path, "fp32_parity": parity}
        if precision == "fp32":
            write_metadata(fp32_path, result)
            return result

        if precision == "fp16":
            fp16_path = convert_onnx_to_fp16(fp32_path, artifact_path(weights_path, "fp16", ".onnx"))
            result["model"] = fp16_path
            print(f"FP16 ONNX: {fp16_path}")
            write_metadata(fp16_path, result)
            return result

        settings = ptq_config("rtdetr")
        quantize_ops = settings["quantize_ops"]
        exclusions = list(settings.get("exclude_patterns", []))
        spec = _resolve_quantize_ops(quantize_ops, RTDETR_SCOPES, exclusions)

        validation_images = int(settings["accuracy_validation_images"])
        print("\n--- Original Model (FP32 ONNX) Evaluation ---")
        print(f"Evaluating FP32 baseline on up to {validation_images} validation images...")
        baseline = evaluate_onnx(fp32_path, "rtdetr", validation_images, gate_manifest)
        if baseline["mAP_50"] <= 0:
            raise RuntimeError("FP32 accuracy gate failed: baseline mAP@0.5 is zero.")

        prepared = preprocess_for_quantization(fp32_path, os.path.join(temp_dir, "prepared.onnx"))
        nodes, op_types = _selected_nodes(prepared, spec)
        candidate = os.path.join(temp_dir, "int8.onnx")
        generic = [op for op in op_types if op in GENERIC_QDQ_OPS]
        print(
            f"Quantizing RT-DETR: {len(nodes)} nodes across {', '.join(op_types)}"
            + (f" ({', '.join(generic)} use ONNX Runtime's generic QDQ path)" if generic else "")
            + "..."
        )
        quantize_onnx_int8(
            prepared,
            candidate,
            "rtdetr",
            op_types=op_types,
            nodes_to_quantize=nodes,
            scope=coverage_scope_id(quantize_ops, exclusions),
        )

        qdq = validate_qdq_for_tensorrt(candidate)
        print(f"Q/DQ graph accepted: {qdq['quantize_nodes']} Q / {qdq['dequantize_nodes']} DQ nodes.")
        print("\n--- INT8 ONNX Accuracy Gate Evaluation ---")
        gate = check_accuracy_gate(
            baseline, evaluate_onnx(candidate, "rtdetr", validation_images, gate_manifest),
            "INT8 ONNX", "rtdetr"
        )
        if not gate["passed"]:
            raise RuntimeError(
                f"INT8 accuracy gate failed (mAP@0.5 drop {gate['map50_drop']:.4f}, "
                f"mAP@0.5:0.95 drop {gate['map50_95_drop']:.4f}). "
                f"{_relaxation_hint(quantize_ops)}; no INT8 artifact was written."
            )

        int8_path = publish(candidate, artifact_path(weights_path, "int8", ".onnx"))
        result.update(
            {
                "model": int8_path,
                "quantize_ops": dict(quantize_ops),
                "excluded_patterns": exclusions,
                "quantized_nodes": len(nodes),
                "quantized_op_types": op_types,
                "qdq": qdq,
                "calibration_images": int(settings["num_calibration_images"]),
                "calibration_method": settings["calibration_method"],
                "fp32_metrics": baseline,
                "int8_gate": gate,
            }
        )
        print(f"INT8 ONNX: {int8_path}")
        write_metadata(int8_path, result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT
# ─────────────────────────────────────────────────────────────────────────────
def build_engine_subprocess(onnx_path: str, engine_path: str, precision: str) -> str:
    """Build a TensorRT engine in a clean child process.

    ``trt.Builder`` fails with "factory function returned nullptr" when it is
    constructed late in a long export process, after ONNX Runtime calibration
    and the forked evaluation dataloaders have run.  A dedicated process also
    releases the builder's considerable workspace as soon as it exits, which
    keeps the peak resident set of a full export bounded.
    """
    command = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "trt_export.py"),
        "--onnx", str(onnx_path),
        "--engine", str(engine_path),
        "--precision", precision,
    ]
    # An empty CUDA_VISIBLE_DEVICES would hide the GPU from the builder.
    env = dict(os.environ)
    if not env.get("CUDA_VISIBLE_DEVICES", "0").strip():
        env.pop("CUDA_VISIBLE_DEVICES")

    print(f"Building the TensorRT engine in a subprocess: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    for stream in (completed.stdout, completed.stderr):
        if stream and stream.strip():
            print(stream.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"TensorRT engine build failed (exit code {completed.returncode}).")
    if not os.path.exists(engine_path):
        raise RuntimeError(f"TensorRT builder reported success but {engine_path} does not exist.")
    return str(engine_path)


def export_engine(weights_path: str, arch: str, precision: str, onnx_result: dict) -> dict:
    """Build and validate a TensorRT engine before publishing it."""
    source = onnx_result.get("model", onnx_result["fp32_onnx"])
    final_engine = str(artifact_path(weights_path, precision, ".engine"))

    # The builder writes both the engine and its provenance sidecar. Keep both
    # staged until every accuracy/evaluation check succeeds so a rejected build
    # cannot replace the last known-good deployment artifact.
    with staging_dir(weights_path) as temp_dir:
        staged_engine = str(Path(temp_dir) / Path(final_engine).name)
        engine = build_engine_subprocess(source, staged_engine, precision)

        result = {
            **onnx_result,
            "format": "engine",
            "engine": engine,
            "engine_source_onnx": source,
        }
        baseline = onnx_result.get("fp32_metrics")
        gate_metrics = None
        # Resolved for every precision, not just INT8: the final artifact
        # evaluation below runs regardless and must score on the same set.
        engine_gate_manifest, engine_gate_targets, engine_gate_fallback = (
            gate_validation_manifest(weights_path)
        )
        if engine_gate_fallback:
            print(f"NOTE: engine gate is not using this checkpoint's own "
                  f"validation set -- {engine_gate_fallback}.")

        if precision == "int8" and baseline is not None:
            validation_images = int(ptq_config(arch)["accuracy_validation_images"])
            print("\n--- TensorRT INT8 Accuracy Gate Evaluation ---")
            print(f"Evaluating the TensorRT INT8 accuracy gate against "
                  f"{engine_gate_manifest} ({engine_gate_targets} targets)...")
            gate_metrics = evaluate_trt(engine, arch, validation_images, engine_gate_manifest)
            gate = check_accuracy_gate(baseline, gate_metrics, "INT8 engine", arch)
            if not gate["passed"]:
                raise RuntimeError(
                    f"TensorRT INT8 accuracy gate failed (mAP@0.5 drop {gate['map50_drop']:.4f}, "
                    f"mAP@0.5:0.95 drop {gate['map50_95_drop']:.4f})."
                )
            result["engine_gate"] = gate

        evaluation_images = int(cfg.export_config.get("evaluation_images", 0))
        evaluation_interval = _validation_subsample_interval(
            evaluation_images, engine_gate_manifest
        )
        if gate_metrics is not None and gate_metrics["subsample_interval"] == evaluation_interval:
            metrics = gate_metrics
        else:
            scope = (
                "the full validation set"
                if evaluation_images <= 0
                else f"up to {evaluation_images} images"
            )
            print("\n--- Final TensorRT Artifact Evaluation ---")
            print(f"Evaluating the final TensorRT artifact on {scope}...")
            metrics = evaluate_trt(engine, arch, evaluation_images, engine_gate_manifest)

        result["evaluation_metrics"] = metrics
        result["evaluation_ground_truth"] = {
            "manifest": engine_gate_manifest,
            "targets": engine_gate_targets,
            "from_checkpoint": not engine_gate_fallback,
            "fallback_reason": engine_gate_fallback or None,
        }
        published_engine = publish(engine, final_engine)

        staged_metadata = Path(str(engine) + ".metadata.json")
        if staged_metadata.exists():
            publish(staged_metadata, str(final_engine) + ".metadata.json")

        result["engine"] = published_engine
        write_metadata(published_engine, result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def export(
    model: str,
    weights: str,
    precision: str,
    formats: list[str],
    yolo_int8_profile: str | None = None,
) -> dict:
    """Export one model to the requested formats at one precision."""
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    arch = "rtdetr" if model == "rtdetr" else "yolo"
    if yolo_int8_profile is not None and (arch != "yolo" or precision != "int8"):
        raise ValueError("yolo_int8_profile only applies to YOLO exports at INT8 precision.")

    if arch == "rtdetr":
        onnx_result = export_rtdetr_onnx(weights, precision)
    else:
        onnx_result = export_yolo_onnx(weights, precision, yolo_int8_profile)
    if "engine" in formats:
        return export_engine(weights, arch, precision, onnx_result)

    final_onnx = onnx_result.get("model", onnx_result["fp32_onnx"])
    evaluation_images = int(cfg.export_config.get("evaluation_images", 0))
    final_manifest, final_targets, _ = gate_validation_manifest(weights)
    scope = "the full validation set" if evaluation_images <= 0 else f"up to {evaluation_images} images"
    print("\n--- Final ONNX Artifact Evaluation ---")
    print(f"Evaluating the final ONNX artifact on {scope}...")
    onnx_result["evaluation_metrics"] = evaluate_onnx(
        final_onnx, arch, evaluation_images, final_manifest
    )
    onnx_result["evaluation_ground_truth"] = {
        "manifest": final_manifest,
        "targets": final_targets,
    }
    write_metadata(final_onnx, onnx_result)
    return onnx_result


def main():
    parser = argparse.ArgumentParser(description="Export detectors to ONNX and TensorRT.")
    parser.add_argument("--model", required=True, choices=["yolo_ultra", "yolo_manual", "rtdetr"])
    parser.add_argument("--weights", required=True, help="Path to the trained weights file.")
    parser.add_argument("--precision", default="fp32", choices=PRECISIONS)
    parser.add_argument(
        "--format",
        nargs="+",
        default=["onnx"],
        choices=FORMATS,
        help="Artifacts to produce. 'engine' always builds its ONNX first.",
    )
    parser.add_argument("--int8", action="store_true", help="Deprecated alias for --precision int8.")
    parser.add_argument("--trt", action="store_true", help="Deprecated alias for --format engine.")
    parser.add_argument(
        "--yolo-int8-profile",
        choices=(*YOLO_INT8_PROFILES, "e2e"),
        default=None,
        help=(
            "YOLO INT8 coverage: 'head_fp32' keeps configured Detect-head exclusions; "
            "'all_conv' quantizes every eligible Conv. 'e2e' aliases 'all_conv'. "
            "Defaults to yolo_ptq_config['int8_profile']."
        ),
    )
    args = parser.parse_args()

    precision = "int8" if args.int8 else args.precision
    formats = list(args.format)
    if args.trt and "engine" not in formats:
        formats.append("engine")
    if args.yolo_int8_profile is not None and (args.model == "rtdetr" or precision != "int8"):
        parser.error("--yolo-int8-profile requires a YOLO model and --precision int8.")

    result = export(
        args.model,
        args.weights,
        precision,
        formats,
        yolo_int8_profile=args.yolo_int8_profile,
    )
    print("\nExport complete:")
    for key in ("fp32_onnx", "model", "engine"):
        if result.get(key):
            print(f"  {key}: {result[key]}")

    metrics = result.get("evaluation_metrics")
    if metrics:
        print("\n  Final artifact evaluation metrics:")
        print(f"    mAP@0.5: {metrics['mAP_50'] * 100:.1f}%")
        print(f"    mAP@0.5:0.95: {metrics['mAP_50_95'] * 100:.1f}%")
        print(
            f"    Average inference time (bs={metrics['batch_size']}): "
            f"{metrics['latency_ms_mean']:.2f} ms/image ({metrics['fps']:.1f} FPS)"
        )
        if metrics.get("compute_latency_ms_mean") is not None:
            print(
                f"    Compute only: {metrics['compute_latency_ms_mean']:.2f} ms/image "
                f"({metrics['compute_fps']:.1f} FPS), "
                f"{metrics['pipeline_overhead_ms_mean']:.2f} ms pipeline overhead"
            )
        print(
            f"    Scope: {metrics['num_images']} images from {metrics['dataset']}, "
            f"condition={metrics['condition']}, conf={metrics['confidence_threshold']}, "
            f"subsample={metrics['subsample_interval']}"
        )


if __name__ == "__main__":
    main()
