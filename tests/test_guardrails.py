"""Fail-closed guardrails: contracts that are silent when they break."""

from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import evaluate as evaluate_module  # noqa: E402
from dataloader import apply_transforms_with_retry  # noqa: E402
from evaluate import (  # noqa: E402
    EVAL_AR_KEYS,
    EVAL_MAX_DETS,
    EvalDataset,
    _compute_coco_metrics,
    _empty_metrics,
    _yolo_idx_to_coco_id,
    build_degradation_transform,
    build_preprocess_transform,
    load_compatible_state_dict,
    load_yolo_manual,
    load_yolo_ultralytics,
    pad_batch_to_static_size,
    run_evaluation,
)
from training_utils import (  # noqa: E402
    normalize_accumulated_gradients,
    resolve_ultralytics_train_imgsz,
)


class _AlwaysDropsBoxes:
    def __init__(self):
        self.params = SimpleNamespace(min_visibility=0.1)
        self.processors = {"bboxes": SimpleNamespace(params=self.params)}
        self.seen_visibility = []

    def __call__(self, **kwargs):
        self.seen_visibility.append(self.params.min_visibility)
        return {
            "image": torch.ones(3, 4, 4),
            "bboxes": [],
            "class_labels": [],
        }


class _DeterministicFallback:
    def __init__(self):
        self.called = False

    def __call__(self, **kwargs):
        self.called = True
        return {
            "image": torch.zeros(3, 8, 12),
            "bboxes": kwargs["bboxes"],
            "class_labels": kwargs["class_labels"],
        }


def test_crop_retry_uses_visibility_and_returns_tensor_fallback():
    transforms = _AlwaysDropsBoxes()
    fallback = _DeterministicFallback()

    image, boxes, labels = apply_transforms_with_retry(
        transforms=transforms,
        fallback_transforms=fallback,
        image=np.zeros((20, 30, 3), dtype=np.uint8),
        boxes=[[2.0, 3.0, 4.0, 5.0]],
        labels=[1],
        min_visibility=0.7,
        max_trials=3,
    )

    assert transforms.seen_visibility == [0.7, 0.7, 0.7]
    assert transforms.params.min_visibility == 0.1
    assert fallback.called
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 8, 12)
    assert boxes == [[2.0, 3.0, 4.0, 5.0]]
    assert labels == [1]


@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape"])
def test_evaluation_checkpoint_loading_fails_closed(fault):
    model = torch.nn.Linear(3, 2)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    if fault == "missing":
        state.pop("bias")
    elif fault == "unexpected":
        state["extra"] = torch.zeros(1)
    else:
        state["weight"] = torch.zeros(3, 2)

    with pytest.raises(RuntimeError, match="incompatible"):
        load_compatible_state_dict(model, state, "checkpoint.pth")


def test_evaluation_checkpoint_loading_accepts_exact_state():
    model = torch.nn.Linear(3, 2)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    load_compatible_state_dict(model, expected, "checkpoint.pth")
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())


def test_automatic_yolo_rejects_rectangular_training_size():
    with pytest.raises(ValueError, match="fixed rectangular training"):
        resolve_ultralytics_train_imgsz(736, 1280)
    assert resolve_ultralytics_train_imgsz(640, 640) == 640


def test_partial_accumulation_gradients_are_renormalized():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(2.0)
    scale = normalize_accumulated_gradients([parameter], target_batches=8, valid_batches=2)
    assert scale == 4.0
    assert parameter.grad.item() == 8.0


class _StubCoco:
    def __init__(self, image):
        self.image = image

    def loadImgs(self, image_id):
        return [self.image]


def test_ultralytics_eval_dataset_returns_bgr_numpy(tmp_path):
    path = tmp_path / "pixel.png"
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :] = [10, 20, 30]
    assert cv2.imwrite(str(path), bgr)
    dataset = EvalDataset(
        _StubCoco({"id": 1, "file_name": str(path)}),
        [1],
        None,
        lambda **kwargs: kwargs,
        "yolo_ultra",
    )

    image, _, _, _ = dataset[0]
    assert np.array_equal(image, bgr)



@pytest.mark.parametrize("condition", ["Clean", "fog", ""])
def test_unknown_evaluation_condition_is_rejected(condition):
    with pytest.raises(ValueError, match="Unknown evaluation condition"):
        build_degradation_transform(condition)


def test_unknown_preprocess_format_is_rejected():
    with pytest.raises(ValueError, match="Unknown model format"):
        build_preprocess_transform("typo")


@pytest.mark.parametrize("class_index", [-1, 10])
def test_invalid_model_class_index_is_rejected(class_index):
    with pytest.raises(ValueError, match="Model produced class index"):
        _yolo_idx_to_coco_id(class_index)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_type": "typo"}, "Unknown model type"),
        ({"conf": 1.1}, "Confidence threshold"),
        ({"subsample_interval": 0}, "Subsample interval"),
        ({"num_workers": -1}, "num_workers"),
        ({"condition": "fog"}, "Unknown evaluation condition"),
        (
            {"model_type": "onnx", "onnx_arch": "typo"},
            "Unknown deployment architecture",
        ),
    ],
)
def test_evaluation_arguments_fail_before_dataset_loading(kwargs, message):
    arguments = {
        "model": None,
        "model_type": "rtdetr",
        "gt_json_path": "does-not-exist.json",
        "condition": "clean",
        "conf": 0.1,
        "device": torch.device("cpu"),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run_evaluation(**arguments)



def _single_image_coco(num_boxes: int):
    """A one-image COCO set whose box count exceeds the middle maxDets budget."""
    from pycocotools.coco import COCO

    boxes = [[10.0 + 20 * i, 10.0, 12.0, 12.0] for i in range(num_boxes)]
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 1, "file_name": "frame.jpg", "width": 4096, "height": 64}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
            for i, box in enumerate(boxes)
        ],
        "categories": [{"id": 0, "name": "pedestrian"}],
    }
    coco.createIndex()
    # One exact detection per box, ranked so the maxDets budget decides how many
    # of them COCOeval is allowed to keep.
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": list(box), "score": 1.0 - i * 1e-4}
        for i, box in enumerate(boxes)
    ]
    return coco, predictions


def test_coco_metrics_use_the_configured_detection_budgets():
    """Every AP is scored at the widest budget and each AR key names its own.

    pycocotools builds `stats[0]` with `_summarize(1)`, whose maxDets defaults to
    100, and `stats[6:9]` at maxDets[0..2]. Reading them positionally scored
    mAP@0.5:0.95 at a narrower budget than every other AP metric and labelled the
    three AR values 1/10/100 regardless of the budgets actually used.
    """
    assert EVAL_AR_KEYS == tuple(f"AR_{n}" for n in EVAL_MAX_DETS)

    # More boxes than the middle budget, so the budgets give different answers.
    coco_gt, predictions = _single_image_coco(num_boxes=EVAL_MAX_DETS[-1] - 20)
    metrics = _compute_coco_metrics(coco_gt, predictions, [1])

    # Recall is capped by the budget, so each AR key must report its own.
    recalls = [metrics[key] for key in EVAL_AR_KEYS]
    assert recalls == sorted(recalls)
    assert recalls[0] < recalls[-1], "AR keys collapsed onto one detection budget"
    assert metrics[EVAL_AR_KEYS[0]] == pytest.approx(EVAL_MAX_DETS[0] / (EVAL_MAX_DETS[-1] - 20), abs=0.02)
    assert metrics[EVAL_AR_KEYS[-1]] == pytest.approx(1.0, abs=0.02)

    # Every detection fits inside the widest budget, so AP there is perfect;
    # reading stats[0] positionally would have scored it at maxDets=100 instead.
    assert metrics["mAP_50_95"] == pytest.approx(1.0, abs=0.02)
    assert metrics["mAP_50_95"] >= metrics["mAP_50"] - 1e-9

    assert set(EVAL_AR_KEYS) <= set(_empty_metrics())


def test_static_batch_padding_only_touches_short_deployment_batches():
    images = torch.arange(3 * 3 * 2 * 2, dtype=torch.float32).reshape(3, 3, 2, 2)
    orig_hws = [(720, 1360, i) for i in range(1, 4)]

    padded = pad_batch_to_static_size(images, orig_hws, "trt", 4)
    assert padded.shape[0] == 4
    assert torch.equal(padded[3], images[2])

    # Full batches, PyTorch runtimes and list batches are all passed through.
    assert pad_batch_to_static_size(images, orig_hws, "trt", 3).shape[0] == 3
    assert pad_batch_to_static_size(images, orig_hws, "yolo_manual", 4).shape[0] == 3
    assert pad_batch_to_static_size([1, 2, 3], orig_hws, "onnx", 4) == [1, 2, 3]


def test_latency_excludes_warmup_inferences(tmp_path, monkeypatch):
    """Runtime start-up must not be charged to the first timed latency sample."""
    import json

    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": str(image_path), "width": 16, "height": 16}],
                "annotations": [],
                "categories": [{"id": 0, "name": "pedestrian"}],
            }
        )
    )

    calls = []

    def fake_infer(model, img_tensors, orig_hws, conf, device, eval_coco=False):
        calls.append(len(orig_hws))
        return []

    monkeypatch.setattr(evaluate_module, "infer_rtdetr", fake_infer)

    def run(warmup):
        calls.clear()
        run_evaluation(
            model=object(),
            model_type="rtdetr",
            gt_json_path=str(gt_path),
            condition="clean",
            conf=0.1,
            device=torch.device("cpu"),
            num_workers=0,
            batch_size=1,
            warmup_batches=warmup,
        )
        return len(calls)

    # One image, so the timed loop is exactly one call plus the warmup passes.
    assert run(0) == 1
    assert run(3) == 4


def test_optional_ultralytics_loaders_raise_clear_errors(monkeypatch):
    monkeypatch.setattr(evaluate_module, "YOLO", None)
    monkeypatch.setattr(evaluate_module, "DetectionModel", None)

    with pytest.raises(ImportError, match="yolo_ultra"):
        load_yolo_ultralytics("weights.pt")
    with pytest.raises(ImportError, match="yolo_manual"):
        load_yolo_manual("weights.pth", torch.device("cpu"))



def test_transform_free_dataset_uses_actual_size_and_remains_collatable(tmp_path):
    import json
    from dataloader import ObjectDetectionDataset, object_detection_collate_fn

    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((10, 20, 3), dtype=np.uint8))
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": str(image_path), "width": 20, "height": 10}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [5.0, 2.5, 10.0, 5.0],
                        "area": 50.0,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 0, "name": "pedestrian"}],
            }
        )
    )

    dataset = ObjectDetectionDataset(
        str(annotation_path), transforms=None, is_training=False
    )
    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 10, 20)
    assert target["boxes"][0].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    images, targets = object_detection_collate_fn([(image, target), (image, target)])
    assert images.shape == (2, 3, 10, 20)
    assert len(targets) == 2


def test_ultralytics_inference_enforces_requested_batch_and_fixed_shape():
    from evaluate import infer_yolo_ultralytics
    import configs.eval_cfg as eval_cfg

    class FakeModel:
        def __init__(self):
            self.model = SimpleNamespace(end2end=False)
            self.kwargs = None

        def predict(self, images, **kwargs):
            self.kwargs = kwargs
            return [None] * len(images)

    model = FakeModel()
    images = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(3)]
    assert infer_yolo_ultralytics(model, images, [1, 2, 3], 0.1) == []
    assert model.kwargs["batch"] == 3
    assert model.kwargs["imgsz"] == (eval_cfg.input_height, eval_cfg.input_width)
    assert model.kwargs["rect"] is False
    assert model.kwargs["iou"] == eval_cfg.nms_iou_threshold


def test_scaler_helper_reports_skipped_optimizer_steps():
    from training_utils import step_optimizer_with_scaler

    class FakeOptimizer:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    class FakeScaler:
        def __init__(self, skip):
            self.skip = skip
            self.scale = 1024.0

        def get_scale(self):
            return self.scale

        def is_enabled(self):
            return True

        def step(self, optimizer):
            if not self.skip:
                optimizer.step()

        def update(self):
            if self.skip:
                self.scale /= 2

    skipped_optimizer = FakeOptimizer()
    assert not step_optimizer_with_scaler(FakeScaler(skip=True), skipped_optimizer)
    assert skipped_optimizer.steps == 0

    stepped_optimizer = FakeOptimizer()
    assert step_optimizer_with_scaler(FakeScaler(skip=False), stepped_optimizer)
    assert stepped_optimizer.steps == 1


# ── Detect anchor-cache precision ────────────────────────────────────────────
# Ultralytics builds the Detect anchor grid on the first inference forward for a
# feature shape and caches it, taking its dtype from the feature maps. Under
# autocast that caches a bf16 grid, and `arange(w) + 0.5` in bf16 loses the
# half-pixel offset above x=128 -- 32 of the 160 P3 columns at 736x1280, where
# 91.9% of VisDrone's objects sit. Measured cost: 18.14 -> 17.17 mAP, with no
# crash, no warning, and an unchanged prediction count.

def test_bf16_autocast_would_corrupt_the_anchor_grid():
    """The precision loss these guards exist to prevent is real, not theoretical."""
    import torch
    import configs.eval_cfg as cfg

    width = cfg.input_width // 8  # P3 grid width at the configured input size
    bf16 = (torch.arange(end=width, dtype=torch.bfloat16) + 0.5).float()
    fp32 = torch.arange(end=width, dtype=torch.float32) + 0.5
    corrupted = (bf16 - fp32).abs()

    assert corrupted.max() >= 0.5, "expected a half-pixel anchor shift under bf16"
    assert corrupted.gt(0).any(), "bf16 must actually damage this grid, or the guard is moot"


def test_assert_fp32_anchor_cache_rejects_a_reduced_precision_grid():
    """A bf16 cache must raise, naming the failure, rather than scoring silently."""
    import torch
    from evaluate import assert_fp32_anchor_cache

    class _Head(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.shape = None
            self.anchors = torch.zeros(2, 4, dtype=dtype)
            self.strides = torch.zeros(1, 4, dtype=dtype)

    class _Model(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Identity(), _Head(dtype))

    with pytest.raises(RuntimeError, match="not float32"):
        assert_fp32_anchor_cache(_Model(torch.bfloat16), "in test")

    # fp32 is accepted, and a model with no Detect head is simply ignored.
    assert_fp32_anchor_cache(_Model(torch.float32), "in test")
    assert_fp32_anchor_cache(torch.nn.Linear(2, 2), "in test")


def test_assert_fp32_anchor_cache_rejects_an_unpopulated_cache():
    """An un-primed head must fail loudly: it would build the grid under autocast."""
    import torch
    from evaluate import assert_fp32_anchor_cache

    class _Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stride = torch.tensor([8.0])
            self.shape = None
            self.anchors = None
            self.strides = None

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Identity(), _Head())

    with pytest.raises(RuntimeError, match="has no cached"):
        assert_fp32_anchor_cache(_Model(), "in test")


def test_enforce_fp32_anchor_cache_keeps_the_grid_fp32_under_autocast():
    """The grid must come out fp32 even when the forward runs in bf16.

    Mimics Detect's contract rather than building a real model: the cache-miss
    condition, the two cached attributes, and the delegated decode.
    """
    import torch
    from evaluate import enforce_fp32_anchor_cache

    class _Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.shape = None
            self.dynamic = False
            self.anchors = None
            self.strides = None
            self.decoded_with = None

        def _get_decode_boxes(self, x):
            # Stands in for Ultralytics' decode: it only reads the cache.
            self.decoded_with = self.anchors.dtype
            return self.anchors

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = _Head()
            self.model = torch.nn.Sequential(torch.nn.Identity(), self.head)

    model = _Model()
    enforce_fp32_anchor_cache(model)
    head = model.head

    feats = [torch.zeros(1, 4, 92, 160, dtype=torch.bfloat16),
             torch.zeros(1, 4, 46, 80, dtype=torch.bfloat16),
             torch.zeros(1, 4, 23, 40, dtype=torch.bfloat16)]
    head._get_decode_boxes({"feats": feats})

    assert head.anchors.dtype == torch.float32
    assert head.strides.dtype == torch.float32
    assert head.decoded_with == torch.float32, "decode ran against a reduced-precision grid"

    # The half-pixel offset the bug destroys must survive.
    xs = head.anchors[0]
    assert torch.isclose(xs.max(), torch.tensor(159.5)), f"anchor grid truncated: max {xs.max()}"

    # A shape change -- the short final batch -- must rebuild in fp32 too.
    feats2 = [f[:1].repeat(2, 1, 1, 1) for f in feats]
    head._get_decode_boxes({"feats": feats2})
    assert head.anchors.dtype == torch.float32, "re-cache on a new shape lost fp32"

    # Idempotent: applying it twice must not double-wrap.
    enforce_fp32_anchor_cache(model)
    enforce_fp32_anchor_cache(model)
    head._get_decode_boxes({"feats": feats})
    assert head.anchors.dtype == torch.float32


def test_enforce_fp32_anchor_cache_fails_closed_on_an_unknown_head():
    """If Ultralytics moves the anchor cache, say so instead of silently not fixing it."""
    import torch
    from evaluate import enforce_fp32_anchor_cache

    class _Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stride = torch.tensor([8.0])
            self.shape = None  # looks like a Detect head, but has no _get_decode_boxes

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Identity(), _Head())

    with pytest.raises(RuntimeError, match="_get_decode_boxes"):
        enforce_fp32_anchor_cache(_Model())
