"""Regression tests for GT, hybrid, and annotation-free teacher training."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from training_regime import resolve_training_regime  # noqa: E402


def _kd_config(*, enabled: bool) -> dict:
    """Small legacy-shaped KD config whose mutation is easy to detect."""

    return {
        "enabled": enabled,
        "disable_on_augs": ["mixup"],
        "pseudo_label": {
            "enabled": True,
            "kd_weight": 0.2,
            "kd_warmup_epochs": 4,
            "kd_ramp_epochs": 4,
            "kd_fade_epochs": 5,
            "kd_cooldown_epochs": 1,
        },
        "pseudo_label_soft": {
            "enabled": True,
            "kd_weight": 0.15,
            "kd_warmup_epochs": 3,
            "kd_ramp_epochs": 2,
            "kd_fade_epochs": 6,
            "kd_cooldown_epochs": 1,
        },
        "feature_based": {
            "enabled": True,
            "kd_weight": 0.09,
            "kd_warmup_epochs": 2,
            "kd_ramp_epochs": 5,
            "kd_fade_epochs": 4,
            "kd_cooldown_epochs": 2,
        },
    }


@pytest.mark.parametrize(
    ("use_gt", "kd_enabled", "expected_mode", "expected"),
    [
        ("full", False, "gt_only", (True, False, False, False, False)),
        ("full", True, "hybrid", (True, False, True, True, False)),
        ("none", False, "teacher_only", (False, True, False, True, False)),
        ("none", True, "teacher_only", (False, True, True, True, False)),
        ("semi_supervised", False, "semi_supervised", (True, True, False, True, True)),
        ("semi_supervised", True, "semi_supervised", (True, True, True, True, True)),
        # Backward compatibility for bool values
        (True, False, "gt_only", (True, False, False, False, False)),
        (True, True, "hybrid", (True, False, True, True, False)),
        (False, False, "teacher_only", (False, True, False, True, False)),
        (False, True, "teacher_only", (False, True, True, True, False)),
    ],
)
def test_the_two_supervision_axes_cover_every_regime(
    use_gt, kd_enabled, expected_mode, expected
):
    """Both axes are independent, so all four combinations are meaningful.

    The teacher base tracks ``use_gt`` alone and the auxiliaries track
    ``kd_config['enabled']`` alone -- there is no cell where one switch
    silently overrides the other.
    """

    regime = resolve_training_regime(
        {
            "ground_truth_supervision": use_gt,
            "teacher_only": {"validation_pseudo_json": "dummy.json"},
            "semi_supervised": {"validation_pseudo_json": "dummy.json"},
        },
        _kd_config(enabled=kd_enabled),
    )

    assert regime.mode == expected_mode
    actual = (
        regime.use_ground_truth,
        regime.use_teacher_pseudo_base,
        regime.use_aux_kd,
        regime.kd_pipeline_enabled,
        regime.is_semi_supervised,
    )
    assert actual == expected


def test_kd_switch_stays_authoritative_when_ground_truth_is_on():
    """Naming a regime must not be able to override the KD switch.

    Resolving `gt_only` with KD enabled to a run with the whole teacher
    pipeline off would contradict the KD config it was handed.
    """

    regime = resolve_training_regime(
        {"ground_truth_supervision": True},
        _kd_config(enabled=True),
    )

    assert regime.mode == "hybrid"
    assert regime.use_aux_kd is True
    assert regime.kd_pipeline_enabled is True



def test_missing_master_knob_is_the_legacy_hybrid_path_without_config_mutation():
    """Old configs must retain both their meaning and every KD schedule value."""

    kd = _kd_config(enabled=True)
    original = deepcopy(kd)

    regime = resolve_training_regime(None, kd)

    assert regime.mode == "hybrid"
    assert regime.use_ground_truth is True
    assert regime.use_teacher_pseudo_base is False
    assert regime.use_aux_kd is True
    assert regime.kd_pipeline_enabled is True
    assert kd == original


@pytest.mark.parametrize("kd_enabled", [False, True])
def test_supervised_runs_ignore_an_invalid_unused_teacher_only_block(kd_enabled):
    """A typo in a section this run does not read must not block it."""

    regime = resolve_training_regime(
        {"ground_truth_supervision": True, "teacher_only": "not-a-mapping"},
        _kd_config(enabled=kd_enabled),
    )
    assert regime.mode == ("hybrid" if kd_enabled else "gt_only")


@pytest.mark.parametrize(
    ("teacher_cfg", "match"),
    [
        ({"teacher_conf_threshold": -0.01}, "conf_threshold"),
        ({"teacher_conf_threshold": float("nan")}, "finite"),
        ({"empty_target_policy": "guess"}, "skip.*background"),
        ({"base_loss": "soft_only"}, "teacher_pseudo_detection"),
    ],
)
def test_resolver_rejects_invalid_teacher_base_settings(teacher_cfg, match):
    with pytest.raises(ValueError, match=match):
        resolve_training_regime(
            {"ground_truth_supervision": False, "teacher_only": teacher_cfg},
            _kd_config(enabled=False),
        )


def _teacher_target_fixture():
    batch_size, queries = 4, 3
    images = torch.arange(batch_size * 3 * 2 * 2, dtype=torch.float32).reshape(
        batch_size, 3, 2, 2
    )
    boxes = torch.zeros(batch_size, queries, 4)
    for image_idx in range(batch_size):
        for query_idx in range(queries):
            boxes[image_idx, query_idx] = torch.tensor(
                [image_idx + 0.1, query_idx + 0.2, 0.3, 0.4]
            )
    classes = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 1, 2]], dtype=torch.long
    )
    # Confidence equal to the threshold is intentionally excluded: the
    # production selection contract is strictly greater-than.
    confidence = torch.tensor(
        [[0.90, 0.50, 0.10], [0.20, 0.30, 0.40],
         [0.49, 0.50, 0.01], [0.80, 0.51, 0.20]],
        dtype=torch.float32,
    )
    batch = {
        "img": images,
        "cls": torch.tensor([[99], [98], [97]]),
        "bboxes": torch.full((3, 4), -123.0),
        "batch_idx": torch.tensor([0, 2, 3]),
    }
    return batch, boxes, classes, confidence


def _assert_pseudo_batches_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_teacher_pseudo_batch_ignores_gt_filters_threshold_and_reindexes():
    pytest.importorskip("ultralytics")
    from train import build_teacher_pseudo_batch, teacher_feature_targets

    batch, boxes, classes, confidence = _teacher_target_fixture()
    keep = torch.tensor([3, 0, 2])

    pseudo, effective, count, skipped = build_teacher_pseudo_batch(
        batch, keep, boxes, classes, confidence, 0.5, "skip"
    )

    assert pseudo is not None
    assert effective.tolist() == [3, 0]
    assert count == 3
    assert skipped == 1
    assert torch.equal(pseudo["img"], batch["img"][[3, 0]])
    assert pseudo["batch_idx"].tolist() == [0, 0, 1]
    assert pseudo["cls"].flatten().tolist() == [9, 1, 0]
    assert torch.equal(
        pseudo["bboxes"], torch.stack([boxes[3, 0], boxes[3, 1], boxes[0, 0]])
    )

    # Change every GT tensor without changing images or teacher output. A pure
    # teacher target builder must be bit-identical.
    poisoned = {
        **batch,
        "cls": torch.tensor([[1], [2], [3], [4], [5]]),
        "bboxes": torch.randn(5, 4) * 10_000,
        "batch_idx": torch.tensor([3, 3, 2, 1, 0]),
    }
    pseudo_poisoned, effective_poisoned, count_poisoned, skipped_poisoned = (
        build_teacher_pseudo_batch(
            poisoned, keep, boxes, classes, confidence, 0.5, "skip"
        )
    )
    _assert_pseudo_batches_equal(pseudo, pseudo_poisoned)
    assert torch.equal(effective, effective_poisoned)
    assert (count, skipped) == (count_poisoned, skipped_poisoned)

    feature_boxes, feature_batch_idx = teacher_feature_targets(pseudo, effective)
    assert torch.equal(feature_boxes, pseudo["bboxes"])
    assert feature_batch_idx.tolist() == [3, 3, 0]


def test_teacher_pseudo_batch_skip_and_background_empty_image_policies():
    pytest.importorskip("ultralytics")
    from train import build_teacher_pseudo_batch, teacher_feature_targets

    batch, boxes, classes, confidence = _teacher_target_fixture()
    silent = torch.tensor([1, 2])

    skipped_batch, skipped_keep, count, skipped = build_teacher_pseudo_batch(
        batch, silent, boxes, classes, confidence, 0.5, "skip"
    )
    assert skipped_batch is None
    assert skipped_keep.numel() == 0
    assert count == 0
    assert skipped == 2

    background_batch, background_keep, count, skipped = build_teacher_pseudo_batch(
        batch, silent, boxes, classes, confidence, 0.5, "background"
    )
    assert background_batch is not None
    assert background_keep.tolist() == [1, 2]
    assert torch.equal(background_batch["img"], batch["img"][silent])
    assert background_batch["cls"].shape == (0, 1)
    assert background_batch["bboxes"].shape == (0, 4)
    assert background_batch["batch_idx"].shape == (0,)
    assert count == 0
    assert skipped == 0

    feature_boxes, feature_batch_idx = teacher_feature_targets(
        background_batch, background_keep
    )
    assert feature_boxes.shape == (0, 4)
    assert feature_batch_idx.shape == (0,)


def test_teacher_feature_mask_excludes_skip_images_from_every_feature_term():
    pytest.importorskip("ultralytics")
    from train import teacher_feature_disable_mask

    augmentation_mask = torch.tensor([False, True, False, False])
    skip_effective = torch.tensor([0, 1])
    assert teacher_feature_disable_mask(
        augmentation_mask, skip_effective
    ).tolist() == [False, True, True, True]

    # Background policy retains every image, so only augmentation exclusions
    # remain.
    background_effective = torch.arange(4)
    assert torch.equal(
        teacher_feature_disable_mask(augmentation_mask, background_effective),
        augmentation_mask,
    )



def test_semi_supervised_subset_loss_scaling_is_linear_in_partition_size():
    pytest.importorskip("ultralytics")
    from train import scale_teacher_pseudo_base_loss

    # Ultralytics returns K * mean_loss for K effective images. For five
    # requested unlabeled images in a batch of ten, the final contribution must
    # be 5 / 10 of that per-image mean, even if only three had teacher targets.
    raw_loss = torch.tensor(15.0)  # K=3, so mean_loss=5
    total = scale_teacher_pseudo_base_loss(
        raw_loss, candidate_count=5, effective_count=3
    )
    assert total.item() == pytest.approx(25.0)
    assert (total / 10).item() == pytest.approx(2.5)


def test_semi_supervised_feature_kd_keeps_labeled_images_and_gt_boxes():
    pytest.importorskip("ultralytics")
    from train import (
        semi_supervised_feature_disable_mask,
        semi_supervised_feature_targets,
    )

    batch = {
        "bboxes": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0, 2]),
    }
    pseudo_batch = {
        "bboxes": torch.tensor([[0.4, 0.4, 0.3, 0.3]]),
        "batch_idx": torch.tensor([0]),
    }
    boxes, batch_idx = semi_supervised_feature_targets(
        batch, pseudo_batch, torch.tensor([1])
    )
    assert torch.equal(boxes, torch.cat((batch["bboxes"], pseudo_batch["bboxes"])))
    assert batch_idx.tolist() == [0, 2, 1]

    mask = semi_supervised_feature_disable_mask(
        torch.tensor([False, False, True, False]),
        torch.tensor([True, False, True, False]),
        torch.tensor([1]),
    )
    # Image 3 is the only teacher-silent unlabeled image. Image 2 remains
    # excluded solely because its augmentation policy explicitly says so.
    assert mask.tolist() == [False, False, True, True]


def test_hard_kd_keeps_a_1d_labeled_subset():
    pytest.importorskip("ultralytics")
    from train import hard_kd_keep_indices

    keep = hard_kd_keep_indices(
        torch.tensor([False, True, False, False]),
        torch.tensor([True, False, True, False]),
    )
    assert keep.tolist() == [0, 2]
    assert keep.ndim == 1


def test_composition_sources_stay_in_their_supervision_partition():
    import dataloader as dl

    dataset = dl.ObjectDetectionDataset.__new__(dl.ObjectDetectionDataset)
    dataset.image_ids = [10, 11, 12, 13]
    dataset._composition_indices = {True: [0, 2], False: [1, 3]}

    assert {dataset._sample_composition_index(True) for _ in range(20)} <= {0, 2}
    assert {dataset._sample_composition_index(False) for _ in range(20)} <= {1, 3}

def test_regime_suffix_preserves_hybrid_names_and_separates_teacher_only():
    pytest.importorskip("ultralytics")
    from train import kd_run_suffix, training_regime_run_suffix

    kd = _kd_config(enabled=True)
    hybrid = resolve_training_regime({"ground_truth_supervision": True}, kd)
    gt_only = resolve_training_regime(
        {"ground_truth_supervision": True}, _kd_config(enabled=False)
    )
    teacher_only = resolve_training_regime(
        {"ground_truth_supervision": False, "teacher_only": {"validation_pseudo_json": "dummy.json"}}, kd
    )

    assert training_regime_run_suffix(hybrid, kd) == kd_run_suffix(kd)
    assert training_regime_run_suffix(gt_only, kd) == ""
    assert training_regime_run_suffix(teacher_only, kd) == "_teacher-only-soft-feat"


def _write_coco_manifest(
    tmp_path: Path, *, annotations: list[dict], name: str = "annotations.json"
) -> Path:
    yy, xx = np.mgrid[:12, :16]
    rgb = np.stack((xx * 8, yy * 12, (xx + yy) * 5), axis=-1).astype(np.uint8)
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    manifest = {
        "images": [
            {
                "id": 7,
                "file_name": str(image_path),
                "width": 16,
                "height": 12,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 0, "name": "pedestrian"}],
    }
    path = tmp_path / name
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _one_annotation() -> list[dict]:
    return [
        {
            "id": 1,
            "image_id": 7,
            "category_id": 0,
            "bbox": [2.0, 3.0, 8.0, 6.0],
            "area": 48.0,
            "iscrowd": 0,
        }
    ]


def _dataset(path: Path, **kwargs):
    pytest.importorskip("ultralytics")
    from dataloader import ObjectDetectionDataset

    return ObjectDetectionDataset(
        str(path),
        transforms=None,
        is_training=False,
        enable_teacher_cache=False,
        emit_teacher_view=False,
        **kwargs,
    )


def test_teacher_only_dataset_never_queries_ground_truth_and_forbids_counts(
    tmp_path, monkeypatch
):
    path = _write_coco_manifest(tmp_path, annotations=_one_annotation())
    dataset = _dataset(path, use_ground_truth=False)

    assert dataset.ignored_ground_truth_count == 1
    assert dataset.coco.anns == {}
    assert dataset.coco.imgToAnns == {7: []}
    assert dataset.coco.catToImgs == {0: []}
    assert dataset.coco.dataset["annotations"] == []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("teacher-only dataset queried ground truth")

    monkeypatch.setattr(dataset.coco, "getAnnIds", forbidden)
    monkeypatch.setattr(dataset.coco, "loadAnns", forbidden)

    _, target = dataset[0]
    assert target["labels"].numel() == 0
    assert target["boxes"].shape == (0, 4)
    with pytest.raises(RuntimeError, match="class counts.*unavailable"):
        dataset.class_instance_counts()


def test_teacher_only_strict_mode_rejects_a_labelled_manifest(tmp_path):
    path = _write_coco_manifest(tmp_path, annotations=_one_annotation())

    with pytest.raises(ValueError, match="annotation-free manifest"):
        _dataset(
            path,
            use_ground_truth=False,
            require_empty_annotations=True,
        )


def test_teacher_only_strict_mode_accepts_an_empty_manifest(tmp_path):
    path = _write_coco_manifest(
        tmp_path, annotations=[], name="annotations_empty.json"
    )

    dataset = _dataset(
        path,
        use_ground_truth=False,
        require_empty_annotations=True,
    )
    _, target = dataset[0]

    assert dataset.ignored_ground_truth_count == 0
    assert target["labels"].numel() == 0
    assert target["boxes"].shape == (0, 4)


@pytest.mark.parametrize("use_ground_truth", [False, True])
def test_every_cached_training_manifest_must_be_covered(
    tmp_path, monkeypatch, use_ground_truth
):
    pytest.importorskip("ultralytics")
    import dataloader as dl
    from kd_cache import TeacherCache

    path = _write_coco_manifest(tmp_path, annotations=[])

    class MissingCache:
        boxes = {}

        def require_coverage(self, images, source):
            raise ValueError(
                "Teacher cache does not cover training manifest: "
                f"{len(images)} selected image IDs are missing"
            )

    monkeypatch.setattr(
        TeacherCache,
        "load",
        classmethod(lambda cls, *_args, **_kwargs: MissingCache()),
    )

    with pytest.raises(ValueError, match="does not cover.*image IDs are missing"):
        dl.ObjectDetectionDataset(
            str(path),
            transforms=None,
            is_training=True,
            use_ground_truth=use_ground_truth,
            enable_teacher_cache=True,
            emit_teacher_view=False,
        )


def test_dataset_default_preserves_existing_labelled_behavior(tmp_path):
    path = _write_coco_manifest(tmp_path, annotations=_one_annotation())

    dataset = _dataset(path, use_ground_truth=True)
    _, boxes, labels, image_id, has_gt = dataset._load_image_and_boxes(0)

    assert dataset.use_ground_truth is True
    assert has_gt is True
    assert dataset.ignored_ground_truth_count == 0
    assert image_id == 7
    assert boxes == [[2.0, 3.0, 8.0, 6.0]]
    assert labels == [0]


def test_validation_targets_defaults_to_ground_truth():
    regime = resolve_training_regime(
        {"ground_truth_supervision": True},
        _kd_config(enabled=False),
    )
    assert regime.validation_targets == "ground_truth"


def test_teacher_only_ignores_validation_targets():
    """A teacher-only run always builds pseudo-annotations for validation."""

    regime = resolve_training_regime(
        {
            "ground_truth_supervision": "none",
            "teacher_only": {
                "validation_targets": "ground_truth",
                "validation_pseudo_json": "dummy.json",
            },
        },
        _kd_config(enabled=False),
    )

    assert regime.mode == "teacher_only"
    assert regime.validation_targets == "teacher"


def test_semi_supervised_validation_targets():
    """Semi-supervised validates on either GT or mixed targets."""
    
    # Defaults to ground_truth
    regime = resolve_training_regime(
        {"ground_truth_supervision": "semi_supervised", "semi_supervised": {"validation_pseudo_json": "dummy.json"}},
        _kd_config(enabled=False),
    )
    assert regime.validation_targets == "ground_truth"

    regime = resolve_training_regime(
        {
            "ground_truth_supervision": "semi_supervised",
            "semi_supervised": {
                "validation_targets": "ground_truth_teacher",
                "validation_pseudo_json": "dummy.json",
            },
        },
        _kd_config(enabled=False),
    )
    assert regime.validation_targets == "ground_truth_teacher"

    # Reject invalid target
    with pytest.raises(ValueError, match="must be 'ground_truth' or 'ground_truth_teacher'"):
        resolve_training_regime(
            {
                "ground_truth_supervision": "semi_supervised",
                "semi_supervised": {
                    "validation_targets": "teacher",
                    "validation_pseudo_json": "dummy.json",
                },
            },
            _kd_config(enabled=False),
        )


def test_teacher_validation_is_recorded_in_checkpoint_metadata():
    """The claim a run can make depends on this, so it has to be on the model."""

    regime = resolve_training_regime(
        {
            "ground_truth_supervision": False,
            "teacher_only": {
                "validation_targets": "teacher",
                "validation_pseudo_json": "data/processed_annotations/v.json",
            },
        },
        _kd_config(enabled=False),
    )
    meta = regime.as_dict()
    assert meta["validation_targets"] == "teacher"
    assert meta["validation_pseudo_json"] == "data/processed_annotations/v.json"


def test_teacher_pseudo_manifest_carries_no_source_annotations(tmp_path, monkeypatch):
    """The generated file must be inspectable and label-free."""

    pytest.importorskip("torch")
    import numpy as np
    import torch
    from kd_cache import (
        CACHE_SCHEMA_VERSION,
        _image_identity,
        write_teacher_pseudo_annotations,
    )

    src = tmp_path / "val.json"
    src.write_text(
        json.dumps(
            {
                "images": [{"id": 5, "file_name": "a.png", "width": 100, "height": 50}],
                "categories": [{"id": 0, "name": "car"}],
                # A real label that must not survive into the output.
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 5,
                        "category_id": 0,
                        "bbox": [1, 2, 3, 4],
                        "area": 12,
                        "iscrowd": 0,
                    }
                ],
            }
        )
    )

    # One confident box, one below threshold, and one running off the edge.
    logits = np.array(
        [[9.0, -9.0], [-9.0, -9.0], [9.0, -9.0]], dtype=np.float16
    )
    boxes = np.array(
        [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.98, 0.5, 0.2, 0.2]],
        dtype=np.float16,
    )
    cache_path = tmp_path / "val.pt"
    torch.save(
        {
            "boxes": {5: boxes},
            "logits": {5: logits},
            "meta": {
                "schema_version": CACHE_SCHEMA_VERSION,
                "state_policy": "prefer_ema_module",
                "state_source": "ema.module",
                "weights_fingerprint": None,
                "input_hw": (50, 100),
                "num_classes": 2,
                "top_k": 300,
                "min_confidence": 0.05,
                "split": "val",
                "image_identities": {
                    5: _image_identity(
                        {"id": 5, "file_name": "a.png", "width": 100, "height": 50}
                    )
                },
            },
        },
        cache_path,
    )

    out = tmp_path / "pseudo.json"
    import kd_cache

    real_replace = kd_cache.os.replace
    replacements = []

    def recording_replace(source, destination):
        source_path = Path(source)
        assert source_path.exists()
        assert source_path.parent == out.parent
        replacements.append((source_path, Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(kd_cache.os, "replace", recording_replace)

    # The fingerprint depends on a real checkpoint file; bypass just that check.
    monkey = kd_cache._weights_fingerprint
    kd_cache._weights_fingerprint = lambda _p: None
    try:
        summary = write_teacher_pseudo_annotations(
            cache_path=str(cache_path),
            source_json=str(src),
            output_json=str(out),
            conf_threshold=0.4,
            weights_path="teacher.pth",
            input_hw=(50, 100),
            num_classes=2,
        )
    finally:
        kd_cache._weights_fingerprint = monkey

    written = json.loads(out.read_text())
    assert len(replacements) == 1
    assert replacements[0][1] == out
    assert summary["annotations"] == 2  # the sub-threshold box is dropped
    assert written["images"] == json.loads(src.read_text())["images"]
    # No annotation may carry the source label's geometry.
    assert all(a["bbox"] != [1, 2, 3, 4] for a in written["annotations"])
    # Every box is clipped inside the frame and non-degenerate.
    for a in written["annotations"]:
        x, y, w, h = a["bbox"]
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert x + w <= 100 + 1e-6 and y + h <= 50 + 1e-6
        assert a["area"] == pytest.approx(w * h)


# --- KD master switch must have something behind it ---------------------------

_NO_METHODS = {
    "enabled": True,
    "pseudo_label": {"enabled": False},
    "pseudo_label_soft": {"enabled": False},
    "feature_based": {"enabled": False},
}


@pytest.mark.parametrize("ground_truth", [True, False])
def test_kd_enabled_with_no_method_is_rejected(ground_truth):
    """Otherwise the run is stamped hybrid and loads a teacher it never calls."""
    with pytest.raises(ValueError, match="no KD method is"):
        resolve_training_regime(
            {
                "ground_truth_supervision": ground_truth,
                "teacher_only": {"validation_pseudo_json": "dummy.json"},
            },
            _NO_METHODS,
        )


@pytest.mark.parametrize("method", ["pseudo_label", "pseudo_label_soft", "feature_based"])
def test_any_single_active_method_is_accepted(method):
    kd = {key: dict(value) if isinstance(value, dict) else value
          for key, value in _NO_METHODS.items()}
    kd[method] = {"enabled": True}
    regime = resolve_training_regime({"ground_truth_supervision": True}, kd)
    assert regime.mode == "hybrid" and regime.use_aux_kd


def test_kd_disabled_with_no_method_stays_a_real_arm():
    """teacher_only + enabled=False is the pseudo-detection-base-only arm."""
    kd = dict(_NO_METHODS, enabled=False)
    regime = resolve_training_regime(
        {
            "ground_truth_supervision": False,
            "teacher_only": {"validation_pseudo_json": "dummy.json"},
        },
        kd,
    )
    assert regime.mode == "teacher_only"
    assert regime.use_teacher_pseudo_base and not regime.use_aux_kd

    supervised = resolve_training_regime({"ground_truth_supervision": True}, kd)
    assert supervised.mode == "gt_only"


# --- validation threshold can be pinned independently -------------------------

def _teacher_only(**teacher_overrides):
    teacher = {
        "validation_targets": "teacher",
        "validation_pseudo_json": "val_pseudo.json",
        **teacher_overrides,
    }
    return resolve_training_regime(
        {"ground_truth_supervision": False, "teacher_only": teacher},
        {"enabled": False},
    )


def test_validation_threshold_tracks_training_by_default():
    regime = _teacher_only(teacher_conf_threshold=0.4)
    assert regime.validation_teacher_conf_threshold == 0.4


def test_validation_threshold_can_be_pinned_while_training_is_swept():
    """A threshold sweep must not also move the ruler it is measured with."""
    scored = {
        _teacher_only(
            teacher_conf_threshold=train_threshold,
            validation_teacher_conf_threshold=0.5,
        ).validation_teacher_conf_threshold
        for train_threshold in (0.3, 0.4, 0.5, 0.7)
    }
    assert scored == {0.5}


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
def test_an_out_of_range_validation_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="validation_teacher_conf_threshold"):
        _teacher_only(teacher_conf_threshold=0.4, validation_teacher_conf_threshold=bad)


def test_a_bad_training_threshold_is_reported_against_its_own_key():
    """The validation knob inherits it, so it must not shadow the real error."""
    with pytest.raises(ValueError, match=r"\['teacher_conf_threshold'\]"):
        _teacher_only(teacher_conf_threshold=float("nan"))
