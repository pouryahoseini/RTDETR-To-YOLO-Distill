"""Regression tests for content-addressed label-budget run artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_artifacts import (  # noqa: E402
    RUN_PROVENANCE_KEY,
    RUN_PROVENANCE_VERSION,
    build_run_provenance,
    budget_run_tag,
    canonical_mapping_digest,
    content_addressed_path,
    file_sha256,
    manifest_summary,
    run_provenance_digest,
    run_provenance_tag,
    verify_run_provenance,
    write_ultralytics_budget_dataset,
)
from label_budget import manifest_fingerprint  # noqa: E402


def _manifest(path: Path, image_paths: list[Path]) -> str:
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": index,
                        "file_name": str(image_path),
                        "width": 4,
                        "height": 4,
                    }
                    for index, image_path in enumerate(image_paths, start=1)
                ],
                "annotations": [],
                "categories": [{"id": 0, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _record(name: str, train: str, val: str) -> dict:
    return {
        "name": name,
        "train": {"fingerprint": manifest_fingerprint(train)},
        "val": {"fingerprint": manifest_fingerprint(val)},
    }


def _run_identity() -> dict:
    return {
        "sampling": {"interval": 1, "seed": 42},
        "teacher": {"sha256": "a" * 64},
        "training_regime": {
            "mode": "teacher_only",
            "use_ground_truth": False,
        },
        "kd": {"enabled": False, "methods": []},
        "train_pool": {
            "manifest": "/data/train.json",
            "fingerprint": "b" * 64,
        },
        "validation": {
            "targets": "ground_truth",
            "manifest": "/data/val.json",
            "fingerprint": "c" * 64,
        },
    }


def test_run_provenance_mapping_is_canonical_and_versioned(tmp_path):
    left = {
        "teacher": {"variants": {"large", "small"}, "path": tmp_path / "t.pth"},
        "kd": {"weight": 0.2, "enabled": True},
    }
    right = {
        "kd": {"enabled": True, "weight": 0.2},
        "teacher": {"path": Path(tmp_path / "t.pth"), "variants": {"small", "large"}},
    }

    assert canonical_mapping_digest(left) == canonical_mapping_digest(right)
    assert run_provenance_digest(left) == run_provenance_digest(right)
    assert run_provenance_tag(left) == run_provenance_tag(right)

    record = build_run_provenance(left)
    assert record["version"] == RUN_PROVENANCE_VERSION
    assert record["digest"] == run_provenance_digest(left)
    assert record["tag"] == run_provenance_tag(left)
    assert record["tag"].endswith(record["digest"])
    assert len(record["digest"]) == 64


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("sampling", "interval"), 2),
        (("teacher", "sha256"), "d" * 64),
        (("training_regime", "mode"), "hybrid"),
        (("training_regime", "mode"), "semi_supervised"),
        (("kd", "enabled"), True),
        (("train_pool", "fingerprint"), "e" * 64),
        (("validation", "targets"), "teacher"),
    ],
)
def test_run_tag_changes_for_every_material_experiment_dimension(
    field_path, replacement
):
    baseline = _run_identity()
    changed = copy.deepcopy(baseline)
    destination = changed
    for field in field_path[:-1]:
        destination = destination[field]
    destination[field_path[-1]] = replacement

    assert run_provenance_tag(changed) != run_provenance_tag(baseline)


def test_exact_run_provenance_verification_is_actionable():
    current = build_run_provenance(_run_identity())
    verify_run_provenance(current, current, "checkpoint.pth")

    with pytest.raises(ValueError, match=rf"no '{RUN_PROVENANCE_KEY}'.*fresh"):
        verify_run_provenance(None, current, "legacy.pth")

    changed_identity = _run_identity()
    changed_identity["teacher"]["sha256"] = "f" * 64
    changed = build_run_provenance(changed_identity)
    with pytest.raises(ValueError, match=r"mismatch.*fields: teacher.*fresh"):
        verify_run_provenance(changed, current, "checkpoint.pth")

    tampered = dict(current)
    tampered["digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest is invalid"):
        verify_run_provenance(tampered, current, "checkpoint.pth")


def test_file_and_manifest_identity_helpers_are_complete(tmp_path):
    teacher = tmp_path / "teacher.pth"
    teacher.write_bytes(b"teacher checkpoint bytes")
    assert file_sha256(teacher) == hashlib.sha256(teacher.read_bytes()).hexdigest()
    assert len(file_sha256(teacher)) == 64

    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    manifest = _manifest(tmp_path / "train.json", [image])
    summary = manifest_summary(manifest)
    assert summary == {
        "manifest": str(Path(manifest).resolve()),
        "fingerprint_schema": "coco-training-content-v1",
        "fingerprint": manifest_fingerprint(manifest),
        "images": 1,
        "annotations": 0,
        "categories": 1,
    }

    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        file_sha256(tmp_path / "missing.pth")


def test_budget_tag_and_artifact_path_include_train_and_val_identity():
    record = {
        "name": "half",
        "train": {"fingerprint": "0123456789abcdef"},
        "val": {"fingerprint": "fedcba9876543210"},
    }

    assert budget_run_tag(record) == "half-01234567-fedcba98"
    assert content_addressed_path(
        "data/val_teacher_pseudo.json", record, "teacher-a", "conf-0.4"
    ).endswith(
        "val_teacher_pseudo_half-01234567-fedcba98_teacher-a_conf-0.4.json"
    )


def test_ultralytics_yaml_uses_exact_manifest_image_lists(tmp_path):
    train_images = [tmp_path / "train-a.jpg", tmp_path / "train-b.jpg"]
    val_images = [tmp_path / "val-a.jpg"]
    for image in [*train_images, *val_images]:
        image.write_bytes(b"placeholder")

    train = _manifest(tmp_path / "train_half.json", train_images)
    val = _manifest(tmp_path / "val_half.json", val_images)
    record = _record("half", train, val)
    result = write_ultralytics_budget_dataset(
        train_manifest=train,
        val_manifest=val,
        output_dir=str(tmp_path / "generated"),
        budget_record=record,
        category_mapping={0: {"name": "car"}},
        project_root=str(tmp_path),
    )

    assert result["train_images"] == 2
    assert result["val_images"] == 1
    train_views = [
        Path(path) for path in Path(result["train_list"]).read_text().splitlines()
    ]
    val_views = [
        Path(path) for path in Path(result["val_list"]).read_text().splitlines()
    ]
    assert all(path.is_symlink() for path in [*train_views, *val_views])
    assert [path.resolve() for path in train_views] == [
        path.resolve() for path in train_images
    ]
    assert [path.resolve() for path in val_views] == [
        path.resolve() for path in val_images
    ]
    for image_view in [*train_views, *val_views]:
        label = (
            image_view.parents[2]
            / "labels"
            / image_view.parent.name
            / f"{image_view.stem}.txt"
        )
        assert label.is_file()
        assert label.read_text() == ""
    yaml_text = Path(result["path"]).read_text()
    assert f'train: "{Path(result["train_list"])}"' in yaml_text
    assert f'val: "{Path(result["val_list"])}"' in yaml_text
    assert "nc: 1" in yaml_text and '0: "car"' in yaml_text


def test_ultralytics_labels_are_materialized_from_the_coco_budget(tmp_path):
    image = tmp_path / "source.jpg"
    image.write_bytes(b"placeholder")
    manifest = Path(_manifest(tmp_path / "budget.json", [image]))
    payload = json.loads(manifest.read_text())
    payload["images"][0].update({"width": 10, "height": 20})
    payload["categories"] = [{"id": 7, "name": "car"}]
    payload["annotations"] = [{
        "id": 1,
        "image_id": 1,
        "category_id": 7,
        "bbox": [2, 4, 4, 8],
        "area": 32,
        "iscrowd": 0,
    }]
    manifest.write_text(json.dumps(payload))
    record = _record("half", str(manifest), str(manifest))

    result = write_ultralytics_budget_dataset(
        train_manifest=str(manifest),
        val_manifest=str(manifest),
        output_dir=str(tmp_path / "generated"),
        budget_record=record,
        category_mapping={0: {"id": 7, "name": "car"}},
        project_root=str(tmp_path),
    )

    image_view = Path(Path(result["train_list"]).read_text().strip())
    label = (
        image_view.parents[2]
        / "labels"
        / "train"
        / f"{image_view.stem}.txt"
    )
    assert label.read_text() == "0 0.4 0.4 0.4 0.4\n"
    assert result["train_annotations"] == 1


def test_ultralytics_yaml_rejects_a_manifest_outside_the_record(tmp_path):
    image = tmp_path / "one.jpg"
    other = tmp_path / "other.jpg"
    image.write_bytes(b"one")
    other.write_bytes(b"other")
    train = _manifest(tmp_path / "train.json", [image])
    val = _manifest(tmp_path / "val.json", [image])
    wrong_train = _manifest(tmp_path / "train_wrong.json", [other])

    with pytest.raises(ValueError, match="does not match.*label-budget record"):
        write_ultralytics_budget_dataset(
            train_manifest=wrong_train,
            val_manifest=val,
            output_dir=str(tmp_path / "generated"),
            budget_record=_record("half", train, val),
            category_mapping={0: {"name": "car"}},
            project_root=str(tmp_path),
        )


def test_ultralytics_entrypoint_uses_generated_budget_yaml(monkeypatch):
    pytest.importorskip("ultralytics")
    import train

    record = {
        "version": 2,
        "name": "quarter",
        "train": {
            "fingerprint": "a" * 64,
            "images": 25,
            "annotations": 100,
        },
        "val": {
            "fingerprint": "b" * 64,
            "images": 10,
            "annotations": 40,
        },
    }
    generated = {
        "path": "/generated/quarter.yaml",
        "run_tag": "quarter-aaaaaaaa-bbbbbbbb",
        "train_images": 25,
        "val_images": 10,
    }
    calls = {}

    class FakeYOLO:
        trainer = None

        def __init__(self, source):
            calls["source"] = source

        def train(self, **kwargs):
            calls["train"] = kwargs

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train.cfg_auto, "input_height", 640)
    monkeypatch.setattr(train.cfg_auto, "input_width", 640)
    monkeypatch.setattr(train.label_budget_meta, "budget_record", lambda _cfg: record)
    monkeypatch.setattr(
        train, "write_ultralytics_budget_dataset", lambda **_kwargs: generated
    )
    # This entry point accepts only gt_only, which it resolves from the live
    # working config. Pinning both supervision axes keeps the test about the
    # generated dataset rather than about whichever experiment is configured on
    # the developer's machine at the time.
    monkeypatch.setattr(
        train.cfg_manual, "training_regime", {"ground_truth_supervision": True}
    )
    monkeypatch.setattr(train.cfg_manual, "kd_config", {"enabled": False})

    train.train_yolo_ultra()

    assert calls["train"]["data"] == generated["path"]
    assert calls["train"]["name"].startswith(
        generated["run_tag"] + "_run-v1-"
    )
    assert calls["train"]["project"].endswith("runs/yolo_ultra")


def test_ultralytics_config_has_a_valid_square_training_size():
    import configs.train_yolo_ultralytics_cfg as cfg
    from training_utils import resolve_ultralytics_train_imgsz

    assert resolve_ultralytics_train_imgsz(cfg.input_height, cfg.input_width) == 992


def test_ultralytics_rejects_teacher_or_hybrid_regimes_instead_of_ignoring_them(
    monkeypatch,
):
    pytest.importorskip("ultralytics")
    import train

    monkeypatch.setitem(
        train.cfg_manual.training_regime, "ground_truth_supervision", False
    )
    with pytest.raises(ValueError, match="use model_format='yolo_manual'"):
        train.train_yolo_ultra()


def test_rtdetr_rejects_teacher_or_hybrid_regimes_instead_of_ignoring_them(
    monkeypatch,
):
    pytest.importorskip("ultralytics")
    import train

    monkeypatch.setitem(
        train.cfg_manual.training_regime, "ground_truth_supervision", False
    )
    with pytest.raises(ValueError, match="use model_format='yolo_manual'"):
        train.train_rtdetr()


def test_manual_and_rtdetr_resume_paths_verify_budget_before_loading_state():
    pytest.importorskip("ultralytics")
    import train

    for entrypoint in (train.train_yolo_manual, train.train_rtdetr):
        source = inspect.getsource(entrypoint)
        assert "verify_budget_match" in source
        assert "verify_run_provenance" in source
        assert "RUN_PROVENANCE_KEY" in source
        # The guarantee is that budgets cannot share a checkpoint file, not that
        # any particular helper produces the name. Asserting the helper's name
        # made this fail when the file name stopped carrying fingerprints, even
        # though the guarantee was untouched.
        assert "budget_named_path" in source


def test_teacher_only_defaults_to_full_unlabeled_image_pool():
    pytest.importorskip("ultralytics")
    from train import training_manifest_for_regime
    from training_regime import resolve_training_regime

    cfg = SimpleNamespace(
        train_json="train_quarter.json", full_train_json="train.json"
    )
    teacher_only = resolve_training_regime(
        {
            "ground_truth_supervision": False,
            "teacher_only": {"validation_pseudo_json": "dummy.json"},
        },
        {"enabled": False},
    )
    teacher_only_override = resolve_training_regime(
        {
            "ground_truth_supervision": False,
            "teacher_only": {
                "train_json": "unlabeled_quarter.json",
                "validation_pseudo_json": "dummy.json",
            },
        },
        {"enabled": False},
    )
    supervised = resolve_training_regime(
        {"ground_truth_supervision": True}, {"enabled": False}
    )

    assert training_manifest_for_regime(teacher_only, cfg) == "train.json"
    assert training_manifest_for_regime(teacher_only_override, cfg) == "unlabeled_quarter.json"
    assert training_manifest_for_regime(supervised, cfg) == "train_quarter.json"


# --- subsampling must not depend on which manifest it is handed ---------------

def test_subsampling_preserves_nesting_between_budgets():
    """Positional slicing broke this: at interval 2, 639 of 773 quarter-training
    images were absent from the selected half set, so two budgets meant to differ
    only in size differed in composition."""
    from experiment_artifacts import subsample_image_ids

    full = list(range(1, 201))
    half = full[:120]
    quarter = half[:60]
    eighth = quarter[:30]
    for interval in (1, 2, 3, 4, 8, 32):
        selected = {
            name: set(subsample_image_ids(ids, interval))
            for name, ids in (("full", full), ("half", half), ("quarter", quarter), ("eighth", eighth))
        }
        assert selected["eighth"] <= selected["quarter"] <= selected["half"] <= selected["full"], interval


def test_subsampling_reproduces_positional_striding_on_a_complete_manifest():
    """Complete manifests carry ids 1..N in file-name order, so historical
    numbers taken at a given interval stay reproducible."""
    from experiment_artifacts import subsample_image_ids

    ids = list(range(1, 501))
    for interval in (1, 2, 3, 5, 6, 8, 32):
        assert subsample_image_ids(ids, interval) == ids[::interval]


def test_subsampling_keeps_the_order_it_was_given():
    from experiment_artifacts import subsample_image_ids

    assert subsample_image_ids([9, 5, 1, 7, 3], 2) == [9, 5, 1, 7, 3]


@pytest.mark.parametrize("interval", [0, -1])
def test_a_non_positive_interval_is_rejected(interval):
    from experiment_artifacts import subsample_image_ids

    with pytest.raises(ValueError, match="must be positive"):
        subsample_image_ids([1, 2, 3], interval)


# --- checkpoints are named for people ----------------------------------------

def _budget_only_record(name):
    return {"name": name, "train": {"fingerprint": "a" * 64},
            "val": {"fingerprint": "b" * 64}}


def test_the_full_budget_keeps_the_traditional_bare_name():
    from experiment_artifacts import budget_named_path

    assert budget_named_path("rtdetr_best.pth", _budget_only_record(None)) == "rtdetr_best.pth"


@pytest.mark.parametrize("budget", ["half", "quarter", "eighth"])
def test_a_reduced_budget_is_named_in_the_file(budget):
    from experiment_artifacts import budget_named_path

    assert budget_named_path("rtdetr_best.pth", _budget_only_record(budget)) == f"rtdetr_best_{budget}.pth"


def test_budgets_do_not_share_a_checkpoint_file():
    from experiment_artifacts import budget_named_path

    names = {budget_named_path("rtdetr_best.pth", _budget_only_record(b))
             for b in (None, "half", "quarter", "eighth")}
    assert len(names) == 4


def test_the_regime_suffix_survives_and_stays_readable():
    from experiment_artifacts import budget_named_path

    assert budget_named_path(
        "yolo_manual_best_teacher-only-soft.pth", _budget_only_record("half")
    ) == "yolo_manual_best_teacher-only-soft_half.pth"


def test_last_resolves_from_best_by_the_usual_substitution():
    """train.py finds the resume checkpoint via save_name.replace('best','last')."""
    from experiment_artifacts import budget_named_path

    best = budget_named_path("rtdetr_best.pth", _budget_only_record("half"))
    assert best.replace("best", "last") == "rtdetr_last_half.pth"


def test_a_directory_component_is_left_alone():
    from experiment_artifacts import budget_named_path

    assert budget_named_path(
        "weights/rtdetr_large/rtdetr_best.pth", _budget_only_record("half")
    ).endswith("rtdetr_large/rtdetr_best_half.pth")
