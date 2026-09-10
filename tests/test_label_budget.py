"""Tests for annotation-budget provenance on checkpoints."""

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from label_budget import (  # noqa: E402
    BUDGET_RECORD_KEY,
    BUDGET_RECORD_VERSION,
    budget_record,
    describe,
    manifest_fingerprint,
    read_budget_record,
    stamp_checkpoint,
    verify_budget_match,
    verify_teacher_budget,
)


def _manifest(path: Path, image_ids: list[int], annotations_per_image: int = 1) -> Path:
    path.write_text(json.dumps({
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": 4, "height": 4}
                   for i in image_ids],
        "annotations": [
            {"id": i * 10 + k, "image_id": i, "category_id": 3,
             "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
            for i in image_ids for k in range(annotations_per_image)
        ],
        "categories": [{"id": 3, "name": "car"}],
    }))
    return path


class _Cfg:
    """Minimal stand-in for a config module."""

    def __init__(self, name, train, val):
        self.label_budget = name
        self._train, self._val = str(train), str(val)

    def _budget_manifest(self, split):
        return self._train if split == "train" else self._val


def test_fingerprint_is_order_and_layout_independent(tmp_path):
    a = _manifest(tmp_path / "train_half.json", [1, 2, 3])
    b = _manifest(tmp_path / "renamed.json", [3, 2, 1])       # same ids, other order
    c = _manifest(tmp_path / "other.json", [1, 2, 4])

    assert manifest_fingerprint(str(a)) == manifest_fingerprint(str(b))
    assert manifest_fingerprint(str(a)) != manifest_fingerprint(str(c))


def test_fingerprint_tracks_annotations_and_record_reports_counts(tmp_path):
    sparse = _manifest(tmp_path / "sparse.json", [1, 2], annotations_per_image=1)
    dense = _manifest(tmp_path / "dense.json", [1, 2], annotations_per_image=5)
    assert manifest_fingerprint(str(sparse)) != manifest_fingerprint(str(dense))

    record = budget_record(_Cfg("half", dense, sparse))
    assert record["train"]["annotations"] == 10
    assert record["val"]["annotations"] == 2


def test_fingerprint_tracks_image_identity_boxes_and_categories(tmp_path):
    original = _manifest(tmp_path / "original.json", [1, 2])
    image_changed = _manifest(tmp_path / "image.json", [1, 2])
    box_changed = _manifest(tmp_path / "box.json", [1, 2])
    category_changed = _manifest(tmp_path / "category.json", [1, 2])

    payload = json.loads(image_changed.read_text())
    payload["images"][0]["file_name"] = "different.jpg"
    image_changed.write_text(json.dumps(payload))
    payload = json.loads(box_changed.read_text())
    payload["annotations"][0]["bbox"] = [0, 0, 2, 1]
    box_changed.write_text(json.dumps(payload))
    payload = json.loads(category_changed.read_text())
    payload["categories"][0]["name"] = "vehicle"
    category_changed.write_text(json.dumps(payload))

    original_fp = manifest_fingerprint(str(original))
    assert original_fp != manifest_fingerprint(str(image_changed))
    assert original_fp != manifest_fingerprint(str(box_changed))
    assert original_fp != manifest_fingerprint(str(category_changed))


def test_missing_manifest_is_reported_by_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        manifest_fingerprint(str(tmp_path / "absent.json"))


def test_matching_budgets_pass(tmp_path):
    train = _manifest(tmp_path / "train_half.json", [1, 2, 3])
    val = _manifest(tmp_path / "val_half.json", [7])
    record = budget_record(_Cfg("half", train, val))

    verify_teacher_budget(dict(record), record, "teacher.pth")


def test_a_teacher_from_a_different_budget_is_rejected(tmp_path):
    half = budget_record(_Cfg("half", _manifest(tmp_path / "t_half.json", [1, 2]),
                              _manifest(tmp_path / "v_half.json", [7])))
    full = budget_record(_Cfg(None, _manifest(tmp_path / "t_full.json", [1, 2, 3, 4]),
                              _manifest(tmp_path / "v_full.json", [7, 8])))

    with pytest.raises(ValueError, match="Annotation-budget mismatch"):
        verify_teacher_budget(full, half, "teacher.pth")


def test_a_teacher_with_different_validation_content_is_rejected(tmp_path):
    train = _manifest(tmp_path / "train.json", [1, 2])
    teacher = budget_record(_Cfg("half", train,
                                 _manifest(tmp_path / "teacher_val.json", [7])))
    run = budget_record(_Cfg("half", train,
                             _manifest(tmp_path / "run_val.json", [8])))
    with pytest.raises(ValueError, match=r"Annotation-budget mismatch.*val"):
        verify_teacher_budget(teacher, run, "teacher.pth")


def test_generic_checkpoint_match_checks_both_splits(tmp_path):
    train = _manifest(tmp_path / "train_generic.json", [1, 2])
    record = budget_record(_Cfg("half", train,
                                _manifest(tmp_path / "val_generic.json", [7])))
    verify_budget_match(record, record, "resume.pth")
    changed = budget_record(_Cfg("half", train,
                                 _manifest(tmp_path / "val_changed.json", [8])))
    with pytest.raises(ValueError, match=r"Annotation-budget mismatch.*val"):
        verify_budget_match(record, changed, "resume.pth")


def test_the_name_alone_cannot_smuggle_a_different_split_through(tmp_path):
    """Redefining a split's seed changes what "half" means; ids catch that."""
    old = budget_record(_Cfg("half", _manifest(tmp_path / "a.json", [1, 2]),
                             _manifest(tmp_path / "va.json", [7])))
    new = budget_record(_Cfg("half", _manifest(tmp_path / "b.json", [3, 4]),
                             _manifest(tmp_path / "vb.json", [7])))

    assert old["name"] == new["name"] == "half"
    with pytest.raises(ValueError, match="Annotation-budget mismatch"):
        verify_teacher_budget(old, new, "teacher.pth")


def test_an_unstamped_teacher_fails_with_the_command_that_fixes_it(tmp_path):
    record = budget_record(_Cfg(None, _manifest(tmp_path / "t.json", [1]),
                                _manifest(tmp_path / "v.json", [2])))
    with pytest.raises(ValueError, match=r"--stamp .*legacy\.pth --budget"):
        verify_teacher_budget(None, record, "legacy.pth")


def test_teacher_identity_is_defined_once_outside_the_kd_block():
    """In teacher-only mode the teacher is the whole supervision signal, not a
    KD detail -- and one value settable in two places can disagree with itself."""
    import configs.train_cfg as cfg

    assert set(cfg.teacher) == {"model", "variant", "weights"}
    assert not any(key.startswith("teacher_") for key in cfg.kd_config), (
        f"teacher identity leaked back into kd_config: {sorted(cfg.kd_config)}"
    )


def test_every_teacher_call_site_reads_the_shared_block_without_a_default():
    """Two call sites once defaulted `variant` differently -- "small" for the live
    teacher, "large" for the cache builder -- so a missing key would have cached
    one architecture and distilled into another."""
    import inspect
    import kd_cache
    import train

    for module in (kd_cache, train, __import__("dataloader")):
        source = inspect.getsource(module)
        assert 'kd_config["teacher_' not in source, module.__name__
        assert "kd_config.get(\"teacher_" not in source, module.__name__
        assert 'teacher.get("variant"' not in source, module.__name__
        assert 'teacher.get("weights"' not in source, module.__name__


def test_a_record_from_an_older_schema_is_read_but_rejected_actionably(tmp_path):
    legacy = {"version": BUDGET_RECORD_VERSION - 1}
    assert read_budget_record({BUDGET_RECORD_KEY: legacy}) == legacy
    assert read_budget_record({BUDGET_RECORD_KEY: "not-a-record"}) is None
    assert read_budget_record({}) is None
    current = budget_record(_Cfg(None, _manifest(tmp_path / "lt.json", [1]),
                                 _manifest(tmp_path / "lv.json", [2])))
    with pytest.raises(ValueError, match="legacy annotation-budget schema"):
        verify_budget_match(legacy, current, "legacy.pth")


def test_budget_record_works_with_train_cfg_public_resolver(tmp_path, monkeypatch):
    import configs.train_cfg as cfg

    _manifest(tmp_path / "train_half.json", [1, 2])
    _manifest(tmp_path / "val_half.json", [3])
    monkeypatch.setattr(cfg, "label_budget", "half")
    monkeypatch.setattr(cfg, "label_budget_splits", {"half": {"fraction": 0.5}})
    monkeypatch.setattr(cfg, "processed_annotations_dir", str(tmp_path))
    record = budget_record(cfg)
    assert record["name"] == "half"
    assert record["train"]["images"] == 2
    assert record["val"]["images"] == 1


def test_stamping_preserves_the_weights(tmp_path):
    checkpoint = tmp_path / "legacy.pth"
    weights = {"conv.weight": torch.ones(2, 2)}
    torch.save({"model": weights, "epoch": 59, "best_map": 0.35}, checkpoint)

    record = budget_record(_Cfg(None, _manifest(tmp_path / "t.json", [1, 2]),
                                _manifest(tmp_path / "v.json", [3])))
    stamp_checkpoint(str(checkpoint), record)

    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert torch.equal(reloaded["model"]["conv.weight"], weights["conv.weight"])
    assert reloaded["epoch"] == 59 and reloaded["best_map"] == 0.35
    assert read_budget_record(reloaded) == record
    verify_teacher_budget(read_budget_record(reloaded), record, str(checkpoint))


def test_describe_reads_cleanly_in_both_states(tmp_path):
    assert describe(None) == "unstamped"
    record = budget_record(_Cfg("quarter", _manifest(tmp_path / "t.json", [1, 2]),
                                _manifest(tmp_path / "v.json", [3])))
    text = describe(record)
    assert "'quarter'" in text and "2 images" in text


# --- validation pool selection ------------------------------------------------

class _Regime:
    def __init__(self, validation_targets, use_ground_truth=False):
        self.validation_targets = validation_targets
        self.use_ground_truth = use_ground_truth
        self.use_teacher_pseudo_base = not use_ground_truth
        self.teacher_only_train_json = None


class _PoolCfg:
    val_json = "data/processed_annotations/val_half.json"
    full_val_json = "data/processed_annotations/val.json"
    train_json = "data/processed_annotations/train_half.json"
    full_train_json = "data/processed_annotations/train.json"


def test_teacher_scored_validation_ignores_the_budget():
    """It spends no annotations, so restricting its images saves nothing."""
    import train

    assert train.validation_manifest_for_regime(
        _Regime("teacher"), _PoolCfg
    ) == _PoolCfg.full_val_json


def test_ground_truth_validation_follows_the_budget():
    """It reads annotations, so those labels are part of what the budget buys."""
    import train

    for use_ground_truth in (True, False):
        regime = _Regime("ground_truth", use_ground_truth=use_ground_truth)
        assert train.validation_manifest_for_regime(regime, _PoolCfg) == _PoolCfg.val_json


def test_teacher_scored_validation_is_the_same_pool_at_every_budget():
    """Constant selection noise across the arms being compared against one another."""
    import configs.base_cfg as base_cfg
    import train

    original = base_cfg.label_budget
    resolved = set()
    try:
        for budget in (None, *sorted(base_cfg.label_budget_splits)):
            base_cfg.label_budget = budget
            cfg = type("Cfg", (), {
                "val_json": base_cfg.budget_manifest("val"),
                "full_val_json": base_cfg.full_val_json,
            })
            resolved.add(train.validation_manifest_for_regime(_Regime("teacher"), cfg))
    finally:
        base_cfg.label_budget = original

    assert len(resolved) == 1, f"teacher-scored validation varied by budget: {resolved}"


def test_the_switch_is_targets_not_supervision():
    """A teacher-only run may still select on labelled validation; then it pays."""
    import train

    teacher_only_but_labelled_val = _Regime("ground_truth", use_ground_truth=False)
    assert train.validation_manifest_for_regime(
        teacher_only_but_labelled_val, _PoolCfg
    ) == _PoolCfg.val_json
