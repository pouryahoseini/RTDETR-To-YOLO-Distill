"""Integrity checks for offline teacher predictions."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _loaded_cache(tmp_path):
    from kd_cache import (
        CACHE_SCHEMA_VERSION,
        CACHE_STATE_POLICY,
        TeacherCache,
        _image_identity,
        _weights_fingerprint,
    )

    image = {"id": 7, "file_name": "images/train/frame.png", "width": 16, "height": 12}
    weights = tmp_path / "teacher.pth"
    weights.write_bytes(b"teacher-state")
    cache_path = tmp_path / "train.pt"
    torch.save(
        {
            "boxes": {7: np.zeros((0, 4), dtype=np.float16)},
            "logits": {7: np.zeros((0, 10), dtype=np.float16)},
            "meta": {
                "schema_version": CACHE_SCHEMA_VERSION,
                "state_policy": CACHE_STATE_POLICY,
                "state_source": "ema.module",
                "weights_fingerprint": _weights_fingerprint(str(weights)),
                "input_hw": (736, 1280),
                "num_classes": 10,
                "min_confidence": 0.05,
                "image_identities": {7: _image_identity(image)},
            },
        },
        cache_path,
    )
    cache = TeacherCache.load(
        str(cache_path),
        str(weights),
        (736, 1280),
        expected_num_classes=10,
        requested_min_confidence=0.5,
    )
    return cache, image


def test_coverage_binds_predictions_to_manifest_image_identity(tmp_path):
    cache, image = _loaded_cache(tmp_path)
    cache.require_coverage([image], "matching manifest")

    reused_id = {**image, "file_name": "images/train/different.png"}
    with pytest.raises(ValueError, match="image identities differ"):
        cache.require_coverage([reused_id], "different manifest")

    with pytest.raises(ValueError, match="image IDs are missing"):
        cache.require_coverage([{**image, "id": 99}], "incomplete manifest")


def test_cache_builder_disables_subsampling_but_keeps_clean_inputs():
    import kd_cache

    source = inspect.getsource(kd_cache.build_teacher_cache)
    assert "is_training=False" in source
    assert "subsample_interval=1" in source


def test_loaded_cache_exposes_schema_v3_image_identities(tmp_path):
    from kd_cache import TeacherCache

    cache, _ = _loaded_cache(tmp_path)
    assert cache.image_identities
    assert cache.meta["schema_version"] >= 3
    cache_path = Path(cache.meta["artifact_path"])
    assert cache.meta["artifact_sha256"] == hashlib.sha256(
        cache_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="changed while the run was being resolved"):
        TeacherCache.load(
            str(cache_path),
            str(tmp_path / "teacher.pth"),
            (736, 1280),
            expected_num_classes=10,
            requested_min_confidence=0.5,
            expected_cache_sha256="0" * 64,
        )


def test_weights_fingerprint_hashes_the_checkpoint_middle(tmp_path):
    """A same-size tensor change in the middle of the file must be detected."""
    from kd_cache import _weights_fingerprint

    block = 1 << 20
    weights = tmp_path / "teacher.pth"
    weights.write_bytes(b"A" * block + b"B" * block + b"C" * block)
    before = _weights_fingerprint(str(weights))

    with weights.open("r+b") as handle:
        handle.seek(block + block // 2)
        handle.write(b"changed-tensor")
        handle.flush()
        os.fsync(handle.fileno())

    after = _weights_fingerprint(str(weights))
    assert len(before) == len(after) == 64
    assert before != after
    assert weights.stat().st_size == 3 * block


def test_atomic_cache_save_replaces_only_a_complete_temp_file(tmp_path, monkeypatch):
    import kd_cache

    destination = tmp_path / "train.pt"
    real_replace = kd_cache.os.replace
    replacements = []

    def recording_replace(source, target):
        source = Path(source)
        assert source.exists()
        assert source.parent == destination.parent
        replacements.append((source, Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(kd_cache.os, "replace", recording_replace)
    kd_cache._atomic_torch_save({"value": torch.arange(4)}, str(destination))

    assert len(replacements) == 1
    temporary, target = replacements[0]
    assert target == destination
    assert not temporary.exists()
    assert torch.equal(
        torch.load(destination, map_location="cpu", weights_only=True)["value"],
        torch.arange(4),
    )
    assert not list(tmp_path.glob(".train.pt.*.tmp"))


def test_atomic_cache_save_preserves_destination_and_cleans_temp_on_error(
    tmp_path, monkeypatch
):
    import kd_cache

    destination = tmp_path / "train.pt"
    destination.write_bytes(b"existing-complete-cache")

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(kd_cache.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="serialization failure"):
        kd_cache._atomic_torch_save({"value": 1}, str(destination))

    assert destination.read_bytes() == b"existing-complete-cache"
    assert not list(tmp_path.glob(".train.pt.*.tmp"))


def test_auto_cache_build_triggers_when_missing(tmp_path, monkeypatch):
    import subprocess
    from kd_cache import default_cache_path, ensure_teacher_caches

    weights = tmp_path / "mock_teacher.pth"
    weights.write_bytes(b"teacher-weights")

    cache_dir = str(tmp_path / "cache")
    train_cache_path = default_cache_path(cache_dir, "train", str(weights))
    val_cache_path = default_cache_path(cache_dir, "val", str(weights))

    # Verify paths initially don't exist
    assert not os.path.exists(train_cache_path)
    assert not os.path.exists(val_cache_path)

    recorded_calls = []

    def mock_subprocess_run(cmd, check=True):
        recorded_calls.append(cmd)
        # Simulate creating the cache files
        os.makedirs(cache_dir, exist_ok=True)
        if "--split" in cmd:
            split_idx = cmd.index("--split") + 1
            split_arg = cmd[split_idx]
            if split_arg in ("train", "both"):
                Path(train_cache_path).write_bytes(b"cached-train")
            if split_arg in ("val", "both"):
                Path(val_cache_path).write_bytes(b"cached-val")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    # Call ensure_teacher_caches
    ensure_teacher_caches(
        cache_dir=cache_dir,
        weights_path=str(weights),
        train_needed=True,
        val_needed=True,
    )

    assert len(recorded_calls) == 1
    assert recorded_calls[0][2] == "--split"
    assert recorded_calls[0][3] == "both"
    assert os.path.exists(train_cache_path)
    assert os.path.exists(val_cache_path)

    # Calling again when caches exist should not trigger subprocess
    recorded_calls.clear()
    ensure_teacher_caches(
        cache_dir=cache_dir,
        weights_path=str(weights),
        train_needed=True,
        val_needed=True,
    )
    assert len(recorded_calls) == 0


