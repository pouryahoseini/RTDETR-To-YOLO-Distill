"""Regression tests for reproducible evaluation-report lineage."""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import evaluate as evaluate_module  # noqa: E402
from experiment_artifacts import (  # noqa: E402
    RUN_PROVENANCE_KEY,
    build_run_provenance,
)
from evaluate import (  # noqa: E402
    _empty_metrics,
    build_evaluation_lineage,
    generate_reports,
    ground_truth_lineage,
    run_evaluation,
)
from label_budget import (  # noqa: E402
    BUDGET_RECORD_KEY,
    BUDGET_RECORD_VERSION,
    MANIFEST_FINGERPRINT_SCHEMA,
    manifest_fingerprint,
)


def _manifest(path: Path, image_path: Path) -> Path:
    path.write_text(json.dumps({
        "images": [{
            "id": 1,
            "file_name": str(image_path),
            "width": 16,
            "height": 16,
        }],
        "annotations": [],
        "categories": [{"id": 0, "name": "pedestrian"}],
    }))
    return path


def _budget_stamp() -> dict:
    summary = {
        "manifest": "data/processed_annotations/train_half.json",
        "fingerprint_schema": MANIFEST_FINGERPRINT_SCHEMA,
        "fingerprint": "a" * 64,
        "images": 2,
        "annotations": 3,
        "categories": 1,
    }
    return {
        "version": BUDGET_RECORD_VERSION,
        "name": "half",
        "train": dict(summary),
        "val": {**summary, "manifest": "data/processed_annotations/val_half.json"},
    }


def test_reports_persist_checkpoint_training_budget_and_ground_truth_lineage(tmp_path):
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((16, 16, 3), dtype=np.uint8))
    gt_path = _manifest(tmp_path / "test.json", image)

    regime = {
        "mode": "teacher_only",
        "use_ground_truth": False,
        "kd_pipeline_enabled": False,
    }
    budget = _budget_stamp()
    run_provenance = build_run_provenance({
        "backend": "rtdetr",
        "sampling": {"train_interval": 1, "validation_interval": 1},
    })
    checkpoint = tmp_path / "student.pth"
    torch.save({
        "model": {},
        "training_regime": regime,
        BUDGET_RECORD_KEY: budget,
        RUN_PROVENANCE_KEY: run_provenance,
    }, checkpoint)

    metrics = _empty_metrics()
    metrics["ground_truth"] = ground_truth_lineage(str(gt_path))
    results = {"test/clean": metrics}
    lineage = build_evaluation_lineage(str(checkpoint), "rtdetr", results)

    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert lineage["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "sha256": expected_sha,
        "size_bytes": checkpoint.stat().st_size,
        "metadata_status": "recorded",
    }
    assert lineage["training_regime"] == regime
    assert lineage["label_budget"] == budget
    assert lineage["run_provenance"] == run_provenance
    assert lineage["ground_truth"] == [{
        **ground_truth_lineage(str(gt_path)),
        "evaluations": ["test/clean"],
    }]

    generate_reports(results, "rtdetr", str(tmp_path), lineage=lineage)
    payload = json.loads(next(tmp_path.glob("eval_rtdetr_*.json")).read_text())
    report = next(tmp_path.glob("eval_rtdetr_*.md")).read_text()

    assert payload["_lineage"] == lineage
    assert str(checkpoint.resolve()) in report
    assert expected_sha in report
    assert "teacher_only" in report
    assert "'half'" in report
    assert run_provenance["tag"] in report
    assert manifest_fingerprint(str(gt_path)) in report
    assert "_lineage" not in next(
        line for line in report.splitlines() if line.startswith("| Metric |")
    )


def test_run_evaluation_records_exact_ground_truth_identity(tmp_path, monkeypatch):
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((16, 16, 3), dtype=np.uint8))
    gt_path = _manifest(tmp_path / "test.json", image)
    monkeypatch.setattr(
        evaluate_module,
        "infer_rtdetr",
        lambda *args, **kwargs: [],
    )

    metrics = run_evaluation(
        model=object(),
        model_type="rtdetr",
        gt_json_path=str(gt_path),
        condition="clean",
        conf=0.1,
        device=torch.device("cpu"),
        num_workers=0,
        batch_size=1,
        warmup_batches=0,
    )

    assert metrics["dataset"] == str(gt_path)  # Backward-compatible field.
    assert metrics["ground_truth"] == {
        "manifest": str(gt_path.resolve()),
        "fingerprint_schema": MANIFEST_FINGERPRINT_SCHEMA,
        "fingerprint": manifest_fingerprint(str(gt_path)),
    }


def test_unstamped_checkpoint_is_reported_as_unknown_not_inferred(tmp_path):
    checkpoint = tmp_path / "legacy.pth"
    torch.save({"model": {}}, checkpoint)

    lineage = build_evaluation_lineage(str(checkpoint), "rtdetr", {})

    assert lineage["checkpoint"]["metadata_status"] == "not_recorded"
    assert lineage["training_regime"] is None
    assert lineage["label_budget"] is None
    assert lineage["ground_truth"] == []
