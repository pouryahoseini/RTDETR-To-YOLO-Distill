"""Record and verify which annotation budget a checkpoint was trained under.

A label-budget experiment produces several teachers that differ only in how many
annotations they were allowed to see. Nothing about a checkpoint says which one
it is: the file name is a convention, the directory is a convention, and both
survive being moved or copied. Distilling a full-annotation teacher into a
student that declares ``label_budget = "half"`` would then produce a perfectly
plausible number for a claim the run does not support, and no error anywhere.

So the budget travels *in* the checkpoint, and a run that reads a teacher checks
it against its own. The comparison is on a canonical fingerprint of the COCO
images, annotations, and categories rather than on the budget's name or JSON
byte layout. A name is only as stable as the config entry behind it -- changing
``label_budget_splits["half"]["seed"]`` redefines what "half" means while
leaving every existing checkpoint claiming to be it. The canonical training
content is what the run actually consumed.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping


# Bump when the meaning of a stamped record changes. A checkpoint carrying an
# older version is surfaced so verification can explain how to upgrade it, but
# it is never trusted as a match.
BUDGET_RECORD_VERSION = 2
MANIFEST_FINGERPRINT_SCHEMA = "coco-training-content-v1"

# Key the record is stored under inside a checkpoint dict.
BUDGET_RECORD_KEY = "label_budget"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def _canonical_records(payload: Mapping[str, Any], field: str) -> list[Any]:
    records = payload.get(field, [])
    if not isinstance(records, list) or any(not isinstance(item, Mapping)
                                            for item in records):
        raise ValueError(f"COCO manifest field {field!r} must be a list of objects.")
    return sorted(records, key=_canonical_json)


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    canonical = {
        field: _canonical_records(payload, field)
        for field in ("images", "annotations", "categories")
    }
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _load_manifest(annotation_path: str) -> Mapping[str, Any]:
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Annotation manifest not found: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"COCO manifest must contain a JSON object: {annotation_path}")
    return payload


def manifest_fingerprint(annotation_path: str) -> str:
    """Canonical identity of COCO training content.

    The hash includes complete image, annotation, and category records. Record
    ordering, object-key ordering, whitespace, and the manifest path do not
    affect it; changes to image identity/metadata, boxes/classes, or the class
    taxonomy do.

    Args:
        annotation_path: Path to a COCO manifest.

    Returns:
        str: A SHA-256 hex digest identifying the training content.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    return _fingerprint_payload(_load_manifest(annotation_path))


def _manifest_summary(annotation_path: str) -> dict[str, Any]:
    payload = _load_manifest(annotation_path)
    return {
        "manifest": annotation_path,
        "fingerprint_schema": MANIFEST_FINGERPRINT_SCHEMA,
        "fingerprint": _fingerprint_payload(payload),
        "images": len(payload.get("images", [])),
        "annotations": len(payload.get("annotations", [])),
        "categories": len(payload.get("categories", [])),
    }


def _resolve_budget_manifest(cfg, split: str) -> str:
    resolver = getattr(cfg, "budget_manifest", None)
    if callable(resolver):
        return resolver(
            split,
            budget=cfg.label_budget,
            budget_splits=cfg.label_budget_splits,
            annotations_dir=cfg.processed_annotations_dir,
        )
    legacy_resolver = getattr(cfg, "_budget_manifest", None)
    if callable(legacy_resolver):
        return legacy_resolver(split)
    raise AttributeError(
        "Config must expose budget_manifest() (or legacy _budget_manifest())."
    )


def budget_record(cfg) -> dict[str, Any]:
    """Describe the annotation budget a run is training under.

    Reads the *budget-resolved* manifests, not whatever manifest the run ends up
    iterating. A teacher-only student deliberately trains over the full image set
    while declaring a reduced budget -- it consumes no annotations, so nothing
    limits which images it sees -- and the quantity that has to match its teacher
    is the budget it claims, not the images it happened to load.

    Args:
        cfg: A config module exposing ``label_budget`` and ``budget_manifest``.

    Returns:
        dict: Record to store in a checkpoint.
    """
    return {
        "version": BUDGET_RECORD_VERSION,
        "name": cfg.label_budget,
        "train": _manifest_summary(_resolve_budget_manifest(cfg, "train")),
        "val": _manifest_summary(_resolve_budget_manifest(cfg, "val")),
    }


def read_budget_record(checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return any structured record, including legacy versions, or None."""
    record = checkpoint.get(BUDGET_RECORD_KEY)
    if not isinstance(record, Mapping):
        return None
    return dict(record)


def describe(record: Mapping[str, Any] | None) -> str:
    """One-line human description of a record, for error messages."""
    if record is None:
        return "unstamped"
    train, val = record.get("train", {}), record.get("val", {})
    return (
        f"{record.get('name')!r} (schema {record.get('version')}; "
        f"train: {train.get('images')} images/{train.get('annotations')} annotations, "
        f"fingerprint {train.get('fingerprint')}; val: {val.get('images')} "
        f"images/{val.get('annotations')} annotations, fingerprint "
        f"{val.get('fingerprint')})"
    )


def verify_budget_match(
    stored_record: Mapping[str, Any] | None,
    run_record: Mapping[str, Any],
    artifact_path: str,
    *,
    artifact_label: str = "Checkpoint",
) -> None:
    """Fail unless a checkpoint and this run have identical train/val budgets."""
    if run_record.get("version") != BUDGET_RECORD_VERSION:
        raise ValueError(
            f"Run annotation-budget record uses schema {run_record.get('version')!r}; "
            f"expected {BUDGET_RECORD_VERSION}. Rebuild it with budget_record()."
        )

    stamp_command = (
        f"python src/label_budget.py --stamp {artifact_path} --budget <name|none>"
    )
    if stored_record is None:
        raise ValueError(
            f"{artifact_label} {artifact_path} carries no annotation-budget record, "
            f"so it cannot be checked against label_budget={run_record.get('name')!r}.\n"
            f"Stamp it with the budget it was actually trained under:\n"
            f"    {stamp_command}"
        )

    stored_version = stored_record.get("version")
    if stored_version != BUDGET_RECORD_VERSION:
        raise ValueError(
            f"{artifact_label} {artifact_path} carries legacy annotation-budget "
            f"schema {stored_version!r}; schema {BUDGET_RECORD_VERSION} is required "
            f"because it fingerprints images, annotations, categories, train, and val.\n"
            f"Restamp it with the budget it was actually trained under:\n"
            f"    {stamp_command}"
        )

    mismatches = []
    for split in ("train", "val"):
        stored_summary = stored_record.get(split, {})
        run_summary = run_record.get(split, {})
        stored_fp = stored_summary.get("fingerprint")
        run_fp = run_summary.get("fingerprint")
        stored_schema = stored_summary.get("fingerprint_schema")
        run_schema = run_summary.get("fingerprint_schema")
        if (stored_schema != MANIFEST_FINGERPRINT_SCHEMA
                or run_schema != MANIFEST_FINGERPRINT_SCHEMA
                or not isinstance(stored_fp, str)
                or not isinstance(run_fp, str)
                or stored_fp != run_fp):
            mismatches.append(
                f"{split} (checkpoint={stored_fp}, run={run_fp})"
            )

    if not mismatches:
        return

    raise ValueError(
        f"Annotation-budget mismatch for {artifact_label.lower()} {artifact_path}: "
        f"{', '.join(mismatches)}. This run declares {describe(run_record)}, but "
        f"the checkpoint records {describe(stored_record)}. Use a checkpoint "
        f"trained on this exact train/val budget, or start a fresh run."
    )


def verify_teacher_budget(teacher_record: Mapping[str, Any] | None,
                          run_record: Mapping[str, Any],
                          teacher_path: str) -> None:
    """Fail unless a teacher was trained under the same budget as this run.

    Args:
        teacher_record: Record read from the teacher checkpoint, or None.
        run_record: This run's record, from `budget_record`.
        teacher_path: Teacher checkpoint path, for the message.

    Raises:
        ValueError: If the teacher is unstamped, or its budget differs.
    """
    verify_budget_match(
        teacher_record, run_record, teacher_path,
        artifact_label="Teacher checkpoint",
    )


def stamp_checkpoint(
    checkpoint_path: str,
    record: Mapping[str, Any],
    *,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a budget record and optional provenance into a checkpoint.

    Used to annotate a checkpoint that carries no budget stamp of its own. The
    weights are rewritten byte-identically, but the file's fingerprint (size
    plus sampled bytes) does change, so any teacher cache built from it must be
    rebuilt -- `TeacherCache.load` refuses the stale one rather than serving
    it.

    Args:
        checkpoint_path: Checkpoint to annotate.
        record: Record from `budget_record`.
        extra_metadata: Additional top-level checkpoint records to persist in
            the same rewrite, such as the resolved training regime and exact
            run provenance for an externally managed training backend.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"{checkpoint_path} is not a checkpoint dict and cannot be stamped."
        )
    checkpoint[BUDGET_RECORD_KEY] = dict(record)
    if extra_metadata is not None:
        checkpoint.update(dict(extra_metadata))
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import configs.base_cfg as cfg

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stamp", metavar="CHECKPOINT",
                        help="Checkpoint to annotate with an annotation budget.")
    parser.add_argument("--budget", default=None,
                        help="Budget the checkpoint was trained under: a name from "
                             "label_budget_splits, or 'none' for the full set.")
    parser.add_argument("--show", metavar="CHECKPOINT",
                        help="Print the budget a checkpoint records.")
    args = parser.parse_args()

    if args.show:
        import torch

        payload = torch.load(args.show, map_location="cpu", weights_only=False)
        print(f"{args.show}: {describe(read_budget_record(payload))}")
    elif args.stamp:
        if args.budget is None:
            parser.error("--stamp requires --budget (use 'none' for the full set).")
        cfg.label_budget = None if args.budget.lower() == "none" else args.budget
        record = budget_record(cfg)
        stamp_checkpoint(args.stamp, record)
        print(f"stamped {args.stamp}: {describe(record)}")
        print("Any teacher cache built from this checkpoint must be rebuilt: "
              "stamping changes the file, and so its fingerprint.")
    else:
        parser.error("Nothing to do: pass --stamp or --show.")
