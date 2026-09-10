"""Content-addressed artifacts for label-budget experiments.

Every external artifact produced by a run must say which labelled train and
validation manifests produced it.  Keeping that rule here prevents the three
training backends from inventing subtly different naming conventions, and it
also gives Ultralytics a budget-aware dataset YAML generated directly from the
same COCO manifests used by the manual loops.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from label_budget import MANIFEST_FINGERPRINT_SCHEMA, manifest_fingerprint


RUN_PROVENANCE_VERSION = 1
RUN_PROVENANCE_KEY = "run_provenance"


def subsample_image_ids(image_ids, interval: int, *, label: str = "subsample_interval"):
    """Take one image in `interval`, by a rule that does not depend on the manifest.

    Positional slicing (``image_ids[::interval]``) sorts whatever manifest it was
    handed and strides *that* list, so which images survive depends on which
    images were there to begin with. This broke provenance verification down the line:
    the validation manifest of a teacher-only run records only 773 annotations
    (because its images were absent from the validation labels), but it verifies
    against a generic teacher record that expects all 1546. Positional striding meant
    639 of the 773 selected quarter-training images were absent from the selected
    half set, and validation lost nesting too. Two budgets that are supposed to
    differ only in size ended up differing in composition, which is the one thing
    the nested draw exists to prevent.

    Selecting on the image id instead makes membership a property of the image,
    so a subset of a manifest yields exactly the subset of that manifest's
    selection. COCO ids here are assigned 1..N in file-name order by
    ``prepare_dataset.yolo_to_coco``, so on a complete manifest this matches a
    positional stride element for element, and on a derived manifest it keeps
    the ids the complete manifest would have chosen.

    Args:
        image_ids: Image ids to filter. Order is preserved.
        interval: Keep one in this many. 1 keeps everything.
        label: Setting name to quote if `interval` is invalid.

    Returns:
        list: The retained ids, in the order given.

    Raises:
        ValueError: If `interval` is not positive.
    """
    interval = int(interval)
    if interval <= 0:
        raise ValueError(f"{label} must be positive, got {interval}.")
    if interval == 1:
        return list(image_ids)
    return [image_id for image_id in image_ids if (int(image_id) - 1) % interval == 0]


def _canonicalize(value: Any) -> Any:
    """Convert supported settings into a deterministic JSON value."""
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Run-provenance mapping keys must all be strings.")
        return {
            key: _canonicalize(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=_canonical_json)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Run provenance cannot contain NaN or infinity.")
        return value
    raise TypeError(
        f"Run provenance contains unsupported value {value!r} "
        f"({type(value).__name__})."
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_mapping_digest(mapping: Mapping[str, Any]) -> str:
    """Return a full SHA-256 over a mapping's canonical JSON representation."""
    if not isinstance(mapping, Mapping):
        raise TypeError("Run provenance identity must be a mapping.")
    canonical = _canonicalize(mapping)
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def run_provenance_digest(identity: Mapping[str, Any]) -> str:
    """Hash a caller-supplied run identity under the current schema version."""
    return canonical_mapping_digest({
        "version": RUN_PROVENANCE_VERSION,
        "identity": identity,
    })


def run_provenance_tag(identity: Mapping[str, Any]) -> str:
    """Return a collision-resistant, versioned filename tag for a run identity."""
    return f"run-v{RUN_PROVENANCE_VERSION}-{run_provenance_digest(identity)}"


def build_run_provenance(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical record stored in checkpoints and report artifacts.

    The caller owns the identity schema and should include every setting that
    can change the meaning or output of a run. Typical fields include sampling,
    teacher checkpoint SHA-256, supervision regime, KD configuration, actual
    training manifest, and validation target/manifest.
    """
    if not isinstance(identity, Mapping):
        raise TypeError("Run provenance identity must be a mapping.")
    canonical_identity = _canonicalize(identity)
    digest = run_provenance_digest(canonical_identity)
    return {
        "version": RUN_PROVENANCE_VERSION,
        "identity": canonical_identity,
        "digest": digest,
        "tag": f"run-v{RUN_PROVENANCE_VERSION}-{digest}",
    }


def _validated_provenance_identity(
    record: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} run-provenance record is not a mapping.")
    version = record.get("version")
    if version != RUN_PROVENANCE_VERSION:
        raise ValueError(
            f"{label} run-provenance schema {version!r} is unsupported; "
            f"expected {RUN_PROVENANCE_VERSION}. Start a fresh run with a "
            f"newly generated provenance record."
        )
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} run provenance has no identity mapping.")
    canonical_identity = _canonicalize(identity)
    expected_digest = run_provenance_digest(canonical_identity)
    if record.get("digest") != expected_digest:
        raise ValueError(
            f"{label} run-provenance digest is invalid; the record may be "
            f"incomplete or modified. Start a fresh run or restore the original "
            f"artifact."
        )
    expected_tag = f"run-v{RUN_PROVENANCE_VERSION}-{expected_digest}"
    if record.get("tag") != expected_tag:
        raise ValueError(
            f"{label} run-provenance tag is invalid; expected {expected_tag}."
        )
    return canonical_identity


def verify_run_provenance(
    stored: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    artifact_path: str,
) -> None:
    """Refuse reuse unless stored and current run identities match exactly."""
    current_identity = _validated_provenance_identity(current, label="Current")
    if stored is None:
        raise ValueError(
            f"Artifact {artifact_path} carries no {RUN_PROVENANCE_KEY!r} record, "
            f"so it cannot be safely reused or resumed. Start a fresh run, or "
            f"select an artifact created with the current run-provenance schema."
        )
    stored_identity = _validated_provenance_identity(stored, label="Stored")
    if _canonical_json(stored_identity) == _canonical_json(current_identity):
        return

    fields = sorted(set(stored_identity) | set(current_identity))
    changed = [
        field for field in fields
        if _canonical_json(stored_identity.get(field))
        != _canonical_json(current_identity.get(field))
    ]
    raise ValueError(
        f"Run-provenance mismatch for artifact {artifact_path}; differing "
        f"identity fields: {', '.join(changed) or '<nested identity>'}. "
        f"Stored digest {stored.get('digest')} does not match current digest "
        f"{current.get('digest')}. Start a fresh run/output path, or select the "
        f"artifact produced by this exact run configuration."
    )


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the complete SHA-256 of a file, suitable for teacher identity."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return absolute COCO path, canonical fingerprint, and record counts."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Annotation manifest not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"COCO manifest must contain a JSON object: {resolved}")
    return {
        "manifest": str(resolved),
        "fingerprint_schema": MANIFEST_FINGERPRINT_SCHEMA,
        "fingerprint": manifest_fingerprint(str(resolved)),
        "images": len(payload.get("images", [])),
        "annotations": len(payload.get("annotations", [])),
        "categories": len(payload.get("categories", [])),
    }


def _safe_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return token or "unnamed"


def budget_run_tag(record: Mapping[str, Any]) -> str:
    """Return a short, stable run tag containing train and validation identity."""

    train_fp = record.get("train", {}).get("fingerprint")
    val_fp = record.get("val", {}).get("fingerprint")
    if not train_fp or not val_fp:
        raise ValueError("A budget record must contain train and val fingerprints.")
    name = "full" if record.get("name") is None else _safe_token(record["name"])
    return f"{name}-{str(train_fp)[:8]}-{str(val_fp)[:8]}"


def budget_named_path(base_path: str, record: Mapping[str, Any]) -> str:
    """Name a checkpoint after the annotation budget that produced it.

    Weights are named for people. A budget name is the one axis that cannot be
    read back from a checkpoint at a glance and that silently ruins an
    experiment when two runs share a file, so it goes in the name; the
    fingerprints and provenance digest do not, because nothing needs them there.
    Cross-budget resume and cross-budget reuse are already refused by the
    records stamped *inside* the checkpoint, which a rename cannot defeat and a
    file name never enforced -- the name only ever kept two files apart.

    The full budget keeps the traditional bare name, matching how
    ``budget_manifest`` resolves ``label_budget = None`` to ``train.json``.

    Args:
        base_path: Name the run would use with no budget, e.g. ``rtdetr_best.pth``.
        record: Budget record from `label_budget.budget_record`.

    Returns:
        str: ``rtdetr_best.pth`` at the full budget, ``rtdetr_best_half.pth`` otherwise.
    """
    name = record.get("name")
    if name is None:
        return base_path
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}_{_safe_token(name)}{path.suffix}"))


def content_addressed_path(
    base_path: str,
    record: Mapping[str, Any],
    *identity_tokens: object,
) -> str:
    """Insert budget and optional producer identity before a path's extension."""

    path = Path(base_path)
    tokens = [budget_run_tag(record), *(_safe_token(v) for v in identity_tokens)]
    return str(path.with_name(f"{path.stem}_{'_'.join(tokens)}{path.suffix}"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_symlink(path: Path, target: Path) -> None:
    """Publish a relative dataset view without copying source image bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() and path.resolve() == target:
        return
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _materialize_ultralytics_split(
    manifest: str,
    project_root: Path,
    dataset_root: Path,
    split: str,
    category_mapping: Mapping[int, Mapping[str, Any]],
) -> tuple[list[str], int]:
    """Create an exact image view and YOLO labels from one COCO manifest."""
    with open(manifest, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = payload.get("images", [])
    if not images:
        raise ValueError(f"Ultralytics manifest contains no images: {manifest}")

    image_ids = [int(image["id"]) for image in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Ultralytics manifest contains duplicate image IDs: {manifest}")

    model_class_by_category = {
        int(category.get("id", model_class)): model_class
        for model_class, (_, category) in enumerate(
            sorted(category_mapping.items(), key=lambda item: item[0])
        )
    }
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = {
        image_id: [] for image_id in image_ids
    }
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in annotations_by_image:
            raise ValueError(
                f"Ultralytics manifest contains annotation {annotation.get('id')} "
                f"for unknown image ID {image_id}: {manifest}"
            )
        annotations_by_image[image_id].append(annotation)

    resolved = []
    annotation_count = 0
    for image in sorted(images, key=lambda item: (str(item.get("file_name", "")), int(item["id"]))):
        raw = Path(image["file_name"])
        source_path = raw if raw.is_absolute() else project_root / raw
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Image {image['id']} from {manifest} does not exist: {source_path}"
            )
        image_id = int(image["id"])
        width, height = int(image["width"]), int(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Image {image_id} has invalid dimensions {width}x{height}: {manifest}"
            )

        # COCO IDs make the view collision-proof even when two source folders
        # contain the same basename. Ultralytics maps /images/ to /labels/, so
        # the generated paths deliberately preserve that directory convention.
        view_name = f"{image_id:012d}_{source_path.name}"
        image_path = dataset_root / "images" / split / view_name
        label_path = dataset_root / "labels" / split / f"{Path(view_name).stem}.txt"
        _atomic_symlink(image_path, source_path)

        lines = []
        for annotation in sorted(
            annotations_by_image[image_id],
            key=lambda item: (int(item.get("id", 0)), _canonical_json(item)),
        ):
            if int(annotation.get("iscrowd", 0)):
                continue
            category_id = int(annotation["category_id"])
            if category_id not in model_class_by_category:
                raise ValueError(
                    f"Annotation {annotation.get('id')} uses category_id "
                    f"{category_id}, absent from category_mapping."
                )
            try:
                left, top, box_width, box_height = map(float, annotation["bbox"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Annotation {annotation.get('id')} has an invalid COCO bbox."
                ) from exc
            values = (left, top, box_width, box_height)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Annotation {annotation.get('id')} has a non-finite bbox."
                )
            right = max(0.0, min(float(width), left + box_width))
            bottom = max(0.0, min(float(height), top + box_height))
            left = max(0.0, min(float(width), left))
            top = max(0.0, min(float(height), top))
            box_width, box_height = right - left, bottom - top
            if box_width <= 0 or box_height <= 0:
                continue
            cx = (left + box_width / 2.0) / width
            cy = (top + box_height / 2.0) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            lines.append(
                f"{model_class_by_category[category_id]} {cx:.10g} {cy:.10g} "
                f"{normalized_width:.10g} {normalized_height:.10g}"
            )
            annotation_count += 1
        _atomic_write_text(label_path, "\n".join(lines) + ("\n" if lines else ""))
        # Keep the symlink path itself in the list. Resolving it would point
        # Ultralytics back at the source /images/ tree and therefore at the old
        # adjacent labels instead of the generated budget-specific labels.
        resolved.append(str(image_path.absolute()))
    return resolved, annotation_count


def write_ultralytics_budget_dataset(
    *,
    train_manifest: str,
    val_manifest: str,
    output_dir: str,
    budget_record: Mapping[str, Any],
    category_mapping: Mapping[int, Mapping[str, Any]],
    project_root: str | None = None,
) -> dict[str, Any]:
    """Write image lists and a YOLO YAML for exactly one annotation budget.

    Ultralytics normally consumes directory paths, which silently expands a
    half/quarter/eighth experiment back to every image. Text-file sources preserve the
    exact image membership. A cache-local symlink view and YOLO labels generated
    directly from COCO ensure the annotations consumed are the ones fingerprinted
    by the budget record, rather than potentially stale adjacent label files.
    """

    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output

    manifests = {"train": train_manifest, "val": val_manifest}
    actual_fingerprints = {
        split: manifest_fingerprint(path) for split, path in manifests.items()
    }
    for split, fingerprint in actual_fingerprints.items():
        expected = budget_record.get(split, {}).get("fingerprint")
        if fingerprint != expected:
            raise ValueError(
                f"{split} manifest {manifests[split]} does not match the run's "
                f"label-budget record (manifest {fingerprint}, record {expected})."
            )

    run_tag = budget_run_tag(budget_record)
    dataset_root = output / f"dataset_{run_tag}"
    train_images, train_annotations = _materialize_ultralytics_split(
        train_manifest, root, dataset_root, "train", category_mapping
    )
    val_images, val_annotations = _materialize_ultralytics_split(
        val_manifest, root, dataset_root, "val", category_mapping
    )
    train_list = output / f"train_{run_tag}.txt"
    val_list = output / f"val_{run_tag}.txt"
    data_yaml = output / f"dataset_{run_tag}.yaml"

    _atomic_write_text(train_list, "\n".join(train_images) + "\n")
    _atomic_write_text(val_list, "\n".join(val_images) + "\n")

    categories = [category_mapping[key]["name"] for key in sorted(category_mapping)]
    yaml_lines = [
        f"path: {json.dumps(str(root))}",
        f"train: {json.dumps(str(train_list.resolve()))}",
        f"val: {json.dumps(str(val_list.resolve()))}",
        f"nc: {len(categories)}",
        "names:",
        *(f"  {index}: {json.dumps(name)}" for index, name in enumerate(categories)),
        "",
    ]
    _atomic_write_text(data_yaml, "\n".join(yaml_lines))
    return {
        "path": str(data_yaml.resolve()),
        "train_list": str(train_list.resolve()),
        "val_list": str(val_list.resolve()),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_annotations": train_annotations,
        "val_annotations": val_annotations,
        "dataset_root": str(dataset_root.resolve()),
        "run_tag": run_tag,
    }
