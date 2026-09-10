"""Resolve the manual YOLO training supervision contract.

The regime is intentionally resolved in one place before the dataloader is
built. Ground-truth and cached-teacher boxes both participate in geometric
augmentation, so switching losses only inside the training loop would be too
late to provide either a clean teacher-only run or a faithful no-KD control.

Supervision is configured on **one** axis -- ``ground_truth_supervision`` --
combined with the existing ``kd_config['enabled']`` switch. The
``gt_only`` / ``hybrid`` / ``teacher_only`` / ``semi_supervised`` names are a
*derived* label on the resolved object, used for checkpoints, run suffixes and
log lines.

Deriving that label rather than configuring it is deliberate. ``gt_only`` and
``hybrid`` differ only in whether KD runs, which ``kd_config['enabled']``
already says, so accepting the name as a third setting would encode one bit
twice and admit a request like "gt_only with KD enabled" that can only be
honoured by overriding one of the two. Keeping it derived leaves
``kd_config['enabled']`` authoritative in every regime.

The axes produce six meaningful combinations:

    GT supervision        KD    label              objective
    "full"  (or True)     off   gt_only            supervised detection loss
    "full"  (or True)     on    hybrid             supervised loss + KD terms
    "none"  (or False)    off   teacher_only       teacher pseudo-detection base alone
    "none"  (or False)    on    teacher_only       that base plus soft/feature auxiliaries
    "semi_supervised"     off   semi_supervised    GT on labeled, teacher on unlabeled
    "semi_supervised"     on    semi_supervised    same + KD auxiliaries

"none" mandates the teacher pseudo-detection base: with no annotations there is
no other source of box/class/DFL supervision, so the base is switched on
regardless of ``kd_config['enabled']``, which then governs only the optional
auxiliaries.

"semi_supervised" trains on the full image pool.  Images within the label budget
receive ground-truth supervision; images outside it receive the teacher
pseudo-detection base.  KD auxiliaries are independently toggleable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings
from typing import Any, Mapping


# The derived label vocabulary. Written to checkpoints and run names, so a
# consumer reading that metadata has a closed set to validate against.
VALID_TRAINING_REGIMES = frozenset({
    "gt_only", "hybrid", "teacher_only", "semi_supervised",
})

# Accepted string values for ``ground_truth_supervision``. True/False are
# normalised to "full"/"none" for backward compatibility.
_GT_SUPERVISION_VALUES = frozenset({"full", "none", "semi_supervised"})

# The concrete KD methods `kd_config['enabled']` switches over. Listed here so
# the master switch can be checked against them rather than trusted on its own.
KD_METHODS = frozenset({"pseudo_label", "pseudo_label_soft", "feature_based"})


def _normalise_gt_supervision(raw: Any) -> str:
    """Normalise ``ground_truth_supervision`` to a canonical string.

    True and False are accepted as backward-compatible synonyms.
    """

    if raw is True:
        return "full"
    if raw is False:
        return "none"
    value = str(raw).lower().strip()
    if value not in _GT_SUPERVISION_VALUES:
        raise ValueError(
            f"training_regime['ground_truth_supervision'] must be one of "
            f"{sorted(_GT_SUPERVISION_VALUES)} (or True/False for backward "
            f"compatibility); got {raw!r}."
        )
    return value


def _derive_mode(gt_supervision: str, kd_enabled: bool) -> str:
    """Name the regime implied by the two independent supervision switches."""

    if gt_supervision == "semi_supervised":
        mode = "semi_supervised"
    elif gt_supervision == "none":
        mode = "teacher_only"
    else:
        mode = "hybrid" if kd_enabled else "gt_only"
    assert mode in VALID_TRAINING_REGIMES
    return mode


def _parse_teacher_settings(
    teacher_cfg: Mapping[str, Any],
    block_name: str,
) -> tuple[float, float, str]:
    """Validate and extract the teacher settings shared by teacher_only and semi_supervised.

    Returns (conf_threshold, validation_conf_threshold, empty_policy).
    """

    if not isinstance(teacher_cfg, Mapping):
        raise TypeError(
            f"training_regime['{block_name}'] must be a mapping."
        )

    conf_threshold = float(teacher_cfg.get("teacher_conf_threshold", 0.5))
    if not math.isfinite(conf_threshold) or not 0.0 <= conf_threshold <= 1.0:
        raise ValueError(
            f"training_regime['{block_name}']['teacher_conf_threshold'] must "
            f"be finite and in [0, 1]; got {conf_threshold!r}."
        )

    raw_validation_threshold = teacher_cfg.get(
        "validation_teacher_conf_threshold", None
    )
    validation_conf_threshold = (
        conf_threshold if raw_validation_threshold is None
        else float(raw_validation_threshold)
    )
    if (not math.isfinite(validation_conf_threshold)
            or not 0.0 <= validation_conf_threshold <= 1.0):
        raise ValueError(
            f"training_regime['{block_name}']['validation_teacher_conf_threshold'] "
            f"must be finite and in [0, 1]; got {validation_conf_threshold!r}."
        )

    empty_policy = str(teacher_cfg.get("empty_target_policy", "skip")).lower()
    if empty_policy not in {"skip", "background"}:
        raise ValueError(
            f"training_regime['{block_name}']['empty_target_policy'] must be "
            f"'skip' or 'background'; got {empty_policy!r}."
        )

    return conf_threshold, validation_conf_threshold, empty_policy


@dataclass(frozen=True)
class ResolvedTrainingRegime:
    """Validated, immutable behavior derived from the user-facing config."""

    mode: str
    use_ground_truth: bool
    is_semi_supervised: bool
    use_teacher_pseudo_base: bool
    use_aux_kd: bool
    kd_pipeline_enabled: bool
    teacher_conf_threshold: float
    validation_teacher_conf_threshold: float
    empty_target_policy: str
    teacher_only_train_json: str | None
    require_empty_annotations: bool
    validation_targets: str
    validation_pseudo_json: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return plain checkpoint metadata without leaking mutable config."""

        return asdict(self)


def resolve_training_regime(
    training_regime: Mapping[str, Any] | None,
    kd_config: Mapping[str, Any] | None,
) -> ResolvedTrainingRegime:
    """Resolve the supervision contract without mutating either config.

    ``ground_truth_supervision`` defaults to ``"full"``, so a config that omits
    the key resolves to plain supervised training with ``kd_config['enabled']``
    deciding whether KD runs alongside it.
    """

    requested = training_regime or {}
    kd = kd_config or {}

    gt_supervision = _normalise_gt_supervision(
        requested.get("ground_truth_supervision", True)
    )
    kd_enabled = bool(kd.get("enabled", False))
    mode = _derive_mode(gt_supervision, kd_enabled)

    is_semi_supervised = gt_supervision == "semi_supervised"
    use_ground_truth = gt_supervision in {"full", "semi_supervised"}

    # ── Defaults (overridden below for teacher_only and semi_supervised) ──
    conf_threshold = 0.5
    empty_policy = "skip"
    validation_targets = "ground_truth"
    validation_conf_threshold = 0.0
    validation_pseudo_json = None
    train_json = None
    require_empty_annotations = False

    # ── teacher_only ─────────────────────────────────────────────────────
    if gt_supervision == "none":
        teacher_cfg = requested.get("teacher_only", {}) or {}
        conf_threshold, validation_conf_threshold, empty_policy = (
            _parse_teacher_settings(teacher_cfg, "teacher_only")
        )

        base_loss = teacher_cfg.get("base_loss", "teacher_pseudo_detection")
        if base_loss != "teacher_pseudo_detection":
            raise ValueError(
                "training_regime['teacher_only']['base_loss'] currently supports "
                f"only 'teacher_pseudo_detection'; got {base_loss!r}."
            )

        # Validation in teacher_only mode always uses teacher predictions, so
        # ``validation_targets`` has nothing left to select. Warn rather than
        # silently ignore a config that sets it to anything else.
        requested_val_targets = teacher_cfg.get("validation_targets")
        if requested_val_targets is not None and str(requested_val_targets).lower() != "teacher":
            warnings.warn(
                "training_regime['teacher_only']['validation_targets'] is "
                "deprecated and ignored. Teacher-only mode always validates "
                "against teacher predictions. Remove the key to silence this "
                "warning.",
                DeprecationWarning,
                stacklevel=2,
            )
        validation_targets = "teacher"

        validation_pseudo_json = teacher_cfg.get("validation_pseudo_json")
        if validation_pseudo_json is not None:
            validation_pseudo_json = str(validation_pseudo_json)
        if not validation_pseudo_json:
            raise ValueError(
                "training_regime['teacher_only']['validation_pseudo_json'] is "
                "required; it names the manifest the teacher's validation "
                "boxes are written to."
            )

        train_json = teacher_cfg.get("train_json")
        if train_json is not None:
            train_json = str(train_json)
            if not train_json.strip():
                raise ValueError(
                    "training_regime['teacher_only']['train_json'] must be a "
                    "non-empty path when explicitly configured; use None to "
                    "select full_train_json."
                )
        require_empty_annotations = bool(
            teacher_cfg.get("require_empty_annotations", False)
        )

    # ── semi_supervised ──────────────────────────────────────────────────
    elif is_semi_supervised:
        semi_cfg = requested.get("semi_supervised", {}) or {}
        conf_threshold, validation_conf_threshold, empty_policy = (
            _parse_teacher_settings(semi_cfg, "semi_supervised")
        )

        validation_targets = str(
            semi_cfg.get("validation_targets", "ground_truth")
        ).lower()
        if validation_targets not in {"ground_truth", "ground_truth_teacher"}:
            raise ValueError(
                "training_regime['semi_supervised']['validation_targets'] must "
                f"be 'ground_truth' or 'ground_truth_teacher'; got "
                f"{validation_targets!r}."
            )

        validation_pseudo_json = semi_cfg.get("validation_pseudo_json")
        if validation_pseudo_json is not None:
            validation_pseudo_json = str(validation_pseudo_json)
        if validation_targets == "ground_truth_teacher" and not validation_pseudo_json:
            raise ValueError(
                "training_regime['semi_supervised']['validation_pseudo_json'] "
                "is required when validation_targets is "
                "'ground_truth_teacher'; it names the manifest the mixed "
                "GT + teacher validation annotations are written to."
            )

    # ── Derived flags ────────────────────────────────────────────────────
    # The pseudo-detection base is mandatory when ground truth is fully off.
    # In semi_supervised mode it is also needed -- but only for unlabeled
    # images, which the training loop gates via the has_gt mask.
    use_teacher_base = gt_supervision in {"none", "semi_supervised"}

    # `enabled` is a master switch over the three concrete methods, not a
    # supervision mode of its own. Turning it on with all three off produces a
    # run that resolves to "hybrid", is stamped hybrid, and loads a ~686 MB
    # teacher it never calls -- while optimising exactly the gt_only objective.
    # Nothing downstream can notice, because the difference is the absence of a
    # loss term. Rejecting it here costs one config read and removes a whole
    # class of ablation result that looks like evidence and is not.
    active_methods = [
        name for name in KD_METHODS
        if isinstance(kd.get(name), Mapping) and kd[name].get("enabled", False)
    ]
    if kd_enabled and not active_methods:
        raise ValueError(
            "kd_config['enabled'] is True but no KD method is: expected at least "
            f"one of {sorted(KD_METHODS)} to have enabled=True. Set one, or set "
            "kd_config['enabled'] = False -- in teacher_only that leaves the "
            "pseudo-detection base running on its own, which is a real arm; in "
            "a supervised run it is plain gt_only."
        )

    use_aux_kd = kd_enabled

    return ResolvedTrainingRegime(
        mode=mode,
        use_ground_truth=use_ground_truth,
        is_semi_supervised=is_semi_supervised,
        use_teacher_pseudo_base=use_teacher_base,
        use_aux_kd=use_aux_kd,
        kd_pipeline_enabled=use_teacher_base or use_aux_kd,
        teacher_conf_threshold=conf_threshold,
        validation_teacher_conf_threshold=validation_conf_threshold,
        empty_target_policy=empty_policy,
        teacher_only_train_json=train_json,
        require_empty_annotations=require_empty_annotations,
        validation_targets=validation_targets,
        validation_pseudo_json=validation_pseudo_json,
    )
