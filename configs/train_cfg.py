import os
from .base_cfg import *

# ─────────────────────────────────────────────────────────────────────────────
# Teacher
# ─────────────────────────────────────────────────────────────────────────────
# The teacher's identity, defined once above every consumer of it.
#
# It sits here rather than inside `kd_config` because with
# `ground_truth_supervision` set to "none" or "semi_supervised" the teacher is
# not a distillation detail -- it is the *entire* supervision signal. The
# pseudo-detection base, both response KD methods, feature KD and
# teacher-scored validation all resolve their teacher from this block, so
# `kd_config["enabled"]` says nothing about whether a teacher is loaded.
#
# It is deliberately not duplicated into `training_regime["teacher_only"]`:
# one value settable in two places is one value that can disagree with itself,
# the same argument that keeps the regime name derived rather than configured.
#
# No key here has a default at the call sites. A config missing `variant` must
# fail loudly rather than let the live teacher and the cache builder each fall
# back to a different architecture -- that would cache one model and distil
# into another.
teacher = {
    "model":   "rtdetr",   # Only "rtdetr" is implemented.
    "variant": "large",    # small (ResNet-18) | large (ResNet-50) | xlarge (ResNet-101)
    "weights": os.path.join(weights_dir, "rtdetr_large", "eighth", "rtdetr_best_eighth.pth"),
    # "weights": os.path.join(weights_dir, "rtdetr_large", "retrained", "rtdetr_best.pth"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Training supervision regime
# ─────────────────────────────────────────────────────────────────────────────
# Supervision has exactly one switch here. Combined with kd_config["enabled"]
# below it selects all six meaningful regimes:
#
#   ground_truth_supervision  kd_config["enabled"]  regime             objective
#   "full"  (or True)         False                 gt_only            supervised loss
#   "full"  (or True)         True                  hybrid             supervised + KD
#   "none"  (or False)        False                 teacher_only       teacher base alone
#   "none"  (or False)        True                  teacher_only       base + soft/feature
#   "semi_supervised"         False                 semi_supervised    GT on labeled, teacher on unlabeled
#   "semi_supervised"         True                  semi_supervised    same + KD auxiliaries
#
# The regime name is derived, not configured. gt_only and hybrid differ only in
# whether KD runs, which kd_config["enabled"] already says, so accepting the
# name as a third setting would encode one bit twice and allow a request like
# "gt_only with KD enabled" that can only be honoured by overriding one of the
# two. Deriving it keeps kd_config["enabled"] authoritative in every regime.
#
# "none" *mandates* the teacher pseudo-detection base: with no annotations
# there is no other source of box/class/DFL supervision, so that base runs
# regardless of kd_config["enabled"], which then governs only the optional
# soft/feature auxiliaries. No annotation reaches a training sample -- ground
# truth is purged before augmentation, class weighting and loss.
#
# "semi_supervised" trains on the full image pool. Images within the label
# budget receive ground-truth supervision; images outside it receive the
# teacher pseudo-detection base. KD auxiliaries are independently toggleable
# and fire on all images as usual.
#
# True and False are accepted as synonyms for "full" and "none" respectively.
training_regime = {
    "ground_truth_supervision": "semi_supervised",

    # Read only when ground_truth_supervision is "none".
    # Validation always uses teacher predictions in this mode (the teacher is
    # the sole source of labels, so scoring against GT would introduce a hidden
    # annotation dependency). The validation_teacher_conf_threshold and the
    # pseudo-annotation output path are configured here.
    "teacher_only": {
        # Declarative identifier, not a menu: this is currently the only
        # implemented teacher-only base objective. Any other value fails fast.
        "base_loss": "teacher_pseudo_detection",
        "teacher_conf_threshold": 0.4,

        # Threshold used to build the teacher's *validation* targets. None
        # tracks `teacher_conf_threshold` above, which is the right default:
        # normally the model should be selected under the conditions it is trained on.
        #
        # Set it to a fixed number when ablating `teacher_conf_threshold`. That
        # knob otherwise moves the training objective and the checkpoint-
        # selection metric at the same time, so each arm is scored by a
        # different ruler and the sweep cannot say which threshold trained the
        # better student -- only which one agreed most with its own targets.
        # Pinning validation makes the comparison a comparison.
        "validation_teacher_conf_threshold": None,
        # "skip" avoids teaching background on images where the teacher is
        # silent. "background" deliberately retains those empty images.
        "empty_target_policy": "skip",
        # Optionally point at a COCO manifest with the same images/categories
        # and image IDs but annotations=[]. If None, full_train_json is used and
        # its annotations are purged before sampling, augmentation, and loss.
        "train_json": None,

        # Where the teacher's validation pseudo-annotations are written.
        # Rebuilt each run so it always matches the threshold above.
        "validation_pseudo_json": os.path.join(
            processed_annotations_dir, "val_teacher_pseudo.json"
        ),
        # Set True for the strongest annotation-free experiment claim. It
        # rejects a manifest containing any annotation instead of purging it.
        "require_empty_annotations": False,
    },

    # Read only when ground_truth_supervision is "semi_supervised".
    # Teacher settings for the unlabeled partition -- same knobs as teacher_only.
    "semi_supervised": {
        "teacher_conf_threshold": 0.4,
        # "skip" avoids teaching background on unlabeled images where the
        # teacher is silent. "background" deliberately retains those empty
        # images as negative examples.
        "empty_target_policy": "skip",

        # What the per-epoch validation scores against.
        #
        #   "ground_truth"          Budgeted val subset, GT labels only.
        #                           The validation set is val_json (the budgeted
        #                           split). Checkpoint selection uses only the
        #                           labels the budget bought.
        #   "ground_truth_teacher"  Full val set with mixed annotations: GT for
        #                           images in the budgeted val split, teacher
        #                           predictions for the rest. Larger validation
        #                           pool, more representative score.
        "validation_targets": "ground_truth_teacher",

        # Threshold used to build the teacher's *validation* targets for the
        # non-budgeted images. None tracks `teacher_conf_threshold` above.
        "validation_teacher_conf_threshold": None,

        # Where the mixed pseudo-annotation manifest is written when
        # validation_targets is "ground_truth_teacher". Contains the budgeted
        # GT annotations plus teacher predictions for the remaining images.
        "validation_pseudo_json": os.path.join(
            processed_annotations_dir, "val_semi_supervised_pseudo.json"
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Distillation Configuration
# ─────────────────────────────────────────────────────────────────────────────
kd_config = {
    # Main KD switch in hybrid mode. In teacher_only it controls the optional
    # soft/feature auxiliaries; the pseudo-detection base remains mandatory.
    # Which teacher these methods distil from is set in `teacher` above.
    "enabled": True,
    # Only "mosaic" and "mixup" are recognised. Listing anything else does not
    # disable KD for it -- the appearance augmentations are handled instead by
    # `feature_based.teacher_clean_view`, which strips them from the teacher's
    # input rather than switching KD off for the sample.
    #
    # This is only the fallback for a method that does not set its own
    # `disable_on_augs`. Each method overrides it below, because they are not
    # equally sensitive: see the note on each for why.
    "disable_on_augs": ["mixup"],
    
    # Can enable multiple KD approaches later.
    #
    # All three methods share one schedule: the term is held at zero for
    # `kd_warmup_epochs`, then ramps linearly to `kd_weight` over the next
    # `kd_ramp_epochs`. `kd_ramp_epochs: 0` switches it on at full strength the
    # instant the warmup elapses -- a step change in the total loss big enough
    # to cost several epochs of validation mAP, which with
    # `early_stopping_patience: 5` can trip early stopping by itself.
    # Symmetrically at the end of training, the term can ramp down linearly to
    # zero over `kd_fade_epochs` and/or be held at zero for `kd_cooldown_epochs`
    # (e.g. matching `close_mosaic_epochs` for pure supervised fine-tuning).
    # ── Teacher confidence threshold ────────────────────────────────────────
    # Measured on 100 val images (5,429 annotations), fraction of ground truth
    # the teacher covers at IoU 0.5 with a box above each threshold:
    #
    #   thresh    small   medium    large      all     dets/img
    #     0.3     71.6%    84.8%    88.0%    76.2%       82.7
    #     0.5     57.0%    80.7%    87.0%    65.4%       46.5
    #     0.7     25.0%    67.7%    82.0%    40.2%       23.8
    #     0.9      0.3%    11.0%    48.0%     5.3%        2.9
    #
    # 0.7 threw away three quarters of the teacher's small-object coverage --
    # and small objects are 70.8% of VisDrone. That is the same population the
    # KD runs lost ground on (AP_small 10.9 -> 9.7, AR_small 27.4 -> 26.0), so
    # the threshold was selecting for precisely the teacher behaviour that hurt.
    # 0.5 nearly doubles small coverage while keeping detections per image
    # (46.5) close to the annotation density (54.3), so it is not simply buying
    # recall with false positives. Per-class or per-size thresholds are the
    # refinement if this pays off.
    "pseudo_label": {
        "enabled": False,
        "teacher_conf_threshold": 0.5,   # see the coverage table above
        "kd_weight": 0.2,                # Multiplier for the teacher's Pass 2 loss
        "kd_warmup_epochs": 4,           # Wait 4 epochs before injecting pseudo-labels
        "kd_ramp_epochs": 4,             # Then ramp 0 -> kd_weight over 4 more
        "kd_fade_epochs": 5,             # Ramp kd_weight -> 0 over the last 4 epochs
        "kd_cooldown_epochs": 1,         # Final epochs to hold the term at zero (0 = disabled)

        # Ground truth is the only authority on object *presence*, so the
        # pseudo target set is built from the annotations and a teacher box is
        # substituted only where the two agree. Building Pass 2 from the
        # teacher's detections alone would score every annotated object that
        # failed to clear `teacher_conf_threshold` as background at a
        # true-positive location -- an error biased towards exactly the small,
        # crowded objects VisDrone is made of.
        #
        # Minimum IoU (same class) for a teacher box to be taken as that GT
        # object's detection. Matching is mutual-best, so each box is used once.
        # Hard KD is GT-anchored: teacher boxes only refine annotations that
        # already exist, and teacher confidence is used for selection, never as
        # a regression target. Neither mosaic nor mixup can therefore corrupt
        # it the way they corrupt an absolute-confidence target, so nothing is
        # excluded.
        "disable_on_augs": [],

        "gt_match_iou": 0.5,
        # What to do with confident teacher boxes matching no annotation:
        #   "drop" -- discard them. Presence is then exactly the ground truth's,
        #             which is correct for a fully-labelled split. Note this
        #             also makes Pass 2 nearly a duplicate of the supervised
        #             pass: with presence fixed and boxes only nudged, there is
        #             little left for a *hard* label to carry.
        #   "add"  -- treat them as extra objects. Only sound where annotations
        #             are known to be incomplete; a teacher false positive
        #             becomes a hard positive label here.
        "unmatched_teacher_boxes": "add",
    },
    "pseudo_label_soft": {
        "enabled": True,                 # Use soft pseudo-labeling (Two-Pass Hybrid)
        "teacher_conf_threshold": 0.5,    # see the coverage table above

        # Temperature behaves very differently here than in textbook KD, which
        # softens a *softmax*: there, T redistributes a fixed unit of
        # probability mass between classes. This head is multi-label sigmoid,
        # so there is no normalisation constraint and T inflates the total mass
        # instead. Measured over the teacher's logits on 100 val images:
        #
        #      T    mean target    targets >0.1    mass/query
        #    1.0         0.0565          13.58%         0.565
        #    1.5         0.1063          34.91%         1.063
        #    2.0         0.1563          66.35%         1.563
        #    2.5         0.1998          93.41%         1.998
        #
        # At 2.5 the student was told that 93% of all class slots carry >0.1
        # probability and each query holds 2.0 units of mass -- 3.5x the
        # teacher's own. Positives are softened down and every background class
        # is pulled up toward 0.5, while the supervised BCE simultaneously
        # drives them to 0. The T^2 factor then amplified that distorted target
        # by 6.25x. Sigmoid outputs are monotonic in the logit at any T, so the
        # relative "dark knowledge" ordering survives T=1.0 intact; the only
        # thing higher T adds here is absolute-level distortion.
        #
        # The fix is not a smaller temperature but a split objective, so that
        # temperature is applied only where it is mathematically sound:
        #
        #   confidence  BCE at T=1 on the teacher's *selected class only*.
        #               Carries absolute calibration. The other nine slots are
        #               left entirely to the supervised loss, so no background
        #               class is ever pulled up off zero.
        #   relation    KL between softmax(student/T) and softmax(teacher/T),
        #               over all classes, scaled by T^2. Softmax mass is fixed
        #               at one unit, so here temperature genuinely redistributes
        #               instead of inflating, and the T^2 correction is the
        #               conventional and correct one. This is what transfers the
        #               dark knowledge -- on VisDrone, that pedestrian/people,
        #               truck/van and tricycle/awning-tricycle are confusable.
        #
        # `temperature` applies to the relation term only; 1.0 reduces it to a
        # plain softmax KL. Because the confidence term is held at T=1, the
        # relation term can carry a genuinely high temperature without the
        # calibration damage that would otherwise rule one out; 2.0 is a
        # reasonable starting point.
        # Soft KD is the one method mixup genuinely breaks. Its confidence term
        # distils an absolute probability the teacher measured on the *clean,
        # unmixed* source, so a box seen at 0.9 there becomes a "predict 0.9"
        # target for an object contributing only r of the blended pixels. The
        # ground truth is immune (presence is binary, and `_load_mixup`
        # concatenates both label sets at full weight by design), as are hard KD
        # and feature KD. Rescaling the target by r would instead make the KD
        # term inconsistent with the supervised term beside it, which does not
        # rescale either -- so the sample is simply excluded.
        "disable_on_augs": ["mixup"],

        "temperature": 2.0,
        "confidence_weight": 1.0,
        "relation_weight": 1.0,           # 0.0 falls back to confidence-only KD

        # Measured on 8 real batches with the split objective above: the raw
        # soft term is 3.36 per image against a supervised 2.90, so
        #
        #   kd_weight 0.002 ->  0.2% of GT loss
        #             0.01  ->  1.2%
        #             0.02  ->  2.3%
        #             0.09  -> 10.4%
        #
        # Weights quoted for a single-term softened-sigmoid objective do not
        # transfer here: that formulation is inflated by T^2=6.25 on top of the
        # probability-mass blow-up documented above, roughly 30x larger for the
        # same nominal weight. Ported directly, a 0.002-0.01 sweep would put
        # every arm under 1.2% -- indistinguishable from no KD at all.
        "kd_weight": 0.20,
        "kd_warmup_epochs": 4,
        "kd_ramp_epochs": 4,
        "kd_fade_epochs": 5,             # Ramp kd_weight -> 0 over the last 4 epochs
        "kd_cooldown_epochs": 1,         # Final epochs to hold the term at zero (0 = disabled)

        # Which detection head receives the soft targets. YOLO26 infers through
        # the one2one branch; one2many is auxiliary and is decayed from 0.8 to
        # 0.1 across training by `E2ELoss`. Distilling one2many alone therefore
        # supervises a branch that is progressively switched off and never runs
        # at inference.
        #   "both"     -- both branches, combined with E2ELoss's own o2m/o2o
        #                 gains so KD follows the same decay as the supervised
        #                 loss. Costs one extra assigner call per step.
        #   "one2one"  -- inference head only.
        #   "one2many" -- auxiliary branch only; kept for ablation.
        # Ignored for non-E2E models, which have a single head.
        "distill_heads": "both",
    },
    "feature_based": {
        "enabled": True,                 # Feature-based KD via projection adapters
        # The loss below sums four balanced terms, so its scale is much larger
        # than a single cosine term would give. Measured on 8 real batches
        # through the trained no-KD student and the trained teacher, at random
        # adapter init: 18.9 against a supervised 2.90 per image. So:
        #
        #   kd_weight 0.01 -> 6.1% of total loss
        #             0.02 -> 11.5%
        #             0.04 -> 20.7%
        #             0.06 -> 28.1%
        #
        # The init value is the wrong thing to calibrate against: the adapter
        # learns fast and the raw loss collapses with it. Measured over the
        # same batches while training the adapter:
        #
        #   adapter steps    raw loss    share of GT loss at 0.02
        #               0       18.90                      13.0%
        #              20        4.32                       3.0%
        #              60        3.56                       2.5%
        #             120        3.16                       2.2%
        #
        # An epoch is ~1600 steps and the adapter converges inside ~100, so a
        # weight tuned to the init value leaves the whole run at ~2% -- far
        # under the intended budget. The ramp below already protects the early
        # phase, so calibrate against the *plateau* instead: 0.09 puts feature
        # KD near 10% of the supervised loss once the adapter has settled.
        #
        # The plateau above was measured on 8 repeated batches, so it is partly
        # overfit and the true full-dataset plateau will be higher -- meaning
        # this weight buys somewhat more than 10%. Sweep 0.04 / 0.09 / 0.18 and
        # select on the gradient-norm ratio, not the scalar loss share.
        "kd_weight": 0.09,               # Beta multiplier for the feature loss
        # Warmup holds the term at zero, then the ramp fades it in. Setting
        # warmup 0 with ramp 4 starts the fade-in immediately; raising the
        # warmup delays it like the other two terms.
        "kd_warmup_epochs": 4,
        "kd_ramp_epochs": 4,
        "kd_fade_epochs": 5,             # Ramp kd_weight -> 0 over the last 4 epochs
        "kd_cooldown_epochs": 1,         # Final epochs to hold the term at zero (0 = disabled)

        # ── Gradient probe ───────────────────────────────────────────────────
        # Log ||g_kd|| / ||g_supervised|| and their cosine on the student's
        # parameters every N steps. This is the quantity to select `kd_weight`
        # on: the loss share above says nothing about how hard the term pulls,
        # because the feature terms differentiate through an L2 normalisation
        # whose gradient scales with 1/||f||. The cosine separates the two ways a
        # KD term can fail to help -- orthogonal (adding information the
        # supervised loss does not carry) versus anti-aligned (spending the step
        # undoing supervision) -- which look identical in every loss scalar.
        # Costs two extra backward passes when it fires, so it is sampled: at
        # ~1600 steps/epoch, 200 gives 8 probes per epoch for well under 1%
        # overhead. 0 disables it. Requires bf16 or fp32; it is skipped under
        # fp16, where the unscaled backward can underflow and report a
        # plausible-looking but wrong ratio.
        "grad_probe_interval": 200,

        # Feature KD cannot use the offline response cache -- it distils the
        # composite the student actually sees, so the teacher must run live on
        # that image. But there is no reason for the teacher to see the cutout
        # holes, rain streaks, motion blur and colour jitter on top of it: it
        # was never trained on those, so what it encodes there is noise, and
        # that noise is what gets distilled. True shows the teacher the same
        # geometry as the student -- resize, LSJ, crop, flip, and the mosaic or
        # mixup composite unless `disable_on_augs` excludes it -- with the
        # appearance augmentations stripped. False feeds the teacher the
        # student's fully augmented image instead.
        "teacher_clean_view": True,

        # Feature KD distils the composite the student actually sees, with the
        # teacher running live on that same image, so there is no calibration
        # mismatch to protect against. "mixup" is a conservative default and
        # [] is equally defensible. Do NOT add "mosaic" while mosaic p is 1.0 --
        # that would disable feature KD for 45 of 50 epochs and leave it active
        # only during the close-mosaic tail.
        "disable_on_augs": ["mixup"],

        # ── Focal-and-global feature distillation ────────────────────────────
        # Two properties this objective is built to avoid. A binary union mask
        # over the GT boxes scales an object's weight with its pixel area, so a
        # bus outvotes a hundred pedestrians; and distilling foreground alone
        # discards the background context that is most of what a big teacher
        # knows. Hence per-instance masks and a separate background term.
        #
        # Per-scale gains, ordered P3, P4, P5. Normalised internally, so these
        # set relative emphasis and do not change the overall magnitude.
        # Measured over the 343,204 training annotations at 736x1280: the median
        # object is 20.9px on a side, and by canonical FPN assignment 91.9% land
        # on P3, 7.1% on P4, 0.9% on P5 (70.8% are COCO-"small", 2.6% "large").
        # An even split would hand P5 a quarter of the loss for under 1% of the
        # objects. These weights normalise to 0.67 / 0.22 / 0.11 -- still far
        # more generous to P4/P5 than the raw counts, because their features
        # feed P3 through the neck and carry the scene context a nano student
        # cannot infer locally.
        # "scale_weights": [6.0, 2.0, 1.0],
        "scale_weights": [1.0, 1.0, 1.0],
        # Projection adapter. "conv1x1" is a single 1x1 convolution and is the
        # default because a deeper adapter is a liability: "dwsep"
        # (1x1 -> depthwise 3x3 -> 1x1) has enough capacity to satisfy the
        # alignment objective internally, letting the loss fall without the
        # student's own features changing.
        "adapter": "conv1x1",            # "conv1x1" | "dwsep"
        # BatchNorm is a poor choice at the physical batch of 4 that KD runs at.
        "adapter_norm": "group",         # "group" | "batch" | "none"
        # Softmax temperature for the spatial/channel attention maps taken from
        # the teacher. Lower sharpens the emphasis onto peak activations.
        "attention_temperature": 0.5,
        # Term gains. Foreground and background are separated so the background
        # can be distilled at a lower, non-zero weight rather than not at all
        # (FGD's central finding: indiscriminate imitation hurts, but excluding
        # background entirely throws away the teacher's context modelling).
        "fg_weight": 1.0,
        "bg_weight": 0.5,
        # Matching the teacher's attention maps themselves, not just the
        # features they weight.
        "attention_weight": 1.0,
        # Global relation term via a learned context block on each scale. This
        # is where cross-architecture KD tends to pay: a nano CNN cannot match
        # a transformer encoder pixel for pixel, but it can learn how the
        # teacher aggregates context. 0.0 disables it and builds no blocks.
        "global_weight": 0.5,
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# DataLoader settings
# ─────────────────────────────────────────────────────────────────────────────
dataloader_config = {
    # "batch_size": 16,
    "batch_size": 4,
    "num_workers": 16,
    "pin_memory": True,
    "frame_subsample_interval": 1,  # Subsample training: use every N-th frame (1 = no subsampling)
    "val_subsample_interval": 1,    # Subsample validation: use every N-th frame (1 = no subsampling)
    "test_subsample_interval": 1,   # Subsample test/evaluation: use every N-th frame (1 = no subsampling)
}

# ─────────────────────────────────────────────────────────────────────────────
# Training Settings
# ─────────────────────────────────────────────────────────────────────────────
shared_train_config = {
    # Shared training settings (RT-DETR and YOLO)
    "seed": 42,
    "epochs": 50 + (10 if model_format == "rtdetr" else 0),
    "early_stopping_patience": 5,
    # "gradient_accumulation_steps": 4,
    "gradient_accumulation_steps": 16,
    "precision": "bf16",  # Options: "fp16", "bf16", "fp32"
    "ema_decay": 0.9992,
    "ema_warmups": 2000,
    "pretrained_lr_ratio": 0.25,       # Initial learning rate multiplier for pretrained parameters
    "fade_pretrained_lr_ratio": True,  # Fade the pretrained LR reduction towards 1.0 (no reduction) as training progresses
    "fade_pretrained_lr_ratio_fraction": 0.2, # Percentage of total epochs (0.0 to 1.0) over which the fade completes
    "resume_training": False,          # Automatically resumes from weights/<model_format>_last.pth
}

# ─────────────────────────────────────────────────────────────────────────────
# YOLO Configuration (Ultralytics Format)
# ─────────────────────────────────────────────────────────────────────────────
yolo_train_config = {
    "model_yaml_path": "yolo26n.yaml",
    "pretrained_model_path": os.path.join(weights_dir, "yolo26n.pt"),

    # Manual Training Hyperparameters
    # Matches official Ultralytics defaults from ultralytics/cfg/default.yaml
    "hyp": {
        # Optimizer Selection
        "optimizer":      "AdamW",  # Options: "SGD", "AdamW"
        
        # Learning Rate and Weight Decay
        # For SGD: lr0=0.01, weight_decay=5e-4, warmup_bias_lr=0.1
        # For AdamW: lr0=0.001, weight_decay=0.01, warmup_bias_lr=0.001
        "lr0":            0.001,    # initial learning rate
        "lrf":            0.01,     # final LR as fraction of lr0 (lr0 * lrf at end)
        "momentum":       0.937,    # SGD momentum / AdamW beta1
        "weight_decay":   0.001,     # L2 regularization
        "warmup_epochs":  3.0,      # linear warmup epochs
        "warmup_momentum": 0.8,     # warmup start momentum → ramps to `momentum`
        "warmup_bias_lr": 0.001,    # warmup start LR for bias params

        # Loss gains (weights applied to each loss component)
        "box":  7.5,    # CIoU box regression loss weight
        "cls":  0.5,    # Classification loss weight
        "dfl":  1.5,    # Normalized L1 box-distance loss weight in YOLO26 (DFL loss weight in Ultralytics v8)

        # Focal Loss (Classification). (0.0 = no Focal Loss, just BCE).
        # Note: yolo_manual model natively uses plain BCE via v8DetectionLoss.
        # A custom wrapper in train.py applies Focal Loss if fl_gamma > 0.
        "fl_gamma": 0.0,    # 1.5

        # Focal Loss positive/negative balance, applied only when fl_gamma > 0.
        # Element-wise: positives are scaled by fl_alpha, negatives by
        # (1 - fl_alpha). This is NOT the same as the `cls` gain above, which
        # scales the whole classification term uniformly. Ultralytics' own
        # FocalLoss defaults to 0.25 (the RetinaNet convention, which weights
        # positives *down* relative to negatives). 0.0 disables it.
        "fl_alpha": 0.25,    # 0.25 to match ultralytics.utils.loss.FocalLoss

        # Inverse-frequency class weighting, ported from Ultralytics' `cls_pw`.
        # Per-class BCE is scaled by (1 / instance_count) ** cls_pw, normalised
        # to mean 1.0 across classes. 0.0 disables (Ultralytics' own default),
        # 1.0 is full inverse frequency, values in between damp it.
        "cls_pw": 0.0,

        # Normalising the weights to mean 1.0 across *classes* does not keep the
        # classification term's magnitude: the loss mass sits on the frequent
        # classes, which receive the smallest weights, so the total shrinks
        # (measured 0.82x / 0.69x / 0.54x at cls_pw 0.25 / 0.5 / 1.0). That
        # would silently change the cls-vs-box balance whenever cls_pw is tuned.
        # True rescales the weighted total back to the unweighted total on every
        # batch, making cls_pw a pure redistribution that is safe to ablate on
        # its own. Set False to reproduce Ultralytics' behaviour exactly, in
        # which case compensate with the `cls` gain above.
        "cls_pw_preserve_magnitude": True,

        # Gradient clipping
        "max_norm": 10.0,
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# RT-DETR Configuration
# ─────────────────────────────────────────────────────────────────────────────
rtdetr_train_config = {
    "variant": "large",  # "small" (ResNet-18) | "large" (ResNet-50) | "xlarge" (ResNet-101)
    "weights": os.path.join(weights_dir, "rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"),
    
    # RT-DETR Training Hyperparameters
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "lr_warmup_steps": 100,  # Number of iterations for linear warmup
    "lr_milestones": [30, 40],  # Epochs where LR is multiplied by lr_gamma
    "lr_gamma": 0.5,          # Multiplier for LR at milestones
    
    # Varifocal Loss parameters (classification head; RTDETRCriterion uses
    # `losses=['vfl', 'boxes']`, so these feed loss_labels_vfl).
    # NOTE: varifocal_alpha is NOT the same quantity as the YOLO `fl_alpha`
    # above. VFL weights negatives by `alpha * pred_score**gamma` and positives
    # by their IoU, so this alpha is a *background* weight; `fl_alpha` is a
    # *foreground* weight. Do not harmonise the two numbers.
    "varifocal_alpha": 0.75,
    "varifocal_gamma": 2.0,

    # Loss-term gains. The RT-DETR counterpart of the YOLO box/cls/dfl gains;
    # `loss_vfl` is the classification term. Upstream RT-DETR hardcodes these
    # (see configs/rtdetr/include/rtdetr_r50vd.yml); they are exposed here so
    # the classification/localisation balance is tunable on both models.
    "loss_weights": {
        "loss_vfl":  1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    },

    # Inverse-frequency class weighting, identical in definition and effect to
    # the YOLO `cls_pw` above -- same counts, same normalisation -- but applied
    # to the per-class VFL map. Not part of upstream RT-DETR. 0.0 disables.
    "cls_pw": 0.0,
    "cls_pw_preserve_magnitude": True,  # see the YOLO entry for the rationale
}

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation settings
# Each key maps directly to a parameter consumed by the dataloader.
# Set a boolean flag to False to disable that augmentation entirely.
# ─────────────────────────────────────────────────────────────────────────────
augmentation = {
    # Normalization values depend on the model's pre-training convention.
    # RT-DETR / DETR: ImageNet mean/std (backbone pre-trained on ImageNet).
    # YOLO:           divide by 255 only — no mean/std subtraction.
    "normalize": {
        "rtdetr":      {"mean": [0.0,   0.0,   0.0], "std": [1.0, 1.0,   1.0]},
        "yolo":        {"mean": [0.0,   0.0,   0.0], "std": [1.0, 1.0,   1.0]},
        "yolo_ultra":  {"mean": [0.0,   0.0,   0.0], "std": [1.0, 1.0,   1.0]},
        "yolo_manual": {"mean": [0.0,   0.0,   0.0], "std": [1.0, 1.0,   1.0]},
    },

    # ── Spatial: Large Scale Jitter (LSJ) ─────────────────────────────────────
    # RT-DETR default: scale [0.1, 2.0], p=1.0 (always applied).
    # Crop retries require at least one box to survive the sampled visibility
    # constraint; 0.0 means only the baseline min_visibility applies.
    "lsj": {
        "enabled": True,
        "scale_limit":             [-0.5, 1.0],   # RandomScale range → [0.5×, 2.0×]
        "p":                       1.0, # 1.0
        # Whether LSJ also runs on mosaic samples. The mosaic already does its
        # own scale-and-crop (`mosaic.scale_limit`), so:
        #   False → LSJ governs non-mosaic samples only (mosaic jitters itself)
        #   True  → both run, independently; the scales compose, and LSJ pads
        #           with grey below 1.0 because the mosaic canvas is already
        #           cropped away. Set `mosaic.scale_limit` to [0.0, 0.0] to make
        #           LSJ the only source of scale jitter.
        "apply_on_mosaic":         False,
        # Additional minimum visible fraction sampled for each crop retry.
        # 0.0 means use the baseline `min_visibility` below; larger values make
        # the crop retain more of at least one object before it is accepted.
        "min_visibility_candidates": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        "max_crop_trials":         50,
        # Minimum fraction of a box that must remain visible after the crop.
        # Albumentations BboxParams.min_visibility is read from here to keep
        # both in sync. RT-DETR uses 0.1; SSD uses ~0.2. 0.1 is more permissive
        # and better for small object diversity.
        "min_visibility":          0.1,
    },

    # ── Photometric Distortion ─────────────────────────────────────────────────
    # Applied BEFORE geometry so it runs on the raw image, matching the
    # SSD / RT-DETR convention.  Values below match YOLOv8 defaults:
    #   brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015.
    # p=1.0: always apply (YOLOv8 / RT-DETR both apply this every sample).
    "photometric_distort": {
        "enabled":    True,
        "p":          1.0, # 1.0
        "brightness": 0.4,    # ±40 % brightness shift   (YOLOv8: hsv_v=0.4)
        "contrast":   0.4,    # ±40 % contrast shift
        "saturation": 0.7,    # ±70 % saturation shift   (YOLOv8: hsv_s=0.7)
        "hue":        0.015,  # ±1.5 % hue rotation      (YOLOv8: hsv_h=0.015)
    },

    # ── Cross-image composition ────────────────────────────────────────────────
    # Mosaic: YOLOv8 uses p=1.0 during the mosaic phase; the close_mosaic_epochs
    # mechanism below disables it for the final epochs automatically.
    "mosaic": {
        "enabled": False if model_format == "rtdetr" else True,
        "p":       0.5,
        # Number of final epochs during which Mosaic and MixUp are disabled so the
        # model can fine-tune on realistic (non-synthetic) images.
        "close_mosaic_epochs": 5,
        # Scale jitter for the mosaic's own scale-and-crop step, in RandomScale
        # convention: [-0.5, 0.5] samples a factor in [0.5x, 1.5x]. The mosaic
        # crops a (H/r, W/r) window and resizes only that, so LSJ does NOT apply
        # to mosaic samples and this range replaces it for them. Factors below
        # 0.5 are clamped -- the window is already the whole canvas there.
        # `lsj.min_visibility` and `lsj.max_crop_trials` are shared by both
        # croppers and apply even when `lsj.enabled` is False.
        "scale_limit": [-0.5, 0.5],
    },

    # MixUp: applied ON TOP of the mosaic image.
    "mixup": {
        "enabled": False if model_format == "rtdetr" else True,
        "p":       0.15,
        "random_distribution_alpha": 8.0,
        "random_distribution_beta":  8.0,
    },

    # ── Standard Spatial Augmentations ────────────────────────────────────────
    "horizontal_flip": {
        "enabled": True,
        "p":       0.5,   # standard 50 % flip — consistent across all detectors
    },

    # ── Cutout / Random Erasing ────────────────────────────────────────────────
    # Forces the model to detect objects from partial features.
    # YOLOv5+: erases up to 8 patches of 32×32 px on a 640×640 input.
    "cutout": {
        "enabled":       True,
        "p":             0.5, # 0.5
        "max_holes":     8,
        "min_holes":     1,
        "max_hole_size": 8,   # px — used for both height and width
        "min_hole_size": 3,    # px
    },

    # ── Hard-Negative / Domain Augmentations ──────────────────────────────────
    # Applied as a OneOf block (only one fires per sample).
    # These simulate target environment conditions: rain, darkness, motion blur.
    "hard_negative": {
        "enabled": True,
        "p":       0.2,   # probability any sub-augment fires (low — these are edge cases)

        "random_rain": {
            "enabled":               True,
            "brightness_coefficient": 0.9,
            "drop_width":            1,
            "blur_value":            3,
        },

        "color_jitter_night": {
            "enabled":    True,
            "brightness": [0.2, 0.5],
            "contrast":   [0.5, 0.8],
        },

        "motion_blur": {
            "enabled":    True,
            "blur_limit": 7,   # kernel size in px (odd number or [min, max] range)
        },
    },
}
