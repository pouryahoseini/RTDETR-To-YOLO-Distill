import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def match_boxes_to_gt(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match predicted boxes to ground truth by mutual-best IoU within a class.

    Used by hard pseudo-label KD to decide which teacher detections correspond
    to an annotated object. A teacher box and a ground-truth box are matched
    only when each is the other's highest-IoU candidate of the same class and
    that IoU clears ``iou_threshold``. Mutual-best is used rather than a greedy
    sweep because it is order-independent, allocates every box at most once,
    and vectorises -- VisDrone images routinely carry hundreds of annotations
    against the teacher's 300 queries, so a Python matching loop would dominate
    the step time.

    Args:
        pred_boxes: Predicted boxes, normalised ``cxcywh``, shape (P, 4).
        pred_cls: Predicted class indices, shape (P,).
        gt_boxes: Ground-truth boxes, normalised ``cxcywh``, shape (G, 4).
        gt_cls: Ground-truth class indices, shape (G,).
        iou_threshold: Minimum IoU for a pair to be considered a match.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Index tensors ``(pred_idx, gt_idx)``
        of equal length, one entry per matched pair. Both are empty when either
        input is empty or nothing clears the threshold.
    """
    empty = torch.zeros(0, dtype=torch.long, device=pred_boxes.device)
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return empty, empty

    def to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(dim=1)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=1)

    pred_xyxy = to_xyxy(pred_boxes.float())
    gt_xyxy = to_xyxy(gt_boxes.float())

    # Pairwise IoU, (P, G).
    lt = torch.max(pred_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    rb = torch.min(pred_xyxy[:, None, 2:], gt_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    area_g = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=0) * (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=0)
    iou = inter / (area_p[:, None] + area_g[None, :] - inter).clamp(min=1e-9)

    # A cross-class pair is never a match. -1 keeps those cells below any real
    # IoU so they can never win an argmax, which 0.0 would not guarantee when a
    # row holds no valid candidate at all.
    same_class = pred_cls.reshape(-1, 1) == gt_cls.reshape(1, -1)
    iou = iou.masked_fill(~same_class, -1.0)

    best_gt = iou.argmax(dim=1)                       # (P,)
    best_pred = iou.argmax(dim=0)                     # (G,)
    pred_idx = torch.arange(iou.shape[0], device=iou.device)
    mutual = best_pred[best_gt] == pred_idx
    good = iou[pred_idx, best_gt] >= iou_threshold
    keep = mutual & good

    return pred_idx[keep], best_gt[keep]


def boxes_to_spatial_mask(
    gt_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Rasterize normalized ``cxcywh`` boxes using half-open pixel bounds.

    A box's lower edge is rounded down and its upper edge is rounded up. The
    resulting slice is ``[y1:y2, x1:x2]``; adding one to an already-integral
    upper edge would incorrectly include an extra row and column.
    """
    spatial_mask = torch.zeros(
        (batch_size, 1, height, width), device=gt_bboxes.device, dtype=torch.float32
    )

    for image_idx in range(batch_size):
        if not bool(valid_mask[image_idx]):
            continue

        boxes = gt_bboxes[batch_idx == image_idx]
        if boxes.numel() == 0:
            continue

        cx, cy, box_width, box_height = boxes.unbind(dim=1)
        x1 = torch.floor((cx - box_width / 2) * width).clamp(0, width).long()
        y1 = torch.floor((cy - box_height / 2) * height).clamp(0, height).long()
        x2 = torch.ceil((cx + box_width / 2) * width).clamp(0, width).long()
        y2 = torch.ceil((cy + box_height / 2) * height).clamp(0, height).long()

        for left, top, right, bottom in zip(x1, y1, x2, y2):
            if right > left and bottom > top:
                spatial_mask[image_idx, 0, top:bottom, left:right] = 1.0

    return spatial_mask


def boxes_to_instance_masks(
    gt_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rasterize boxes into a per-instance *scale* mask and a binary foreground mask.

    A plain binary union of the ground-truth boxes makes every foreground pixel
    count the same, so an object's influence on the loss is proportional to its
    area. On VisDrone that is exactly backwards: a bus covers hundreds of
    feature pixels while a pedestrian at P5 covers one, so the large and
    frequent classes dominate a dataset whose whole difficulty is small
    objects. The scale mask instead carries ``1 / area`` inside each box, so
    summing it over an object's pixels yields 1 regardless of the object's
    size -- FGD's "scale mask" -- and every instance contributes equally.

    Where boxes overlap their weights are added. This preserves one unit of
    total mask mass per instance even in crowded scenes: using the maximum at
    an overlap while still dividing the loss by ``n_instances`` silently
    underweighted overlapping objects (two identical boxes produced one unit
    of mask mass but a denominator of two).

    Args:
        gt_bboxes: Normalized ``cxcywh`` boxes, shape (N, 4).
        batch_idx: Image index per box, shape (N,).
        batch_size: Number of images in the batch.
        height: Feature-map height at this scale.
        width: Feature-map width at this scale.
        valid_mask: Per-image bool tensor; False images contribute nothing.

    Returns:
        tuple: ``(scale_mask, fg_mask, n_instances)`` where both masks are
        ``(B, 1, H, W)`` float tensors and ``n_instances`` counts the boxes
        that occupied at least one pixel on a valid image.
    """
    device = gt_bboxes.device
    scale_mask = torch.zeros(
        (batch_size, 1, height, width), device=device, dtype=torch.float32
    )
    fg_mask = torch.zeros_like(scale_mask)
    n_instances = 0

    rows = torch.arange(height, device=device).view(1, -1)
    cols = torch.arange(width, device=device).view(1, -1)
    # Rasterizing box-by-box in Python costs ~0.4 s per step at VisDrone's crowd
    # density (measured, 800 boxes over three scales), which dwarfs the loss
    # itself. Boxes are painted as one broadcast instead, chunked so peak memory
    # stays bounded no matter how crowded the frame is.
    chunk_size = 256

    for image_idx in range(batch_size):
        if not bool(valid_mask[image_idx]):
            continue

        boxes = gt_bboxes[batch_idx == image_idx]
        if boxes.numel() == 0:
            continue

        cx, cy, box_width, box_height = boxes.unbind(dim=1)
        x1 = torch.floor((cx - box_width / 2) * width).clamp(0, width).long()
        y1 = torch.floor((cy - box_height / 2) * height).clamp(0, height).long()
        x2 = torch.ceil((cx + box_width / 2) * width).clamp(0, width).long()
        y2 = torch.ceil((cy + box_height / 2) * height).clamp(0, height).long()

        # Degenerate boxes cover no pixel and must not count as instances.
        keep = (x2 > x1) & (y2 > y1)
        if not bool(keep.any()):
            continue
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        n_instances += int(keep.sum())

        weights = 1.0 / ((x2 - x1) * (y2 - y1)).float()

        for start in range(0, x1.shape[0], chunk_size):
            stop = start + chunk_size
            cx1, cy1, cx2, cy2 = x1[start:stop], y1[start:stop], x2[start:stop], y2[start:stop]

            # Half-open bounds, matching `boxes_to_spatial_mask`.
            inside_rows = (rows >= cy1.view(-1, 1)) & (rows < cy2.view(-1, 1))  # (n, H)
            inside_cols = (cols >= cx1.view(-1, 1)) & (cols < cx2.view(-1, 1))  # (n, W)
            covered = inside_rows.unsqueeze(2) & inside_cols.unsqueeze(1)       # (n, H, W)

            chunk_weights = covered * weights[start:stop].view(-1, 1, 1)
            # Addition preserves one unit of total mask mass per instance.
            scale_mask[image_idx, 0] += chunk_weights.sum(dim=0)
            fg_mask[image_idx, 0] = torch.maximum(
                fg_mask[image_idx, 0], covered.any(dim=0).float()
            )

    return scale_mask, fg_mask, n_instances


def spatial_channel_attention(
    features: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FGD-style spatial and channel attention maps for one feature map.

    Both maps are softmax-normalized and rescaled to mean 1.0, so using them as
    loss weights redistributes emphasis without changing the loss's overall
    magnitude. The maps are always taken from the *raw* features: magnitude is
    what tells you which locations and channels the network considers
    important, and it is discarded by the direction-only error metric the
    weights are applied to.

    Args:
        features: Feature map, shape (B, C, H, W).
        temperature: Softmax temperature. Lower values sharpen the maps.

    Returns:
        tuple: ``(spatial, channel)`` of shapes (B, 1, H, W) and (B, C, 1, 1).
    """
    b, c, h, w = features.shape
    magnitude = features.abs()

    spatial = magnitude.mean(dim=1).view(b, -1)             # (B, H*W)
    spatial = (spatial / temperature).softmax(dim=1) * (h * w)
    spatial = spatial.view(b, 1, h, w)

    channel = magnitude.mean(dim=(2, 3))                    # (B, C)
    channel = (channel / temperature).softmax(dim=1) * c
    channel = channel.view(b, c, 1, 1)

    return spatial, channel


class GlobalContextBlock(nn.Module):
    """Global context block used to compare student and teacher pixel relations.

    The focal term is strictly local: it compares each pixel to the teacher's
    pixel at the same location. What a large teacher mostly knows that a nano
    student does not is *relational* -- how a rooftop's context makes the
    object on it a car rather than a van. This block pools the whole map into
    one context vector via a learned attention, transforms it, and adds it
    back, so matching its output between the two networks matches how each
    aggregates global context (FGD's global loss).

    Args:
        channels (int): Feature channels in and out.
        ratio (float): Bottleneck ratio for the transform.
    """

    def __init__(self, channels: int, ratio: float = 0.5):
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.context_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        mask = self.context_mask(features).view(b, 1, h * w).softmax(dim=-1)
        context = torch.bmm(features.view(b, c, h * w), mask.transpose(1, 2))
        return features + self.transform(context.view(b, c, 1, 1))


def _build_adapter(
    student_channels: int, teacher_channels: int, adapter: str, adapter_norm: str
) -> nn.Module:
    """Build one projection adapter.

    A deep adapter is a liability rather than a feature: with enough capacity
    it can satisfy the alignment objective internally, leaving the student's
    own features untouched, so the loss falls without anything being taught.
    The default is therefore the smallest projection that can change channel
    count at all.

    BatchNorm is avoided by default because KD runs at a physical batch of 4,
    where its statistics are too noisy to normalize with.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
    ]

    if adapter == "dwsep":
        # Deeper alternative, retained for ablation against the 1x1 default.
        layers += [
            _build_norm(teacher_channels, adapter_norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                teacher_channels, teacher_channels, kernel_size=3,
                padding=1, groups=teacher_channels, bias=False,
            ),
            _build_norm(teacher_channels, adapter_norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1, bias=False),
        ]
    elif adapter != "conv1x1":
        raise ValueError(f"Unknown adapter {adapter!r}; expected 'conv1x1' or 'dwsep'.")

    layers.append(_build_norm(teacher_channels, adapter_norm))
    return nn.Sequential(*[m for m in layers if not isinstance(m, nn.Identity)])


def _build_norm(channels: int, norm: str) -> nn.Module:
    """Normalization layer for an adapter, tolerant of odd channel counts."""
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        groups = math.gcd(32, channels)
        return nn.GroupNorm(max(1, groups), channels)
    raise ValueError(f"Unknown adapter_norm {norm!r}; expected 'group', 'batch' or 'none'.")


class FeatureKD(nn.Module):
    """Focal-and-global feature distillation between a detector's neck and a teacher's.

    Three design choices distinguish it from a plain foreground-masked cosine
    loss, each addressing a way that simpler objective fails on this data:

    1. Foreground masks are per-instance (`boxes_to_instance_masks`), not a
       binary union of boxes. A union mask scales an object's weight with its
       pixel area -- backwards for VisDrone, where the objects that matter are
       the small ones. Every instance gets equal say regardless of size, and
       `scale_weights` can emphasise the small-object level (P3) further.
    2. Foreground and background are separate terms, joined by a global
       relational term (`GlobalContextBlock`). Distilling foreground alone
       discards the background context that is most of what a large teacher
       knows and a nano student does not.
    3. The adapter defaults to a single 1x1 projection with GroupNorm. A deeper
       adapter has enough capacity to satisfy the objective on its own without
       teaching the student anything, and BatchNorm is unusable at the physical
       batch of 4 that KD runs at.

    The per-pixel error stays direction-based: both feature maps are
    L2-normalised along the channel axis before differencing, so the loss never
    asks a CNN neck to reproduce a transformer encoder's activation magnitudes.
    Magnitude is not thrown away, though -- it is what the attention weights are
    computed from.

    Args:
        student_channels: Channel dims of the student's multi-scale features.
        teacher_channels: Channel dims of the teacher's features, same length.
        scale_weights: Per-scale gains, ordered like the feature lists. None
            weights every scale equally. Weights are normalised, so they set
            relative emphasis only.
        adapter: ``"conv1x1"`` (default) or ``"dwsep"`` for the older adapter.
        adapter_norm: ``"group"`` (default), ``"batch"`` or ``"none"``.
        attention_temperature: Softmax temperature for the attention maps.
        fg_weight: Gain on the foreground feature term.
        bg_weight: Gain on the background feature term.
        attention_weight: Gain on the attention-transfer term.
        global_weight: Gain on the global relation term. 0 disables it and
            builds no context blocks.
    """

    def __init__(
        self,
        student_channels: list[int],
        teacher_channels: list[int],
        scale_weights: list[float] | None = None,
        adapter: str = "conv1x1",
        adapter_norm: str = "group",
        attention_temperature: float = 0.5,
        fg_weight: float = 1.0,
        bg_weight: float = 0.5,
        attention_weight: float = 1.0,
        global_weight: float = 0.5,
    ):
        super().__init__()
        if not student_channels:
            raise ValueError("student_channels must not be empty")
        if len(student_channels) != len(teacher_channels):
            raise ValueError(
                "Channel list lengths must match. "
                f"Got {len(student_channels)} and {len(teacher_channels)}."
            )
        if scale_weights is None:
            scale_weights = [1.0] * len(student_channels)
        if len(scale_weights) != len(student_channels):
            raise ValueError(
                f"scale_weights must have one entry per scale ({len(student_channels)}); "
                f"got {len(scale_weights)}."
            )
        if any(w < 0 for w in scale_weights):
            raise ValueError("scale_weights must be non-negative.")
        if sum(scale_weights) <= 0:
            raise ValueError("scale_weights must not sum to zero.")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")

        self.adapters = nn.ModuleList([
            _build_adapter(sc, tc, adapter, adapter_norm)
            for sc, tc in zip(student_channels, teacher_channels)
        ])
        self.scale_weights = list(scale_weights)
        self.attention_temperature = attention_temperature
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.attention_weight = attention_weight
        self.global_weight = global_weight

        # Per-term, per-scale breakdown of the last forward, as a detached
        # (n_scales, 4) tensor in `term_names` order. Each entry already carries
        # its own gain, its scale gain and the `weight_sum` normalisation, so the
        # grand total equals the scalar `forward` returned: column sums give the
        # split between the four objectives, row sums the split between P3/P4/P5.
        # Without this only the sum is observable, and a run cannot be diagnosed
        # -- four very different balances produce the same scalar.
        self.term_names = ("fg", "bg", "attn", "global")
        self.last_terms: torch.Tensor | None = None

        # One block per scale, shared between student and teacher so the
        # comparison is of the features, not of two different poolers.
        self.context_blocks = (
            nn.ModuleList([GlobalContextBlock(tc) for tc in teacher_channels])
            if global_weight > 0 else None
        )

    def forward(
        self,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor],
        gt_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        disable_kd_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the focal-and-global distillation loss across every scale.

        Args:
            student_feats: Student neck outputs, one tensor per scale (P3..P5).
            teacher_feats: Teacher encoder outputs, matching scales and shapes
                after projection.
            gt_bboxes: Ground truth in normalized ``cxcywh``, shape (N, 4).
            batch_idx: Image index per box, shape (N,).
            disable_kd_mask: Bool tensor (B); True excludes that image.

        Returns:
            torch.Tensor: Scalar loss, a `scale_weights`-weighted mean over the
            scales that had foreground to distil.
        """
        if len(student_feats) != len(teacher_feats) or len(student_feats) != len(self.adapters):
            raise ValueError("Number of feature scales must match the number of adapters.")
        if not student_feats:
            raise ValueError("At least one feature scale is required.")

        batch = student_feats[0].shape[0]
        device = student_feats[0].device

        if disable_kd_mask is None:
            valid_mask = torch.ones(batch, dtype=torch.bool, device=device)
        else:
            valid_mask = ~disable_kd_mask.to(device=device, dtype=torch.bool).reshape(-1)
            if valid_mask.numel() != batch:
                raise ValueError(
                    f"disable_kd_mask must contain one value per image ({batch}), "
                    f"got {valid_mask.numel()}."
                )
        n_scales = len(self.adapters)
        if not valid_mask.any():
            self.last_terms = torch.zeros(n_scales, 4, device=device)
            return student_feats[0].sum() * 0.0

        image_weight = valid_mask.float().view(batch, 1, 1, 1)
        total = None
        weight_sum = 0.0
        # Scales skipped for want of foreground keep their zero row, so the
        # logged breakdown stays shape-stable across steps.
        per_scale_terms = [torch.zeros(4, device=device) for _ in range(n_scales)]

        for i, (s_feat, t_feat, adapter) in enumerate(zip(student_feats, teacher_feats, self.adapters)):
            s_proj = adapter(s_feat)
            if s_proj.shape != t_feat.shape:
                raise ValueError(
                    f"Feature scale {i} shape mismatch after projection: "
                    f"Student projected shape {s_proj.shape} != Teacher shape {t_feat.shape}."
                )

            _, channels, height, width = s_proj.shape
            scale_mask, fg_mask, n_instances = boxes_to_instance_masks(
                gt_bboxes.to(device), batch_idx.to(device), batch, height, width, valid_mask
            )
            if n_instances == 0:
                continue

            # fp32 for the whole comparison: the normalisation and the softmaxes
            # below are both sensitive to bf16's mantissa.
            s_proj = s_proj.float()
            t_feat = t_feat.float()

            # Attention comes from the teacher's raw magnitudes, before the
            # direction-only normalisation discards them.
            t_spatial, t_channel = spatial_channel_attention(t_feat, self.attention_temperature)
            s_spatial, s_channel = spatial_channel_attention(s_proj, self.attention_temperature)

            s_norm = F.normalize(s_proj, p=2, dim=1)
            t_norm = F.normalize(t_feat, p=2, dim=1)
            # Per-channel squared error; summed over channels this is
            # 2 * (1 - cosine), so it is bounded and architecture-agnostic.
            sq_err = (s_norm - t_norm) ** 2
            weighted_err = sq_err * t_spatial * t_channel * image_weight

            # Every term below is normalized to the same unit -- the per-pixel
            # error *summed over channels*, which is 2 * (1 - cosine) and so
            # sits near 1 whatever the feature width. Mixing units here is easy
            # and silently fatal: a term left summing over H*W drowns the rest
            # and the configured gains stop meaning anything.
            #
            # Foreground: `scale_mask` sums to 1 per object, so dividing by the
            # instance count makes this a mean over objects rather than pixels.
            fg_loss = (weighted_err * scale_mask).sum() / n_instances

            # Background: the same quantity, averaged over background pixels.
            bg_mask = (1.0 - fg_mask) * image_weight
            bg_loss = (weighted_err * bg_mask).sum() / bg_mask.sum().clamp(min=1.0)

            valid_images = valid_mask.sum().clamp(min=1)
            attn_loss = (
                ((s_spatial - t_spatial).abs() * image_weight).sum()
                / (valid_images * height * width)
                + ((s_channel - t_channel).abs() * image_weight).sum()
                / (valid_images * channels)
            )

            if self.context_blocks is not None:
                block = self.context_blocks[i]
                # Applied to the normalised features so this term, like the
                # others, cannot be dominated by a magnitude mismatch, and
                # averaged over pixels so it stays in the same unit as the
                # foreground and background terms.
                global_diff = (block(s_norm) - block(t_norm)) ** 2
                global_loss = (global_diff * image_weight).sum() / (
                    valid_images * height * width
                )
            else:
                global_loss = torch.zeros((), device=device)

            terms = torch.stack((
                self.fg_weight * fg_loss,
                self.bg_weight * bg_loss,
                self.attention_weight * attn_loss,
                self.global_weight * global_loss,
            ))
            scale_loss = terms.sum()

            gain = self.scale_weights[i]
            contribution = scale_loss * gain
            per_scale_terms[i] = terms.detach() * gain
            total = contribution if total is None else total + contribution
            weight_sum += gain

        if total is None or weight_sum == 0.0:
            # No scale had foreground to distil. Return a graph-connected zero
            # so the caller can add it unconditionally.
            self.last_terms = torch.zeros(n_scales, 4, device=device)
            return student_feats[0].sum() * 0.0

        # Normalising by the contributing weight keeps the magnitude independent
        # of how many scales happened to hold a ground-truth box this batch.
        self.last_terms = torch.stack(per_scale_terms) / weight_sum
        return total / weight_sum
