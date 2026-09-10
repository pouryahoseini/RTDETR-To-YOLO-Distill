"""Small training helpers that do not depend on Ultralytics internals."""

from collections.abc import Iterable

import torch


def resolve_ultralytics_train_imgsz(input_height: int, input_width: int) -> int:
    """Return a valid automatic-training size or reject a rectangular contract.

    Ultralytics' trainer accepts one training dimension and silently replaces a
    two-dimensional ``imgsz`` with its maximum. Failing here prevents a configured
    rectangular experiment from unexpectedly becoming square.
    """
    height, width = int(input_height), int(input_width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Input dimensions must be positive, got {height}x{width}.")
    if height != width:
        raise ValueError(
            "Automatic Ultralytics training only supports a square `imgsz`, but "
            f"the configured input is {height}x{width}. Use model_format='yolo_manual' "
            "for fixed rectangular training, or configure equal height and width."
        )
    return height


@torch.no_grad()
def normalize_accumulated_gradients(
    parameters: Iterable[torch.nn.Parameter],
    target_batches: int,
    valid_batches: int,
) -> float:
    """Renormalize a partial accumulation window to the mean of valid batches."""
    target_batches = int(target_batches)
    valid_batches = int(valid_batches)
    if target_batches <= 0 or valid_batches <= 0:
        raise ValueError("target_batches and valid_batches must both be positive.")
    scale = target_batches / valid_batches
    if scale != 1.0:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return scale


def step_optimizer_with_scaler(scaler, optimizer) -> bool:
    """Step an AMP optimizer and report whether parameters were updated.

    GradScaler.step silently skips optimizer.step when unscaled gradients
    contain Inf/NaN values. State that advances per optimizer update (EMA,
    iteration-based warmup, and similar counters) must therefore only be
    advanced when the scale did not back off.

    Returns:
        bool: True when the optimizer step ran, otherwise False.
    """
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return not scaler.is_enabled() or float(scaler.get_scale()) >= scale_before

