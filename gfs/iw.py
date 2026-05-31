from __future__ import annotations
import torch


def importance_weights(
    labels: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    max_w: float = 50.0,
) -> torch.Tensor:
    """Compute per-sample importance weights w = q[y] / p[y], normalized to mean 1.

    Args:
        labels: per-sample class indices, shape (B,)
        q: target distribution, shape (K,)
        p: current sampling distribution, shape (K,)
        max_w: upper clamp value for weight explosion prevention

    Returns:
        weights: shape (B,), mean ~1
    """
    w = q[labels] / p[labels].clamp(min=1e-8)
    w = w / w.mean()
    return w.clamp(max=max_w)
