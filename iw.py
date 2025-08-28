from __future__ import annotations
import numpy as np
import torch
from typing import Sequence

def importance_weights(labels: torch.Tensor, target_probs: Sequence[float], current_probs: torch.Tensor, max_w: float | None = None) -> torch.Tensor:
    """
    Args:
        labels: tensor of class indices
        target_probs: target q (sequence length K)
        current_probs: current sampling probs p (tensor length K)
        max_w: optional clip value for importance weights (useful for stability)
    Returns:
        weights normalized to mean 1 (after optional clipping)
    """
    Compute importance weights w_i = q_i / p_i for each label in the batch.
    Returns weights normalized to mean 1.
    """
    Args:
        labels: tensor of class indices
        target_probs: target q (sequence length K)
        current_probs: current sampling probs p (tensor length K)
        max_w: optional clip value for importance weights (useful for stability)
    Returns:
        weights normalized to mean 1 (after optional clipping)
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.long)
    q = torch.as_tensor(target_probs, dtype=torch.float32, device=labels.device)
    p = current_probs.to(labels.device).float()
    w = q[labels] / p[labels]
    if max_w is not None:
        w = torch.clamp(w, max=float(max_w))
    return w / (w.mean() + 1e-12)
