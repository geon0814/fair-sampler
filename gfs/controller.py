from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional, Sequence


class SimpleFairController:
    """Feedback-controlled class probability manager for imbalanced training.

    Tracks the gap between a target distribution and the observed per-class
    usage in recent batches, then adjusts sampling probabilities to close
    the deficit over time.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 30.0,
        lr: float = 1.0,
        class_counts: Optional[Sequence[int]] = None,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.lr = lr
        self.scores = torch.zeros(num_classes)
        self._step_count: int = 0
        if class_counts is not None:
            sqrt_freq = torch.tensor(class_counts, dtype=torch.float).sqrt()
            self.target = sqrt_freq / sqrt_freq.sum()
        else:
            self.target = torch.ones(num_classes) / num_classes

    def get_class_probs(self) -> torch.Tensor:
        """Return current per-class sampling probabilities (sums to 1)."""
        probs = F.softmax(self.alpha * self.scores, dim=0)
        probs = probs + 0.02
        return probs / probs.sum()

    def step(self, labels: torch.Tensor) -> None:
        """Update scores based on observed class distribution in a batch."""
        counts = torch.bincount(labels, minlength=self.num_classes).float()
        usage = counts / counts.sum()
        error = self.target - usage
        self.scores = self.scores + self.lr * error
        self.scores = self.scores - self.scores.mean()
        self.scores = self.scores.clamp(-1.0, 1.0)
        self._step_count += 1

    def reset(self) -> None:
        """Reset scores to zero while keeping the target distribution."""
        self.scores = torch.zeros(self.num_classes)
        self._step_count = 0

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "scores": self.scores.clone(),
            "target": self.target.clone(),
            "step_count": self._step_count,
            "alpha": self.alpha,
            "lr": self.lr,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint dict."""
        self.scores = state["scores"].clone()
        self.target = state["target"].clone()
        self._step_count = state["step_count"]
        self.alpha = state["alpha"]
        self.lr = state["lr"]

    def log(self, epoch: int, step: int, loss: float) -> None:
        p = self.get_class_probs()
        p_str = [f"{x:.2f}" for x in p.tolist()]
        print(
            f"[Ep {epoch + 1} | Step {step + 1}] loss={loss:.4f} "
            f"probs={p_str}  min={p.min():.3f} max={p.max():.3f}"
        )
