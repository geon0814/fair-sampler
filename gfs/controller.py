from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional, Sequence


class SimpleFairController:
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
        if class_counts is not None:
            sqrt_freq = torch.tensor(class_counts, dtype=torch.float).sqrt()
            self.target = sqrt_freq / sqrt_freq.sum()
        else:
            self.target = torch.ones(num_classes) / num_classes

    def get_class_probs(self) -> torch.Tensor:
        probs = F.softmax(self.alpha * self.scores, dim=0)
        probs = probs + 0.02
        probs = probs / probs.sum()
        return probs

    def step(self, labels: torch.Tensor) -> None:
        counts = torch.bincount(labels, minlength=self.num_classes).float()
        usage = counts / counts.sum()
        error = self.target - usage
        self.scores += self.lr * error
        self.scores -= self.scores.mean()
        self.scores = self.scores.clamp(-1.0, 1.0)

    def log(self, epoch: int, step: int, loss: float) -> None:
        p = self.get_class_probs()
        p_str = [f"{x:.2f}" for x in p.tolist()]
        print(f"[Ep {epoch+1} | Step {step+1}] loss={loss:.4f} probs={p_str}")
