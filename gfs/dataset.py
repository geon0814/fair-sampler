from __future__ import annotations
import torch
from torch.utils.data import Dataset, Subset
from typing import Optional, Sequence


class AugmentedDataset(Dataset):
    """Wraps a Subset and applies Gaussian noise augmentation to tail classes.

    A class is a tail class if its sample count is below ``threshold``.
    When ``threshold`` is None it defaults to ``max(class_counts) // 10``.
    Tail samples receive additive Gaussian noise clamped to [0, 1].
    """

    def __init__(
        self,
        subset: Subset,
        class_counts: Sequence[int],
        threshold: Optional[int] = None,
        noise_std: float = 0.1,
    ):
        self.subset = subset
        self.noise_std = noise_std
        if threshold is None:
            threshold = max(class_counts) // 10
        self.threshold = threshold
        self.is_tail = torch.tensor([
            class_counts[subset.dataset.targets[i].item()] < threshold
            for i in subset.indices
        ])

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        if self.is_tail[idx]:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0.0, 1.0)
        return x, y
