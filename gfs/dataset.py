from __future__ import annotations
import torch
from torch.utils.data import Dataset, Subset
from typing import Sequence


class AugmentedDataset(Dataset):
    """Wraps a Subset and applies Gaussian noise augmentation to tail classes.

    A class is considered a tail class if its sample count is below ``threshold``.
    Tail samples receive additive Gaussian noise (std=0.1) clamped to [0, 1].
    """

    def __init__(self, subset: Subset, class_counts: Sequence[int], threshold: int = 100):
        self.subset = subset
        self.is_tail = torch.tensor([
            class_counts[subset.dataset.targets[i].item()] < threshold
            for i in subset.indices
        ])

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        if self.is_tail[idx]:
            x = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
        return x, y
