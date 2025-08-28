from __future__ import annotations
import random
from typing import Iterator, List, Dict, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import Sampler

from .controller import FeedbackController

class FeedbackBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that uses FeedbackController to sample class-balanced batches adaptively.
    It yields a list of indices for each batch. After training that batch,
    call controller.step_update(labels, losses) to update statistics.

    Args:
        labels: 1D array-like of dataset labels in [0, K-1]
        batch_size: number of samples per batch
        steps_per_epoch: how many batches to yield per epoch (len = steps_per_epoch)
        controller: FeedbackController instance controlling class probabilities
        replacement: sample with replacement within each class pool
    """
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        steps_per_epoch: int,
        controller: FeedbackController,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        assert self.labels.ndim == 1
        self.N = self.labels.size
        self.K = controller.K
        assert self.labels.min() >= 0 and self.labels.max() < self.K, "labels out of range"
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.controller = controller
        self.replacement = bool(replacement)
        self.rng = generator if generator is not None else torch.Generator()

        # map class -> list of indices
        self.per_class_indices: Dict[int, List[int]] = {k: [] for k in range(self.K)}
        for idx, c in enumerate(self.labels):
            self.per_class_indices[int(c)].append(int(idx))

        # fallback if some classes empty
        for k in range(self.K):
            if len(self.per_class_indices[k]) == 0:
                raise ValueError(f"class {k} has no samples.")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            p = self.controller.get_probs()
            # sample classes for the batch according to p
            # draw class counts via multinomial
            counts = np.random.multinomial(self.batch_size, p)
            batch_indices: List[int] = []
            for cls, cnt in enumerate(counts):
                if cnt <= 0:
                    continue
                pool = self.per_class_indices[cls]
                if self.replacement:
                    chosen = np.random.choice(pool, size=cnt, replace=True)
                else:
                    if cnt > len(pool):
                        # if not enough samples, fallback to replacement for the remainder
                        chosen = np.random.choice(pool, size=cnt, replace=True)
                    else:
                        chosen = np.random.choice(pool, size=cnt, replace=False)
                batch_indices.extend(int(i) for i in chosen.tolist())
            # shuffle within batch
            random.shuffle(batch_indices)
            yield batch_indices
