from __future__ import annotations
import torch
from torch.utils.data import Sampler
from typing import Iterator, List

from .controller import SimpleFairController


class FairSampler(Sampler):
    """Batch sampler that draws samples according to controller class probabilities."""

    def __init__(
        self,
        labels: torch.Tensor,
        controller: SimpleFairController,
        batch_size: int,
        steps_per_epoch: int,
    ):
        self.labels = labels
        self.controller = controller
        self.batch_size = batch_size
        self.steps = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps):
            class_probs = self.controller.get_class_probs()
            sample_weights = class_probs[self.labels]
            sample_weights = sample_weights / sample_weights.sum()
            indices = torch.multinomial(sample_weights, self.batch_size, replacement=True)
            yield indices.tolist()

    def __len__(self) -> int:
        return self.steps
