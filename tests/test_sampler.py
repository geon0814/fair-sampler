import torch
from gfs import SimpleFairController, FairSampler


def test_batch_count(num_classes, labels):
    ctrl = SimpleFairController(num_classes)
    steps = 10
    sampler = FairSampler(labels, ctrl, batch_size=32, steps_per_epoch=steps)
    assert len(list(sampler)) == steps


def test_batch_size(num_classes, labels):
    ctrl = SimpleFairController(num_classes)
    sampler = FairSampler(labels, ctrl, batch_size=32, steps_per_epoch=5)
    for batch in sampler:
        assert len(batch) == 32


def test_indices_in_range(num_classes, labels):
    ctrl = SimpleFairController(num_classes)
    sampler = FairSampler(labels, ctrl, batch_size=16, steps_per_epoch=5)
    n = len(labels)
    for batch in sampler:
        assert all(0 <= idx < n for idx in batch)


def test_len(num_classes, labels):
    ctrl = SimpleFairController(num_classes)
    steps = 7
    sampler = FairSampler(labels, ctrl, batch_size=32, steps_per_epoch=steps)
    assert len(sampler) == steps


def test_generator_reproducibility(num_classes, labels):
    ctrl = SimpleFairController(num_classes)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s1 = FairSampler(labels, ctrl, batch_size=16, steps_per_epoch=3, generator=g1)
    s2 = FairSampler(labels, ctrl, batch_size=16, steps_per_epoch=3, generator=g2)
    for b1, b2 in zip(s1, s2):
        assert b1 == b2
