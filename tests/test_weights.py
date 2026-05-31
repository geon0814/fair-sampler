import torch
from gfs import importance_weights


def test_output_shape():
    labels = torch.tensor([0, 1, 2, 1, 0])
    q = torch.tensor([0.2, 0.3, 0.5])
    p = torch.tensor([0.4, 0.3, 0.3])
    w = importance_weights(labels, q, p)
    assert w.shape == labels.shape


def test_mean_approximately_one():
    labels = torch.tensor([0, 1, 2, 1, 0])
    q = torch.ones(3) / 3
    p = torch.tensor([0.5, 0.3, 0.2])
    w = importance_weights(labels, q, p)
    assert abs(w.mean().item() - 1.0) < 1e-4


def test_clamping_applied():
    labels = torch.tensor([2])
    q = torch.tensor([0.33, 0.33, 0.34])
    # class 2의 샘플링 확률이 매우 낮음 → weight가 크게 올라가지만 clamp로 막힘
    p = torch.tensor([0.99, 0.005, 0.005])
    w = importance_weights(labels, q, p, max_w=5.0)
    assert w.max().item() <= 5.0


def test_uniform_q_and_p_gives_ones():
    K = 4
    labels = torch.tensor([0, 1, 2, 3])
    q = torch.ones(K) / K
    p = torch.ones(K) / K
    w = importance_weights(labels, q, p)
    assert torch.allclose(w, torch.ones(K), atol=1e-5)


def test_no_division_by_zero():
    # p에 0이 있어도 clamp(min=1e-8) 덕분에 안 터져야 함
    labels = torch.tensor([0, 1])
    q = torch.tensor([0.5, 0.5])
    p = torch.tensor([1.0, 0.0])
    w = importance_weights(labels, q, p)
    assert torch.all(torch.isfinite(w))
