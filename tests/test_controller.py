import torch
from gfs import SimpleFairController


def test_init_uniform(num_classes):
    ctrl = SimpleFairController(num_classes)
    assert ctrl.target.shape == (num_classes,)
    assert torch.allclose(ctrl.target, torch.ones(num_classes) / num_classes)
    assert torch.all(ctrl.scores == 0)


def test_init_with_class_counts(num_classes, class_counts):
    ctrl = SimpleFairController(num_classes, class_counts=class_counts)
    assert ctrl.target.shape == (num_classes,)
    assert abs(ctrl.target.sum().item() - 1.0) < 1e-5
    # sqrt-freq: 큰 클래스일수록 target 확률도 큼
    assert ctrl.target[0] > ctrl.target[-1]


def test_get_class_probs_sums_to_one(num_classes):
    ctrl = SimpleFairController(num_classes)
    p = ctrl.get_class_probs()
    assert abs(p.sum().item() - 1.0) < 1e-5


def test_get_class_probs_all_positive(num_classes):
    ctrl = SimpleFairController(num_classes)
    p = ctrl.get_class_probs()
    assert torch.all(p > 0)


def test_step_updates_scores(num_classes):
    ctrl = SimpleFairController(num_classes)
    labels = torch.tensor([0, 0, 0, 1, 2])
    ctrl.step(labels)
    assert not torch.all(ctrl.scores == 0)
    assert ctrl._step_count == 1


def test_step_scores_clamped(num_classes):
    # lr 크게 줘도 clamp(-1, 1) 보장
    ctrl = SimpleFairController(num_classes, lr=100.0)
    for _ in range(20):
        ctrl.step(torch.zeros(50, dtype=torch.long))
    assert ctrl.scores.max() <= 1.0
    assert ctrl.scores.min() >= -1.0


def test_reset_keeps_target(num_classes, class_counts):
    ctrl = SimpleFairController(num_classes, class_counts=class_counts)
    original_target = ctrl.target.clone()
    ctrl.step(torch.zeros(10, dtype=torch.long))
    ctrl.reset()
    assert torch.all(ctrl.scores == 0)
    assert ctrl._step_count == 0
    assert torch.allclose(ctrl.target, original_target)


def test_state_dict_roundtrip(num_classes, class_counts):
    ctrl = SimpleFairController(num_classes, class_counts=class_counts)
    ctrl.step(torch.tensor([0, 1, 2, 3, 4]))
    state = ctrl.state_dict()

    ctrl2 = SimpleFairController(num_classes)
    ctrl2.load_state_dict(state)

    assert torch.allclose(ctrl.scores, ctrl2.scores)
    assert torch.allclose(ctrl.target, ctrl2.target)
    assert ctrl._step_count == ctrl2._step_count
    assert ctrl.alpha == ctrl2.alpha
    assert ctrl.lr == ctrl2.lr
