import numpy as np
from gfs.controller import FeedbackController

def test_probs_sum_to_one():
    c = FeedbackController(num_classes=3, target_probs=[0.2,0.5,0.3], alpha=5.0)
    p = c.get_probs()
    assert abs(p.sum() - 1.0) < 1e-8

def test_updates_change_probs():
    c = FeedbackController(num_classes=2, target_probs=[0.5,0.5], alpha=8.0)
    # make class 0 over-sampled
    labels = np.array([0]*100 + [1]*10)
    c.step_update(labels=labels, losses=None)
    p = c.get_probs()
    # since class 1 is under-sampled, its prob should be > 0.5
    assert p[1] > 0.5
