from __future__ import annotations
import math
from typing import Optional, Sequence, Dict, List
import numpy as np

class FeedbackController:
    """
    Geon's Feedback Controller.
    Maintains per-class counts and EWMA losses; outputs per-class sampling probabilities.
    p_i(t) = softmax(alpha * (lambda1 * delta_i(t) + lambda2 * ewma_loss_i(t)))
    where delta_i(t) = q_i * t - n_i(t)  (deficit vs target).

    Args:
        num_classes: K
        target_probs: length-K target distribution q (defaults to uniform)
        alpha: aggressiveness (temperature for softmax)
        lambda1: weight for deficit term
        lambda2: weight for EWMA loss term
        ewma_beta: smoothing for EWMA (0..1), larger = smoother
    """
    def __init__(
        self,
        num_classes: int,
        target_probs: Optional[Sequence[float]] = None,
        alpha: float = 4.0,
        lambda1: float = 1.0,
        lambda2: float = 0.0,
        ewma_beta: float = 0.9,
        min_prob: float = 1e-6,
    ):
        assert num_classes >= 2
        self.K = num_classes
        if target_probs is None:
            self.q = np.ones(self.K, dtype=np.float64) / self.K
        else:
            q = np.asarray(target_probs, dtype=np.float64)
            assert q.shape == (self.K,)
            assert np.all(q >= 0)
            s = q.sum()
            assert s > 0, "target_probs sum must be > 0"
            self.q = q / s

        self.alpha = float(alpha)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.ewma_beta = float(ewma_beta)
        self.min_prob = float(min_prob)

        self.n = np.zeros(self.K, dtype=np.int64)      # counts
        self.t = 0                                      # total draws
        self.ewma_loss = np.zeros(self.K, dtype=np.float64)
        self._probs_cache = np.ones(self.K, dtype=np.float64) / self.K

    def reset(self):
        self.n[:] = 0
        self.t = 0
        self.ewma_loss[:] = 0.0
        self._probs_cache[:] = 1.0 / self.K

    def step_update(self, labels: np.ndarray, losses: Optional[np.ndarray] = None):
        """
        Update internal stats after a batch was drawn/seen.
        labels: shape (B,) ints in [0, K-1]
        losses: optional (B,) loss values to update EWMA per class
        """
        labels = np.asarray(labels)
        assert labels.ndim == 1
        assert labels.size > 0
        assert np.all((0 <= labels) & (labels < self.K))

        # counts
        for c in labels:
            self.n[c] += 1
            self.t += 1

        # ewma by class
        if losses is not None:
            losses = np.asarray(losses, dtype=np.float64)
            assert losses.shape == labels.shape
            # per-class average loss for this batch
            for k in range(self.K):
                mask = (labels == k)
                if mask.any():
                    batch_mean = float(losses[mask].mean())
                    self.ewma_loss[k] = self.ewma_beta * self.ewma_loss[k] + (1 - self.ewma_beta) * batch_mean

        # refresh probs
        self._probs_cache = self._compute_probs()

    def _compute_probs(self) -> np.ndarray:
        if self.t == 0:
            return np.ones(self.K, dtype=np.float64) / self.K

        delta = self.q * self.t - self.n.astype(np.float64)  # deficit vs target
        score = self.lambda1 * delta + self.lambda2 * self._normalize_ewma(self.ewma_loss)
        logits = self.alpha * score
        # softmax
        logits = logits - logits.max()  # stability
        expv = np.exp(logits)
        p = expv / expv.sum()
        # avoid degenerate zeros
        eps = self.min_prob
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()
        return p

    def _normalize_ewma(self, v: np.ndarray) -> np.ndarray:
        # scale to roughly [-1,1] using median absolute deviation or std
        s = v.std()
        if s <= 1e-12:
            return np.zeros_like(v)
        return (v - v.mean()) / (s + 1e-12)

    def get_probs(self) -> np.ndarray:
        """Return current per-class sampling probabilities (length K)."""
        return self._probs_cache.copy()

    def importance_weight_for_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Importance weights w_i = q_i / p_i(t).
        Returns per-sample weights for given labels.
        """
        labels = np.asarray(labels)
        p = self.get_probs()
        w = self.q[labels] / p[labels]
        # normalize weights to mean 1 for numerical stability
        return w / (w.mean() + 1e-12)
