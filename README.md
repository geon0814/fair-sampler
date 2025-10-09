# Fair Sampler
*By Geon *

**Feedback-controlled sampler applying negative feedback to keep class distribution fair and stable during AI training.**  
Forces fast convergence to target ratios while preserving unbiased estimates via **importance weighting**.

## Install
```bash
pip install -e .
```

## Quick Start
```python
from gfs.controller import FeedbackController
from gfs.batch_sampler import FeedbackBatchSampler
from gfs.iw import importance_weights
```

## Core Idea
- Negative feedback to counter imbalance: under-sampled classes get higher sampling probability.
- Softmax control: `p_i(t) = softmax(alpha * (lambda1 * deficit_i + lambda2 * ewma_loss_i))`
- Unbiased training with importance weights: `w_i = target_q[i] / p_i(t)` (normalized to mean 1).

## Stability Tips
- Use a **minimum probability floor** (now default `min_prob=1e-6`) to avoid extreme importance weights.
- (Optional) **Clip importance weights** via `importance_weights(..., max_w=50.0)` in very aggressive settings.
- Consider scheduling `alpha` during training (warm-up/cool-down).

## Reproducibility
```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
```
## Example
The `train_mnist.py` script shows how to integrate **Fair Sampler** into a simple training loop.  
(MNIST dataset is used here only as a demonstration — you can replace it with any dataset.)

➡️ [train_mnist.py](train_mnist.py)
