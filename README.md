# 📦 Fair-Sampler  
*By Geon*

**Adaptive, feedback-controlled sampler that applies negative feedback to keep class distribution fair and stable during AI training.**  
Forces rapid convergence to target ratios — while preserving unbiased estimates via **importance weighting**.
// 실전 롱테일/드리프트 환경에서 샘플링 비율을 실시간 제어하고, 중요도 가중치로 편향 없이 학습

![Rebalance (99:1 → 50:50)](assets/fair_rebalance.png)
![Drift→Recovery (10 cycles)](assets/drift_recovery.png)

## Install
```bash
pip install -e .
```

## Quick Start
```python
from gfs.controller import FeedbackController
from gfs.batch_sampler import FeedbackBatchSampler
from gfs.iw import importance_weights

controller = FeedbackController(num_classes=K, target_probs=[1/K]*K, alpha=8.0)
sampler = FeedbackBatchSampler(labels, batch_size=256, steps_per_epoch=100, controller=controller)

for batch_indices in sampler:
    y = labels[batch_indices]                        # class ids for the batch
    p = controller.get_probs()                       # current sampling probs
    w = importance_weights(y, controller.q, p)       # unbiased IW (mean≈1)
    loss = (w * criterion(model(x[batch_indices]), torch.as_tensor(y))).mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    controller.step_update(y, losses=None)           # or pass per-sample loss for EWMA(loss)
```

## Core Idea
- **Negative feedback** counteracts imbalance: under-sampled classes get higher sampling probability.
- **Softmax control**: `p_i(t) = softmax(α * (λ1 * deficit_i + λ2 * ewma_loss_i))`
- **Unbiased training**: `w = q[labels] / p[labels]`, normalized to mean 1.
// p는 매 step 적응, w는 그 p에 정렬되어 bias 0

## Stability Tips
- Set a **probability floor**: `min_prob=1e-6` (default) to avoid extreme weights.
- If drift is wild: `importance_weights(..., max_w=50.0)`.
- Consider **α warm-up/cool-down** for smooth adaptation.
// 초기 급격 튐 방지

## Reproducibility
```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
```

## Example
See `train_mnist.py` for a runnable demo (replace with CIFAR-LT/medical/streaming as needed).
// 예시는 데모용, 어떤 데이터에도 바로 적용 가능
