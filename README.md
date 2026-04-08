# gfs — Geon's Fair Sampler

Feedback-controlled batch sampler for training on long-tail class distributions.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from gfs import SimpleFairController, FairSampler, AugmentedDataset, importance_weights

# 1. Controller — tracks class deficit and adjusts sampling probabilities
controller = SimpleFairController(
    num_classes=10,
    class_counts=[5000, 2500, 1250, 600, 300, 150, 75, 40, 20, 10],  # optional
)

# 2. Sampler — draws class-balanced batches
sampler = FairSampler(
    labels=train_labels,       # torch.Tensor of class indices
    controller=controller,
    batch_size=256,
    steps_per_epoch=39,
)

# 3. Dataset wrapper — applies Gaussian noise to tail classes
aug_ds = AugmentedDataset(subset, class_counts=class_counts, threshold=100)

loader = DataLoader(aug_ds, batch_sampler=sampler)

# 4. Training loop
loss_fn = nn.CrossEntropyLoss(reduction='none')
q = torch.ones(10) / 10   # target distribution

for x, y in loader:
    p = controller.get_class_probs()
    w = importance_weights(y.cpu(), q, p).to(device)
    loss = (w * loss_fn(model(x), y)).mean()
    loss.backward()
    optimizer.step()
    controller.step(y.cpu())
```

## How It Works

Each step, the controller measures the gap between the target class distribution and
the observed usage in the current batch (deficit). Per-class scores are updated by
the deficit, mean-centered, and clamped to [-1, 1] to prevent drift.
Sampling probabilities are derived via softmax over `alpha * scores`, with a minimum
floor of 0.02 to ensure no class is completely starved.

Importance weights correct the loss for the sampling bias: samples drawn at higher
probability are down-weighted so the gradient remains an unbiased estimate of the
loss under the target distribution.

## Parameters

### `SimpleFairController`

| Parameter | Default | Description |
|---|---|---|
| `num_classes` | — | Number of classes K |
| `alpha` | `30.0` | Softmax temperature. Higher = more aggressive redistribution |
| `lr` | `1.0` | Score update step size |
| `class_counts` | `None` | If provided, uses sqrt-frequency target; otherwise uniform (1/K) |

### `FairSampler`

| Parameter | Default | Description |
|---|---|---|
| `labels` | — | `torch.Tensor` of per-sample class indices |
| `controller` | — | `SimpleFairController` instance |
| `batch_size` | — | Samples per batch |
| `steps_per_epoch` | — | Batches yielded per epoch |

### `AugmentedDataset`

| Parameter | Default | Description |
|---|---|---|
| `subset` | — | `torch.utils.data.Subset` |
| `class_counts` | — | List of per-class sample counts |
| `threshold` | `100` | Classes with fewer samples than this receive noise augmentation |

### `importance_weights`

| Parameter | Default | Description |
|---|---|---|
| `labels` | — | Per-sample class indices, shape `(B,)` |
| `q` | — | Target distribution, shape `(K,)` |
| `p` | — | Current sampling distribution, shape `(K,)` |
| `max_w` | `50.0` | Upper clamp to prevent weight explosion |

## Notes

- Recommended imbalance ratio: **10:1 or lower**. Beyond that, tail classes have
  too few samples for meaningful generalization regardless of sampling strategy.
- `class_counts` enables a sqrt-frequency target, which is a middle ground between
  uniform (1/K) and inverse-frequency. It boosts tail sampling without completely
  suppressing head classes.
- The oscillation in sampling probabilities is intentional — the controller
  continuously corrects for recent imbalance rather than converging to a fixed distribution.
