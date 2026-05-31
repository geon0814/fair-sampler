# gfs — Geon's Fair Sampler

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Feedback-controlled batch sampler for training on long-tail class distributions.

## Why gfs?

Standard DataLoader shuffles uniformly, so rare classes appear in proportion to their size — a class with 10 samples gets ~0.1% of batches in a 10,000-sample dataset. gfs continuously monitors the class distribution inside each batch and pushes sampling probabilities toward a target distribution, while importance weighting keeps the gradient estimate unbiased.

## Install

```bash
pip install -e .
# with test dependencies
pip install -e ".[test]"
# with visualization
pip install -e ".[plot]"
```

## Quick Start

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gfs import SimpleFairController, FairSampler, AugmentedDataset, importance_weights

# 1. Controller — class deficit를 추적하고 샘플링 확률을 조정
controller = SimpleFairController(
    num_classes=10,
    class_counts=[5000, 2500, 1250, 600, 300, 150, 75, 40, 20, 10],  # optional
)

# 2. Sampler — class-balanced 배치 생성
sampler = FairSampler(
    labels=train_labels,        # torch.Tensor, 샘플별 클래스 인덱스
    controller=controller,
    batch_size=256,
    steps_per_epoch=39,
    generator=torch.Generator().manual_seed(42),  # optional, 재현성용
)

# 3. Dataset wrapper — tail class에 Gaussian noise 증강
aug_ds = AugmentedDataset(subset, class_counts=class_counts)

loader = DataLoader(aug_ds, batch_sampler=sampler)

# 4. Training loop
loss_fn = nn.CrossEntropyLoss(reduction="none")
q = torch.ones(10) / 10   # target distribution (여기선 uniform)

for x, y in loader:
    p = controller.get_class_probs()
    w = importance_weights(y.cpu(), q, p).to(device)
    loss = (w * loss_fn(model(x), y)).mean()
    loss.backward()
    optimizer.step()
    controller.step(y.cpu())
```

## How It Works

1. **Controller**: 매 배치마다 관측된 클래스 분포와 목표 분포의 차이(deficit)로 per-class 스코어를 업데이트합니다. 스코어는 mean-centering 후 `[-1, 1]`에 clamp되어 발산을 막고, `softmax(α·scores)` + 0.02 floor로 최종 확률을 계산합니다.

2. **Sampler**: 컨트롤러의 클래스 확률을 각 샘플에 매핑한 가중치로 `torch.multinomial`을 호출해 배치를 구성합니다.

3. **Importance Weighting**: over-sampling된 클래스 샘플은 down-weight되어 `w = q[y] / p[y]`의 가중 손실을 만들고, 그래디언트 추정이 목표 분포 `q` 아래에서 unbiased하게 유지됩니다.

### Controller Behavior

99:1로 불균형한 데이터에서 컨트롤러가 50:50 목표를 향해 빠르게 수렴합니다.

![Adaptation from 99:1 toward 50:50](https://raw.githubusercontent.com/geon0814/fair-sampler/main/assets/fair_rebalance.png)

목표에 도달한 뒤에도 배치 단위 노이즈로 인해 진동(oscillation)이 발생하지만, 컨트롤러가 매 배치마다 즉시 보정합니다. 이 진동은 의도된 동작입니다.

![Drift and Recovery cycles](https://raw.githubusercontent.com/geon0814/fair-sampler/main/assets/drift_recovery.png)

## API Reference

### `SimpleFairController`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `num_classes` | — | 클래스 수 K |
| `alpha` | `30.0` | Softmax 온도. 높을수록 더 공격적으로 재분배 |
| `lr` | `1.0` | 스코어 업데이트 step size |
| `class_counts` | `None` | 제공 시 sqrt-frequency 목표 분포 사용, 없으면 uniform (1/K) |

추가 메서드:
- `reset()` — 스코어를 0으로 초기화 (목표 분포는 유지)
- `state_dict()` / `load_state_dict(state)` — 체크포인트 저장/복원

### `FairSampler`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `labels` | — | 샘플별 클래스 인덱스 `torch.Tensor` |
| `controller` | — | `SimpleFairController` 인스턴스 |
| `batch_size` | — | 배치당 샘플 수 |
| `steps_per_epoch` | — | 에폭당 배치 수 |
| `replacement` | `True` | 복원 추출 여부 |
| `generator` | `None` | `torch.Generator` (재현성용) |

### `AugmentedDataset`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `subset` | — | `torch.utils.data.Subset` |
| `class_counts` | — | 클래스별 샘플 수 리스트 |
| `threshold` | `None` | tail 기준 샘플 수. `None`이면 `max(class_counts) // 10` 자동 사용 |
| `noise_std` | `0.1` | 증강 노이즈 표준편차 |

### `importance_weights`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `labels` | — | 샘플별 클래스 인덱스, shape `(B,)` |
| `q` | — | 목표 분포, shape `(K,)` |
| `p` | — | 현재 샘플링 분포, shape `(K,)` |
| `max_w` | `50.0` | weight 폭발 방지 상한 clamp |

## Run Tests

```bash
pytest tests/ -v
```

## Notes

- **권장 imbalance ratio: 10:1 이하**. 그 이상이면 tail class 샘플 수 자체가 너무 적어 샘플링 전략과 무관하게 일반화가 어렵습니다.
- `class_counts`를 제공하면 sqrt-frequency 목표를 사용합니다. uniform (1/K)과 inverse-frequency의 중간으로, head class를 완전히 억제하지 않으면서 tail을 부스팅합니다.
- 샘플링 확률이 진동하는 것은 의도된 동작입니다 — 컨트롤러는 고정 분포로 수렴하지 않고 최근 imbalance를 지속적으로 보정합니다.
