import torch
import pytest
from torch.utils.data import Dataset, Subset


class _FakeImageDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def num_classes() -> int:
    return 5


@pytest.fixture
def class_counts() -> list:
    # [500, 200, 100, 50, 10]  — max=500, default threshold=50
    return [500, 200, 100, 50, 10]


@pytest.fixture
def simple_subset(class_counts: list) -> Subset:
    targets_list = []
    for cls, count in enumerate(class_counts):
        targets_list.extend([cls] * count)
    targets = torch.tensor(targets_list)
    data = torch.rand(len(targets), 1, 4, 4)  # 값이 [0,1]이어야 clamp 테스트가 정확함
    ds = _FakeImageDataset(data, targets)
    return Subset(ds, list(range(len(targets))))


@pytest.fixture
def labels(class_counts: list) -> torch.Tensor:
    labels_list = []
    for cls, count in enumerate(class_counts):
        labels_list.extend([cls] * count)
    return torch.tensor(labels_list)
