import torch
from gfs import AugmentedDataset


def test_len(simple_subset, class_counts):
    ds = AugmentedDataset(simple_subset, class_counts)
    assert len(ds) == len(simple_subset)


def test_default_threshold_auto_computed(simple_subset, class_counts):
    ds = AugmentedDataset(simple_subset, class_counts)
    # max=500, so threshold = 500 // 10 = 50
    assert ds.threshold == max(class_counts) // 10


def test_tail_detection(simple_subset, class_counts):
    ds = AugmentedDataset(simple_subset, class_counts, threshold=300)
    # class 0 (500): not tail → is_tail[0] = False
    assert not ds.is_tail[0].item()
    # class 1 starts at index 500 (count=200 < 300): tail
    assert ds.is_tail[500].item()


def test_output_shape(simple_subset, class_counts):
    ds = AugmentedDataset(simple_subset, class_counts)
    x, y = ds[0]
    assert x.shape == simple_subset[0][0].shape


def test_tail_pixels_clamped(simple_subset, class_counts):
    # noise_std 크게 줘도 [0, 1] 밖으로 안 나가야 함
    ds = AugmentedDataset(simple_subset, class_counts, noise_std=10.0)
    for idx in ds.is_tail.nonzero(as_tuple=True)[0][:10]:
        x, _ = ds[idx.item()]
        assert x.min() >= 0.0
        assert x.max() <= 1.0


def test_noise_std_zero_identity(simple_subset, class_counts):
    # noise_std=0이면 tail 샘플도 원본과 동일 (rand data는 이미 [0,1])
    ds = AugmentedDataset(simple_subset, class_counts, noise_std=0.0)
    for idx in ds.is_tail.nonzero(as_tuple=True)[0][:5]:
        x_aug, _ = ds[idx.item()]
        x_orig, _ = simple_subset[idx.item()]
        assert torch.allclose(x_aug, x_orig)
