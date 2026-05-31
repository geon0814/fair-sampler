import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import ssl

from gfs import SimpleFairController, FairSampler, AugmentedDataset, importance_weights

ssl._create_default_https_context = ssl._create_unverified_context

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

SEED = 42
LONGTAIL_COUNTS = [5000, 2500, 1250, 600, 300, 150, 75, 40, 20, 10]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# [1] Long-tail Dataset
# -----------------------------
def make_longtail_subset(ds):
    targets = ds.targets
    indices = []
    for cls, count in enumerate(LONGTAIL_COUNTS):
        cls_indices = (targets == cls).nonzero(as_tuple=True)[0].tolist()
        indices.extend(cls_indices[:count])
    return Subset(ds, indices)


# -----------------------------
# [2] Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# [3] Evaluate per-class accuracy
# -----------------------------
def evaluate(model, test_loader, device, num_classes=10):
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            y_cpu = y.cpu()
            preds_cpu = preds.cpu()
            for c in range(num_classes):
                mask = y_cpu == c
                total[c] += mask.sum()
                correct[c] += (preds_cpu[mask] == c).sum()
    model.train()
    return (correct / total.clamp(min=1)).tolist()


# -----------------------------
# [4] Train
# -----------------------------
def train_fair(ds_train, device, epochs=7):
    set_seed(SEED)
    aug_ds = AugmentedDataset(ds_train, class_counts=LONGTAIL_COUNTS)
    labels = torch.tensor([ds_train.dataset.targets[i].item() for i in ds_train.indices])

    controller = SimpleFairController(
        num_classes=10, alpha=30.0, lr=1.0, class_counts=LONGTAIL_COUNTS
    )
    sampler = FairSampler(
        labels=labels,
        controller=controller,
        batch_size=256,
        steps_per_epoch=39,
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = DataLoader(aug_ds, batch_sampler=sampler)

    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    q = torch.ones(10) / 10

    for ep in range(epochs):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            p = controller.get_class_probs()
            w = importance_weights(y.cpu(), q, p).to(device)
            loss = (w * loss_fn(model(x), y)).mean()
            loss.backward()
            opt.step()
            controller.step(y.cpu())
            if (step + 1) % 20 == 0:
                controller.log(ep, step, loss.item())

    return model


def train_vanilla(ds_train, device, epochs=7):
    set_seed(SEED)
    loader = DataLoader(ds_train, batch_size=128, shuffle=True)

    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    return model


def plot_comparison(acc_fair, acc_vanilla):
    classes = list(range(10))
    x = range(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], [a * 100 for a in acc_fair], width, label="Fair", color="#4c9be8")
    ax.bar([i + width / 2 for i in x], [a * 100 for a in acc_vanilla], width, label="Vanilla", color="#e8834c")

    ax.set_xlabel("Class (train samples)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-class accuracy: FairSampler vs Vanilla")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c}\n({LONGTAIL_COUNTS[c]})" for c in classes])
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("saved results.png")
    plt.show()


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    tfm = transforms.ToTensor()
    ds_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    ds_test = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    ds_train = make_longtail_subset(ds_full)
    test_loader = DataLoader(ds_test, batch_size=256)

    print(f"\nLong-tail train set: {LONGTAIL_COUNTS}")
    print(f"Total train samples: {sum(LONGTAIL_COUNTS)}\n")

    print("=" * 50)
    print("Run A: FairSampler + importance weight")
    print("=" * 50)
    model_fair = train_fair(ds_train, device)
    acc_fair = evaluate(model_fair, test_loader, device)

    print("\n" + "=" * 50)
    print("Run B: Vanilla shuffle")
    print("=" * 50)
    model_vanilla = train_vanilla(ds_train, device)
    acc_vanilla = evaluate(model_vanilla, test_loader, device)

    print("\n" + "=" * 50)
    print("Per-class accuracy comparison")
    print("=" * 50)
    print(f"{'Class':<8} {'Samples':>8} {'Fair':>8} {'Vanilla':>10} {'Delta':>8}")
    print("-" * 44)
    for c in range(10):
        delta = acc_fair[c] - acc_vanilla[c]
        sign = "+" if delta >= 0 else ""
        print(
            f"{c:<8} {LONGTAIL_COUNTS[c]:>8} "
            f"{acc_fair[c]:>7.1%} {acc_vanilla[c]:>9.1%} "
            f"{sign}{delta:>6.1%}"
        )
    print("-" * 44)
    print(
        f"{'Mean':<8} {'':>8} "
        f"{sum(acc_fair)/10:>7.1%} {sum(acc_vanilla)/10:>9.1%}"
    )

    if HAS_MATPLOTLIB:
        plot_comparison(acc_fair, acc_vanilla)

    print("\ndone")


if __name__ == "__main__":
    main()
