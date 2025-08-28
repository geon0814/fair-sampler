"""
Example: Train a tiny MLP on MNIST with Geon's Feedback Sampler/Optimizer (GFS/GFO).

Note: Requires internet for dataset download on first run (torchvision).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from gfs.controller import FeedbackController
from gfs.batch_sampler import FeedbackBatchSampler
from gfs.iw import importance_weights
import numpy as np

# ---- Model ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)

    # labels
    labels = np.array(train_ds.targets, dtype=np.int64)
    K = 10
    target = np.ones(K) / K  # uniform target

    # Controller: emphasize deficit (lambda1) and optionally loss (lambda2)
    controller = FeedbackController(
        num_classes=K,
        target_probs=target,
        alpha=6.0,      # aggressiveness: try 4..10
        lambda1=1.0,    # deficit term
        lambda2=0.5,    # loss term weight (try 0 first, then increase)
        ewma_beta=0.95,
    )

    # Sampler: create adaptive batches
    batch_size = 256
    steps_per_epoch = len(train_ds) // batch_size
    sampler = FeedbackBatchSampler(labels=labels, batch_size=batch_size,
                                   steps_per_epoch=steps_per_epoch, controller=controller)

    # We'll wrap the dataset to get items by index from the sampler
    class IndexedDS(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            x, y = self.base[idx]
            return x, y, idx

    train_ix = IndexedDS(train_ds)
    # DataLoader with our batch sampler
    loader = DataLoader(train_ix, batch_sampler=sampler, num_workers=2, pin_memory=True)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")  # per-sample loss
    optimz = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        class_counts = np.zeros(K, dtype=np.int64)

        for step, batch in enumerate(loader):
            x, y, idx = batch
            x, y = x.to(device), y.to(device)

            # current probs from controller
            cur_p = torch.from_numpy(controller.get_probs())

            logits = model(x)
            per_sample_loss = criterion(logits, y)

            # importance weights for unbiasedness
            iw = importance_weights(y, controller.q, cur_p).to(device)
            loss = (iw * per_sample_loss).mean()

            optimz.zero_grad(set_to_none=True)
            loss.backward()
            optimz.step()

            running_loss += float(loss.item())

            # update controller stats AFTER we used current probs
            controller.step_update(labels=y.detach().cpu().numpy(),
                                   losses=per_sample_loss.detach().cpu().numpy())

            # track counts just to log
            for c in y.detach().cpu().numpy():
                class_counts[int(c)] += 1

            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1} Step {step+1} Loss {running_loss/100:.4f}  p={controller.get_probs()}")
                running_loss = 0.0

        print(f"[Epoch {epoch+1}] class counts:", class_counts, " probs:", controller.get_probs())

if __name__ == "__main__":
    main()
