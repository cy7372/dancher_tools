import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    @property
    def status(self) -> str:
        return f"{self.counter}/{self.patience}"


class CombinedLoss(nn.Module):
    def __init__(self, losses: list, weights: list | None = None):
        super().__init__()
        self.losses = nn.ModuleList([l() if isinstance(l, type) else l for l in losses])

        if weights is None:
            weights = [1.0 / len(self.losses)] * len(self.losses)
        elif len(weights) != len(self.losses):
            raise ValueError("Length of weights must match the number of loss functions.")

        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets) -> torch.Tensor:
        total = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total += weight * loss_fn(inputs, targets)
        return total
