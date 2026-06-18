import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_ddp() -> bool:
    return "RANK" in os.environ


def ddp_info() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Initializes process group if DDP."""
    if is_ddp():
        import torch.distributed as dist
        from datetime import timedelta

        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        return dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    return 0, 1, 0


def ddp_cleanup():
    if is_ddp():
        import torch.distributed as dist

        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Slice parsing
# ---------------------------------------------------------------------------

def parse_slices(slices_json: str | None) -> dict[int, slice] | None:
    if slices_json is None:
        return None
    import json
    raw = json.loads(slices_json)
    return {int(k): slice(v[0], v[1]) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Loss functions for SR / regression tasks
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier loss: sqrt(x^2 + epsilon^2).

    Differentiable L1 with smooth gradient near zero. Standard in image SR.
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon2 = epsilon ** 2

    def forward(self, inputs, targets) -> torch.Tensor:
        diff = inputs - targets
        return torch.mean(torch.sqrt(diff * diff + self.epsilon2))


class GradientLoss(nn.Module):
    """L1 on spatial gradients (DH, DW, and DT differences).

    Penalizes edge misalignment — sharpens boundaries (coastlines, fronts).
    """

    def forward(self, inputs, targets) -> torch.Tensor:
        total = 0.0
        for dim in (-3, -2, -1):
            in_diff = inputs.diff(dim=dim)
            tgt_diff = targets.diff(dim=dim)
            total = total + F.l1_loss(in_diff, tgt_diff)
        return total / 3.0


class CoastWeightedLoss(nn.Module):
    """Charbonnier weighted by proximity to land (target == 0).

    Pixels where target is 0 are land; their neighbors get higher weight
    to emphasize coastal accuracy. Default weight 3.0 for coastal pixels.
    """

    def __init__(self, coastal_weight: float = 3.0, epsilon: float = 1e-3):
        super().__init__()
        self.coastal_weight = coastal_weight
        self.epsilon2 = epsilon ** 2

    def forward(self, inputs, targets) -> torch.Tensor:
        diff = inputs - targets
        base = torch.sqrt(diff * diff + self.epsilon2)

        land = (targets == 0).float()
        ocean = 1.0 - land
        # Coastal = ocean pixel within 1 pixel of land (max-pool of land mask)
        # Pad to keep spatial dims, pool with kernel 3 to capture 1-ring neighbors.
        land_padded = F.pad(land, (1, 1, 1, 1, 1, 1))
        coastal_zone = F.max_pool3d(land_padded, kernel_size=3, stride=1)
        coastal_zone = (coastal_zone - land).clamp(0, 1)  # coastal only, exclude land
        weight = 1.0 + (self.coastal_weight - 1.0) * coastal_zone * ocean
        return torch.mean(base * weight)


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_criterion(spec: str) -> nn.Module:
    """Build a loss function from a short spec string.

    Examples:
        "mse"                       -> F.mse_loss wrapped
        "l1"                        -> F.l1_loss wrapped
        "charbonnier"               -> CharbonnierLoss
        "charbonnier+gradient"      -> CombinedLoss(charb + grad, weights [1, 0.1])
        "charbonnier+coast"         -> CombinedLoss(charb + coast, weights [1, 1])
    """
    spec = spec.strip().lower()
    parts = spec.split("+")

    registry = {
        "mse": lambda: _FunctionalLoss(F.mse_loss),
        "l1": lambda: _FunctionalLoss(F.l1_loss),
        "charbonnier": CharbonnierLoss,
        "charb": CharbonnierLoss,
        "gradient": GradientLoss,
        "grad": GradientLoss,
        "coast": CoastWeightedLoss,
        "coastal": CoastWeightedLoss,
    }

    for p in parts:
        if p not in registry:
            raise ValueError(f"Unknown loss '{p}'. Available: {list(registry.keys())}")

    losses = [registry[p]() for p in parts]
    if len(losses) == 1:
        return losses[0]

    # Default combination weights: anchor loss = 1, others = 0.1
    weights = [1.0] + [0.1] * (len(losses) - 1)
    return CombinedLoss(losses, weights)


class _FunctionalLoss(nn.Module):
    """Wrap an arbitrary functional (e.g. F.mse_loss) as an nn.Module."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, inputs, targets) -> torch.Tensor:
        return self.fn(inputs, targets)
