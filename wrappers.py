"""
Model wrappers for dancher_tools Core.

SRWrapper — wraps any nn.Module for standard forward-pass training.
EvalMixin — denormalizes output during evaluation for physical-unit metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import Core


class EvalMixin:
    """Denormalize model output before computing eval metrics."""

    target_mean: float
    target_std: float

    def _eval_step(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self(inputs)
        outputs_denorm = outputs * self.target_std + self.target_mean
        loss = F.l1_loss(outputs_denorm, targets)
        step_metrics = {name: fn(outputs_denorm, targets) for name, fn in self.metrics.items()}
        return loss, step_metrics


class SRWrapper(EvalMixin, Core):
    """Wraps any nn.Module for use with dancher_tools Core."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
