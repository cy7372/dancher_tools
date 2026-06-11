import os
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import CombinedLoss, EarlyStopping


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name: str | None = None
        self.last_epoch: int = 0
        self.best_val: float | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.criterion: nn.Module | None = None
        self.metrics: dict = {}
        self._logger: logging.Logger | None = None
        self._log_dir: str | None = None

    def compile(
        self,
        criterion: nn.Module | list[nn.Module],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        metrics: dict[str, callable] | None = None,
        loss_weights: list[float] | None = None,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(criterion, list):
            match len(criterion):
                case 0:
                    raise ValueError("Criterion list cannot be empty.")
                case 1:
                    self.criterion = criterion[0]
                case _:
                    self.criterion = CombinedLoss(losses=criterion, weights=loss_weights)
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise TypeError("Criterion should be a callable or a list of callables.")

        self.metrics = {}
        if metrics is not None:
            for name, fn in metrics.items():
                if not callable(fn):
                    raise ValueError(f"Metric '{name}' is not callable.")
                self.metrics[name] = fn

        self._log("info", f"Compiled with metrics: {list(self.metrics.keys())}")

    # ── Logging ──────────────────────────────────────────────

    def _setup_logger(self, log_dir: str):
        if self._logger is not None and self._log_dir == log_dir:
            return

        self._logger = logging.getLogger(f"dancher.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        self._logger.addHandler(console)

        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"), encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        self._logger.addHandler(file_handler)
        self._log_dir = log_dir

    def _log(self, level: str, msg: str):
        if self._logger is not None:
            getattr(self._logger, level)(msg)
        else:
            print(msg)

    # ── Helpers ──────────────────────────────────────────────

    def _filename(self, mode: str) -> str:
        table = {
            "latest": f"{self.model_name}_latest.pth",
            "best": f"{self.model_name}_best.pth",
            "epoch": f"{self.model_name}_epoch_{self.last_epoch}.pth",
        }
        if mode not in table:
            raise ValueError(f"Invalid mode '{mode}'. Use: {list(table.keys())}")
        return table[mode]

    # ── Training ─────────────────────────────────────────────

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 500,
        model_save_dir: str = "./checkpoints/",
        patience: int = 15,
        delta: float = 0.01,
        save_interval: int = 1,
        grad_clip: float | None = 1.0,
        higher_is_better: bool = True,
    ):
        self._setup_logger(model_save_dir)

        early_stopping = EarlyStopping(patience=patience, delta=delta)
        device = next(self.parameters()).device
        start_epoch = getattr(self, "last_epoch", 0)
        total_epochs = start_epoch + num_epochs
        first_metric = list(self.metrics.keys())[0] if self.metrics else None

        self._log("info", f"Training epoch {start_epoch + 1} -> {total_epochs}")

        for epoch in range(start_epoch + 1, total_epochs + 1):
            self.last_epoch = epoch
            self._log("info", f"\nEpoch {epoch}/{total_epochs}")
            self.train()
            running_loss = 0.0

            for batch in tqdm(train_loader, desc="Training", leave=False):
                inputs, targets = batch[0].to(device), batch[1].to(device)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=grad_clip
                    )
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            self._log("info", f"Train Loss: {epoch_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()
                self._log("info", f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

            if save_interval > 0 and epoch % save_interval == 0:
                self.save(model_dir=model_save_dir, mode="latest")

            val_loss, val_metrics = self.evaluate(val_loader, verbose=True)
            val_score = val_metrics.get(first_metric) if first_metric else None

            if val_score is not None:
                if self.best_val is None:
                    improved = True
                elif higher_is_better:
                    improved = val_score > self.best_val
                else:
                    improved = val_score < self.best_val
                if improved:
                    self.best_val = val_score
                    self.save(model_dir=model_save_dir, mode="best")
                    self._log(
                        "info", f"New best model: {first_metric}={self.best_val:.4f}"
                    )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                self._log("info", "Early stopping triggered")
                break

        self.load(model_dir=model_save_dir, mode="best")
        self._log("info", f"Training complete. Best {first_metric}: {self.best_val:.4f}")

    # ── Evaluation ───────────────────────────────────────────

    def evaluate(
        self, data_loader, verbose: bool = False
    ) -> tuple[float, dict[str, float]]:
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0.0
        metric_results = {name: [] for name in self.metrics}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = self(inputs)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                for name, fn in self.metrics.items():
                    metric_results[name].append(fn(outputs, targets))

        avg_loss = total_loss / len(data_loader)
        avg_metrics = {
            name: round(float(np.mean(vals)), 4) for name, vals in metric_results.items()
        }

        if verbose:
            self._log("info", f"Val Loss: {avg_loss:.4f} | Metrics: {avg_metrics}")
        return avg_loss, avg_metrics

    # ── Predict ──────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        device = next(self.parameters()).device
        self.eval()
        return self(*args, **kwargs)

    # ── Save / Load / Transfer ───────────────────────────────

    def save(self, model_dir: str = "./checkpoints", mode: str = "latest"):
        os.makedirs(model_dir, exist_ok=True)

        save_path = os.path.join(model_dir, self._filename(mode))
        save_dict = {
            "epoch": self.last_epoch,
            "model_state_dict": self.state_dict(),
            "best_val": self.best_val,
        }
        if self.optimizer is not None:
            save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        try:
            torch.save(save_dict, save_path, pickle_protocol=4)
            self._log("info", f"Saved to {save_path}")
        except Exception as e:
            self._log("error", f"Failed to save: {e}")

    def load(
        self,
        model_dir: str = "./checkpoints",
        mode: str = "latest",
        specified_path: str | None = None,
    ):
        load_path = specified_path or os.path.join(model_dir, self._filename(mode))

        if not os.path.exists(load_path):
            self._log("info", f"No checkpoint at {load_path}, starting from scratch")
            self.last_epoch = 0
            self.best_val = None
            return

        checkpoint = torch.load(load_path, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.last_epoch = checkpoint.get("epoch", 0)
        self.best_val = checkpoint.get("best_val", None)

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._log(
            "info",
            f"Loaded from {load_path}, epoch={self.last_epoch}, best_val={self.best_val}",
        )

    def transfer(self, specified_path: str, strict: bool = False):
        if not specified_path:
            raise ValueError("Transfer path not specified.")
        if not os.path.exists(specified_path):
            raise FileNotFoundError(f"Transfer path not found: {specified_path}")

        self._log("info", f"Transferring from {specified_path}")
        checkpoint = torch.load(specified_path, weights_only=False)
        src_state = checkpoint.get("model_state_dict", checkpoint)
        dst_state = self.state_dict()

        new_state = {}
        missing = []
        size_mismatch = []

        for name, param in dst_state.items():
            if name in src_state:
                if src_state[name].size() == param.size():
                    new_state[name] = src_state[name]
                else:
                    size_mismatch.append(name)
            else:
                missing.append(name)

        self.load_state_dict(new_state, strict=False)
        self._log(
            "info",
            f"Transfer: {len(new_state)} matched, {len(missing)} missing, "
            f"{len(size_mismatch)} size mismatch",
        )
        if missing:
            self._log("info", f"Missing: {missing}")
        if size_mismatch:
            self._log("info", f"Size mismatch: {size_mismatch}")
