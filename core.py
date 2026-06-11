import csv
import os
import tempfile
import logging
import time

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
        self.best_loss: float | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.criterion: nn.Module | None = None
        self.metrics: dict = {}
        self._logger: logging.Logger | None = None
        self._log_dir: str | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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

    # ── CSV log ──────────────────────────────────────────────

    def _csv_path(self, save_dir: str) -> str:
        return os.path.join(save_dir, "training_log.csv")

    def _write_csv_header(self, save_dir: str):
        header = ["epoch", "train_loss", "val_loss", "lr", "time"]
        header.extend(self.metrics.keys())
        with open(self._csv_path(save_dir), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def _append_csv_row(self, save_dir: str, row: list):
        with open(self._csv_path(save_dir), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ── Helpers ──────────────────────────────────────────────

    def _filename(self, mode: str) -> str:
        table = {
            "latest": f"{self.model_name}_latest.pth",
            "best": f"{self.model_name}_best.pth",
        }
        if mode not in table:
            raise ValueError(f"Invalid mode '{mode}'. Use: {list(table.keys())}")
        return table[mode]

    def _weights_filename(self) -> str:
        return f"{self.model_name}_weights.pth"

    def count_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def summary(self, max_layers: int = 20):
        params = self.count_params()
        total, trainable = params["total"], params["trainable"]
        frozen = total - trainable
        dev = self.device

        rows = [
            f"Model: {self.model_name}",
            f"Device: {dev}",
            f"Params: {total:,} total, {trainable:,} trainable, {frozen:,} frozen",
        ]

        trainable_modules = {}
        for name, module in self.named_modules():
            if list(module.children()):
                continue
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                trainable_modules[name or type(module).__name__] = n

        if trainable_modules:
            rows.append("")
            rows.append(f"{'Layer':<40} {'Params':>12}")
            rows.append("-" * 52)
            sorted_modules = sorted(trainable_modules.items(), key=lambda x: -x[1])
            for name, n in sorted_modules[:max_layers]:
                rows.append(f"{name:<40} {n:>12,}")
            if len(sorted_modules) > max_layers:
                rows.append(f"... and {len(sorted_modules) - max_layers} more layers")

        msg = "\n".join(rows)
        self._log("info", msg)
        return params

    def freeze(self, layers: list[str] | None = None):
        if layers is None:
            for p in self.parameters():
                p.requires_grad = False
            self._log("info", "Froze all parameters")
        else:
            for name, param in self.named_parameters():
                if any(name.startswith(l) for l in layers):
                    param.requires_grad = False
            self._log("info", f"Froze layers matching: {layers}")

    def unfreeze(self, layers: list[str] | None = None):
        if layers is None:
            for p in self.parameters():
                p.requires_grad = True
            self._log("info", "Unfroze all parameters")
        else:
            for name, param in self.named_parameters():
                if any(name.startswith(l) for l in layers):
                    param.requires_grad = True
            self._log("info", f"Unfroze layers matching: {layers}")

    # ── Training hooks (override for custom training steps) ───

    def _training_step(self, batch) -> torch.Tensor:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self(inputs)
        return self.criterion(outputs, targets)

    def _eval_step(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        step_metrics = {name: fn(outputs, targets) for name, fn in self.metrics.items()}
        return loss, step_metrics

    def _step_scheduler(self, val_loss: float | None = None):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _current_lr(self) -> float:
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        return 0.0

    # ── Training ─────────────────────────────────────────────

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 500,
        model_save_dir: str = "./checkpoints/",
        patience: int = 15,
        delta: float = 0.01,
        min_delta: float = 0.0,
        save_interval: int = 1,
        grad_clip: float | None = 1.0,
    ):
        self._setup_logger(model_save_dir)
        self._write_csv_header(model_save_dir)

        early_stopping = EarlyStopping(patience=patience, delta=delta)
        start_epoch = getattr(self, "last_epoch", 0)
        total_epochs = start_epoch + num_epochs

        self._log("info", f"Training epoch {start_epoch + 1} -> {total_epochs}")
        self.summary()

        t_start = time.time()

        for epoch in range(start_epoch + 1, total_epochs + 1):
            self.last_epoch = epoch
            self._log("info", f"\nEpoch {epoch}/{total_epochs}")
            self.train()
            running_loss = 0.0
            t_epoch = time.time()

            for batch in tqdm(train_loader, desc="Training", leave=False):
                self.optimizer.zero_grad()
                loss = self._training_step(batch)
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=grad_clip
                    )
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            epoch_time = time.time() - t_epoch
            lr = self._current_lr()

            self._log(
                "info",
                f"Train Loss: {epoch_loss:.4f} | LR: {lr:.8f} | Time: {epoch_time:.1f}s",
            )

            if save_interval > 0 and epoch % save_interval == 0:
                self.save(model_dir=model_save_dir, mode="latest")

            val_loss, val_metrics = self.evaluate(val_loader, verbose=True)

            if self.best_loss is None or val_loss < self.best_loss - min_delta:
                self.best_loss = val_loss
                self.save(model_dir=model_save_dir, mode="best")
                self._save_weights(model_save_dir)
                self._log("info", f"New best model: val_loss={self.best_loss:.4f}")

            self._step_scheduler(val_loss)
            new_lr = self._current_lr()
            if new_lr != lr:
                self._log("info", f"LR: {lr:.8f} -> {new_lr:.8f}")

            metric_vals = [round(v, 6) for v in val_metrics.values()]
            self._append_csv_row(
                model_save_dir,
                [epoch, f"{epoch_loss:.6f}", f"{val_loss:.6f}", f"{new_lr:.8f}", f"{epoch_time:.1f}"] + metric_vals,
            )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                self._log("info", f"Early stopping triggered (patience={early_stopping.counter}/{patience})")
                break

        total_time = time.time() - t_start
        self.load(model_dir=model_save_dir, mode="best")
        self._log(
            "info",
            f"Training complete. Best val_loss: {self.best_loss:.4f} | "
            f"Total time: {total_time / 60:.1f}min",
        )

    # ── Evaluation ───────────────────────────────────────────

    def evaluate(
        self, data_loader, verbose: bool = False
    ) -> tuple[float, dict[str, float]]:
        self.eval()
        total_loss = 0.0
        metric_results = {name: [] for name in self.metrics}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                loss, step_metrics = self._eval_step(batch)
                total_loss += loss.item()
                for name, val in step_metrics.items():
                    metric_results[name].append(val)

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
        self.eval()
        args = tuple(a.to(self.device) if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return self(*args, **kwargs)

    # ── Save / Load / Transfer ───────────────────────────────

    def save(self, model_dir: str = "./checkpoints", mode: str = "latest"):
        os.makedirs(model_dir, exist_ok=True)

        save_path = os.path.join(model_dir, self._filename(mode))
        save_dict = {
            "epoch": self.last_epoch,
            "model_state_dict": self.state_dict(),
            "best_val": self.best_loss,
        }
        if self.optimizer is not None:
            save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        self._atomic_save(save_dict, save_path)
        self._log("info", f"Saved to {save_path}")

    def _save_weights(self, model_dir: str):
        path = os.path.join(model_dir, self._weights_filename())
        self._atomic_save(self.state_dict(), path)

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
            self.best_loss = None
            return

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.last_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_val", None)

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._log(
            "info",
            f"Loaded from {load_path}, epoch={self.last_epoch}, best_loss={self.best_loss}",
        )

    def load_weights(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        self._log("info", f"Loaded weights from {path}")

    def transfer(self, specified_path: str, strict: bool = False):
        if not specified_path:
            raise ValueError("Transfer path not specified.")
        if not os.path.exists(specified_path):
            raise FileNotFoundError(f"Transfer path not found: {specified_path}")

        self._log("info", f"Transferring from {specified_path}")
        checkpoint = torch.load(specified_path, map_location="cpu", weights_only=True)
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

    @staticmethod
    def _atomic_save(obj, path: str):
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(obj, f)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
