import json
import os
import tempfile
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
        self._ddp_rank: int = 0
        self._ddp_world_size: int = 1
        self._ddp_local_rank: int = 0
        self._ddp_wrapped = None
        self._use_amp: bool = False
        self._grad_scaler: torch.amp.GradScaler | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def is_main(self) -> bool:
        return self._ddp_rank == 0

    # ── DDP ──────────────────────────────────────────────────

    def _setup_ddp(self):
        """Initialize DDP if detected. Sets rank/device info and moves model."""
        from .utils import is_ddp, ddp_info

        if is_ddp():
            rank, world_size, local_rank = ddp_info()
            self._ddp_rank = rank
            self._ddp_world_size = world_size
            self._ddp_local_rank = local_rank
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            self.to(device)
        return self._ddp_world_size > 1

    def _setup_dataloaders(self, train_data, val_data, batch_size: int):
        """Accept DataLoader, Dataset, or DataModule. Returns (train_loader, val_loader)."""
        from torch.utils.data import Dataset, DataLoader
        from .data import DataModule

        if isinstance(train_data, DataModule):
            if train_data.train_ds is None:
                train_data.setup()
            return train_data.train_dataloader(), train_data.val_dataloader()

        if isinstance(train_data, Dataset):
            from .utils import is_ddp

            sampler = None
            shuffle = True
            if is_ddp():
                from torch.utils.data.distributed import DistributedSampler

                sampler = DistributedSampler(train_data, shuffle=True)
                shuffle = False
            train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=shuffle,
                num_workers=2,
                pin_memory=True,
            )
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
            return train_loader, val_loader

        return train_data, val_data

    def compile(
        self,
        criterion: nn.Module | list[nn.Module],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        metrics: dict[str, callable] | None = None,
        loss_weights: list[float] | None = None,
        amp: bool = False,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler

        if amp:
            self._use_amp = True
            self._grad_scaler = torch.amp.GradScaler("cuda")
            self._log("AMP enabled (float16 autocast)")

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

        if self.metrics:
            self._log(f"Compiled with metrics: {list(self.metrics.keys())}")

    # ── Logging ──────────────────────────────────────────────

    def _log(self, msg: str):
        if self.is_main:
            print(msg, flush=True)

    # ── JSONL log ─────────────────────────────────────────────

    def _log_path(self, save_dir: str) -> str:
        return os.path.join(save_dir, "training_log.jsonl")

    def _append_log_row(self, save_dir: str, row: dict):
        if not self.is_main:
            return
        with open(self._log_path(save_dir), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Helpers ──────────────────────────────────────────────

    def _filename(self, mode: str) -> str:
        table = {
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
        self._log(msg)
        return params

    def freeze(self, layers: list[str] | None = None):
        if layers is None:
            for p in self.parameters():
                p.requires_grad = False
            self._log("Froze all parameters")
        else:
            for name, param in self.named_parameters():
                if any(name.startswith(l) for l in layers):
                    param.requires_grad = False
            self._log(f"Froze layers matching: {layers}")

    def unfreeze(self, layers: list[str] | None = None):
        if layers is None:
            for p in self.parameters():
                p.requires_grad = True
            self._log("Unfroze all parameters")
        else:
            for name, param in self.named_parameters():
                if any(name.startswith(l) for l in layers):
                    param.requires_grad = True
            self._log(f"Unfroze layers matching: {layers}")

    # ── Training hooks (override for custom training steps) ───

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — goes through DDP wrapper when active."""
        if self._ddp_wrapped is not None:
            return self._ddp_wrapped(x)
        return self(x)

    def _training_step(self, batch) -> torch.Tensor:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self._forward(inputs)
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

    def _ddp_all_reduce(self, value: float) -> float:
        """Average a scalar across all DDP ranks. No-op in single-process."""
        if self._ddp_world_size <= 1:
            return value
        import torch.distributed as dist
        t = torch.tensor(value, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / self._ddp_world_size).item()

    # ── Training ─────────────────────────────────────────────

    def fit(
        self,
        train_data,
        val_data=None,
        num_epochs: int = 500,
        batch_size: int = 2,
        model_save_dir: str = "./checkpoints/",
        patience: int = 15,
        delta: float = 0.01,
        min_delta: float = 0.0,
        grad_clip: float | None = 1.0,
    ):
        # ── DDP init + wrap ──
        wrapped = self._setup_ddp()
        if wrapped:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self._ddp_wrapped = DDP(self, device_ids=[self._ddp_local_rank])
        else:
            self._ddp_wrapped = None

        # ── Resolve data ──
        train_loader, val_loader = self._setup_dataloaders(train_data, val_data, batch_size)

        if self.is_main:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self._log(f"Trainable params: {n_params:,}")
            if wrapped:
                self._log(f"DDP: {self._ddp_world_size} GPUs, rank={self._ddp_rank}")
            self._log(f"Device: {self.device}")

        early_stopping = EarlyStopping(patience=patience, delta=delta)
        start_epoch = getattr(self, "last_epoch", 0)
        total_epochs = start_epoch + num_epochs

        self._log(f"Training epoch {start_epoch + 1} -> {total_epochs}")
        t_start = time.time()

        try:
            for epoch in range(start_epoch + 1, total_epochs + 1):
                self.last_epoch = epoch

                if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

                self.train()
                running_loss = 0.0
                t_epoch = time.time()

                for batch in tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}/{total_epochs}",
                    leave=False,
                    disable=not self.is_main,
                ):
                    self.optimizer.zero_grad()
                    if self._use_amp:
                        with torch.amp.autocast("cuda"):
                            loss = self._training_step(batch)
                        self._grad_scaler.scale(loss).backward()
                        if grad_clip is not None:
                            self._grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.parameters(), max_norm=grad_clip
                            )
                        self._grad_scaler.step(self.optimizer)
                        self._grad_scaler.update()
                    else:
                        loss = self._training_step(batch)
                        loss.backward()
                        if grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.parameters(), max_norm=grad_clip
                            )
                        self.optimizer.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)
                epoch_loss = self._ddp_all_reduce(epoch_loss)
                epoch_time = time.time() - t_epoch

                val_loss, val_metrics = self.evaluate(val_loader, verbose=False)

                is_best = self.best_loss is None or val_loss < self.best_loss - min_delta
                if is_best:
                    self.best_loss = val_loss
                    self.save(model_dir=model_save_dir, mode="best")
                    self._save_weights(model_save_dir)

                if self._ddp_world_size > 1:
                    import torch.distributed as dist
                    dist.barrier()

                self._step_scheduler(val_loss)
                new_lr = self._current_lr()

                # Compact epoch summary
                best_flag = " *" if is_best else ""
                parts = [
                    f"[{epoch}/{total_epochs}]",
                    f"train={epoch_loss:.4f}",
                    f"val={val_loss:.4f}{best_flag}",
                    f"lr={new_lr:.2e}",
                    f"{epoch_time:.0f}s",
                ]
                if val_metrics:
                    parts.append(" ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                if early_stopping.counter > 0:
                    parts.append(f"patience={early_stopping.counter}/{patience}")
                self._log(" | ".join(parts))

                self._append_log_row(
                    model_save_dir,
                    {
                        "epoch": epoch,
                        "train_loss": round(epoch_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "lr": new_lr,
                        "time": round(epoch_time, 1),
                        "best": is_best,
                        **{k: round(v, 6) for k, v in val_metrics.items()},
                    },
                )

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    self._log(f"Early stopping triggered (patience={early_stopping.counter}/{patience})")
                    break

            total_time = time.time() - t_start
            self.load(model_dir=model_save_dir, mode="best")
            self._log(
                f"Training complete. Best val_loss: {self.best_loss:.4f} | "
                f"Total time: {total_time / 60:.1f}min",
            )
        finally:
            from .utils import ddp_cleanup
            del self._ddp_wrapped
            self._ddp_wrapped = None
            ddp_cleanup()

    # ── Evaluation ───────────────────────────────────────────

    def evaluate(
        self, data_loader, verbose: bool = False
    ) -> tuple[float, dict[str, float]]:
        self.eval()
        total_loss = 0.0
        metric_results = {name: [] for name in self.metrics}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False, disable=not self.is_main):
                if self._use_amp:
                    with torch.amp.autocast("cuda"):
                        loss, step_metrics = self._eval_step(batch)
                else:
                    loss, step_metrics = self._eval_step(batch)
                total_loss += loss.item()
                for name, val in step_metrics.items():
                    metric_results[name].append(val)

        avg_loss = total_loss / len(data_loader)
        avg_loss = self._ddp_all_reduce(avg_loss)

        avg_metrics = {
            name: round(self._ddp_all_reduce(float(np.mean(vals))), 4)
            for name, vals in metric_results.items()
        }

        if verbose:
            self._log(f"Val Loss: {avg_loss:.4f} | Metrics: {avg_metrics}")
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

    def save(self, model_dir: str = "./checkpoints", mode: str = "best"):
        if not self.is_main:
            return
        os.makedirs(model_dir, exist_ok=True)

        save_path = os.path.join(model_dir, self._filename(mode))
        save_dict = {
            "epoch": self.last_epoch,
            "model_state_dict": self.state_dict(),
            "best_loss": self.best_loss,
        }
        if self.optimizer is not None:
            save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        if self._grad_scaler is not None:
            save_dict["scaler_state_dict"] = self._grad_scaler.state_dict()

        self._atomic_save(save_dict, save_path)

    def _save_weights(self, model_dir: str):
        if not self.is_main:
            return
        path = os.path.join(model_dir, self._weights_filename())
        self._atomic_save(self.state_dict(), path)

    def load(
        self,
        model_dir: str = "./checkpoints",
        mode: str = "best",
        specified_path: str | None = None,
    ):
        load_path = specified_path or os.path.join(model_dir, self._filename(mode))

        if not os.path.exists(load_path):
            self._log(f"No checkpoint at {load_path}, starting from scratch")
            self.last_epoch = 0
            self.best_loss = None
            return

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.last_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", None)

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self._grad_scaler is not None and "scaler_state_dict" in checkpoint:
            self._grad_scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self._log(
            f"Loaded from {load_path}, epoch={self.last_epoch}, best_loss={self.best_loss}",
        )

    def load_weights(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        self._log(f"Loaded weights from {path}")

    def transfer(self, specified_path: str, strict: bool = False):
        if not specified_path:
            raise ValueError("Transfer path not specified.")
        if not os.path.exists(specified_path):
            raise FileNotFoundError(f"Transfer path not found: {specified_path}")

        self._log(f"Transferring from {specified_path}")
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
            f"Transfer: {len(new_state)} matched, {len(missing)} missing, "
            f"{len(size_mismatch)} size mismatch",
        )
        if missing:
            self._log(f"Missing: {missing}")
        if size_mismatch:
            self._log(f"Size mismatch: {size_mismatch}")

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
