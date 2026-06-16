import os
import time
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import is_ddp


# ---------------------------------------------------------------------------
# Slice helper
# ---------------------------------------------------------------------------

def _apply_slices(data: np.ndarray, slices_dict: dict) -> np.ndarray:
    full_slices = [slices_dict.get(i, slice(None)) for i in range(data.ndim)]
    return data[tuple(full_slices)]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std < 1e-8:
        return data - mean
    return (data - mean) / std


def denormalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return data * std + mean


def denormalize_tensor(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return tensor * std + mean


# ---------------------------------------------------------------------------
# Stats persistence
# ---------------------------------------------------------------------------

def save_stats(stats: dict, path: str) -> None:
    arr = np.array([
        stats["input"]["mean"], stats["input"]["std"],
        stats["target"]["mean"], stats["target"]["std"],
    ], dtype=np.float32)
    np.save(path, arr)


def load_stats(path: str) -> Dict[str, Dict[str, float]]:
    arr = np.load(path)
    return {
        "input": {"mean": float(arr[0]), "std": float(arr[1])},
        "target": {"mean": float(arr[2]), "std": float(arr[3])},
    }


class DataModule:
    """Base class for data setup. Override setup() to create train_ds / val_ds.

    Usage:
        class MyData(DataModule):
            def setup(self):
                self.train_ds = ...
                self.val_ds = ...

        dm = MyData(batch_size=4, num_workers=2)
        dm.setup()
        train_loader = dm.train_dataloader()  # auto DDP-aware
    """

    def __init__(
        self,
        batch_size: int = 2,
        val_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ds = None
        self.val_ds = None

    def setup(self):
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup() first or set train_ds")
        sampler = None
        shuffle = True
        if is_ddp():
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(self.train_ds, shuffle=True)
            shuffle = False
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup() first or set val_ds")
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False)


class MmapArrayDataset(Dataset):
    """Memory-mapped paired numpy dataset with on-the-fly normalization.

    Avoids loading full arrays into RAM. Stats (mean/std) are computed on the
    mmap'd data; normalization and dtype conversion happen per-sample in
    __getitem__.

    Shapes are expected to be (N, ...) — __getitem__ returns tensors with
    an added channel dimension: (1, ...).
    """

    def __init__(
        self,
        input_path: str,
        target_path: str,
        input_stats: Optional[Dict[str, float]] = None,
        normalize_target: bool = True,
        add_channel_dim: bool = True,
        input_slices: Optional[Dict[int, slice]] = None,
        target_slices: Optional[Dict[int, slice]] = None,
    ):
        s = time.time()
        steps = ["mmap input", "mmap target", "input stats", "target stats"]
        pbar = tqdm(steps, desc="Dataset", unit="step", leave=True)

        inp_size = os.path.getsize(input_path) / 1e9
        pbar.set_postfix_str(f"input ({inp_size:.1f} GB)")
        inp_raw = np.load(input_path, mmap_mode="r")
        if input_slices:
            inp_raw = np.array(_apply_slices(np.asarray(inp_raw), input_slices))
        pbar.update(1)

        tgt_size = os.path.getsize(target_path) / 1e9
        pbar.set_postfix_str(f"target ({tgt_size:.1f} GB)")
        tgt_raw = np.load(target_path, mmap_mode="r")
        if target_slices:
            tgt_raw = np.array(_apply_slices(np.asarray(tgt_raw), target_slices))
        pbar.update(1)

        pbar.set_postfix_str("computing...")
        if input_stats is not None:
            self.input_mean = input_stats["mean"]
            self.input_std = input_stats["std"]
        else:
            self.input_mean = float(np.mean(inp_raw))
            self.input_std = float(np.std(inp_raw))
        pbar.update(1)

        self.normalize_target = normalize_target
        pbar.set_postfix_str("computing...")
        if normalize_target:
            self.target_mean = float(np.mean(tgt_raw))
            self.target_std = float(np.std(tgt_raw))
        else:
            self.target_mean = 0.0
            self.target_std = 1.0
        pbar.update(1)

        self._inp_raw = inp_raw
        self._tgt_raw = tgt_raw
        self._need_cast = (inp_raw.dtype != np.float32) or (tgt_raw.dtype != np.float32)
        self._add_channel = add_channel_dim

        elapsed = time.time() - s
        inp_shape = "x".join(str(d) for d in inp_raw.shape)
        tgt_shape = "x".join(str(d) for d in tgt_raw.shape)
        pbar.set_postfix_str(f"in[{inp_shape}] tgt[{tgt_shape}] {elapsed:.1f}s")
        pbar.close()

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        return {
            "input": {"mean": self.input_mean, "std": self.input_std},
            "target": {"mean": self.target_mean, "std": self.target_std},
        }

    def __len__(self) -> int:
        return self._inp_raw.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = self._inp_raw[idx]
        tgt = self._tgt_raw[idx]
        if self._need_cast:
            inp = inp.astype(np.float32)
            tgt = tgt.astype(np.float32)
        inp = (inp - self.input_mean) / self.input_std
        tgt = (tgt - self.target_mean) / self.target_std
        if self._add_channel:
            inp = inp[np.newaxis]
            tgt = tgt[np.newaxis]
        return (
            torch.from_numpy(np.ascontiguousarray(inp)),
            torch.from_numpy(np.ascontiguousarray(tgt)),
        )
