from torch.utils.data import DataLoader

from .utils import is_ddp


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
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.batch_size = batch_size
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
        return DataLoader(self.val_ds, batch_size=1, shuffle=False)
