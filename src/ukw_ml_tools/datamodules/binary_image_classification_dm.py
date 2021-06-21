from typing import Optional
from typing import Tuple

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from ..datasets.binary_image_classification_ds import BinaryImageClassificationDS


class BinaryImageClassificationDM(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str,
        scaling: int,
        num_classes: int,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_df = pd.read_csv(data_dir, index_col=0)
        self.file_paths = self.data_df["file_path"].tolist()
        self.labels = self.data_df["label"].tolist()
        self.scaling = scaling
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = num_classes

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        # print(paths)

        dataset = BinaryImageClassificationDS(self.file_paths, self.labels, self.scaling)
        total = len(dataset)
        n_train = int(total * self.train_val_test_split[0])
        n_val = int(total * self.train_val_test_split[1])
        n_test = int(total * self.train_val_test_split[2])

        # print(total, n_train, n_va)

        while (n_train + n_val + n_test) < total:
            n_train += 1
        while (n_train + n_val + n_test) > total:
            n_train -= 1
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, (n_train, n_val, n_test)
        )

        print("len train", len(self.data_train))
        print("len val", len(self.data_val))
        print("len test", len(self.data_test))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
