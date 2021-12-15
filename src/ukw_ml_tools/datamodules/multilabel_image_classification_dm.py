from typing import Optional
from typing import Tuple
from ..classes.train_data import TrainData

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from ..datasets.binary_image_classification_ds import BinaryImageClassificationDS


class MultilabelImageClassificationDM(LightningDataModule):

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
        train_data: TrainData,
        scaling: int,
        test_size: Tuple[float, float, float] = 0.2,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.train_data_object = train_data

        # self.paths = np.array([_.as_posix() for _ in train_data.paths])
        # self.labels = np.array(train_data.labels)
        self.scaling = scaling
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = train_data.n_classes()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        # sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
        # train_indexes, test_indexes = [_ for _ in sss.split(
        #     self.train_data_object.paths,
        #     self.train_data_object.labels
        # )][0]

        # x_train, x_test, y_train, y_test = self.train_data_object.group_strat_split(
        #     test_size=self.test_size)
        labels = np.array(self.train_data_object.labels)
        paths = np.array([_.as_posix() for _ in self.train_data_object.paths])
        is_val = np.array(self.train_data_object.is_val)

        x_train = paths[~is_val]
        x_test = paths[is_val]
        y_train = labels[~is_val]
        y_test = labels[is_val]


        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        y_train = torch.from_numpy(y_train).long()

        self.train_dataset = BinaryImageClassificationDS(
            x_train, y_train, self.scaling
        )

        self.test_dataset = BinaryImageClassificationDS(
            x_test, y_test, self.scaling
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler = self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         shuffle=False,
    #     )
