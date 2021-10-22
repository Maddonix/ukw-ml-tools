from typing import List
from .binary_image_classification_ds import BinaryImageClassificationDS
from torch.utils.data import Dataset


def load_binary_image_classification_dataset(paths: List, ids: List, scaling: int, training: bool) -> Dataset:
    dataset = BinaryImageClassificationDS(paths, ids, scaling=scaling, training=training)

    return dataset
