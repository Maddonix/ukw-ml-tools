from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
import albumentations as A
import numpy as np

img_augmentations = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.MotionBlur(p=0.3, blur_limit=(10, 20)),
    A.RandomBrightnessContrast(p=0.5)
])

cropping_large = A.Compose([
    A.CenterCrop(width = 1024, height = 1024)
])

cropping_small = A.Compose([
    A.CenterCrop(width = 720, height = 720)
])

img_transforms = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


class ToolDetectionDataset(Dataset):

    def __init__(self, paths, labels, scaling:int = 75, training: bool = True):
        self.paths = paths
        self.scaling = scaling
        self.training = training

        if training:
            self.labels = labels
            assert len(paths) == len(labels)
        
        self.classes = {
            0: "non_tool",
            1: "tool"
        }
        
    def __getitem__(self,idx):
        img = cv2.imread(self.paths[idx])
        width = int(1024 * self.scaling / 100) #img.shape[1]
        height = int(1024 * self.scaling / 100) #img.shape[0]
        if img.shape[1] > 1000:
            img = cropping_large(image=img)["image"]
        else:
            img = cropping_small(image=img)["image"]
        dim = (width, height)
        img = np.flip(img, axis = -1)
        if self.training:
            img = img_augmentations(image = img)["image"]


        img = cv2.resize(img, dsize= dim, interpolation = cv2.INTER_AREA)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if self.training:
            return img_transforms(img), torch.tensor(self.labels[idx])
        else:
            return img_transforms(img), self.paths[idx]
    
    def __len__(self):
        return len(self.paths)