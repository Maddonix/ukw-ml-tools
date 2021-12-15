import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
# from .utils import crop_img

def crop_img(img):
    no_crop = False
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], mask = None, histSize = [100], ranges = [0,256])
    n_max_bin = np.argmax(hist)
    threshold = 255/100 * (n_max_bin+1)
    if threshold <= 13:
        threshold = 13
    if threshold >= 100:
        threshold = 30

    mask[mask > threshold] = 255
    mask[mask <=threshold] = 0

    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        contour = contours[0]
    elif len(contours) == 0:
        no_crop = True
    else:
        contour = max(contours, key = cv2.contourArea)

    if not no_crop:
        x,y,w,h = cv2.boundingRect(contour)
    else:
        x = 0
        y = 0
        _shape = img.shape
        
        h = _shape[0]
        w = _shape[1]
        # no_crop = False

    img = img[y:y+h, x:x+w]

    delta = w-h

    if delta < 0:
        # x axis needs to be expanded
        _padding = [(0,0), (abs(delta),0), (0,0)]
    else: 
        # y axis needs to be expanded
        _padding = [(delta,0), (0,0), (0,0)]

    img = np.pad(img, _padding)

    return img




img_augmentations = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.MotionBlur(p=0.3, blur_limit=(10, 20)),
        A.RandomBrightnessContrast(p=0.5),
    ]
)

cropping_large = A.Compose([A.CenterCrop(width=1024, height=1024)])

cropping_small = A.Compose([A.CenterCrop(width=720, height=720)])

img_transforms = transforms.Compose([transforms.ToTensor()])


class BinaryImageClassificationDS(Dataset):
    def __init__(self, paths, labels, scaling: int = 75, training: bool = True):
        self.paths = paths
        self.scaling = scaling
        self.training = training

        self.labels = labels
        assert len(paths) == len(labels)

        self.classes = {0: "negative", 1: "positive"}

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        width = int(1024 * self.scaling / 100)  # img.shape[1]
        height = int(1024 * self.scaling / 100)  # img.shape[0]

        # if img.shape[1] >= 1024 and img.shape[0] >= 1024:
        #     img = cropping_large(image=img)["image"]
        # elif img.shape[1] >= 720 and img.shape[0] >= 720:
        #     img = cropping_small(image=img)["image"]
        img = crop_img(img)
        dim = (width, height)

        # Switch BGR to RGB
        # img = np.flip(img, axis=-1)
        
        if self.training:
            img = img_augmentations(image=img)["image"]

        img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
        img = cv2.normalize(
            img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        if self.training:
            return img_transforms(img), torch.tensor(self.labels[idx])
        else:
            return img_transforms(img), self.labels[idx]

    def __len__(self):
        return len(self.paths)
