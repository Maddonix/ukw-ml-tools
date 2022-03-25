import cv2
from torch.utils.data import Dataset
import numpy as np
from .binary_image_classification_ds import img_augmentations, img_transforms
import torch

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

class ClassificationDataset(Dataset):
    def __init__(
        self, paths, labels, train, crop, prediction = False, target_dim = 800
    ):
        self.paths = paths
        self.labels = labels
        self.train = train
        self.prediction = prediction
        self.target_x = target_dim
        self.target_y = target_dim
        self.crop = crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        ymin, ymax, xmin, xmax = self.crop[idx]
        img = img[ymin:ymax, xmin:xmax, :]
        if img.shape[1] >= img.shape[0]:
            img = image_resize(img, width = self.target_x)
        else:
            img = image_resize(img, height = self.target_y)

        pad_y = self.target_y - img.shape[0]
        pad_x = self.target_x - img.shape[1]
        img = np.pad(img, [[0,pad_y], [0,pad_x], [0,0]])
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train:
            img = img_augmentations(image=img)["image"]

        img = img_transforms(image=img)["image"]
        img = torch.tensor(img)
        img = torch.swapaxes(img, 0, 2)

        if self.prediction:
            label = self.labels[idx]
        else:
            label = torch.Tensor(self.labels[idx])

        return img, label


def get_dataset(images):
    image_ids = [str(_["_id"]) for _ in images]
    paths = [_["metadata"]["path"] for _ in images]
    crop = [_["metadata"]["crop"] for _ in images]
    ds = ClassificationDataset(paths, image_ids, crop = crop,train = False, prediction = True)

    return ds