from ukw_ml_tools.datasets.binary_image_classification_ds import img_augmentations, img_transforms
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

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
        self, paths, labels, train, prediction = False, finetune = False
    ):
        self.paths = paths
        self.labels = labels
        self.train = train
        self.prediction = prediction
        self.target_x = 1920
        self.target_y = 1920
        self.finetune = finetune

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        # print(img.shape)
        pad_y = self.target_y - img.shape[0]
        pad_x = self.target_x - img.shape[1]
        if self.finetune:
            _dim = 1920
            img = image_resize(img, width = _dim)
            pad_y = _dim - img.shape[0]
            pad_x = _dim - img.shape[1]
            img = np.pad(img, [[0,pad_y], [0,pad_x], [0,0]])
            
        else:
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