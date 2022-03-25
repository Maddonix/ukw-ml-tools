import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
# import warnings


def crop_img(img, crop):
    # crop is: ymin, ymax, xmin, xmax
    ymin, ymax, xmin, xmax = crop
    img = img[ymin:ymax, xmin:xmax, :]

    y, x, _ = img.shape
    delta = x-y
    if delta > 0:
        _padding = [(abs(delta), 0), (0, 0), (0, 0)]
        img = np.pad(img, _padding)
    elif delta < 0:
        _padding = [(0, 0), (abs(delta),0), (0, 0)]
        img = np.pad(img, _padding)

    return img

    # no_crop = False
    # mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([img], [0], mask=None, histSize=[100], ranges=[0, 256])
    # n_max_bin = np.argmax(hist)
    # threshold = 255/100 * (n_max_bin+1)
    # if threshold <= 13:
    #     threshold = 13
    # if threshold >= 100:
    #     threshold = 30

    # mask[mask > threshold] = 255
    # mask[mask <= threshold] = 0

    # contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) == 1:
    #     contour = contours[0]
    # elif len(contours) == 0:
    #     no_crop = True
    # else:
    #     contour = max(contours, key=cv2.contourArea)

    # if not no_crop:
    #     x, y, w, h = cv2.boundingRect(contour)
    # else:
    #     x = 0
    #     y = 0
    #     _shape = img.shape

    #     h = _shape[0]
    #     w = _shape[1]

    # img = img[y:y+h, x:x+w]

    # delta = w-h

    # if delta < 0:
    #     # x axis needs to be expanded
    #     _padding = [(0, 0), (abs(delta), 0), (0, 0)]
    # else:
    #     # y axis needs to be expanded
    #     _padding = [(delta, 0), (0, 0), (0, 0)]

    # img = np.pad(img, _padding)

    # return img


img_augmentations = A.Compose([
    # Rotates the image 90, 180 or 270 degrees
    A.RandomRotate90(p = 0.5),
    # Flips the image vertically, horizontally or both
    A.Flip(p = 0.5),
    # Interchanges x and y dimension of the image
    A.Transpose(p = 0.5),
    # Applies a gaussian noise filter with a variability limit of 10 - 50 and mean of 0 
    A.GaussNoise(p = 0.2),
    # Applies one of the included blur algorithms
    A.OneOf([
        A.MotionBlur(),
        A.MedianBlur(blur_limit=3),
        A.Blur(blur_limit=3),
    ], p=0.2),
    # Randomly applies one of: shift image on x or y axis, rescale image, rotate image
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.1),
    # Randomly appplies one of the distortion algorithm
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        A.augmentations.PiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        # Apply contrast limited adaptive histogram equalization
        A.CLAHE(clip_limit=2),
        # Increases image sharpness and overlays it with the original image
        A.Sharpen(),
        # Replaces pixels by highliths and shadows and overlays it with the original image
        A.Emboss(),
        # Applies random brightness and contrast values
        A.RandomBrightnessContrast()           
    ], p=0.3),
    # Randomly shift hue and saturation of the image
    # A.HueSaturationValue(p=0.3),
    # Randomly cut out an image section
    # A.Cutout(p = 0.1)
])

img_transforms = A.Compose([
    A.Normalize(
        mean=(0.45211223, 0.27139644, 0.19264949),
        std=(0.31418097, 0.21088019, 0.16059452),
        max_pixel_value=255)
])

class BinaryImageClassificationDS(Dataset):
    def __init__(self, paths, labels, crop, scaling: int = 75, training: bool = True, skip_augmentation = False, **kwargs):
        self.paths = paths
        self.scaling = scaling
        self.crop = crop
        self.training = training
        self.skip_augmentation = skip_augmentation
        if "swapaxes" in kwargs:
            self.swapaxes = kwargs["swapaxes"]
        else: self.swapaxes = None

        self.labels = labels
        # assert len(paths) == len(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        width = int(1024 * self.scaling / 100) 
        height = int(1024 * self.scaling / 100)

        img = crop_img(img, self.crop[idx])
        dim = (width, height)
        # FIXME INTER AREA?
        img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.training:
            if not self.skip_augmentation:
                img = img_augmentations(image=img)["image"]

        img = img_transforms(image=img)["image"]
        img = torch.tensor(img)
        img = torch.swapaxes(img, 0, 2)

        return img, self.labels[idx]