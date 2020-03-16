import os
import cv2
import PIL
import torch
import pandas
import torchvision

import numpy as np

import matplotlib.pyplot as plt

from torchvision import transforms

class GaussianBlur(object):

    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max

        self.kernel_size = kernel_size

    def __call__(self, sample):

        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class SimCLRDataTransform(object):

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):

        xi = self.transform(sample)
        xj = self.transform(sample)

        return xi, xj

class SimCLRDataset(torch.utils.data.Dataset):

    def __init__(self, image_root, transforms=None):
        self.image_root = image_root
        self.transforms = transforms
        self.file_list = []
        for root, dirs, files in os.walk(self.image_root):
            for fname in files:
                full_fname = os.path.join(root, fname)
                self.file_list.append(full_fname)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.file_list[idx])
        if self.transforms:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':

    input_shape = (448, 448)
    ratio = 0.8
    batch_size = 32
    num_workers = 4

    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose([
        transforms.Resize(size=input_shape[0]),
        transforms.CenterCrop(size=input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=7),
        transforms.ToTensor()])

    data_augmentation = SimCLRDataTransform(data_transforms)
    dataset = SimCLRDataset(
        image_root='data',
        transforms=data_augmentation)
    idx = np.random.choice(len(dataset))
    (xi, xj) = dataset[idx]
    xi = xi.permute([1, 2, 0]).detach().numpy()
    xj = xj.permute([1, 2, 0]).detach().numpy()
    plt.imsave('xi.png', xi)
    plt.imsave('xj.png', xj)