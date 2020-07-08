import os

import numpy as np
import torch
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
from torchvision import transforms

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, 'data')
# can define different image transformations for pre-processing

transformations = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
"""
transformations = transforms.Compose([
    # transforms.RandomResizedCrop(224),

    transforms.ToTensor(),
])

"""
mnist_trainset = dataset.MNIST(
    root=DATA_DIR, train=True, download=True, transform=transformations)
mnist_testset = dataset.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transformations)

# size 10 of the classes here


def one_hot_encoding(vector):
    t = np.zeros((len(vector), 10), dtype=int)
    t[np.arange(len(vector)), vector] = 1
    return t
