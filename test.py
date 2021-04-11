import os
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
from authcode.make2 import *
import torch

a = torch.randn(32,45,17)
b = nn.MaxPool2d(kernel_size=2,stride=2)
c =b(a)
print(c.shape)