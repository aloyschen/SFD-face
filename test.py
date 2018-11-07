import cv2
import numpy as np
import torch
import torch.nn as nn
from anchor import generate_anchors
import config
import torchvision.transforms as transforms
from model import build_SFD

# Detector('./checkpoint.pth.tar').infer('./test.jpg')
a = torch.Tensor([[0.1, 0.2], [0.3, 0.1]])
print(torch.max(a, dim = 1))
c = np.array([[1], [2], [3]])
b = np.array([[1, 1, 1]])
print(c[b])