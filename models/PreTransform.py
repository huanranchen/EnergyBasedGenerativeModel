import torch
from torchvision import transforms


def cifar10_normalize():
    return transforms.Normalize(((0.4914, 0.4822, 0.4465)), (0.2470, 0.2435, 0.2616))
