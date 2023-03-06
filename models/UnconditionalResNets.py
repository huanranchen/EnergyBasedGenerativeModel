from .SmallResolutionModel.resnet import resnet32
import torch
from torch import nn
from .PreTransform import cifar10_normalize
from torch import Tensor


class UnconditionalResNet32(nn.Module):
    def __init__(self):
        super(UnconditionalResNet32, self).__init__()
        self.transform = cifar10_normalize()
        self.cnn = resnet32(num_classes=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform(x)
        x = self.cnn(x)
        return x
