from .SmallResolutionModel.resnet import resnet32
from .SmallResolutionModel import Wide_ResNet, IGEBM
import torch
from torch import nn
from .PreTransform import cifar10_normalize
from torch import Tensor


class UnconditionalResNet32(nn.Module):
    def __init__(self):
        super(UnconditionalResNet32, self).__init__()
        # self.transform = cifar10_normalize()
        # self.cnn = Wide_ResNet(num_classes=1)
        # self.cnn = resnet32(num_classes=1)
        self.cnn = IGEBM()

    def forward(self, x: Tensor) -> Tensor:
        # x = self.transform(x)
        x = (x - 0.5) * 2
        x = self.cnn(x)
        return x
