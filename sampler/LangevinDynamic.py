import torch
from torch import nn
from torch import Tensor
import math


class EnergyBasedLangevinDynamicSampler():
    def __init__(self, model: nn.Module, img_size=(3, 32, 32)):
        self.model = model
        self.img_size = img_size
        self.device = torch.device('cuda')

    @torch.enable_grad()
    def get_grad(self, x: Tensor) -> Tensor:
        x.requires_grad = True
        x.grad = None
        target = self.model(x)
        target = target.sum()
        # print(target)
        target.backward()
        grad = x.grad.clone()
        x.grad = None
        x.requires_grad = False
        return grad

    @torch.no_grad()
    def sample(self, x: Tensor, step=60, lam=0.0025, step_size=10):
        for t in range(1, step + 1):
            grad = self.get_grad(x)
            grad = self.clamp(grad, min=-0.03, max=0.03)
            # x = x + step_size * grad + torch.randn_like(x) * lam
            x = x + 10 * grad + torch.randn_like(x) * lam
            # x = x - 1 / 255 * grad.sign() + 1e-4 * torch.randn_like(x)
            # print(grad)
            # print(torch.mean(torch.randn_like(x) * math.sqrt(lam)), torch.mean(lam / 2 * grad))
            x = self.clamp(x)
        return x.detach()

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @staticmethod
    def clamp(x: Tensor, min=0., max=1.) -> Tensor:
        x = torch.clamp(x, min=min, max=max)
        return x
