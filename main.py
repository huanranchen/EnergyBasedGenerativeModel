from Solver import MCMCSolver
from sampler import EnergyBasedLangevinDynamicSampler
from data import get_CIFAR10_train
from models import UnconditionalResNet32
import torch
from torchvision import transforms

to_img = transforms.ToPILImage()

loader = get_CIFAR10_train(batch_size=256)
model = UnconditionalResNet32().cuda().eval()
# model.load_state_dict(torch.load('model.pth'))

sampler = EnergyBasedLangevinDynamicSampler(model)
solver = MCMCSolver(model, sampler)
solver.train(loader)

# x = torch.rand(1, 3, 32, 32).cuda()
# x = sampler.sample(x, step=6000)
# x = to_img(x.squeeze())
# x.save('test.png')
