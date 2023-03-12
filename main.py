from Solver import MCMCSolver
from sampler import EnergyBasedLangevinDynamicSampler
from data import get_CIFAR10_train, get_CIFAR10_test
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


model.eval()


#
x, _ = next(iter(loader))
x = x[:1].cuda()
print(model(x.cuda()).sum())
x = sampler.sample(x, step=600)
print(model(x.cuda()).sum(), x.shape)

#
x = torch.rand(1, 3, 32, 32).cuda()
print(model(x.cuda()).sum())
x = sampler.sample(x, step=600)
print(model(x.cuda()).sum(), x.shape)
x = to_img(x[0].squeeze())
x.save('test.png')
