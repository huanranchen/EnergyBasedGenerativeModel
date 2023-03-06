from Solver import MCMCSolver
from sampler import EnergyBasedLangevinDynamicSampler
from data import get_CIFAR10_train
from models import UnconditionalResNet32

loader = get_CIFAR10_train(batch_size=256)
model = UnconditionalResNet32()
sampler = EnergyBasedLangevinDynamicSampler(model)
solver = MCMCSolver(model, sampler)
solver.train(loader)