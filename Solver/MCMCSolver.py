import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
import random


class MCMCSolver():
    def __init__(self,
                 model: nn.Module,
                 sampler: Callable):
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.sampler = sampler

    def train(self,
              loader: DataLoader,
              total_epoch=2000,
              lr=1e-4,
              uncondition_prob=1,
              buffer_size=10000,
              ):
        self.buffer = torch.rand(64, *self.sampler.img_size, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(1, total_epoch + 1):
            pbar = tqdm(loader)
            epoch_loss = 0
            self.model.train()
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.cuda(), y.cuda()
                selected = torch.randint(low=0, high=self.buffer.shape[0] - 1,
                                         size=(round(x.shape[0] * 0.95),))
                unselected = set(list(range(self.buffer.shape[0]))) - set(selected.numpy().tolist())
                unselected = torch.tensor(list(unselected), device=self.device)
                negative_buffer = self.buffer[selected]
                rand_buffer = self.initial_distribution_sample(round(x.shape[0] * 0.05))
                self.buffer = self.buffer[unselected]
                negative = torch.cat([negative_buffer, rand_buffer], dim=0)
                negative = self.sampler(negative)
                self.buffer = torch.cat([self.buffer, negative], dim=0)
                positive = self.model(x)
                negative = self.model(negative)
                # print(torch.mean(positive), torch.mean(negative))
                if random.random() < uncondition_prob:  # uncondition
                    regulation_term = torch.mean(positive ** 2) + torch.mean(negative ** 2)
                    loss = torch.mean(positive - negative)
                    loss = loss + regulation_term
                else:
                    pass  # condition
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            torch.save(self.model.state_dict(), 'model.pth')
            if self.buffer.shape[0] > buffer_size:
                self.buffer = self.buffer[:buffer_size]

    def initial_distribution_sample(self, batch_size):
        # x0 = torch.randn(batch_size, *self.img_size, device=self.device)
        # x0 = x0 * torch.tensor([0.2470, 0.2435, 0.2616], device=self.device).view(1, 3, 1, 1) + \
        #      torch.tensor([0.4914, 0.4822, 0.4465], device=self.device).view(1, 3, 1, 1)
        x0 = torch.rand(batch_size, *self.sampler.img_size, device=self.device)
        return x0
