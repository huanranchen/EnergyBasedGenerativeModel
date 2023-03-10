import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import List, Callable, Tuple
from tqdm import tqdm
from attacks import AdversarialInputAttacker
from copy import deepcopy
from torch import multiprocessing


def test_transfer_attack_acc(attacker: Callable, loader: DataLoader,
                             target_models: List[nn.Module],
                             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> List[float]:
    transfer_accs = [0] * len(target_models)
    denominator = 0
    # count = 0
    # to_img = transforms.ToPILImage()
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        # ori_x = x.clone()
        x = attacker(x, y)
        # temp = to_img(x[0])
        # temp.save(f'./what/mi/4/adv_{count}.png')
        # temp = to_img(ori_x[0])
        # temp.save(f'./what/mi/4/ori_{count}.png')
        # temp = to_img(x[0]-ori_x[0])
        # temp.save(f'./what/mi/4/perturb_{count}.png')
        # count += 1
        with torch.no_grad():
            denominator += x.shape[0]
            for i, model in enumerate(target_models):
                pre = model(x)  # N, D
                pre = torch.max(pre, dim=1)[1]  # N
                transfer_accs[i] += torch.sum(pre == y).item()

    transfer_accs = [1 - i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print('-' * 100)
        print(model.__class__,  model.model.__class__, transfer_accs[i])
        print('-' * 100)
    return transfer_accs


def test_autoattack_acc(model: nn.Module, loader: DataLoader):
    from autoattack import AutoAttack
    adversary = AutoAttack(model, eps=8 / 255)
    xs, ys = [], []
    for x, y in tqdm(loader):
        xs.append(x)
        ys.append(y)
    x = torch.concat(xs, dim=0)
    y = torch.concat(ys, dim=0)
    adversary.run_standard_evaluation(x, y)


def test_transfer_attack_acc_with_batch(get_attacker: Callable,
                                        batch_x: torch.tensor,
                                        batch_y: torch.tensor,
                                        get_target_models: Callable,
                                        batch_size: int = 1,
                                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> List[
    float]:
    attacker = get_attacker()
    target_models = get_target_models()
    transfer_accs = [0] * len(target_models)
    denominator = 0
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    xs = list(torch.split(batch_x, batch_size, dim=0))
    ys = list(torch.split(batch_y, batch_size, dim=0))
    attacker.to(device)

    for model in target_models:
        model.to(device)
    for x, y in tqdm(zip(xs, ys)):
        x = attacker(x, y)
        with torch.no_grad():
            denominator += x.shape[0]
            for i, model in enumerate(target_models):
                pre = model(x)  # N, D
                pre = torch.max(pre, dim=1)[1]  # N
                transfer_accs[i] += torch.sum(pre == y).item()

    transfer_accs = [1 - i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print('-' * 100)
        print(model.__class__, transfer_accs[i])
        print('-' * 100)
    return transfer_accs


def test_transfer_attack_acc_distributed(get_attacker: Callable,
                                         loader: DataLoader,
                                         get_target_models: Callable,
                                         batch_size: int = 1,
                                         num_gpu: int = torch.cuda.device_count()):
    def list_mean(x: list) -> float:
        return sum(x) / len(x)

    print(f'available gpu num {num_gpu}')
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    xs, ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)
    xs, ys = list(torch.split(xs, xs.shape[0] // num_gpu, dim=0)), list(torch.split(ys, ys.shape[0] // num_gpu, dim=0))
    pool = multiprocessing.Pool(processes=num_gpu)
    results = [pool.apply_async(func=test_transfer_attack_acc_with_batch,
                                args=(
                                    get_attacker,
                                    xs[i], ys[i],
                                    get_target_models
                                ),
                                kwds=(
                                    {'batch_size': batch_size,
                                     'device': torch.device(f'cuda:{num_gpu - i - 1}')
                                     }
                                )
                                ) for i in range(num_gpu)
               ]
    pool.close()
    pool.join()
    # print(results)
    # results = [list_mean([results[target_model_id][j] for j in range(len(results))])
    #            for target_model_id in range(len(results[0]))]
    # for i, model in enumerate(target_models):
    #     print('-' * 100)
    # print(model.__class__, model.model.__class__, results[i])
    # print('-' * 100)
    return results
