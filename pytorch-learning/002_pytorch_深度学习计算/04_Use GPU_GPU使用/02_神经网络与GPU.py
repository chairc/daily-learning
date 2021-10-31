#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


X = torch.ones(2, 3, device=try_gpu())
print(f'X    ->  \n{X}')

net = nn.Sequential(
    nn.Linear(3, 1)
)

net = net.to(device=try_gpu())
print(f'net ->  \n{net}')
print(f'net(X) ->  \n{net(X)}')
print(f'net[0].weight.data.device   ->  {net[0].weight.data.device}')
