#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 内置初始化器
print('********内置初始化器********')


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(f'net[0].weight.data[0]   ->  {net[0].weight.data[0]}')
print(f'net[0].bias.data[0]   ->  {net[0].bias.data[0]}')

# 参数初始化为指定常数
print('********参数初始化为指定常数********')


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(f'net[0].weight.data[0]   ->  {net[0].weight.data[0]}')
print(f'net[0].bias.data[0]   ->  {net[0].bias.data[0]}')

# 自定义初始化
print('********自定义初始化********')


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(f'net[0].weight[:2]   ->  \n{net[0].weight[:2]}')
