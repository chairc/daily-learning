#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.nn import init

# 默认初始化
print('********默认初始化********')
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

print(f'net ->  \n{net}')
X = torch.rand(2, 4)
Y = net(X).sum()

# 访问模型参数
print('********访问模型参数********')
print(f'net.named_parameters类型  ->  {type(net.named_parameters())}')
for name, param in net.named_parameters():
    print(name, param.size())

# 初始化模型参数
print('********初始化模型参数********')
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)

# 自定义初始化方法
print('********自定义初始化方法********')


def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)

# 共享模型参数
print('********共享模型参数********')
linear = nn.Linear(1, 1, bias=False)
net1 = nn.Sequential(linear, linear)
print(f'net1 ->  \n{net1}')

for name, param in net1.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

# 两个线性层是一个对象
print(id(net1[0]) == id(net1[1]))
print(id(net1[0].weight) == id(net1[1].weight))

# 由于模型参数中包含了梯度，在进行反向传播计算时，共享参数梯度是叠加状态
x = torch.ones(1, 1)
y = net1(x).sum()
print(f'y   ->  {y}')
y.backward()
print(f'net1[0].weight.grad  ->  {net1[0].weight.grad}')
