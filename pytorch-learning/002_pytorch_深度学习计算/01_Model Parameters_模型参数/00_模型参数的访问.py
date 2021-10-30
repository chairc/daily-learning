#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
X = torch.rand(2, 4)
print(f'net ->  \n{net}')

# 访问参数，例如检查第二个全连接层，这个全连接层包括两个参数，权重和偏置
print('********访问参数，例如检查第二个全连接层，这个全连接层包括两个参数，权重和偏置********')
print(f'第二个全连接层参数   ->  \n{net[2].state_dict()}')

# 参数偏置的类型
print('********参数偏置的类型********')
print(f'type(net[2].bias)   ->  {type(net[2].bias)}')

# 参数偏置
print('********参数偏置********')
print(f'net[2].bias   ->  {net[2].bias}')

# 参数偏置的数据
print('********参数偏置的数据********')
print(f'net[2].bias.data   ->  {net[2].bias.data}')

# 访问每个参数的梯度，因为还没设置网络的反向传播，所以梯度暂时为None，处于初始状态
print('********访问每个参数的梯度，因为还没设置网络的反向传播，所以梯度暂时为None，处于初始状态********')
print(f'net[2].weight.grad == None  ->  {net[2].weight.grad == None}')
print(f'net[2].bias.grad == None  ->  {net[2].bias.grad == None}')

# 一次性访问所有参数
print('********一次性访问所有参数********')
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 从嵌入块中收集参数
print('********从嵌入块中收集参数********')


def block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )


def block2():
    net1 = nn.Sequential()
    for i in range(4):
        # 在net1中添加block1()
        net1.add_module(f'block {i}', block1())
    return net1


# 组合之后嵌套的网络rgnet中有4个block1的Sequential和一个Linear
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(f'rgnet   ->  \n{rgnet}')
print(f'rgnet(X)   ->  \n{rgnet(X)}')

# 访问嵌套块中的参数
print('********访问嵌套块中的参数********')
print(f'访问第一个主块中的第二个子块的第一层偏置    ->  {rgnet[0][1][0].bias.data}')
