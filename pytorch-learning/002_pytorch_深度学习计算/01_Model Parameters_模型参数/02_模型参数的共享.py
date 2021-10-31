#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
"""
    参数共享可以节省内存
    对于图像识别中的CNN，共享参数使网络能够在图像中的任何地方而不是仅在某个区域中查找给定的功能。
    对于RNN，它在序列的各个时间步之间共享参数，因此可以很好地推广到不同序列长度的示例。
    对于自动编码器，编码器和解码器共享参数。 在具有线性激活的单层自动编码器中，共享权重会在权重矩阵的不同隐藏层之间强制正交。
"""
X = torch.rand(2, 4)
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 1)
)

print(f'net  ->  \n{net}')
print(f'net(X)  ->  \n{net(X)}')
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保是同一个对象，不是只有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
