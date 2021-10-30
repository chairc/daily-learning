#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn


# 两个参数：weight权重，bias偏移项
# in_units输入数量，units输出数量
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
print(f'linear.weight   ->  \n{linear.weight}')

# 使用自定义层直接进行正向传播
print(f'执行正向传播计算    ->  \n{linear(torch.rand(2, 5))}')

# 自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(f'net ->  \n{net}')
print(f'net(torch.rand(2,64))   ->  \n{net(torch.rand(2, 64))}')
