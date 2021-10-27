#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

# 不含模型参数的自定义层
print('********不含模型参数的自定义层********')


class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        # x值减去x均值
        return x - x.mean()


# 实例化CenteredLayer，做向前计算
centered_layer = CenteredLayer()
# 均值=(1.+2.+3.+4.+5.)/5.=3.，输出为[-2.,-1.,0.,1.,2.]
print(centered_layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

# 构造一个更复杂的模型
net1 = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
print(f'net1 ->  \n{net1}')

y = net1(torch.rand(4, 8))
print(f'y.shape    ->  {y.shape}')
print(f'y.mean().item() ->  {y.mean().item()}')
