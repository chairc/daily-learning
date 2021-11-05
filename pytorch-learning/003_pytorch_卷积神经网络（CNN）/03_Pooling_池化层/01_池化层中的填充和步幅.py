#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(f'X  ->  \n{X}')

# 框架中默认步幅和池化窗口大小相同，如果我们使用池化窗口为3，那么步幅为(3,3)
pool2d = nn.MaxPool2d(3)
print(f'pool2d  ->  {pool2d}')
print(f'pool2d(X)  ->  {pool2d(X)}')

# 设置填充为1，步幅为2的，池化窗口为3
pool2d_1 = nn.MaxPool2d(3, padding=1, stride=2)
print(f'pool2d_1  ->  {pool2d_1}')
print(f'pool2d_1(X)  ->  \n{pool2d_1(X)}')

# 设置任意大小的池化窗口，填充和步幅
pool2d_2 = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
print(f'pool2d_2  ->  {pool2d_2}')
print(f'pool2d_2(X)  ->  \n{pool2d_2(X)}')

