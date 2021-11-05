#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(f'X  ->  \n{X}')
X = torch.cat((X, X + 1), 1)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(f'pool2d(X)  ->  \n{pool2d(X)}')
