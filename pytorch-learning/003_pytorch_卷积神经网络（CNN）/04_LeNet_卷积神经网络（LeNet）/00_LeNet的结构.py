#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


net = nn.Sequential(
    Reshape(),
    # 输入通道为1，输出通道为6的卷积层   (28+4-5+1)*(28+4-5+1) = 28*28
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # 平均池化层2x2，移动不重叠   28/2 = 14
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 输入通道为6，输出通道为16的卷积层  (14-5+1)*(14-5+1) = 10*10
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # 平均池化层2x2，移动不重叠   10/2 = 5
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 将四维卷积层变为一维向量
    nn.Flatten(),
    # 16个输出再乘以5*5的池化层转为向量为16*5*5 = 400
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    # 输出0到9
    nn.Linear(84, 10)
)

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__}output shape\t->\t{X.shape}')
