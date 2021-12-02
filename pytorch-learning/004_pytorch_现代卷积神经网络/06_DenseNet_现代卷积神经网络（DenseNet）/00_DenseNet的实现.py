#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from d2lzh_pytorch import torch_d2l as d2l

"""
    DenseNet主要是通过构建稠密块和过渡块
    DenseNet通过过度层来控制网络的维度，减少通道数
"""


def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


# 稠密块体
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                conv_block(
                    num_channels * i + input_channels, num_channels
                )
            )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for block in self.net:
            Y = block(X)
            # 连接通道维度上每个块的输入输出
            X = torch.cat((X, Y), dim=1)
        return X


# 定义有2个输出通道的10的DenseBlock
block1 = DenseBlock(2, 3, 10)
X1 = torch.randn(4, 3, 8, 8)
Y = block1(X1)
print(f'Y.shape  ->  {Y.shape}')


# 过度层
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


# 对稠密块的输出通道10的过渡层，输出通道减为10，高宽均为一半
block2 = transition_block(23, 10)
print(f'Y.shape  ->  {block2(Y).shape}')

# DenseNet模型
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blocks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blocks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一块稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块间加入转换层，并且通道数减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blocks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(
    b1,
    *blocks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10)
)

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
