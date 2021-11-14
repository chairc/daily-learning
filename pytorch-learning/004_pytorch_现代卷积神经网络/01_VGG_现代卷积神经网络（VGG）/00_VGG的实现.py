#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from d2lzh_pytorch import torch_d2l as d2l

"""
    VGG使用可重复使用的卷积块来构建深度卷积神经网络
    不同的卷积块个数和超参数可以得到不同复杂度的变种

"""


# 定义一个VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        ))
        layers.append(nn.ReLU())
        # 下一层的输入通道等于上一层的输出通道
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 5个卷积块，第1、2个块有1个卷积层，第3、4、5个块有2个卷积层，一共有8个卷积层，加上3个全连接层，一共有11层结构，称作VGG-11
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1
    # 卷积层部分，把VGG块拼接到一块
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blocks, nn.Flatten(),
        # 全连接层
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


net = vgg(conv_arch)

print(f'net  ->  \n{net}')

X = torch.randn(size=(1, 1, 224, 224))
for block in net:
    X = block(X)
    print(f'{block.__class__.__name__}output shape\t->\t{X.shape}')

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
