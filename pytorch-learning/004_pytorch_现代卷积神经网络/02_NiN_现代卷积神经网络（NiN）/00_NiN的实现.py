#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from d2lzh_pytorch import torch_d2l as d2l

"""
    VGG和NiN的块之间主要结构差异：
        NiN块以一个普通卷积层开始，后面是两个1×1的卷积层。
        这两个1×1卷积层充当带有ReLU激活函数的逐像素全连接层。
        第一层的卷积窗口形状通常由用户设置。随后的卷积窗口形状固定为1×1。
    NiN：网络中的网络
    NiN架构：
        1.无全连接层
        2.交替使用NiN块和步幅为2的最大池化层，逐步减小高宽和增大通道数
        3.最后使用全局平均池化层得到输出，其输入通道数是类别数
    NiN总结：
        NiN块使用卷积层加两个1xl卷积层,后者对每个像素增加了非线性性
        NiN使用全局平均池化层来替代VGG和AlexNet中的全连接层，不容易过拟合，更少的参数个数
"""


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


net = nn.Sequential(
    # 向下取整输出：(224-11+4)/4 = 54.25  ->  54*54  ->  1,96,54,54
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    # 每次通道数不变，行高和列宽减半，向下取整输出：(54-3+2)/2 = 26.5  ->  26*26  ->  26*26*96
    nn.MaxPool2d(3, stride=2),
    # 向下取整输出：(26+4-5+1)/1 = 26  ->  26*26  ->  96,256,26,26
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    # 每次通道数不变，行高和列宽减半，向下取整输出：(26-3+2)/2 = 12  ->  12*12  ->  12*12*256
    nn.MaxPool2d(3, stride=2),
    # 向下取整输出：(12+2-3+1)/1 = 12  ->  12*12  ->  256,384,12,12
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    # 每次通道数不变，行高和列宽减半，向下取整输出：(12-3+2)/2 = 5  ->  5*5  ->  5*5*384
    nn.MaxPool2d(3, stride=2),
    # 防止过拟合
    nn.Dropout(0.5),
    # fashion_MNIST的输出样本为10
    # 向下取整输出：(5+2-3+1)/1 = 5  ->  5*5  ->  384,10,5,5
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 全局平均池化层
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    nn.Flatten())

print(f'net  ->  \n{net}')

# size(批量=1,灰度=1,图片行高=224,图片列宽=224)
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__}output shape\t->\t{X.shape}')

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
