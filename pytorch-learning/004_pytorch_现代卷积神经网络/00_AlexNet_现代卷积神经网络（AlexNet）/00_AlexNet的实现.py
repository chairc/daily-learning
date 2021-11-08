#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from d2lzh_pytorch import torch_d2l as d2l

# 使用Fashion-MNIST的单通道数据集，没使用ImageNet的RGB三通道数据集
# 此处设计与原版有部分区别
net = nn.Sequential(
    # 卷积层C1
    # 使用11*11的卷积核并使用步幅为4来捕捉对象，以减小输出的行高和列宽
    # AlexNet初始输入224*224*3，这里如果按224输入则下面会变为54*54，所以按227输入
    # 由于使用Fashion-MNIST，我们将输入图片的通道设为1，若果为ImageNet则设置为3
    # nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    # 卷积：(227+2-11+4)/4 = 55.5  ->  55*55  ->  55*55*96
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    # 池化：(55-3+2)/2 = 27  ->  27*27*96
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 当前输出：27*27*96

    # 卷积层C2
    # 减小卷积窗口，使用填充为2来使输入和输出的行高和列宽一致，增大输出通道
    # 卷积：(27+4-5+1)/1 = 27  ->  27*27  ->  27*27*256
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    # 池化：(27-3+2)/2 = 13  ->  13*13
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 当前输出：13*13*256

    # 使用三个连续的卷积层和较小的卷积窗口
    # 卷积层C3
    # 卷积：(13+2-3+1)/1 = 13  ->  13*13*384
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    # 当前输出：13*13*384

    # 卷积层C4
    # 卷积：(13+2-3+1)/1 = 13  ->  13*13*384
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    # 当前输出：13*13*384

    # 卷积层C5
    # 卷积：(13+2-3+1)/1 = 13  ->  13*13*256
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    # 当前输出：13*13*256
    # 卷积：(13-3+2)/2 = 6  -> 6*6*256
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 当前输出：6*6*256

    # 全连接层FC6
    nn.Flatten(),
    # 使用dropout减少过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 1000), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Fashion-MNIST输出为10个类别
    nn.Linear(1000, 10)
)

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__}output shape\t->\t{X.shape}')

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
