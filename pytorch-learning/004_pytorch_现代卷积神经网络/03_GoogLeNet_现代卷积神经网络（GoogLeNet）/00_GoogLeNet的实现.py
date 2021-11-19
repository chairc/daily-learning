#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from d2lzh_pytorch import torch_d2l as d2l

"""
    InceptionV1版本每一个Inception块中有c1~c4路径：
        c1： 1*1 Conv
        c2： 1*1 Conv，3*3 Conv padding=1
        c3： 1*1 Conv，5*5 Conv padding=2
        c4： 3*3 MaxPooling padding=1,1*1 Conv
"""


class Inception(nn.Module):
    # c1 c2 c3 c4是每一个Inception块中的路径（Inception块中有四条）
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 路径1的第1层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 路径2的第1层与第2层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 路径3的第1层与第2层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 路径4的第1层和第2层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    # 输入通道是192
    # 下一个输出通道c1=64 c2=128 c3=32 c4=32之和为256
    Inception(192, 64, (96, 128), (16, 32), 32),
    # 输入通道是256
    # 下一个输出通道c1=128 c2=192 c3=96 c4=64之和为480
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    # 480
    # 下一个输出通道c1=192 c2=208 c3=48 c4=64之和为512
    Inception(480, 192, (96, 208), (16, 48), 64),
    # 输入通道是512
    # 下一个输出通道c1=160 c2=224 c3=64 c4=64之和为512
    Inception(512, 160, (112, 224), (24, 64), 64),
    # 输入通道是512
    # 下一个输出通道c1=128 c2=256 c3=64 c4=64之和为512
    Inception(512, 128, (128, 256), (24, 64), 64),
    # 输入通道是512
    # 下一个输出通道c1=112 c2=288 c3=64 c4=64之和为528
    Inception(512, 112, (144, 288), (32, 64), 64),
    # 输入通道是528
    # 下一个输出通道c1=256 c2=320 c3=128 c4=128之和为832
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(
    # 输入通道是832
    # 下一个输出通道c1=256 c2=320 c3=128 c4=128之和为832
    Inception(832, 256, (160, 320), (32, 128), 128),
    # 输入通道是832
    # 下一个输出通道c1=384 c2=384 c3=128 c4=128之和为1024
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

# 组合网络模型
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

print(f'net  ->  \n{net}')
X = torch.randn(1, 1, 96, 96)
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__}output shape\t->\t{X.shape}')

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
