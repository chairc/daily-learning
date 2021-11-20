#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from d2lzh_pytorch import torch_d2l as d2l


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 残差块path1中第1个卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        # 残差块path1中第2个卷积层
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # 如果use_1x1conv是True，残差块path2中1个卷积层（用于改变通道数和分辨率，变换相应形状在后进行相加运算），否则为None（将输入添加到输出）
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        # conv1  ->  batch normalization  ->  relu  ->  conv2  ->  batch normalization
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # path1和path2相加运算
        Y += X
        # 再进行relu
        return F.relu(Y)


blk1 = Residual(3, 3)
X1 = torch.rand(4, 3, 6, 6)
Y1 = blk1(X1)
print(f'Y1.shape  ->  {Y1.shape}')

# 增加输出通道，减半输出的高和宽
blk2 = Residual(3, 6, use_1x1conv=True, strides=2)
X2 = torch.rand(4, 3, 6, 6)
Y2 = blk2(X2)
print(f'Y2.shape  ->  {Y2.shape}')

# ResNet模型
# 卷积层：输出通道64 步幅2 7x7卷积核
# 池化层：步幅2 3x3最大池化层
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(
    *resnet_block(64, 64, 2, first_block=True)
)
b3 = nn.Sequential(
    *resnet_block(64, 128, 2)
)
b4 = nn.Sequential(
    *resnet_block(128, 256, 2)
)
b5 = nn.Sequential(
    *resnet_block(256, 512, 2)
)

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__}output shape\t->\t{X.shape}')

# 模型训练
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
