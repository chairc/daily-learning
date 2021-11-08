#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

"""
    步幅是指行/列滑动的步长，可以成倍减少输出形状
    Nh为输入的行高，Nw为输入的列宽，Kh为卷积核的行高，Kw为卷积核的列宽，Pw为填充的行数，Ph为填充的列数
    ⌊⌋表示向下取整
    给定步幅高度Sh,和宽度Sw的步幅，输出形状是
    ⌊(Nh - Kh+ Ph +Sh) / Sh⌋ * ⌊(Nw - Kw + Pw+ Sw) / Sw⌋
    如果Ph = Kh-1，Pw= Kw - 1
    ⌊(Nh + Sh - 1)/s⌋ * ⌊(Nw + Sw - 1)/Sw⌋
    如果输入高度和宽度可以被步幅整除(Nh / Sh) * (Nw / Sw)

"""


# 定义一个计算卷积层的函数
def comp_conv2d(conv2d, X):
    # 设置批量大小与通道都为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


print('********简单步幅********')
conv2d_1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(f'conv2d_1.kernel_size  ->  {conv2d_1.kernel_size}')
# 设置输入矩阵为8*8
X = torch.rand(size=(8, 8))
print(f'X.shape  ->  {X.shape}')
# (8 - 3 + 2 + 2)/2 * (8 - 3 + 2 + 2)/2 = [4,4]
print(f'comp_conv2d(conv2d_1,X).shape  ->  {comp_conv2d(conv2d_1, X).shape}')

print('********复杂步幅********')
conv2d_2 = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(f'conv2d_2.kernel_size  ->  {conv2d_2.kernel_size}')
# (8 - 3 + 0 + 3)/3 * (8 - 5 + 2 + 4)/4 = [2,2]
print(f'comp_conv2d(conv2d_2,X).shape  ->  {comp_conv2d(conv2d_2, X).shape}')
