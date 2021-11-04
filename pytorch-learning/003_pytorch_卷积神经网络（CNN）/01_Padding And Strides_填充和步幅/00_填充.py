#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

"""
    填充后，输出矩阵 = （输入矩阵的行高 - 卷积核的行高 + 填充的行数 + 1）*（输入矩阵的列宽 - 卷积核的列宽 + 填充的列数 + 1）
    通常来说，填充的行数和列数为卷积核的行高和列宽 - 1，那么代入上面的式子会发现输出和输入的形状不会变化
    当卷积核的行高为奇数时，在上下两侧填充（填充的行数/2）
    当卷积核的行高为偶数时（很少使用），在上侧填充（填充的行数/2，上侧比下侧多一行），在下侧填充（填充的行数/2，下侧比上侧少一行）


"""


# 定义一个计算卷积层的函数
def comp_conv2d(conv2d, X):
    # 设置批量大小与通道都为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


# 设置卷积核为3*3，填充，每边都填充1行1列所以总填充数为2行2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
print(f'conv2d.kernel_size  ->  {conv2d.kernel_size}')
# 设置输入矩阵为8*8
X = torch.rand(size=(8, 8))
print(f'X.shape  ->  {X.shape}')
# (8 - 3 + 2 + 1) * (8 - 3 + 2 + 1)
print(f'comp_conv2d(conv2d, X).shape  ->  {comp_conv2d(conv2d, X).shape}')
