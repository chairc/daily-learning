#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

"""
    互相关运算就是卷积窗口从输入张量的左上角开始，从左到右、从上到下滑动。 当卷积窗口滑动到新一个位置时，
    包含在该窗口中的部分张量与卷积核张量进行按元素相乘，得到的张量再求和得到一个单一的标量值，由此我们得出了这一位置的输出张量值。
    详细实现过程可参考corr2d这个方法
"""


# 计算二维互相关运算
def corr2d(X, K):
    # 获取卷积核K的形状，也就是获取高和宽（行和列）
    h, w = K.shape
    # 获取输出层的矩阵格式（输入的行高 - 卷积核的行高 + 1，输入的列宽 - 卷积核的列宽 + 1）
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 遍历输出层进行乘积求和
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 输出矩阵使用哈达玛积每个元素求和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(f'corr2d  ->  \n{corr2d(X, K)}')


# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
