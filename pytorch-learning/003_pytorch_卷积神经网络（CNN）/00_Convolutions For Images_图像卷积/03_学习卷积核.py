#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn


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


# 假设黑色为0，白色为1
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(f'X   ->  \n{X}')

# 构造一个行高为1，列宽为2的卷积核K，1.0和-1.0的原因时在计算哈达玛积的时候可以在分界线的地方得到1或者-1，其余没分界线的地方哈达玛积为0
K = torch.tensor([[1.0, -1.0]])

# 进行二维互相关运算
Y = corr2d(X, K)
print(f'Y   ->  \n{Y}')

# 通过学习由X生成Y的卷积核
# 使用nn自带的Conv2d构造一个卷积层，创建一个1个输出通道（黑白）和卷积核大小为行高为1，列宽为2的卷积核，忽略bias
con2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 使用思维输入输出格式（批量大小、通道、行高、列宽）
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

# 10次迭代
for i in range(10):
    Y_hat = con2d(X)
    # 均方损失
    l = (Y_hat - Y) ** 2
    con2d.zero_grad()
    l.sum().backward()
    # 卷积核迭代（手写），3e-2是手动设置的学习率
    con2d.weight.data[:] -= 3e-2 * con2d.weight.grad
    print(f'batch  ->  {i + 1}, loss  ->  {l.sum():.3f}')

# 卷积核权重张量，所学的卷积核的权重张量与设置的卷积核K = torch.tensor([[1.0, -1.0]])的误差相差很小了
print(f'卷积核权重张量  ->  {con2d.weight.data.reshape((1, 2))}')

print(f'所学卷积核与设置卷积核K误差为  ->  {torch.sub(K, con2d.weight.data.reshape((1, 2)))}')
