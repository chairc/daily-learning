#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


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


def corr2d_multi_in(X, K):
    # 先遍历X和K的第0个维度（通道维度）,x和k是输入通道的矩阵，再加在一起
    return sum(corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    # 迭代K的第0维元素，输入X是个3d的（输入通道，输入矩阵行高，输入矩阵列宽），卷积核K是个4d的（输出通道，输入通道，卷积核行高，卷积核列宽）
    # k是个3d的Tensor，最后将所有的结果叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out_1x1(X, K):
    # 获取输入矩阵的输入通道数，矩阵行高，矩阵列宽
    c_i, h, w = X.shape
    # 获取卷积核K的第一个参数输出通道数
    c_o = K.shape[0]
    # Y = (c_o, c_i) * (c_i, h * w) -> (c_0, h * w)
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


# (3,3,3) 输入通道为3的3*3矩阵
X = torch.normal(0, 1, (3, 3, 3))
# (2,3,1,1) 输出通道为2，输入通道为3的1*1矩阵
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
# 1e-6几乎为0，小于则表示基本上1x1卷积核全连接层完全一样
print(f'float(torch.abs(Y1 - Y2).sum()) < 1e-6  ->  {float(torch.abs(Y1 - Y2).sum()) < 1e-6}')
