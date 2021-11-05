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


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(f'corr2d_multi_in(X, K)  ->  \n{corr2d_multi_in(X, K)}')
