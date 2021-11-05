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


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

# 将卷积核张量K与K中每个元素加1的K+1和K+2连接，构造一个具有3个输出通道的卷积核
K = torch.stack((K, K + 1, K + 2), 0)
# K当前为3个输出通道，2个输入通道的2*2矩阵
print(f'K.shape  ->  {K.shape}')

print(f'corr2d_multi_in_out(X, K)  ->  \n{corr2d_multi_in_out(X, K)}')
