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


# 假设黑色为0，白色为1
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(f'X   ->  \n{X}')

# 构造一个行高为1，列宽为2的卷积核K，1.0和-1.0的原因时在计算哈达玛积的时候可以在分界线的地方得到1或者-1，其余没分界线的地方哈达玛积为0
K = torch.tensor([[1.0, -1.0]])

# 进行二维互相关运算
Y = corr2d(X, K)
print(f'Y   ->  \n{Y}')

# 假如我们对这个X进行转置，则不能正确判断
Y_t = corr2d(X.t(), K)
print(f'Y_t   ->  \n{Y_t}')

# 如果我们把这个卷积核K转置
Y_t_2 = corr2d(X.t(), K.t())
print(f'Y_t_2   ->  \n{Y_t_2}')
