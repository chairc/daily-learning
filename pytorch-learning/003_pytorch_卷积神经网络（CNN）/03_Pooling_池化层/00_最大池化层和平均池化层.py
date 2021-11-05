#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

"""
    池化层返回窗口中最大或平均值
    缓解卷积层会位置的敏感性
    同样有窗口大小、填充、和步幅作为超参数


"""


# 池化层的正向传播
def pool2d(X, pool_size, mode='max'):
    # 获取池化窗口的行高和列宽
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 最大池化
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            # 平均池化
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


# 验证二维最大池化层输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(f'pool2d(X, (2, 2))  ->  \n{pool2d(X, (2, 2))}')

# 验证二维平均池化层输出
print(f"""pool2d(X, (2, 2),'avg')  ->  \n{pool2d(X, (2, 2), 'avg')}""")
