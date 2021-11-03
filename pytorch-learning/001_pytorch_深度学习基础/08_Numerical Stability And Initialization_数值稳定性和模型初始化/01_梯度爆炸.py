#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch

"""
    值超出值域(infinity)
        对于16位浮点数尤为严重（数值区间6e-5 - 6e4)
    对学习率敏感
        如果学习率太大->大参数值->更大的梯度
        如果学习率太小->训练无进展
        我们可能需要在训练过程不断调整学习率

"""

# 当这种情况是由于深度网络的初始化所导致时，我们没有机会让梯度下降优化器收敛
# 均值为0，方差为1 4行4列矩阵
M = torch.normal(0, 1, size=(4, 4))
print(f'矩阵M ->  \n{M}')
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))
print(f'矩阵M×100个矩阵  ->  \n{M}')
