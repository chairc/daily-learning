#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch

# 判断是否可用CUDA支持的GPU
# device = ('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA是否可用 ->  ', torch.cuda.is_available())
array1 = torch.randn(4, 5)
# 将张量array1移动到GPU上得到张量array2，也可具体到某一个cuda，比如cuda:0、cuda:1
array2 = array1.to('cuda')
print('当前array2 ->\n', array2)
# 显示张量array2中的device信息
print(array2.device)
# 直接在array2的GPU中创建张量array3，张量array3和张量array2的device相同
array3 = torch.randn(5, 6, device=array2.device)
print('当前array3 ->\n', array3)
# CPU中的张量与GPU中的张量不能运算，存在不同GPU的张量也不能运算，同一地方的张量可以运算（当前为4行5列的矩阵乘5行6列的矩阵）
print('当前array2*array3 ->\n', torch.mm(array2, array3))
