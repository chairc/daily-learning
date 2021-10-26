#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from time import time

# 矢量计算表达式
print('********矢量计算表达式********')
# 先定义两个1000维向量
array1 = torch.ones(1000)
array2 = torch.ones(1000)
start = time()
# 将两个向量直接做矢量加法
array4 = array1 + array2
print(time() - start)