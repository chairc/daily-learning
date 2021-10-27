#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch

# 显示化指定array1可导
array1 = torch.randn(2, 3, requires_grad=True)
print('开启requires_grad当前array1   ->\n', array1)
# 获取array1的requires_grad属性
print(array1.requires_grad)
# array1的2次幂
array2 = array1 ** 2
print('当前array2   ->\n', array2)
print(array2.grad_fn)
# 关闭array1的可导属性
array1.requires_grad = False
print(array1.requires_grad)
print('关闭requires_grad当前array1   ->\n', array1)

