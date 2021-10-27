#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch

# 一元操作
array1 = torch.randn(3, 4)
print('当前array1 ->\n', array1)

# 绝对值操作
print('********绝对值操作********')
print(torch.abs(array1))

# 取e指数操作
print('********取e指数操作********')
print(torch.exp(array1))

# sigmoid函数值
print('********sigmoid函数值********')
print(torch.sigmoid(array1))

# 截断元素[0,1]之间
print('********截断元素[0,1]之间********')
print(torch.clamp(array1, min=0, max=1))

print('\n\n')

# 二元操作
array2 = torch.tensor([[1, 2], [3, 4]])
array3 = torch.tensor([[5, 6], [7, 8]])
print('当前array2 ->  \n', array2)
print('当前array3 ->  \n', array3)

# 逐元素加法，可以写为array2+array3或者torch.add(array2,array3)
print('********逐元素加法array2+array3********')
print(torch.add(array2, array3, out=None))

print('********逐元素加法array2+10********')
print(torch.add(array2, 10, out=None))

# 逐元素乘法，可以写为array2*array3或者torch.mul(array2,array3)
print('********逐元素乘法array2*array3********')
print(torch.mul(array2, array3))

# 矩阵乘法
print('********矩阵乘法********')
print(torch.mm(array2, array3))

print('\n\n')

# 张量操作总结值，如：最大指标，平均值，方差，求和...
array4 = torch.randn(3, 4)
print('当前array4 ->  \n', array4)

# 全局最大值指标
print('********全局最大值指标********')
print(torch.argmax(array4))

# 按照dim=1维度，即按照行求最大值指标（矩阵的每一行找最大值，输出则为每一行最大值的位置）
print('********按照行求最大值指标********')
print(torch.argmax(array4, dim=1))

# 按照dim=0维度，即按照列求最大值指标（矩阵的每一列找最大值，输出则为每一列最大值的位置）
print('********按照列求最大值指标********')
print(torch.argmax(array4, dim=0))

print('********求和********')
print(torch.sum(array4))

print('********按行求和********')
print(torch.sum(array4, dim=1))

print('********按列求和********')
print(torch.sum(array4, dim=0))

print('\n\n')

# 比较操作
"""
比较算子     符号     中文解释
    eq  ->  ==      等于
    gt  ->  >       大于
    lt  ->  <       小于
    ne  ->  !=      不等于
    ge  ->  >=      大于等于
    le  ->  <=      小于等于
    equal           
    kthvalue
    max
    min
    sort
    topk
"""
array5 = torch.randn(3, 4)
print('当前array5 -> \n', array5)

# array5矩阵中的元素大于等于1，可以写为torch.ge(array5,1)或是array5>=1
print('********大于等于********')
print(torch.ge(array5, 1))  # array5>=1

# 返回最大值和对应的指标，按行比较（第一个返回最大值，第二个返回最大值位置）
print('********返回最大值和对应的指标********')
print(torch.max(array5, dim=1))

# 返回按照dim=1（行）最大k=2（从大到小k个）的值和指标（第一个返回为k个从大到小的值，第二个返回为k个从大到小的值的位置）
print('********返回按照dim=1最大k=2的值和指标********')
print(torch.topk(array5, dim=1, k=2))

# 返回按照dim=0（列）最大k=2（从大到小k个）的值和指标
print('********返回按照dim=0最大k=2的值和指标********')
print(torch.topk(array5, dim=0, k=2))

print('\n\n')

# 其他运算操作
array6 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('当前array6 -> \n', array6)

# 按照dim=1（行）进行累加
print('********按照dim=1（行）进行累加********')
"""
    a1,a1+a2,a1+a2+a3
    b1,b1+b2,b1+b2+b3
"""
print(torch.cumsum(array6, dim=1))

# 按照dim=1（行）进行累乘
print('********按照dim=1（行）进行累乘********')
"""
    a1,a1*a2,a1*a2*a3
    b1,b1*b2,b1*b2*b3
"""
print(torch.cumprod(array6, dim=1))

# 按照dim=0（列）进行累加
print('********按照dim=0（列）进行累加********')
"""
    a1      a2      a3
    a1+b1   a2+b2   a3+b3
"""
print(torch.cumsum(array6, dim=0))

# 按照dim=0（列）进行累乘
print('********按照dim=0（列）进行累乘********')
"""
    a1      a2      a3
    a1*b1   a2*b2   a3*b3
"""
print(torch.cumprod(array6, dim=0))

print('\n\n')

array7 = torch.randn(3, 3)
print('当前array7 -> \n', array7)
# 对角化操作
print('********对角化操作********')
# 如果diag中为矩阵时，则取对角化
print(torch.diag(array7))
# 如果diag中为向量时，则对角化操作
print(torch.diag(torch.tensor([1, 2, 3])))

# 取下三角矩阵
print('********取下三角矩阵********')
print(torch.tril(array7))

# 求迹（线性代数）
print('********求迹********')
print(torch.trace(array7))

print('\n\n')

# 特殊运算子

array8 = torch.randn(2, 3)
array9 = torch.randn(3, 4)
print('当前array8 -> \n', array8)
print('当前array9 -> \n', array9)

# 代表按照j维度进行求和，相当于c_ij=∑_j a_ij*b_jk
print('********代表按照j维度进行求和********')
print(torch.einsum('ij,jk->ik', array8, array9))
