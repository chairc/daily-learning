#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np

"""
    Tensor可以理解为多维数组，高维矩阵，或是numpy中的ndarray
"""

# 初始化一个Tensor
print('********初始化一个Tensor********')
print(torch.tensor([[1., -1.], [1., -1.]]))

# 初始化一个Tensor，从numpy的array转化而来
print('********初始化一个Tensor，从numpy的array转化而来********')
print(torch.tensor(np.array([[1., -1.], [1., -1.]])))

# Tensor的类型依赖于初始化列表和numpy的矩阵类型
print('********Tensor的类型依赖于初始化列表和numpy的矩阵类型********')
array1 = torch.tensor([[1, -1], [1, -1]])
array2 = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
print('array1   ->  ', array1.dtype)
print('array2   ->  ', array2.dtype)

# 初始化不同类型的张量
print('********初始化全0的张量，类型为int32********')
print(torch.zeros([2, 4], dtype=torch.int32))

print('********初始化全1的张量，类型为float64********')
print(torch.ones([3, 4], dtype=torch.float64))

print('********初始化对角矩阵，类型为int32********')
print(torch.eye(4, dtype=torch.int32))

print('********生成[start,end]等距向量********')
print(torch.arange(start=0, end=1, step=0.2))

print('********生成[start,end]等距向量********')
print(torch.linspace(start=0, end=1, steps=5))

print('********生成未初始化的指定矩阵********')
print(torch.empty(3, 4))

print('********利用指定值填充矩阵********')
print(torch.full(size=[2, 3], fill_value=0.5))

print('********生成[0,1)均匀随机采样矩阵********')
print(torch.rand(2, 5))

print('********生成[low,high)随机采样整数矩阵********')
print(torch.randint(low=0, high=10, size=(3, 4)))

print('********生成标准正态分布采样矩阵********')
print(torch.randn(3, 4))

print('********_like方法，生成和array3一样的全零矩阵和全1矩阵********')
array3 = torch.rand(3, 4)
print('全0矩阵 —>  \n', torch.zeros_like(array3))
print('全1矩阵 ->  \n', torch.ones_like(array3))

# 对于Tensor的操作是Tensor类的成员方法，一般用'_'结尾
print('********对于Tensor类的成员方法********')
array4 = torch.randn(3, 4)
print('当前array4   ->  \n', array4)
# 将array4矩阵清零，并返回结果赋予到array5矩阵中
array5 = array4
array4.zero_()
print('使用zero_()方法后的array4  ->  \n', array4)
print('未使用zero_()方法后的array5  ->  \n', array5)
# 该方法都是对于矩阵是原地操作，即对原始张量进行更改
# array4和array5两个矩阵实际上是共享同一块数据存储部分
# 当对任意一个元素进行更改时，两个矩阵都会改变
array4[1][2] = 1
print('当对任意一个元素进行更改时的array4  ->  \n', array4)
print('当对任意一个元素进行更改时array5  ->  \n', array5)

print('********对Tensor使用深拷贝********')
array6 = torch.randn(3, 4)
# 使用clone方法可以对Tensor进行深拷贝
array7 = array6.clone()
print('当前array6   ->  \n', array6)
print('当前array7   ->  \n', array7)
# 对array6进行清0操作
array6.zero_()
# 结果array6变为0矩阵，而array7的数值不会改变
print('使用zero_()方法后的array6  ->  \n', array6)
print('未使用zero_()方法后的array7  ->  \n', array7)
