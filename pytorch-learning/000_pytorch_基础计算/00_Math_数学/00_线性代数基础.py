#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch

# 标量
print('********标量********')
"""
    标量是基本数学对象
    标量是有一个元素的张量表示，可以理解为一个标量就是一个单独的数
"""
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(f'x + y   ->  {x + y}')
print(f'x - y   ->  {x - y}')
print(f'x * y   ->  {x * y}')
print(f'x / y   ->  {x / y}')
print(f'x ** y   ->  {x ** y}')

# 向量
print('********向量********')
"""
    向量是基本数学对象
    可以将向量视为标量值组成的列表，可以理解为一个向量就是一列数，比如[x1,x2,x3,x4]T
"""
# 生成0到3四个元素
z = torch.arange(4)
print(f'z   ->  {z}')
# 任意取得其中一个元素
print(f'取得z的第3个元素   ->  {z[2]}')

# 长度、维度和形状
print('********长度、维度和形状********')
print(f'取得张量的长度   ->  {len(z)}')
print(f'取得向量的长度   ->  {z.shape}')

# 矩阵
print('********矩阵********')
"""
    矩阵是基本数学对象
    矩阵在数据结构中是非常有用的，它允许我们组织具有不同变化模式的数据
"""
# 生成一个从0到19的20个元素的5行4列的矩阵
X = torch.arange(20).reshape(5, 4)
print(f'X   ->  \n{X}')
# 访问转置
print(f'X的转置   ->  \n{X.T}')
# 创建一个新的对称矩阵并列出它的转置，进行比较（对称矩阵的转置等于它本身）
Y = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(f'对称矩阵Y与Y的转置比较   ->  \n{Y == Y.T}')

# 张量
print('********张量********')
"""
    张量是基本数学对象
    张量是基于向量和矩阵的推广，可以将标量视为零阶张量，矢量视为一阶张量，那么矩阵就是二阶张量
    张量为我们提供了描述具有任意数量轴的n维数组的“通用”方法
    张量乘以或加上一个标量不会改变长相的形状，张量的每个元素都将与标量相乘或者相加
"""
# 创建3个2x4的矩阵
X1 = torch.arange(24).reshape(3, 2, 4)
print(f'X1   ->  \n{X1}')
# 设置一个标量，将标量与张量X1相乘和相加
a = 2
print(f'X1 + a   ->  \n{X1 + a}')
print(f'(X1 * a).shape   ->  {(X1 * a).shape}')

# 降维
print('********降维********')
"""
    一个张量可以通过sum和mean沿指定轴降低维度
    默认情况下，sum()函数会沿所有的轴降低张量的维度
    以矩阵X为例，调用函数时可以根据指定轴axis进行降维
"""
x1 = torch.arange(4, dtype=torch.float32)
print(f'x1   ->  {x1}')
print(f'x1.sum()   ->  {x1.sum()}')
print(f'X   ->  \n{X}')
print(f'沿X轴axis=0求和   ->  {X.sum(axis=0)}')
print(f'沿X轴axis=0求和后的shape   ->  {X.sum(axis=0).shape}')

# 点积
print('********点积********')
"""
    例如：
        X1 = [1,2,3,4]
        X2 = [1,0,1,1]
        那么X1与X2的点积就是1 * 1 + 2 * 0 + 3 * 1 + 4 * 1 = 1 + 0 + 3 + 4 =8
"""
y1 = torch.ones(4, dtype=torch.float32)
print(f'x1   ->  {x1}')
print(f'y1   ->  {y1}')
print(f'torch.dot(x1,y1)   ->  {torch.dot(x1, y1)}')

# 矩阵——向量积
print('********矩阵——向量积********')
"""
    torch.mv()是矩阵和向量相乘
"""
# 生成一个从0到19的20个元素的5行4列的矩阵
X2 = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x3 = torch.arange(4, dtype=torch.float32)
print(f'X2.shape   ->  {X2.shape}')
print(f'x3.shape   ->  {x3.shape}')
print(f'torch.mv(X2,x3) ->  {torch.mv(X2, x3)}')

# 矩阵——矩阵乘法
"""
    例如：
            | a1  a2  a3 |          | b1  b2 |
        A = | a4  a5  a6 |      B = | b3  b4 |
            | a7  a8  a9 |          | b5  b6 |
            
                 | a1*b1+a2*b3+a3*b5  a1*b2+a2*b4+a3*b6 |
        C = AB = | a4*b1+a5*b3+a6*b5  a4*b2+a5*b4+a6*b6 |
                 | a7*b1+a8*b3+a9*b5  a7*b2+a8*b4+a9*b6 |
"""

X3 = torch.arange(20, dtype=torch.float32).reshape(5, 4)
X4 = torch.ones(4, 3, dtype=torch.float32)
print(f'X3   ->  \n{X3}')
print(f'X4   ->  \n{X4}')
print(f'torch.mm(X3,X4) ->  {torch.mm(X3, X4)}')

# 范数
print('********范数********')
"""
    线性代数中最有用的一些运算符是范数（norm），简单来说范数就是告诉我们一个向量有多大
    性质：
        1. f(αx) = |α|f(x)，常数因子α缩放向量的所有元素，范数也会按相同的常数因子的绝对值缩放
        2. 三角不等式：f(x+y) ≤ f(x) + f(y)
        3. 范数必须是非负的：f(x) ≥ 0（该性质要求范数最小为0，当且仅当向量全有0构成
    
    比如，欧几里得距离就是一个范数，假设n维向量x中的元素是x1，x2，x3，x4...xn
    那么它的L2范数就是向量元素平方之和的平方根
    ||x||₂ = √((x1)²+(x2)²+...+(xn-1)²+(xn)²)
    L1范数表示向量元素的绝对值之和
"""
# ||x||₂ = √((x1)²+(x2)²+...+(xn-1)²+(xn)²)
u = torch.tensor([3.0, -4.0, 3.0])
print(f'torch.norm(u) ->  {torch.norm(u)}')
# L1范数
print(f'torch.abs(u).sum() ->  {torch.abs(u).sum()}')

