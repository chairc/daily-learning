#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch

# item()：方法从包含单个值的张量中获取pytorch的值
print('********item()：方法从包含单个值的张量中获取pytorch的值********')
loss1 = torch.tensor(3)
print(loss1.shape)
print(loss1.item())
loss2 = torch.tensor([3])
# loss2是一个1维张量
print(loss2.shape)
# 在loss2中获取一个0维张量
print(loss2[0])
# 获取为pytorch中的数值
print(loss2.item())

# numpy()：和numpy的转换
print('********numpy()：和numpy的转换********')
array1 = torch.randn(2, 3)
print('当前array1   ->\n', array1)
# 转换为numpy.ndarray格式
array2 = array1.numpy()
print('当前array2   ->\n', array2)
# 利用from_numpy方法从numpy数据创建新的张量
array3 = torch.from_numpy(array2)
print('当前array3   ->\n', array3)

# scatter()：按照指定索引指标赋值给张量
print('********scatter()：按照指定索引指标赋值给张量********')
array4 = torch.zeros(3, 4)
index = torch.tensor([[1], [2], [0]])
print(index.shape)
# 按照维度1以index所示标签赋值1.0
print(array4.scatter_(1, index, 1.0))

# view()：将一个张量转换成另外一个形状
print('********view()：将一个张量转换成另外一个形状********')
array5 = torch.randn(2, 3)
# 将2行3列的矩阵转换为6行1列
array6 = array5.view(6, 1)
print('当前array5   ->\n', array5)
print('当前array6   ->\n', array6)
# 修改array5中1行2列的元素，array6也会变化
array5[0, 1] = 10
print('修改1行2列的元素后array6   ->\n', array6)
