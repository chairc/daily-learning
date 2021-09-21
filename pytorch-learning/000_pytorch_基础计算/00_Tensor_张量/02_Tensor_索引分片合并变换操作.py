#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch

# 索引分片操作

array1 = torch.randn(3, 4)
print('当前array1 ->\n', array1)

# 按照列维度，取第0列和第2列,torch.index_select(array1, dim=1, index=torch.tensor([0, 2]))或array1[:,[0,2]]
print('********按照列维度，取第0列和第2列********')
print(torch.index_select(array1, dim=1, index=torch.tensor([0, 2])))
print(array1[:, [0, 2]])

print('\n\n')

# 合并操作：包含拼接concatenate，堆叠stack，分块chunk，拆分split

array2 = torch.zeros(2, 3)
array3 = torch.zeros(2, 3).fill_(1)
array4 = torch.zeros(2, 3).fill_(2)
array5 = torch.cat([array2, array3, array4], dim=0)
print('当前array2 ->\n', array2)
print('当前array3 ->\n', array3)
print('当前array4 ->\n', array4)
print('当前array5 ->\n', array5)

# 按照dim=0进行拼接
print('********按照dim=0进行拼接********')
print(torch.cat([array2, array3, array4], dim=0))

# 按照dim=0进行拼接，不增加维度，保持二维
print('********按照dim=0进行拼接，不增加维度，保持二维********')
print(torch.cat([array2, array3, array4], dim=0).shape)

# 按照dim=1进行拼接
print('********按照dim=1进行拼接********')
print(torch.cat([array2, array3, array4], dim=1))

# 按照dim=0进行堆叠
print('********按照dim=0进行堆叠********')
print(torch.stack([array2, array3, array4], dim=0))

# 按照dim=0进行堆叠，堆叠后，增加额外维度
print('********按照dim=0进行堆叠，堆叠后，增加额外维度********')
print(torch.stack([array2, array3, array4], dim=0).shape)

# 按照行维度进行分三块
print('********按照行维度进行分三块********')
print(torch.chunk(array5, 3, dim=0))

# 按照行维度拆分为[1,3,2]宽度的3块
print('********按照行维度拆分为[1,3,2]宽度的3块********')
print(torch.split(array5, [1, 3, 2], dim=0))

print('\n\n')

# 变换操作

array6 = torch.randn(3, 4)
print('当前array6 ->\n', array6)

# 转置操作，维度0和维度1互换
print('********转置操作，维度0和维度1互换********')
print(torch.transpose(array6, dim0=1, dim1=0))

# 在张量array6上新增一个维度
print('********在张量array6上新增一个维度********')
array7 = torch.unsqueeze(array6, dim=2)
print(array7.shape)
print(torch.squeeze(array7).shape)
