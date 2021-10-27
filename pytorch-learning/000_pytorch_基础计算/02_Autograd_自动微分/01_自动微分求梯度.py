#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch

print('********loss为标量时，求梯度********')
array1 = torch.randn(2, 3, requires_grad=True)
# 模拟最终输出损失值
loss1 = array1.sum()
# 调用.backward()，进行反转
loss1.backward()
# 查看梯度
print('当前array1的梯度  ->\n', array1.grad)

print('********loss为向量时，求梯度********')
array2 = torch.randn(2, 3, requires_grad=True)
# array2不是标量，是向量
loss2 = array2.sum(dim=0)
print(loss2.shape)
# 反向传播输入与loss2形状相同
loss2.backward(torch.FloatTensor([1, 2, 3]))
# 查看梯度
print('当前array2的梯度  ->\n', array2.grad)

print('********自动反向求导处理循环和动态分支，求梯度********')
array3 = torch.randn(2, 3, requires_grad=True)
loss3 = array3.abs().sum()
# 由loss3的值控制循环次数
while loss3 < 100:
    loss3 = loss3 ** 2
print('当前loss3  ->', loss3)
loss3.backward()
# 查看梯度
print('当前array3的梯度  ->\n', array3.grad)

# 该做法一般用在模型测试阶段，此时已经无需训练，可以节省在前项计算时暂存的中间结果，对于GPU可以节省显存，节省计算量达到加速目的
print('********关闭张量自动求导********')
array4 = torch.randn(2, 3, requires_grad=True)
# 即使array4可导，loss4也关闭了自动求导
with torch.no_grad():
    loss4 = array4.sum()
# loss4张量无对应的反向求导函数
print(loss4)
print('loss4的requires_grad  ->', loss4.requires_grad)

# Tensor类中的detach()方法
print('********Tensor类中的detach()方法********')
array5 = torch.randn(2, 3, requires_grad=True)
# array5和array6数值相同，但array6不可导
array6 = array5.detach()
# 输出array6的requires_grad属性
print('array6的requires_grad ->', array6.requires_grad)
# 计算梯度时仅对可导的array5进行计算，array6不计算并输出None不作为显示
loss5 = array5.sum()
loss5.backward()
print('当前array5的梯度  ->\n', array5.grad)
print('当前array6的梯度  ->\n', array6.grad)
