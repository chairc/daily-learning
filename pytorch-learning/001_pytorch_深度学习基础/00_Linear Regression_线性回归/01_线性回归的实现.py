#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

"""
关于IPython的使用方法：
    1、安装 ipython 之后，查看 PyCharm 设置，确保 Console 的通用设置 Use IPython if available 选项打勾。
    2、打开 Python Console 窗口(Tools-Python Console)，屏幕下方就可见 ipython 的交互式界面了。
    3、选中代码行，一行、多行皆可。然后鼠标右键，点选 Execute Line in Console。
"""

# 生成数据集
print('********生成数据集********')
# 特征数为2
num_inputs = 2
# 样本数为1000
num_examples = 1000
# 回归模型真是权重为[2, -3.4]
true_w = [2, -3.4]
# 偏差值为4.2
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

print(features[0], labels[0])


def use_svg_display():
    # 用矢量图表示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图片尺寸
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

# 读取数据集
print('********读取数据集********')


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本读取顺序是随机的
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个batch
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x, y)
    break

# 初始化模型参数
print('********初始化模型参数********')
# 初始化均值为0，标准差为0.01的正态分布随机数，偏差初始化为0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 后续训练中需要对参数的梯度进行迭代，因此设置requires_grad为True
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型
print('********定义模型********')


def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义损失函数
print('********定义损失函数********')


def squared_loss(y_hat, y):
    # 返回的是向量
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
print('********定义优化算法********')


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 训练模型
print('********训练模型********')
"""
    在训练中，我们将多次迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征X和标签y），
    通过调用反向函数backward计算小批量随机梯度，并调用优化算法sgd迭代模型参数。由于我们之前设批量大
    小batch_size为10，每个小批量的损失l的形状为(10, 1)。回忆一下自动求梯度一节。由于变量l并不是一
    个标量，所以我们可以调用.sum()将其求和得到一个标量，再运行l.backward()得到该变量有关模型参数的
    梯度。注意在每次更新完参数后不要忘了将参数的梯度清零。

    在一个迭代周期（epoch）中，我们将完整遍历一遍data_iter函数，并对训练数据集中所有样本都使用一次
    （假设样本数能够被批量大小整除）。这里的迭代周期个数num_epochs和学习率lr都是超参数，分别设3和0.03。
    在实践中，大多超参数都需要通过反复试错来不断调节。虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长。
"""
lr = 0.03
# 训练迭代周期
num_epochs = 3
net = linreg
loss = squared_loss

# 训练模型一共需要num_epochs个迭代周期
for epoch in range(num_epochs):
    # 每个迭代周期会使用训练数据集中所有样本一次，X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        # l是小批量X和y的损失
        l = loss(net(X, w, b), y).sum()
        # 小批量的损失对模型参数求梯度
        l.backward()
        # 使用小批量随机梯度下降迭代模型参数
        sgd([w, b], lr, batch_size)

        # 清除梯度
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# 训练结束，利用学到的参数训练来生成训练集的真实参数
print('********训练结束，利用学到的参数训练来生成训练集的真实参数********')
print(true_w, '\n', w)
print(true_b, '\n', b)
