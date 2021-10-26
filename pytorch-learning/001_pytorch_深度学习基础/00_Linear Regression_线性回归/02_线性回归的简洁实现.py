#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
# 神经网络模型
import torch.nn as nn
# 初始化
from torch.nn import init
# 优化算法
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

# 定义模型
print('********定义模型********')


# nn是利用autograd来定义的模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


# 输出模型
net = LinearNet(num_inputs)
print(net)

net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可传入到其它层
)
print(net)
print(net[0])

# 查看模型所有可学习的参数
for param in net.parameters():
    print(param)

# 初始化模型参数
print('********初始化模型参数********')
net1 = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
# mean：正态分布均值
# std：正态分布标准差
init.normal_(net1[0].weight, mean=0, std=0.01)
# init.normal_(net.linear.weight, mean=0, std=0.01)
# 也可修改bias的data：net[0].bias.data.fill_(0)
init.constant_(net1[0].bias, val=0)
# init.constant_(net.linear.bias, val=0)

# 定义损失参数
print('********定义损失参数********')
loss = nn.MSELoss()

# 定义优化算法
print('********定义优化算法********')
# SGD：小批量随机梯度下降
# lr：learning rate 学习率
optimizer = optim.SGD(net1.parameters(), lr=0.03)
print(optimizer)
# 如果要调整学习率，我们可以构建一个优化器
for param_group in optimizer.param_groups:
    # 学习率是之前的2倍
    param_group['lr'] *= 2

# 训练模型
print('********训练模型********')
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net1(X)
        l = loss(output, y.view(-1, 1))
        # 梯度清零，等价于net.zero_grad()
        optimizer.zero_grad()
        l.backward()
        # step指明批量大小
        optimizer.step()
    print('epoch %d,loss %f' % (epoch, l.item()))

dense = net1[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
