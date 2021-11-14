#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torchvision
import sys
from torch import nn
import d2lzh_pytorch.torch_d2l as d2l


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


# 原始net
net = nn.Sequential(
    Reshape(),
    # 输入通道为1，输出通道为6的卷积层   (28+4-5+1)*(28+4-5+1) = 28*28
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # 平均池化层2x2，移动不重叠   28/2 = 14
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 输入通道为6，输出通道为16的卷积层  (14-5+1)*(14-5+1) = 10*10
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # 平均池化层2x2，移动不重叠   10/2 = 5
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 将四维卷积层变为一维向量
    nn.Flatten(),
    # 16个输出再乘以5*5的池化层转为向量为16*5*5 = 400
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    # 输出0到9
    nn.Linear(84, 10)
)

# 改进后net_new，减少一个Linear
net_new = nn.Sequential(
    Reshape(),
    # 输入通道为1，输出通道为6的卷积层   (28+4-5+1)*(28+4-5+1) = 28*28
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # 平均池化层2x2，移动不重叠   28/2 = 14
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 输入通道为6，输出通道为16的卷积层  (14-5+1)*(14-5+1) = 10*10
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # 平均池化层2x2，移动不重叠   10/2 = 5
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 将四维卷积层变为一维向量
    nn.Flatten(),
    # 16个输出再乘以5*5的池化层转为向量为16*5*5 = 400
    nn.Linear(16 * 5 * 5, 84), nn.Sigmoid(),
    # 输出0到9
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        # 设置为评估模式
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)。"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')


# 原始学习率与迭代次数
lr, num_epochs = 0.9, 10
# 改进后学习率与迭代次数
lr_new, num_epochs_new = 0.9, 20
# 原始训练
# train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# 改进后训练
train_ch6(net_new, train_iter, test_iter, num_epochs_new, lr_new, d2l.try_gpu())