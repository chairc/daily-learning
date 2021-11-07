#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import torch.nn as nn
import d2lzh_pytorch.torch_d2l as d2l

dropout1, dropout2 = 0.2, 0.5
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    # 在第一个全连接层后加一个丢弃层
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    # 在第二个全连接层后加一个丢弃层
    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
