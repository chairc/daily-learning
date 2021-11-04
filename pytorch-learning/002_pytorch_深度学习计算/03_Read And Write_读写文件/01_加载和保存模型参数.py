#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(2, 20)
Y = net(X)
print(f'X   ->  \n{X}')
print(f'net   ->  \n{net}')
print(f'net(X)   ->  \n{Y}')
torch.save(net.state_dict(), 'mlp.params')

clone = MLP()
# 从保存到文件加载参数时可能会遇到字典不匹配的问题，可以加上strict=False
clone.load_state_dict(torch.load('mlp.params'))
# 遍历输出
clone.eval()

# 两个实例具有相同的模型参数，在输入相同的X时，两个实例的计算结果应该相同
Y_clone = clone(X)
print(Y_clone == Y)
