#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from collections import OrderedDict

# 继承Module类来构造模型
print('********继承Module类来构造模型********')


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        # 隐藏层
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        # 输出层
        self.output = nn.Linear(256, 10)

    # 定义模型的向前计算，根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


X = torch.rand(2, 784)
net1 = MLP()
print(f'net1 ->  \n{net1}')
print(f'net1(X) ->  \n{net1(X)}')

# Module的子类
print('********Module的子类********')


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].item():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


net2 = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(f'net2 ->  \n{net2}')
print(f'net2(X) ->  \n{net2(X)}')

# ModuleList类
print('********ModuleList类********')
# 以一个List形式输入
net3 = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
# 在这个List中增加一个值
net3.append(nn.Linear(256, 10))
print(f'net3[-1] ->  \n{net3}')
print(f'net3 ->  \n{net3}')

# ModuleDict类
print('********ModuleDict类********')
# 类似于字典的键值对的形式
net4 = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
# 将output与对应的值添加进字典中
net4['output'] = nn.Linear(256, 10)
print('访问net4[\'linear\']   ->  ', net4['linear'])
print('访问net4.output    ->  ', {net4.output})
print(f'net4 ->  \n{net4}')

# 构造复杂的模型
print('********构造复杂的模型********')


class fancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(fancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层
        x = self.linear(x)
        # 控制流
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


X_fancy = torch.rand(2, 20)
net5 = fancyMLP()
print(f'net5 ->  \n{net5}')
print(f'net5(X_fancy) ->  {net5(X_fancy)}')


class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


net6 = nn.Sequential(NestMLP(), nn.Linear(30, 20), fancyMLP())

X_nest = torch.rand(2, 40)
print(f'net6 ->  \n{net6}')
print(f'net6(X_nest) ->  {net6(X_nest)}')
