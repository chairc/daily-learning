#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib_inline.backend_inline
import torch
import numpy as np
import matplotlib.pylab as plt

"""
    多层感知机十分适合处理表格数据
    激活函数：
        1. ReLU函数
        ReLU(x) = max(x,0)
        输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1，当输入值精确等于0时，ReLU函数不可导
        ReLU减轻了困扰以往神经网络的梯度消失问题
        2. sigmoid函数
        sigmoid(x) = 1/1+exp(−x)
        将输出视作二分类问题的概率时，sigmoid仍然被广泛用作输出单元上的激活函数
        3. tanh函数
        tanh(x) = 1−exp(−2x)/1+exp(−2x)
        当输入接近0时，tanh函数的导数接近最大值1。与我们在sigmoid函数图像中看到的类似，输入在任一方向上越远离0点，导数越接近0。
"""

def use_svg_display():
    # 该方法已弃用
    # display.set_matplotlib_formats('svg')
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 激活函数
def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)


# 绘制ReLU函数
def draw_relu():
    y = x.relu()
    xyplot(x, y, 'relu')


# 绘制ReLU函数的导数
def draw_relu_grad(x, y):
    y.sum().backward()
    xyplot(x, x.grad, 'grad of relu')


draw_relu()
draw_relu_grad(x=x, y=x.relu())


# 绘制sigmoid函数
def draw_sigmoid(x):
    y = x.sigmoid()
    xyplot(x, y, 'sigmoid')


# 绘制sigmoid函数的导数
def draw_sigmoid_grad(x, y):
    x.grad.zero_()
    y.sum().backward()
    xyplot(x, x.grad, 'grad of sigmoid')


draw_sigmoid(x)
draw_sigmoid_grad(x=x, y=x.sigmoid())


# 绘制tanh函数
def draw_tanh(x):
    y = x.tanh()
    xyplot(x, y, 'tanh')


# 绘制tanh函数的导数
def draw_tanh_grad(x, y):
    x.grad.zero_()
    y.sum().backward()
    xyplot(x, x.grad, 'grad of tanh')


draw_tanh(x)
draw_tanh_grad(x=x, y=x.tanh())
