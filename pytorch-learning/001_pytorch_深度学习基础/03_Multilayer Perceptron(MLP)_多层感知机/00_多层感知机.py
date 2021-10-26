#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib_inline.backend_inline
import torch
import numpy as np
import matplotlib.pylab as plt


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
