#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib_inline.backend_inline
import matplotlib.pyplot as plt
import torch
import numpy as np

# 导数和微分
print('********导数和微分********')
"""
    可微与和可导的关系：可微一定可导，可导不一定可微
    d[Cf(x)]/dx = Cdf(x)/dx
    d[f(x) ± g(x)]/dx = df(x)/dx ± dg(x)/dx
    d[f(x)*g(x)]/dx = f(x)*d[g(x)]/dx + g(x)*d[f(x)]/dx
    d[f(x)/g(x)]/dx = [g(x)*d[f(x)]/dx - f(x)*d[g(x)]/dx]/[g(x)]²
"""


# 先定义一个函数
def f(x):
    return 3 * x ** 2 - 4 * x


# 导数定义求极限 f是函数 h是变化量
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


# 令x=1让h无限接近0，求出f(x)的极限
h = 0.1
for i in range(5):
    print(f'h   ->  {h:.5f}, numerical limit    ->  {numerical_lim(f, 1, h):.5f}')
    h *= 0.1


def use_svg_display():
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果 `X`（张量或列表）有 1 个轴，则返回 True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# 以绘制函数u = f(x)及其在x = 1处的切线y = 2x − 3，其中系数2是切线的斜率
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
