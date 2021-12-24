#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from d2lzh_pytorch import torch_d2l as d2l

"""
    卷积、全连接、池化层都只考虑不随意线索
    注意力机制则显示的考虑随意线索：
        1.随意线索被称之为查询(query)
        2.每个输入是一个值(value)和不随意线索(key)的对
        3.通过注意力池化层来有偏向性的选择选择某些输入

"""


# 显示矩阵热图
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    global pcm
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


# 生成对角矩阵
attention_weights_1 = torch.eye(10).reshape((1, 1, 10, 10))
# 生成自定义矩阵
attention_weights_2 = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 0]]).reshape((1, 1, 3, 3))
# 生成随机矩阵
attention_weights_3 = torch.rand(10, 10).reshape((1, 1, 10, 10))
# 显示注意力权重
show_heatmaps(attention_weights_1, xlabel='Keys', ylabel='Queries')
show_heatmaps(attention_weights_2, xlabel='Keys', ylabel='Queries')
show_heatmaps(attention_weights_3, xlabel='Keys', ylabel='Queries')
