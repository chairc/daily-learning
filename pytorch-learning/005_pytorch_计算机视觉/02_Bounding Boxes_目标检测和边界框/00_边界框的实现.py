#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from d2lzh_pytorch import torch_d2l as d2l

# 显示图片
d2l.set_figsize()
image = d2l.plt.imread('../../Datasets/img/catdog.jpg')
d2l.plt.imshow(image)

dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]


def bbox_to_rect(bbox, color):
    # 边界框转换为matplotlib格式
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1], fill=False, edgecolor=color,
        linewidth=2
    )


fig = d2l.plt.imshow(image)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
