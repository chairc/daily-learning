#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from d2lzh_pytorch import torch_d2l as d2l

"""
    在多个尺度下，我们可以生成不同尺寸的锚框来检测不同尺寸的目标
    通过定义特征图的形状，我们可以决定任何图像上均匀采样的锚框的中心
    使用输入图像在某个感受野区域内的信息，来预测输入图像上与该区域位置相近的锚框类别和偏移量
"""

image = d2l.plt.imread('../../Datasets/img/catdog.jpg')
# 获取高和宽
h, w = image.shape[:2]


# 显示锚框
def display_anchors(feature_map_w, feature_map_h, size):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    feature_map = torch.zeros((1, 10, feature_map_h, feature_map_w))
    # 生成锚框，生成的都是0~1的数值
    # ratios固定为锚框生成的宽高比分别为1:1, 2:1, 1:2
    anchors = d2l.multibox_prior(feature_map, sizes=size, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(image).axes, anchors[0] * bbox_scale)


# 探测小目标
# 锚框size设为0.15，特征图的高和宽设置为4
display_anchors(feature_map_w=4, feature_map_h=4, size=[0.15])

# 特征图减半，使用较大的锚框来检测较大的目标
display_anchors(feature_map_h=2, feature_map_w=2, size=[0.4])

display_anchors(feature_map_w=1, feature_map_h=1, size=[0.8])
