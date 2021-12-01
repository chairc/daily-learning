#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torchvision
from torch import nn
from d2lzh_pytorch import torch_d2l as d2l

"""
    数据增强通过变形数据来获取多样性从而使得模型泛化更好
"""
root = 'C:/Users/lenovo/Desktop/Testing environment/pytorch learning/Datasets/img'

d2l.set_figsize()
img = d2l.Image.open(f'{root}/cat1.jpg')
d2l.plt.imshow(img)


# 图片，图片增广方法，生成多少行，生成多少列，图片大小
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)


# 左右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())

# 裁剪图片，裁剪面积0.1~1.0，高宽比0.5~2随机取值
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 改变亮度，亮度为(1-0.5)~(1+0.5)范围
apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))

# 改变色调
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))

# 随机改变颜色
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 多种结合的图像增广方法
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    shape_aug,
    color_aug
])
apply(img, augs)
