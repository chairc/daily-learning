#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn

# 指定用于存储和计算的设备，如CPU和GPU
print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1'))

# 查询GPU的数量
print(f'GPU数量   ->  {torch.cuda.device_count()}')


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(f'try_gpu(), try_gpu(10), try_all_gpus()  ->  {try_gpu(), try_gpu(10), try_all_gpus()}')
