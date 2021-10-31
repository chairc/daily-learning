#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)

# 保存张量到当前目录，并作为输入
print('********保存张量作为输入********')
print(f'x   ->  {x}')
torch.save(x, 'x-file')

# 将文件数据读取到内存
print('********将文件数据读取到内存********')
torch.load('x-file')
print(f'读取文件x-file    ->  {x}')

# 储存张量列表，并读取纸内存
print('********储存张量列表，并读取纸内存********')
y = torch.zeros(4)
print(f'y   ->  {y}')
torch.save([x, y], 'x-y-files')
x_load, y_load = torch.load('x-y-files')
print(f'读取文件x-y-files   ->  {x_load, y_load}')

# 写入或读取从字符串映射到张量的字典
my_dict = {'x': x, 'y': y}
print(f'my_dict   ->  {my_dict}')
torch.save(my_dict, 'my-dict-file')
my_dict_load = torch.load('my-dict-file')
print(f'读取文件my-dict-file    ->  {my_dict_load}')
