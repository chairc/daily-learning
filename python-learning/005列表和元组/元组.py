#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 元组
# 创建方法：使用逗号分隔一些值，会自动创建元组

# 1.tuple()函数
print('********tuple()函数********')
print(tuple(['hello', 'world']))
print(tuple('hello'))
# 参数是元组
print(tuple(('hello', 'world')))

# 2.元组基本操作
# 访问元组：使用下标索引访问元组
print('********访问元组********')
a = ('hello', 'world', 2021, 2022)
b = (1, 2, 3, 4, 5, 6, 7)
print(a[1])
print(a[3])
# 从b[1]开始到元组第五个元素
print(b[1:5])

# 修改元组：元组中的元素值不允许修改，但可以对元组进行连接组合
print('********修改元组********')
c = ('hello', 'world')
d = (2021, 2022)
print(c + d)

# 删除元组：元组中的元素值不允许删除，但是可以del整个元组
print('********删除元组********')
e = ('hello', 'world')
del e

# 元组索引、截取
print('********元组索引、截取********')
f = ('hello', 'world', 'hi')
print(f[2])
print(f[-2])
print(f[1:])