#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 字典方法

# clear()：删除字典内的所有项
print('********clear()********')
a = {'小明': '1001', '小强': '1002', '小华': '1003', '小峰': '1004'}
a.clear()
print(a)
print('%d' % len(a))

# copy()：返回一个具有相同键值对的新字典
print('********copy()********')
b = {'小明': '1001', '小强': '1002', '小华': '1003', '小峰': '1004'}
c = {'info1': ['小明', '1001', '男'], 'info2': ['小强', '1002', '女']}
b_copy = b.copy()
c_copy = c.copy()
print(b_copy)
print(c_copy['info2'])

# fromkeys()：创建一个新字典
# dict.fromkeys(seq[,value])：dict为指定字典；seq为字典键值列表；value为可选参数，设置序列seq的值
print('********fromkeys()********')
seq = {'name', 'age', 'sex'}
info1 = dict.fromkeys(seq)
print('------')
# 新字典输出，不带value
print('%s' % info1)
print('------')
# 新字典输出，带value
info2 = dict.fromkeys(seq, 10)
print('%s' % info2)

# get()：返回指定键的值，如果不在字典中，返回默认值
print('********get()********')
d = {'info1': ['小明', '1001', '男'], 'info2': ['小强', '1002', '女']}
print('------')
# get在字典中的值
print(d.get('info1'))
print('------')
# get不在字典中的值，默认None
print(d.get('info3'))
print('------')
# get不在字典中的值，自定义
print(d.get('info3', '不在字典中'))

# key in dict：判断键是否存在于字典中
print('********key in dict********')
e = {'info1': ['小明', '1001', '男'], 'info2': ['小强', '1002', '女']}
print('%s' % ('info1' in e))
print('%s' % ('info3' in e))

# items()：遍历键值元组数组
print('********items()********')
f = {'小明': '1001', '小强': '1002', '小华': '1003', '小峰': '1004'}
g = {'info1': ['小明', '1001', '男'], 'info2': ['小强', '1002', '女']}
print(f.items())
print(g.items())

# keys()：返回一个字典的所有键
print('********keys()********')
h = {'小明': '1001', '小强': '1002', '小华': '1003', '小峰': '1004'}
i = {'info1': ['小明', '1001', '男'], 'info2': ['小强', '1002', '女']}
print('%s' % h.keys())
print('%s' % i.keys())

# setdefault()：用于获得与给定键相关联的值，如果键不存在于字典中，则会添加键并设置默认值
print('********setdefault()********')
j = {'小明': '1001', '小强': '1002', '小华': '1003'}
print('%s' % j.setdefault('小华'))
print('------')
# 小峰不在字典中，添加小峰并设置默认值None
print('%s' % j.setdefault('小峰'))
print('%s' % j)

# update()：把字典dict2的键值对更新到dict1里，提供的字典与旧字典有相同的键会被覆盖
print('********update()********')
k = {'小明': '1001', '小强': '1002'}
l = {'小强': '1222', '小华': '1003', '小峰': '1004'}
# k与l会合并到k中，k中的小强的value值会被l中的小强的value值覆盖
k.update(l)
print('%s' % k)

# values()：返回字典中的所有值
print('********values()********')
m = {'小明': '1001', '小强': '1002', '小华': '1003', '小峰': '1004'}
n = {'info1': ['小明', '1001', '男'], 'info2': ['小强', '1002', '女']}
print(m.values())
print(n.values())