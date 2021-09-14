#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 列表
# list的特点：查找和插入时间随着元素的增加而增加；占用空间小，浪费内存少

# 1.更新列表
# 元素赋值
print('********元素赋值********')
a = [1, 2, 3, 4, 5]
a[1] = 10
print(a)
a[2] = 'hello'
print(a)
string = [None] * 5
string[2] = 'hello'
print(string)

# 增加元素，只能在队尾添加元素
print('********增加元素********')
b = [1, 2, 3]
b.append(4)
print(b)
b.append('hello')
print(b)

# 删除元素
print('********删除元素********')
c = ['a', 'b', 'c', 'd']
print(len(c))
del c[1]
print(c)
print(len(c))

# 分片赋值，在任意位置添加元素
print('********分片赋值********')
print(list('女排夺冠了'))
boil = list('女排夺冠了')
print(boil)

print(list('hi,bro'))
show = list('hi,bro')
show[3:] = list('man')
print(show)

# 2.嵌套列表
a = ['a', 'b', 'c']
b = [1, 2, 3]
mix = [a, b]
print(mix)
print(mix[0])
print(mix[1])

# 3.列表方法
# append()：在队尾添加元素
print('********append()********')
a = [1, 2, 3]
a.append(4)
print(a)

# count()：统计某个元素在列表中出现的次数
print('********count()********')
b = [1, 2, 2, 2, 3, 4]
print(b.count(2))

# extend()：在列表末尾一次性追加另一个序列中的多个值
print('********extend()********')
c = ['hello', 'world']
d = ['你好', '世界']
c.extend(d)
print(c)

# index()：从列表中找出某个值第一个匹配项索引位置
print('********index()********')
e = ['hello', 'world', '你好', '世界']
print(e.index('hello'))
print(e.index('你好'))

# insert()：将对象插入列表
print('********insert()********')
num = [1, 2, 3]
print(num)
num.insert(2, 2.5)
print(num)

# pop()：移除列表中的一个元素（默认最后一个元素），并返回该元素的值
print('********pop()********')
f = ['hello', 'world', '你好', '世界']
# 无参，默认最后一个
f.pop()
print(f)
f.pop(1)
print(f)

# remove()：用于移除列表中某个值的第一个匹配项
print('********remove()********')
g = ['hello', 'world', '你好', '世界']
g.remove('world')
print(g)

# reverse()：用于反向列表中的元素
print('********reverse()********')
h = [1, 2, 3, 4, 5]
h.reverse()
print(h)

# sort()：用于对列表进行排序；sorted()：可以获取列表副本进行排序
print('********sort()&sorted()********')
i = [4, 2, 5, 1, 9, 6, 0, 3, 7, 8]
i_i = sorted(i)
print(i_i)
print(i)
i.sort()
print(i)

# clear()：用于清空列表
print('********clear()********')
j = ['hello', 'world', '你好', '世界']
j.clear()
print(j)

# copy()：用于复制列表
print('********copy()********')
k = ['hello', 'world', '你好', '世界']
k_copy = k.copy()
print(k_copy)

# 高级排序：自定义比较方法
print('********高级排序********')
l = ['hello', 'hi', 'ok', 'fine']
m = [4, 2, 1, 3]
# 根据长度排序
l.sort(key=len)
print(l)
# 排序后逆序
m.sort(reverse=True)
print(m)