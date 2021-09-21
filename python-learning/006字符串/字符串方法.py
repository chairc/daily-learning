#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 字符串方法

# find()：检测字符串中是否包含子字符串，如果包含返回索引值，否则返回-1
print('********find()********')
a = 'hello,world!bro'
# 返回b的位置为12（0是第一个）
print(a.find('bro'))
print(a.find('hello'))
print(a.find('hi'))

# join()：将序列中的元素以指定字符连接成一个新的字符串，操作对象必须是字符串
print('********join()********')
b = ['1', '2', '3', '4']
c = '+'
print(c.join(b))

# lower()：将字符串中的所有大写转为小写
print('********lower()********')
d = 'ABCdEfgh'
print(d.lower())

# upper()：将字符串中的所有小写转为大写
print('********upper()********')
e = 'ABCdEfgh'
print(e.upper())

# swapcase()：将字符串中的所有小写转为大写
print('********swapcase()********')
f = 'ABCdEfgh'
print(f.swapcase())

# replace()：把旧的字符串替换成新字符串
print('********replace()********')
g = 'hello,world'
h = 'aaaaaaaa'
print(g.replace('hello', 'hi'))
# replace(old_str,new_str,x)将字符串替换x次，理论不超过x次
print(h.replace('a', 'b', 1))
print(h.replace('a', 'b', 3))

# split()：指定分隔符对字符串进行切片，该方法是join的逆方法，将字符串分割成序列
# str.split(st='',num=string.count(str))  str代表指定检索的字符串；st代表分隔符，默认空格；num代表分割次数
print('********split()********')
i = 'hello world'
j = 'hello,world'
# 不使用分隔符
print(i.split())
# 根据o分割
print(i.split('o'))
# 根据逗号分割
print(j.split(','))

# strip()：移除字符串头尾指定的字符（默认空格）
# str.strip([chars])  str为指定的检索字符串；chars为移除字符串头尾指定字符
print('********strip()********')
k = '---hello world---'
# 指定移除-
print(k.strip('-'))

# translate()：根据参数table给出的表转换字符串的字符，将要过滤的字符放入到del参数中
# str.translate(table[,deletechars])
# str为指定的检索字符串；table为翻译表，翻译后通过maketrans方法转换而来；deletechars为字符串中要过滤的字符列表
print('********translate()********')
# a与1，b与2，依次对应，当需要转换的字符表中含有a时，将其替换为1，以此类推
intab = 'adefs'
outtab = '12345'
trantab = str.maketrans(intab, outtab)
st = 'just do it'
print(st.translate(trantab))