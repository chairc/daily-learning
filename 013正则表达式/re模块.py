#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re

# re模块


# re.match()：从字符串的起始位置匹配一个模式，匹配成功返回对象，失败则返回None
# 用法：re.match(pattern,string,flag=0)：pattern指匹配的正则表达式；string是指要匹配的字符；flags是指标志位，控制匹配方式
print('********re.match()********')
# 在起始位置开始匹配
print(re.match('hello', 'hello world').span())
# 不在起始位置开始匹配
print(re.match('world', 'hello world'))

# re.search()：用于扫描整个字符串并返回第一个成功匹配的字符
# 用法：re.search(pattern,string,flags=0)：pattern指匹配的正则表达式；string是指要匹配的字符；flags是指标志位，控制匹配方式
print('********re.search()********')
# 在起始位置开始匹配
print(re.search('hello', 'hello world').span())
# 不在起始位置开始匹配
print(re.search('world', 'hello world').span())

"""
re.match()与re.search()区别：
    re.match()只匹配字符串开始的字符，如果开始不符合正则表达式，匹配就会失败，函数返回None
    re.search()会匹配整个字符串，直到找到一个匹配的对象，匹配结束后没找到匹配值才会返回None
"""
