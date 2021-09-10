#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 定义函数
"""
def 函数名(参数列表):
   函数体
"""


# 一般函数
def hello_world():
    print('hello world')


# 带有返回值的函数
def mix_number(num1, num2):
    num = num1 + num2
    return num


# 定义空函数
def do_nothing():
    pass


# 带有默认值的函数
def default_value_function(var1, var2=10, var3='这是默认参数'):
    print(var1)
    print(var2)
    print(var3)


print('********一般函数********')
hello_world()
print('********带有返回值的函数********')
print(mix_number(10, 20))
print('********定义空函数********')
do_nothing()
print('********带有默认值的函数********')
default_value_function(1)
default_value_function(2, 20, '自定参数')
