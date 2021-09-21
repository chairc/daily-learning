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


# 可变参数的函数
def variable_parameter_function1(val, *vartuple):
    print(val)
    for i in vartuple:
        print(f'可变参数：{i}')
    return


# 处理关键字的可变函数，关键字参数传入kwargs字典中（kwargs是传入值的复制）
def variable_parameter_function2(name, number, **kwargs):
    print(f'名称：{name}，学号：{number}，其他：{kwargs}')


# 组合参数的函数
def exp(p1, p2, df=0, *vart, **kwargs):
    print(f'p1={p1},p2={p2},df={df},vart={vart},kwargs={kwargs}')


# 递归函数
def fact(n):
    if n == 1:
        return 1
    return n * fact(n - 1)


# 匿名函数
lambda_val = lambda x, y: x + y


print('********一般函数********')
hello_world()
print('********带有返回值的函数********')
print(mix_number(10, 20))
print('********定义空函数********')
do_nothing()
print('********带有默认值的函数********')
default_value_function(1)
print('------')
default_value_function(2, 20, '自定参数')
print('********可变参数的函数********')
variable_parameter_function1('小明')
print('------')
variable_parameter_function1('小明', 22)
print('------')
variable_parameter_function1('小明', 22, '1002')
print('------')
other = {'城市': '济南', '爱好': '编程'}
variable_parameter_function2('小明', '1002', 城市=other['城市'], 爱好=other['爱好'])
print('********组合参数的函数********')
exp(1, 2)
print('------')
exp(1, 2, c=3)
print('------')
exp(1, 2, 3, 'a', 'b')
print('------')
exp(1, 2, 3, 'abc', x=9)
print('********递归函数********')
print(f'{fact(5)}')
print('********匿名函数********')
print(f'{lambda_val(1, 2)}')