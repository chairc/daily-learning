#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
异常处理提供了try...except...else语句
try:
    <语句>
except <异常名称1>
    <语句>
except <异常名称2>
    <语句>
else:
    <语句>

如果try语句执行时没有发生异常，就会执行else语句后的语句
"""


# 异常中的else
def else_exception(x, y):
    try:
        a = x / y
    except:
        print('异常！除数不能为0')
    else:
        print('a=', a)


print('********异常中的else********')
else_exception(2, 1)
else_exception(2, 0)


# 自定义异常：异常命名以Error结尾
class CustomError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return '这是自定义异常'


def custom_error():
    try:
        raise CustomError
    except CustomError as e:
        print(e)


print('********自定义异常********')
custom_error()


# finally子句
def use_finally(x, y):
    try:
        a = x / y
    except ZeroDivisionError:
        print('异常！除数不能为0')
    finally:
        print('finally继续执行')


print('********finally子句********')
use_finally(2, 0)
