#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Python中支持处理多个异常
try:
    <语句>
except <异常名称1>
    <语句>
except <异常名称2>
    <语句>
.....

一个try语句可能包含多个except子句，分别处理不同异常，但最多只有一个分之except语句会被执行

"""


# 捕捉多个异常
def mult_exception1(x, y):
    try:
        a = x / y
        b = name
    except ZeroDivisionError:
        print('执行ZeroDivisionError')
    except NameError:
        print('执行NameError')


# 捕捉多个异常
def mult_exception2(x, y):
    try:
        b = name
        a = x / y
    except ZeroDivisionError:
        print('执行ZeroDivisionError')
    except NameError:
        print('执行NameError')


# 捕捉多个异常
print('********捕捉多个异常********')
# 此语句执行ZeroDivisionError异常
mult_exception1(2, 0)
# 此语句执行NameError异常
mult_exception2(2, 0)


# 一个块捕捉多个异常
def model_exception(x, y):
    try:
        b = name
        a = x / y
    except(ZeroDivisionError, NameError, TypeError):
        print('发生了ZeroDivisionError或NameError或TypeError')


print('********一个块捕捉多个异常********')
model_exception(2, 0)


# 捕捉对象
def object_exception1(x, y):
    try:
        b = name
        a = x / y
    except(ZeroDivisionError, NameError, TypeError) as e:
        print(e)


def object_exception2(x, y):
    try:
        a = x / y
        b = name
    except(ZeroDivisionError, NameError, TypeError) as e:
        print(e)


print('********捕捉对象********')
object_exception1(2, 0)
object_exception2(2, 0)

