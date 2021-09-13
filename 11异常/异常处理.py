#!/usr/bin/env python
# -*- coding:utf-8 -*-

def exp_exception(x, y):
    try:
        a = x / y
        print('a=', a)
        return a
    except Exception:
        print('异常！除数不能为0')


exp_exception(2, 0)
