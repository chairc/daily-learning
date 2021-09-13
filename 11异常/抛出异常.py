#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Python使用raise语句抛出一个指定异常
"""
    重要异常类型
    Exception:常规错误基类
    AttributeError:对象没有这个属性
    IOError:输入输出操作失败
    IndexError:序列中没有此索引（index）
    KeyError:映射中没有此键
    NameError:未声明/初始化对象（没有属性）
    SyntaxError:Python语法错误
    SystemError:一般解释器系统错误
    ValueError:传入无效参数
"""

try:
    raise NameError('这是一个声明错误')
except NameError:
    print('声明错误发生了')
    # 如果不加raise，则输出声明错误发生后结束
    raise