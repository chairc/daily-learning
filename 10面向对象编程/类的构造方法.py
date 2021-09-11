#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 类的构造方法
class HelloXiaoMingClass(object):
    # 类的构造方法
    def __init__(self, name):
        self.name = name

    def hello_xiaoming(self):
        return 'hello,' + self.name


# 类的一般初始化
class DefaultInitClass(object):
    def __init__(self):
        print('初始化中，这是__init__方法')

    def show(self):
        print('类中定义方法，需要实例化调用')


# 类的带参数的初始化
class ParamInitClass(object):
    def __init__(self):
        print('初始化中，这是不带参数的__init__方法')

    def __init__(self, param):
        print(f'初始化中，这是带参数的__init__方法，参数值{param}')


# 调用类的构造方法
print('********类的构造方法********')
hello_xiaoming = HelloXiaoMingClass('xiaoming')
print(f'{hello_xiaoming.hello_xiaoming()}')

# 类的初始化
print('********类的初始化********')
default_init = DefaultInitClass()
print('初始化结束')
default_init.show()
print('------')
ParamInitClass('hello')
print('实例化结束')
