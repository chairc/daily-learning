#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 类的创建
class HelloWorldClass(object):
    # 类的属性
    i = 123

    # 类的hello world方法
    def hello_world(self):
        return 'hello world'


# 类的使用
print('********类的使用********')
hello_world = HelloWorldClass()
# 调用类的属性
print(f'{hello_world.i}')
# 调用类的方法
print(f'{hello_world.hello_world()}')

