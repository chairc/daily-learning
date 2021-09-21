#!/usr/bin/env python
# -*- coding:utf-8 -*-
class Animal(object):

    def run(self):
        print('Animal正在跑步...')

    def jump(self):
        print('Animal正在跳跃')


class Dog(Animal):
    # 对Animal中的run进行覆盖
    def run(self):
        print('Dog正在跑步...')


class Cat(Animal):
    # 对Animal中的run进行覆盖
    def run(self):
        print('Cat正在跑步...')


# 多态
print('********多态********')
dog = Dog()
dog.run()
print('------')
cat = Cat()
cat.run()