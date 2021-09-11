#!/usr/bin/env python
# -*- coding:utf-8 -*-
class Animal(object):

    def run(self):
        print('Animal正在跑步...')

    # 这是私有__run方法，子类不能继承父类该方法
    def __run(self):
        print('这是一个私有run方法')

    # 通过类的函数调用类中私有方法
    def get_private_run(self):
        print('正在调用私有run方法')
        self.__run()

    def jump(self):
        print('Animal正在跳跃')


class Dog(Animal):
    def eat(self):
        print('Dog正在吃饭')


class Cat(Animal):
    pass


# 类的继承
print('********类的继承********')
dog = Dog()
dog.run()
dog.eat()
dog.jump()
print('------')
cat = Cat()
cat.run()
cat.jump()
cat.get_private_run()
