#!/usr/bin/env python
# -*- coding:utf-8 -*-

class Student(object):

    def __init__(self, name):
        self.name = name

    # __str__
    def __str__(self):
        return f'学生名称：{self.name}'


class Fib(object):
    def __init__(self):
        # 初始化两个计数器a,b
        self.a, self.b = 0, 1

    # __iter__
    def __iter__(self):
        # 实例化本身就是迭代自己，返回自己
        return self

    def __next__(self):
        # 计算下一个值
        self.a, self.b = self.b, self.a + self.b
        # 退出循环条件
        if self.a > 100:
            raise StopIteration()
        return self.a


print('********__str__********')
print(Student('小明'))
print('********__iter__********')
for n in Fib():
    print(n)