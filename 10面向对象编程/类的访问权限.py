#!/usr/bin/env python
# -*- coding:utf-8 -*-
class StudentPublicClass(object):
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def show(self):
        print(f'学生：{self.name}，成绩：{self.grade}')


class StudentPrivateClass(object):
    def __init__(self, name, grade):
        # 在参数前加 __ python会自动设置为私有变量
        self.__name = name
        self.__grade = grade

    def show(self):
        print(f'学生：{self.__name}，成绩：{self.__grade}')

    def update_grade(self, grade):
        self.__grade = grade


class PublicPrivateMethodClass(object):
    def __init__(self):
        pass

    # 私有方法
    def __private_function(self):
        print('这是一个私有方法')

    # 公有方法
    def public_function(self):
        print('这是一个公有方法')
        print('公有方法调用私有方法')
        self.__private_function()
        print('公有方法调用私有方法结束')


# 类的访问权限
print('********类的访问权限********')
student_public = StudentPublicClass('小明', 100)
student_public.show()
# 可在外部修改，使用
student_public.grade = 99
student_public.show()
print('------')
student_private = StudentPrivateClass('小刚', 99)
student_private.show()
# 需要用类中方法进行修改
student_private.update_grade(100)
student_private.show()
print('------')
public_private_method = PublicPrivateMethodClass()
public_private_method.public_function()