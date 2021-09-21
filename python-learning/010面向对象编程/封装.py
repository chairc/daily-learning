#!/usr/bin/env python
# -*- coding:utf-8 -*-
class Student(object):
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade


student = Student('小明', 100)


# 封装到方法中
def student_info(student):
    print(f'学生：{student.name}，成绩：{student.grade}')


student_info(student)
