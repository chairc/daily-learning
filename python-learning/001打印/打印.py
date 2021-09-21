#!/usr/bin/env python
# -*- coding:utf-8 -*-
print("hello world")
print(123)
print(12.3)
print(1 + 2)
print(100 / 3)
# 只取整数部分
print(100 // 3)
print(100 % 3)

print(3.3 * 101)
print(int(3.3 * 101))
print(float(3.3 * 100))
print(float(int(352.1)))

# 复数complex(x)，实部x，虚部0
print(complex(1))
# 复数complex(x,y)，实部x，虚部y
print(complex(1, 2))

# 赋值
x = 'xiao'
print(x)

# type函数确定变量类型
print(type(123))
print(type(1.0))
print(type('xxx'))

# 输出到文件
# fp = open("C:/Users/lenovo/Desktop/测试环境/python学习/001打印/print.txt", "a+")
# print("这是一个测试打印!", file=fp)
# fp.close()

# 不换行输出
print("hello", "world")
