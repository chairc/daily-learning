#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 字符串格式化

# 1.字符串格式化符号为%
print('********字符串格式化符号为%********')
print('我是%d级研究生' % 2021)

# 2.字符串格式化元组
print('********字符串格式化元组********')
print('今年是%d年，我是%s，出生于%d年' % (2021, '学生', 1998))

# 3.字段宽度、精度
print('********字段宽度、精度********')
# 字段宽度为10
print('圆周率为%10f' % 3.1415926)
# 字段宽度为10，保留2位小数
print('圆周率为%10.2f' % 3.1415926)
# 没有字段宽度，保留2位小数
print('圆周率为%.2f' % 3.1415926)
# 打印前5个字符
print('%.5s' % 'hello,world')

# 4.符号、对齐和0填充
print('********符号、对齐和0填充********')
# 字段宽度为10，用0填充，保留两位小数
print('圆周率为%010.2f' % 3.1415926)
# 左边空格填充
print('圆周率为%10.2f' % 3.1415926)
# 右边空格填充
print('圆周率为%-10.2f' % 3.1415926)