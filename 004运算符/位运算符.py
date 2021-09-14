#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 位运算符
a = 60
b = 13
c = 0
# 按位与，相应位为1，则为1，否则为0
c = a & b
print(c)
# 按位或，相应位有一个是1，则为1
c = a | b
print(c)
# 按位异或，相应位相反，则为1
c = a ^ b
print(c)
# 按位取反,1变0,0变1
c = ~a
print(c)
# 左移，高位丢弃，低位补0
c = a << 2
print(c)
# 右移
c = a >> 2
print(c)