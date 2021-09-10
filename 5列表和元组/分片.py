#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 分片
number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 从左开始计数输出第2,3个元素
print(number[1:3])
# 从右开始计数输出倒数第2,3个元素
print(number[-3:-1])
# 从左开始输出从第1个元素输出到第9个
print(number[0:9])
print(number[-9:0])
print(number[-10:])
# 第一个元素输出，输出全部结果
print(number[0:])
# 最后一个元素为第一个，输出为空
print(number[:0])
# 获取整个数组
print(number[:])
# 步长，默认为1，范围从1到10，输出间隔为1
print(number[0:10:1])
# 步长为2，范围从1到10，输出间隔为2
print(number[0:10:2])
# 步长的前两个索引可为空
print(number[::3])
# 步长为2，范围为反向从10到0，输出间隔为2
print(number[10:0:-2])
# 第2个索引为0，取不到第1个元素
print(number[10:0:-1])
# 设置第2个索引为空，可以取到第1个元素
print(number[10::-1])
