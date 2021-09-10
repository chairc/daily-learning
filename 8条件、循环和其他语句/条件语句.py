#!/usr/bin/env python
# -*- coding:utf-8 -*-

# if条件语句
a = 3
if 4 < a < 8:
    if a == 7:
        print('这是7')
    elif a == 6:
        print('这是6')
    else:
        print('这是5')
elif 0 < a <= 4:
    if a == 4:
        print('这是4')
    elif a == 3:
        print('这是3')
    elif a == 2:
        print('这是2')
    else:
        print('这是1')
else:
    print('这不在1和7的范围内')
