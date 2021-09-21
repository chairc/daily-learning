#!/usr/bin/env python
# -*- coding:utf-8 -*-

import datetime
import time

# 2.datetime模块

# today()：返回当前本地时间
print('********today()********')
print(f'{datetime.datetime.today()}')

# now([tz])：返回一个datetime对象，tz参数是指时区的本地时间
print('********now([tz])********')
print(f'{datetime.datetime.now()}')

# utcnow()：返回一个当前UTC时间的datetime对象
print('********utcnow()********')
print(f'{datetime.datetime.utcnow()}')

# fromtimestamp(timestamp[,tz])：根据时间戳创建一个datetime对象
print('********fromtimestamp(timestamp[,tz])********')
print(f'{datetime.datetime.fromtimestamp(time.time())}')

# utcfromtimestamp(timestamp)：根据UTC时间戳创建一个datetime对象
print('********utcfromtimestamp(timestamp)********')
print(f'{datetime.datetime.utcfromtimestamp(time.time())}')

# strptime(date_string,format)：将格式字符串转换为datetime对象
print('********strptime(date_string,format)********')
dt = datetime.datetime.now()
print(f"{dt.strptime(str(dt), '%Y-%m-%d %H:%M:%S.%f')}")

# strftime(format)：将字符串转换为datetime对象
print('********strftime(format)********')
dt = datetime.datetime.now()
print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}")