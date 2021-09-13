#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
1.格式化符号

    %y 两位数的年份表示（00-99）
    %Y 四位数的年份表示（000-9999）
    %m 月份（01-12）
    %d 月内中的一天（0-31）
    %H 24小时制小时数（0-23）
    %I 12小时制小时数（01-12）
    %M 分钟数（00-59）
    %S 秒（00-59）
    %a 本地简化星期名称
    %A 本地完整星期名称
    %b 本地简化的月份名称
    %B 本地完整的月份名称
    %c 本地相应的日期表示和时间表示
    %j 年内的一天（001-366）
    %p 本地A.M.或P.M.的等价符
    %U 一年中的星期数（00-53）星期天为星期的开始
    %w 星期（0-6），星期天为 0，星期一为 1，以此类推。
    %W 一年中的星期数（00-53）星期一为星期的开始
    %x 本地相应的日期表示
    %X 本地相应的时间表示
    %Z 当前时区的名称
    %% %号本身

2.时间元组
    0	tm_year	2008
    1	tm_mon	1 到 12
    2	tm_mday	1 到 31
    3	tm_hour	0 到 23
    4	tm_min	0 到 59
    5	tm_sec	0 到 61 (60或61 是闰秒)
    6	tm_wday	0到6 (0是周一)
    7	tm_yday	1 到 366(儒略历)
    8	tm_isdst	-1, 0, 1, -1是决定是否为夏令时的旗帜

3.时间格式转化
    转化格式-1                            使用方法
    Format string   ->  struct_time     strptime()
    struct_time     ->  Format string   strftime()
    struct_time     ->  Timestamp       mktime()
    Timestamp       ->  struct_time     localtime()/gmtime()

    转化格式-2                                        使用方法
    struct_time     ->  %a %b %d %H:%M:%S %Y串       asctime()
    Timestamp       ->  %a %b %d %H:%M:%S %Y串       ctime()

"""
# 时间包
import time

# 1.time模块

# time()：用于返回当前时间的时间戳
print('********time()********')
print(f'{time.time()}')

# localtime([secs])：用于格式化时间戳为本地时间
print('********localtime([secs])********')
print(f'{time.localtime()}')

# gmtime([secs])：用于将一个时间转换为UTC时区
print('********gmtime([secs])********')
print(f'{time.gmtime()}')

# mktime(t)：用于执行与gmtime()、localtime()相反操作，接受struct_time对象作为参数，返回用秒数表示时间的浮点数
print('********mktime(t)********')
t = (2021, 9, 12, 16, 15, 10, 6, 255, 0)
print(f'{time.mktime(t)}')

# asctime([t])：用于接受时间元组并返回一个可读长度为24个字符的字符串
print('********asctime([t])********')
current_time = time.localtime()
print(f'{time.asctime(current_time)}')

# ctime([secs])：用于把一个时间戳（按秒计算的浮点数）转化为time.asctime()的形式
print('********ctime([secs])********')
print(f'{time.ctime()}')

# sleep(secs)：用于推迟调用线程的运行，secs为挂起时间
print('********sleep(secs)********')
print(f'{time.ctime()}')
time.sleep(1)
print(f'{time.ctime()}')

# clock()被perf_counter()替代：用于以浮点数计算的秒数返回当前CPU时间，来衡量不同程序的耗时，Python3.8之后被perf_counter()替代
print('********clock()被perf_counter()替代********')


def sleep():
    time.sleep(1)


# clock被perf_counter替代
t1 = time.perf_counter()
sleep()
print(f't1:{(time.perf_counter() - t1)}')

t2 = time.time()
sleep()
print(f't2:{(time.time() - t2)}')

# strftime(format[,t])：用于接收时间元组，并返回以可读字符串表示的当地时间，格式由参数format决定
print('********strftime(format[,t])********')
t = (2021, 9, 12, 16, 15, 10, 6, 255, 0)
t = time.mktime(t)
print(time.strftime('%b %d %Y %H:%M:%S', time.gmtime(t)))

# strptime(string[,format])：用于根据指定的格式把一个时间字符串解析为时间元组
print('********strptime(string[,format])********')
struct_time = time.strptime("11 Sep 12", "%d %b %y")
print(f'{struct_time}')
