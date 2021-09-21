#!/usr/bin/env python
# -*- coding:utf-8 -*-

# while循环
print('********while循环********')
n = 1
while n <= 3:
    print('这是第%d遍输出' % n)
    n += 1

# for循环
print('********for循环********')
a = ['A', 'B', 'C']
m = 1
for i in a:
    print(f'当前第{m}次输出字母为{i}')
    m += 1

# 循环遍历字典
print('********循环遍历字典********')
tups = {'name': '小智', 'number': '1002'}
for tup in tups:
    print(f'{tup}:{tups[tup]}')
print('------')
# 序列解包
for key, value in tups.items():
    print(f'{key}:{value}')
print('------')
# 序列索引迭代
tups2 = ['aaa', 'bbb', 'ccc']
for i in range(len(tups2)):
    print('%s' % tups2[i])

# break和continue
print('********break和continue********')
for i in range(10):
    if i == 5:
        print('当前为continue语句')
        continue
    elif i == 7:
        print('当前为break语句')
        break
    print(f'这是第{i+1}次输出')

# 循环中的else子句
print('********循环中的else子句********')
for num in range(10, 20):  # 迭代 10 到 20 之间的数字
    for i in range(2, num):  # 根据因子迭代
        if num % i == 0:  # 确定第一个因子
            j = num / i  # 计算第二个因子
            print('%d 等于 %d * %d' % (num, i, j))
            break  # 跳出当前循环
    else:  # 循环的 else 部分
        print(num, '是一个质数')

# pass语句：空语句，保持程序结构的完整性
print('********pass语句********')
name = 'xiaogang'
if name == 'xiaoming':
    print('ok')
elif name == 'xiaogang':
    pass
else:
    print('None')