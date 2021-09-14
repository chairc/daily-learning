#!/usr/bin/env python
# -*- coding:utf-8 -*-
money = 10
total = 0
for x in range(0, int(money / 5) + 1):
    for y in range(0, int(money / 2) + 1):
        for z in range(0, int(money / 1) + 1):
            for m in range(0, int(money / 0.5) + 1):
                for n in range(0, int(money / 0.25) + 1):
                    if 5 * x + 2 * y + 1 * z + 0.5 * m + 0.25 * n == money:
                        total += 1
                        print(f'这是第{total}种方法，组合为{x}个5元，{y}个2元，{z}个1元，{m}个0.5元，{n}个0.25元')
