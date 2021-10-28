#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.distributions as multinomial
import matplotlib_inline.backend_inline
import matplotlib.pyplot as plt


# 所需方法
def use_svg_display():
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# 基本概率论
print('********基本概率论********')
"""
    概率分布中抽取样本称为抽样（sampling）
    概率分配给一些离散选择的分布称为多项分布（multinational distribution）
    
    概率论公理：
        概率可以被认为是将集合映射到真实值的函数。在给定的样本空间S中，事件A的概率表示为P(A)，满足以下属性：
            1. 对于任意事件A，概率不会是负数，P(A) ≥ 0
            2. 整个样本空间S的概率为1，P(S) = 1
            3. 对于互斥事件的任意⼀个可数序列A1, A2, . . .，序列中任意⼀个事件发⽣的概率等于它们各⾃发⽣的概率之和
            
    随机变量：
        随机变量可以使任何数量，并且是不确定性的
        离散随机变量（如骰子的每一面）和连续随机变量（如人的身高体重）是有区别的
        
    联合概率：
        联合概率P(A = a, B = b)，对于任何a和b的取值，P(A = a, B = b) ≤ P(A = a)，如果同时发生A=a和B=b，A=a必须发生，B=b必须发生，反之也同理。
        A=a和B=b同时发生的可能性不大于A=a或者B=b单独发生的可能性
        
    条件概率：
        比率：0 ≤ P(A = a, B = b)/P(A = a) ≤ 1，我们称这个比率为条件概率
        使用P(B=b|A=a)表示，B=b的概率，前提是A=a已经发生
        
    贝叶斯定理：
        使⽤条件概率的定义，我们可以得出统计学中最有⽤和最著名的⽅程之⼀：Bayes定理（Bayes’theorem）
        它如下所⽰。通过构造，我们有乘法规则，P(A, B) = P(B | A)P(A)。根据对称性，这也适⽤于P(A, B) = P(A | B)P(B)。
        假设P(B) > 0，求解其中⼀个条件变量，我们得到P(A | B) = P(B | A)P(A)/P(B)
        其中P(A, B)是⼀个联合分布，P(A | B)是⼀个条件分布。这种分布可以在给定值A = a, B = b上进⾏求值
        
"""
# 传入一个概率向量
fair_probs = torch.ones([6]) / 6
# 进行1次样本随机出现在6个元素中任意一个位置
print(f'1次样本随机 ->  {multinomial.Multinomial(1, fair_probs).sample()}')

# 例如进行10次样本随机
print(f'10次样本随机 ->  {multinomial.Multinomial(10, fair_probs).sample()}')

# 例如进行1000次样本随机
counts_1000 = multinomial.Multinomial(1000, fair_probs).sample()
print(f'1000次样本随机 ->  {counts_1000}')
# 对1000次样本随机进行估计，使得6个元素和为1
print(f'1000次样本随机作为真实估计 ->  {counts_1000 / 1000}')

"""
    如果我们的样本随机足够多，那么这6条线越会趋向于1/6
"""
# 对500次样本实验每组抽取10个样本
counts_500_10 = multinomial.Multinomial(10, fair_probs).sample((500,))
print(f'500次样本随机中抽取10个样本    ->  \n{counts_500_10}')
# dim=0是纵向压缩，500组样本实验按照列求和算出每列总数
cum_counts = counts_500_10.cumsum(dim=0)
# cum_counts.sum(dim=1, keepdims=True)是按照行计算算出每行的样本个数
# estimates是对样本数的估计
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
set_figsize((6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
    plt.axhline(y=0.167, color='black', linestyle='dashed')
    plt.gca().set_xlabel('Number of experiments')
    plt.gca().set_ylabel('Estimated probability')
    plt.legend()
