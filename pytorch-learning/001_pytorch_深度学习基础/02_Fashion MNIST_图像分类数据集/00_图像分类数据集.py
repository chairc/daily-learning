#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython import display
import time
import sys

# 获取数据集
print('********获取数据集********')
root_url = 'C:/Users/lenovo/Desktop/Testing environment/pytorch learning/Datasets/FashionMNIST'

mnist_train = torchvision.datasets.FashionMNIST(root=root_url, train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=root_url, train=False, download=True,
                                               transform=transforms.ToTensor())
print('mnist_train类型    ->  ', type(mnist_train))
print('mnist_train长度    ->  ', len(mnist_train))
print('mnist_test长度    ->  ', len(mnist_test))

# 访问任意一个样本
feature, label = mnist_train[0]
print('mnist_train样本1feature ->', feature.shape)
print('mnist_train样本1label   ->', label)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def use_svg_display():
    # 用矢量图表示
    display.set_matplotlib_formats('svg')


def show_fashion_mnist(images, labels):
    use_svg_display()
    # _表示忽略的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 前十个样本的图像内容和文本标签
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取小批量样本
print('********读取小批量样本********')
# 批量大小
batch_size = 256
if sys.platform.startswith('win'):
    # 0表示不需要额外的进程来加速读取数据
    num_workers = 0
else:
    num_workers = 4
# 训练/测试迭代器
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('训练数据所需时间 ->  %.2f sec' % (time.time() - start))
