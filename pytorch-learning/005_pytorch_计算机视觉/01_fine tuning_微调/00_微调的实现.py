#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
import torchvision
from torch import nn
from d2lzh_pytorch import torch_d2l as d2l

"""
    微调通过使用在大数据上得到的预训练好的模型来初始化模型权重来完成提升精度
    预训练模型质量很重要
    微调涌常速度更快、精度更高
    一般来说，微调参数使用较小的学习率，从头开始训练的输出层用更大的学习率

"""
# 获取数据，下载失败需要手动复制到浏览器
# http://d2l-data.s3-accelerate.amazonaws.com/hotdog.zip
# d2l.DATA_HUB['hotdog'] = (d2l.DATA_URLDATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
# data_dir = d2l.download_extract('hotdog')

image_path = '../../Datasets/data/hotdog'

# 读取数据
train_images = torchvision.datasets.ImageFolder(f'{image_path}/train')
test_images = torchvision.datasets.ImageFolder(f'{image_path}/test')

# 显示前8后8张照片
hotdogs = [train_images[i][0] for i in range(8)]
not_hotdogs = [train_images[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# 使用3个RGB通道的均值和标准偏差
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)
# 数据增强
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

# 定义和初始化模型
finetune_net = torchvision.models.resnet18(pretrained=True)
# 修改模型最后一层输出
finetune_net.fc = nn.Linear(
    finetune_net.fc.in_features, 2
)
nn.init.xavier_uniform(finetune_net.fc.weight)


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(f'{image_path}/train', transform=train_augs),
        batch_size=batch_size,
        shuffle=True
    )
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(f'{image_path}/test', transform=test_augs),
        batch_size=batch_size
    )
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        # 最后一层用十倍的学习率进行较快的学习
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


# 使用预训练
# train_fine_tuning(finetune_net, 5e-5)

# 未使用预训练
scrtch_net = torchvision.models.resnet18()
scrtch_net.fc = nn.Linear(
    scrtch_net.fc.in_features, 2
)
train_fine_tuning(scrtch_net, 5e-4, param_group=False)
