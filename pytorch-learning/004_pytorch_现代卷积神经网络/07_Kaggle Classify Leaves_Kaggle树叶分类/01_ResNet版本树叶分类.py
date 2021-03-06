#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib_inline.backend_inline
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

"""
    RGB图、数据增强、k折交叉验证（未做）
"""

# 模式
mode_is_train_and_valid = True

# 主路径
root = 'C:/Users/lenovo/Desktop/Testing environment/pytorch learning/Datasets/data/kaggle_classify_leaves'
# 训练路径
train_path = f'{root}/train.csv'
# 测试路径
test_path = f'{root}/test.csv'
# 图片路径
image_path = f'{root}/'
# 模型路径
# model_path = './model_path/ResNet_Model'
model_path = 'F:/Model/Classify Leaves'

if mode_is_train_and_valid:
    # 时间戳
    time_name = int(time.time())
    # 保存模型路径
    save_model_path = f'{model_path}/{time_name}'
    # 在模型路径下创建时间戳文件夹
    os.mkdir(save_model_path)
    # 载入模型路径
    load_model_path = f'{save_model_path}/best_accuracy_model.pth'
    # 保存格式
    save_csv_path = f'{save_model_path}/submission.csv'
    # 保存损失和准确率图片地址
    save_loss_accuracy_jpg_path = f'{save_model_path}/loss_accuracy.jpg'
else:
    # 载入模型路径
    load_model_path = f'{model_path}/1637732396/epoch_69_loss_0.308_accuracy_0.918.pth'
    # 保存格式
    save_csv_path = f'{model_path}/1637732396/submission.csv'


# 绘图
def use_svg_display():
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def draw_svg(x_label, y_label, x1_vals, y1_vals, x2_vals, y2_vals, x3_vals, y3_vals, x4_vals, y4_vals, legend=None,
             figsize=(8, 6)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 绘制点
    plt.plot(x1_vals, y1_vals, '-', x2_vals, y2_vals, '--', x3_vals, y3_vals, '-', x4_vals, y4_vals, '--')
    plt.legend(legend)
    plt.grid(linestyle='-.')
    plt.savefig(save_loss_accuracy_jpg_path)


def set_figsize(figsize=(8, 6)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 读取文件
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
print(f'训练样本量  ->  {train_data.shape}')
print(f'测试样本量  ->  {test_data.shape}')
print(train_data['label'].value_counts())
train_label = sorted(list(set(train_data['label'])))
print(f'train_label  ->  {train_label}')
num_label = len(train_label)
print(f'前十个train_label  ->  {train_label[:10]}')
# 把label转换成数字
num_label_to_num = dict(zip(train_label, range(num_label)))
print(f'label转换成数字  ->  {num_label_to_num}')
# 数字转换成label
num_to_label = {v: k for k, v in num_label_to_num.items()}
print(f'数字转换成label  ->  {num_to_label}')


# 数据处理
class ClassifyLeavesDataSet(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode
        # 数据信息，pd读取csv文件不包含表头
        self.data_info = pd.read_csv(csv_path, header=None)
        # 数据总长度
        self.data_length = len(self.data_info.index - 1)
        # 训练数据集长度，在训练集中选择80%作为训练数据集，20%作为验证数据集
        self.train_length = int(self.data_length * (1 - valid_ratio))
        if mode == 'train':
            # 读取第一列图片路径，不包括头
            self.train_image_path = np.array(self.data_info.iloc[1:self.train_length, 0])
            # 读取第二列树叶类型，不包括头
            self.train_image_label = np.array(self.data_info.iloc[1:self.train_length, 1])
            # 图片路径和图片类型的数组
            self.image_array = self.train_image_path
            self.label_array = self.train_image_label
        elif mode == 'valid':
            # 读取第一列图片路径，不包括头
            self.valid_image_path = np.array(self.data_info.iloc[self.train_length:, 0])
            # 读取第二列树叶类型，不包括头
            self.valid_image_label = np.array(self.data_info.iloc[self.train_length:, 1])
            # 图片路径和图片类型的数组
            self.image_array = self.valid_image_path
            self.label_array = self.valid_image_label
        elif mode == 'test':
            # 读取测试图片路径
            self.test_image_path = np.array(self.data_info.iloc[1:, 0])
            # 图片路径数组
            self.image_array = self.test_image_path
        self.real_length = len(self.image_array)
        print(f'完成读取{mode}数据集中的{self.real_length}个样本')

    def __getitem__(self, item):
        # 获取单独一个图像路径
        one_image_path_name = self.image_array[item]
        # 读取这个路径上的图片
        read_image = Image.open(self.file_path + one_image_path_name)
        # 对图片进行数据增强
        if self.mode == 'train':
            # 如果模式是train，随机进行翻转
            transform = transforms.Compose([
                # 随机裁剪图片，所得图片为原始面积的0.08到1之间，高宽比在3/4和4/3之间，缩放图片以创建224*224的新图像
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor()
            ])
        else:
            # 如果模式是valid和test
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        # 输出数据增强后的图片
        read_image = transform(read_image)
        if self.mode == 'test':
            return read_image
        else:
            label = self.label_array[item]
            # 把返回的图片的label转换为对应的数字
            number_label = num_label_to_num[label]
        # 返回索引后对应的图片数据和对应的label数字
        return read_image, number_label

    def __len__(self):
        return self.real_length


train_dataset = ClassifyLeavesDataSet(train_path, image_path, mode='train')
valid_dataset = ClassifyLeavesDataSet(train_path, image_path, mode='valid')
test_dataset = ClassifyLeavesDataSet(test_path, image_path, mode='test')
print(f'train_dataset  ->  {train_dataset}')
print(f'valid_dataset  ->  {valid_dataset}')
print(f'test_dataset  ->  {test_dataset}')

# 定义dataloader
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=16, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'当前使用的设备为  ->  {device}')


# pytorch里resnet模型
def resnet_model():
    # ResNet18
    # resnet_model = torchvision.models.resnet18(pretrained=True)
    # ResNet34
    # resnet_model = torchvision.models.resnet34(pretrained=True)
    # ResNet50
    resnet_model = torchvision.models.resnet50(pretrained=True)
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, 176)
    )
    return resnet_model


# pytorch的ResNet
model = resnet_model()
model = model.to(device)
model.device = device


# 训练和验证
def train_and_valid():
    # 设置模型参数
    lr, num_epochs, weight_decay = 1e-4, 100, 1e-3
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_accuracy = 0.0
    best_epoch = 0
    # 图片显示的平均值
    train_loss_show_list = []
    train_accuracy_show_list = []
    valid_loss_show_list = []
    valid_accuracy_show_list = []
    for epoch in range(num_epochs):
        # 训练
        print(f'\n————————————————————第 {epoch + 1} 次训练开始————————————————————')
        # 模型在训练模式
        model.train()
        # 训练的损失和准确率暂存列表
        train_loss = []
        train_accuracy = []
        for batch in tqdm(train_dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            l_train = loss(logits, labels)
            optimizer.zero_grad()
            l_train.backward()
            optimizer.step()
            # 计算当前批次的准确度
            accuracy_train = (logits.argmax(dim=-1) == labels).float().mean()
            # 训练损失与准确度不断叠加
            train_loss.append(l_train.item())
            train_accuracy.append(accuracy_train)
        # 训练损失与训练准确度求平均
        train_loss_mean = sum(train_loss) / len(train_loss)
        train_accuracy_mean = sum(train_accuracy) / len(train_accuracy)
        # 把训练平均值和准确率输入进最终显示数组中
        train_loss_show_list.append(train_loss_mean)
        train_accuracy_show_list.append(train_accuracy_mean.data.cpu().numpy())
        print(f'train epochs  ->  {epoch + 1}, train loss  ->  {train_loss_mean:.3f}, '
              f'train accuracy  ->  {train_accuracy_mean:.3f}')
        torch.save(model.state_dict(),
                   f'{save_model_path}/epoch_{epoch + 1}_loss_{train_loss_mean:.3f}_accuracy_{train_accuracy_mean:.3f}.pth')
        print(f'已保存训练{epoch + 1}模型到{model_path}中')
        print(f'————————————————————第 {epoch + 1} 次训练结束————————————————————')

        # 验证
        print(f'————————————————————第 {epoch + 1} 次验证开始————————————————————')
        model.eval()
        # 验证的损失和准确率暂存列表
        valid_loss = []
        valid_accuracy = []
        for batch in tqdm(valid_dataloader):
            images, labels = batch
            images = images.to(device)
            with torch.no_grad():
                logits = model(images)
            # 计算损失
            labels = labels.to(device)
            l_valid = loss(logits, labels)
            accuracy_valid = (logits.argmax(dim=-1) == labels).float().mean()
            valid_loss.append(l_valid.item())
            valid_accuracy.append(accuracy_valid)
        valid_loss_mean = sum(valid_loss) / len(valid_loss)
        valid_accuracy_mean = sum(valid_accuracy) / len(valid_accuracy)
        # 把验证平均值和准确率输入进最终显示数组中
        valid_loss_show_list.append(valid_loss_mean)
        valid_accuracy_show_list.append(valid_accuracy_mean.data.cpu().numpy())
        print(f'valid epochs  ->  {epoch + 1}, valid loss  ->  {valid_loss_mean:.3f}, '
              f'valid accuracy  ->  {valid_accuracy_mean:.3f}')
        if valid_accuracy_mean > best_accuracy:
            best_accuracy = valid_accuracy_mean
            best_epoch = epoch
            torch.save(model.state_dict(), f'{save_model_path}/best_accuracy_model.pth')
            print(f'已保存{best_accuracy}模型到{model_path}中')
        print(f'————————————————————第 {epoch + 1} 次验证结束————————————————————')
        print(f'第 {epoch + 1} 次训练与验证结束，效果最好的模型为第 {best_epoch + 1} 次准确率为{best_accuracy:.3f}的模型')
    # 输出图片
    train_loss_show_list = np.array(train_loss_show_list)
    valid_loss_show_list = np.array(valid_loss_show_list)
    # 绘图
    x_val = range(1, num_epochs + 1)
    draw_svg(
        'epochs',
        'loss/accuracy',
        x_val,
        train_loss_show_list,
        x_val,
        train_accuracy_show_list,
        x_val,
        valid_loss_show_list,
        x_val,
        valid_accuracy_show_list,
        ['train loss', 'train accuracy', 'valid loss', 'valid accuracy'],
        figsize=(16, 12)
    )


# 预测
def predict():
    print(f'\n————————————————————预测开始————————————————————')
    # 读取保存的最好的模型文件
    model.load_state_dict(torch.load(load_model_path))
    model.eval()
    # 使用列表存储预测值
    predictions = []
    for batch in tqdm(test_dataloader):
        images = batch
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    predictions_num_to_label = []
    # 遍历预测值将数字转换为标签
    for i in predictions:
        predictions_num_to_label.append(num_to_label[i])
    # 读取测试数据
    test_data = pd.read_csv(test_path)
    test_data['label'] = pd.Series(predictions_num_to_label)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(save_csv_path, index=False)
    print(f'————————————————————预测结束，csv文件生成成功！————————————————————')


if mode_is_train_and_valid:
    train_and_valid()
    predict()
else:
    predict()
