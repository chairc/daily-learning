#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib_inline.backend_inline
import matplotlib.pyplot as plt

# 获取数据集
print('********获取数据集********')
root = 'C:/Users/lenovo/Desktop/Testing environment/pytorch learning/Datasets/data/kaggle_house_pred'
print(f'pytorch version ->  {torch.__version__}')
torch.set_default_tensor_type(torch.FloatTensor)

# 读取文件
train_data = pd.read_csv(f'{root}/kaggle_house_pred_train.csv')
test_data = pd.read_csv(f'{root}/kaggle_house_pred_test.csv')

# 训练数据和测试数据的样本量和特征量
print(train_data.shape)
print(test_data.shape)

# 测试前4个样本的前4个特征和后2个特征和标签
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 将所有训练数据与测试数据的79个特征样本连结
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 预处理数据
print('********预处理数据********')
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个数值特征的均值变为0，可直接用0来代替缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 训练模型
print('********训练模型********')
loss = nn.MSELoss()


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


# 对数均方根误差实现
def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        # 公式
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证
print('********K折交叉验证********')


def use_svg_display():
    # 该方法已弃用
    # display.set_matplotlib_formats('svg')
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    # plt.show()


def set_figsize(figsize=(8, 6)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # 切片函数
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        # 恰好i==j，直接载入测试集
        if j == i:
            X_valid, y_valid = X_part, y_part
        # 载入训练集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        # 将各个训练集相加
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls,
                     ['train', 'valid'])
        print(f'fold ->  {i:d}, train rmse ->  {train_ls[-1]:f}, valid rmse    ->  {valid_ls[-1]:f}')
    return train_l_sum / k, valid_l_sum / k


# 模型选择
print('********模型选择********')
k, num_epochs, lr, weight_decay, batch_size = 5, 150, 12, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-fold validation: avg train rmse ->  {train_l:f}, avg valid rmse ->  {valid_l:f}')

# 模型预测
print('********模型预测********')


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print(f'train rmse  ->  {train_ls[-1]:f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv(f'{root}/submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
