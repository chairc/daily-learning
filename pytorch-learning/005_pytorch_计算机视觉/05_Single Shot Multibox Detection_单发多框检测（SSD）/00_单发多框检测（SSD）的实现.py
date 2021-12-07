#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2lzh_pytorch import torch_d2l as d2l


# 预测锚框的类别
# 预测总像素数 * 每个像素的锚框数 * (每个锚框预测的类别 + 背景类)
# 注：预测总像素数 = 图片的宽 * 图片的高
def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    特征图每个像素对应a锚框，每个锚框对应q个分类，单个像素就要a*(q+1)个预测信息
    这个信息，通过卷积核的多个通道来存储，所以这里进行卷操作
    图像分类只预测分类情况，所以接全连接层这里单个像素的预测结果太多，就用多个通道霁行

    :param num_inputs: 输入通道数
    :param num_anchors: 锚框的数量
    :param num_classes: 类别的数量
    :return:
    """
    # (num_classes + 1)：输入的类别 + 背景类
    # num_anchors * (num_classes + 1)：对每一个锚框预测有多少类
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


# 边界框预测层
def bbox_predictor(num_inputs, num_anchors):
    """
    预测到真实的偏移，需要为每个锚框预测4个偏移量

    :param num_inputs: 输入通道数
    :param num_anchors: 锚框的数量
    :return:
    """
    # num_anchors * 4：以每一个像素为中心的锚框大小乘4
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


# 连结多尺度的预测
def forward(x, block):
    return block(x)


# 举例子
# batch_size = 2， 通道数 = 8， feature_map = 20 * 20， 通道数 = 8，每个像素生成锚框数 = 5， 类别数 = 10
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
# batch_size = 2， 通道数 = 16， feature_map = 10 * 10， 通道数 = 16，每个像素生成锚框数 = 3， 类别数 = 10
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# 结果：[2, 55, 20, 20]  ->  55 = 5 * (10 + 1)
# 类别预测输出20 * 20 * 55个预测
print(f'Y1.shape  ->  {Y1.shape}')
# 结果：[2, 33, 20, 20]  ->  33 = 3 * (10 + 1)
# 类别预测输出10 * 10 * 33个预测
print(f'Y2.shape  ->  {Y2.shape}')


# 将四维变为二维
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


print(f'concat_preds([Y1, Y2]).shape  ->  {concat_preds([Y1, Y2]).shape}')


# 一般用于变换通道数，高和宽减半块
def down_sample_blk(in_channels, out_channels):
    blk = []
    # 进行两次卷积、批量归一化、ReLU
    for _ in range(2):
        # 卷积不改变高宽
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        # 不改变通道
        in_channels = out_channels
    # 最大池化层窗口为2，步长为2，高宽减半
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# 例如将高宽为20，通道数为3的Tensor放入down_sample_blk中，输出为通道数为10的高宽减半为10
print(f'forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape  ->  '
      f'{forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape}')


# 基本网络块
def base_net():
    blk = []
    # 通道数
    num_filters = [3, 16, 32, 64]
    # 循环num_filters - 1次，输出num_filters - 1 个down_sample_blk，原始图片会减少8倍
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


# 例如将高宽为256，通道数为3的Tensor放入base_net中，base_net有三个down_sample_blk
# 输出为通道数为num_filters列表最后一个数64，高宽减半为32
print(f'forward(torch.zeros((2, 3, 256, 256)), base_net()).shape  ->  '
      f'{forward(torch.zeros((2, 3, 256, 256)), base_net()).shape}')


# 完整的单发多框检测模型由五个模块组成
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


# 给每个块进行前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """
    参考SSD原理图可以看出每个block后都会生成锚框 + 类别预测 + 边界框预测
    :param X: 当前输入
    :param blk: 当前网络块
    :param size: 当前分辨率下锚框的尺度
    :param ratio: 当前分辨率下锚框的宽高比
    :param cls_predictor: 当前分辨率下类别预测
    :param bbox_predictor: 当前分辨率下边界框预测
    :return: 
    """
    # blk(X)算出特征图
    Y = blk(X)
    # 给定特征图的高宽后生成一些锚框
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


# 设置每个尺度的超参数的两个比例值的列表，每个尺度第一个数每次加0.17
# 注：0.272是根据 √(0.2 * 0.37)得出
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
# 模型5层，固定高宽比
ratios = [[1, 2, 0.5]] * 5
# 每个像素为中心生成4个锚框
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        # 每个阶段的输入通道
        idx_to_in_channels = [64, 128, 128, 128, 128]
        # 五个阶段，对每个阶段进行赋值
        for i in range(5):
            # setattr设置属性
            # 赋值语句self.blk_i = get_blk(i)，下面的setattr以此类推
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        # 五个阶段，将每个阶段的值使用getattr取出来
        # 并将每个阶段取出来的值代入blk_forward这个方法，得到每个阶段的返回值（锚框、分类预测，边界框预测）组成列表
        for i in range(5):
            # getattr(self, 'blk_%d' % i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # 所有锚框合并在一起
        anchors = torch.cat(anchors, dim=1)
        # 将cls_preds变为矩阵
        cls_preds = concat_preds(cls_preds)
        # 将cls_preds变为三维的，将最后一个类拿出来
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        # 将bbox_preds变为矩阵
        bbox_preds = concat_preds(bbox_preds)
        # 拿到每一层好的输出并且合并
        return anchors, cls_preds, bbox_preds


net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors  ->  ', anchors.shape)
print('output class preds  ->  ', cls_preds.shape)
print('output bbox preds  ->  ', bbox_preds.shape)

# 读取数据集和初始化
batch_size = 32
# 获取数据，下载失败需要手动复制到浏览器
# http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip
train_iter, _ = d2l.load_data_bananas(batch_size)
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评价函数
# 分类损失，交叉熵损失函数，不将loss叠加
cls_loss = nn.CrossEntropyLoss(reduction='none')
# 边界框损失，L1损失函数（预测值和真实值之差的绝对值），不将loss叠加
bbox_loss = nn.L1Loss(reduction='none')


# 计算loss
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """

    :param cls_preds: 类别预测
    :param cls_labels: 类别标签
    :param bbox_preds: 边界框预测
    :param bbox_labels: 边界框标签
    :param bbox_masks: 掩码变量，令负类锚框和填充锚框不参与损失的计算
    :return:
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(
        # cls_preds是一个三维的，-1的意思是将前面的批量大小维与中间锚框个数维放在一起，将每个锚框变为一个样本，cls_labels同理
        cls_preds.reshape(-1, num_classes),
        cls_labels.reshape(-1)
    ).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(
        # bbox_masks当锚框对应的是背景框时，mask=0，否则mask=1
        bbox_preds * bbox_masks,
        bbox_labels * bbox_masks
    ).mean(dim=1)
    return cls + bbox


# 评估，对分类精度
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 乘bbox_masks是防止非背景锚框的预测错误
    # bbox_masks当锚框对应的是背景框时，mask=0，否则mask=1
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


# 训练模型
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    # 训练模式
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        # Y：真实五体的边界框
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    # 输出趋势图（如果要显示趋势图则注释该图，要显示该图则注释趋势图）
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err  ->  {cls_err:.2e}, bbox mae  ->  {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')

# 预测目标
X = torchvision.io.read_image('../../Datasets/img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()


def predict(X):
    # 预测模式
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # 使用非极大抑制预测边界框
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


output = predict(X)


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


# 所有置信度不低于0.9的边界框，输出最终锚框识别结果（如果要显示趋势图则注释该图，要显示该图则注释趋势图）
display(img, output.cpu(), threshold=0.9)
