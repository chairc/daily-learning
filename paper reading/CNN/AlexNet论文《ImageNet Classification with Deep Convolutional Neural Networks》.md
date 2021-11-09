# AlexNet论文整理

___

## 题目
《ImageNet Classification with Deep Convolutional Neural Networks》

## 简介
AlexNet属于一个更大更深的LeNet  

改进有以下三点：
1. 增加了dropout层（丢弃层）
2. 激活函数从Sigmoid变为ReLu，作用是减缓梯度消失
3. 增加了MaxPooling（最大池化层），作用是取最大值，梯度相对较大，比较好训练  

这个AlexNet论文具体段落分析在PDF中，md文件仅做总结

## Abstract

> We trained a large, deep convolutional neural network to classify the 1.2 million
high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 dif-
ferent classes. On the test data, we achieved top-1 and top-5 error rates of 37.5%
and 17.0% which is considerably better than the previous state-of-the-art. The
neural network, which has 60 million parameters and 650,000 neurons, consists
of five convolutional layers, some of which are followed by max-pooling layers,
and three fully-connected layers with a final 1000-way softmax. To make train-
ing faster, we used non-saturating neurons and a very efficient GPU implemen-
tation of the convolution operation. To reduce overfitting in the fully-connected
layers we employed a recently-developed regularization method called “dropout”
that proved to be very effective. We also entered a variant of this model in the
ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%,
compared to 26.2% achieved by the second-best entry.

总结下就是介绍了AlexNet这个模型使用了一个又大又深的卷积神经网络分类了120万张高分辨率的图片，
并且参加了ImageNet里的一些比赛并取得了比较好的成绩。介绍了这个神经网络的参数、神经元、结构等，并
使用了高性能的GPU进行训练，接着又介绍了解决过拟合使用了丢弃法。

## 1 Introduction

在`Introduction`这个章节中，**第一段**说识别物体主要使用在机器学习方法中，收集比较大的数据集和更强
的模型，用更好的技术去防止过拟合。然后第一段后部分说我们使用了ImageNet这个数据集，吹了一波这个
数据集的图片多，分辨率好还有种类多。**第二段**说CNN是个很好的模型，我们需要用它来做深度学习（`注意：
这里并没有与其它的方法做比较，只提CNN，在自己写论文时这里应该加上作比较`）。**第三段**提到CNN不好
训练什么的，需要很大的资源开销，但是在这里有GPU进行计算，并方便训练CNN。**第四段**提到论文的新技术、
方法等等，总结就是四部分，一是用GPU做2D卷积；二是网络新功能；三是有效方法介绍过拟合；四是深层的网络
很重要。**第五段**讲GPU。

## 2 The Dataset

在`The Dataset`这个章节中，**第一段**讲了ImageNet这个数据集图片多，分辨率好还有种类多。**第二段**
讲ILSVRC这个比赛。**第三段**讲图片分辨率不同，然后这个模型把这些图片按照一定的方式划分256x256的格
式，并以原始图片输入。

## 3 The Architecture

### 3.1 ReLU Nonlinearity

![ReLU vs tanh](./img/ImageNet Classification with Deep Convolutional Neural Networks/ReLU vs tanh.png)

在`ReLU Nonlinearity`这个章节里，主要就是说ReLu函数比传统的那些tanh、sigmoid函数要好，在CNN
中有更快的表现。图中的实线代表ReLU，虚线代表tanh

### 3.2 Training on Multiple GPUs

在`Training on Multiple GPUs`这个章节里，讲了用多个GPU进行训练。

### 3.3 Local Response Normalization

在`Local Response Normalization`这个章节里，讲了输入归一化，那个公式基本没人用目前。

### 3.4 Overlapping Pooling

在`Overlapping Pooling`这个章节里，CNN中的池化层总结了同一核图中相邻神经元组的输出，然后作者发现
在训练中观察到使用重叠池化层的模型不易过拟合。

### 3.5 Overall Architecture

![Overall Architecture](./img/ImageNet Classification with Deep Convolutional Neural Networks/Overall Architecture.png)

在`Overall Architecture`这个章节里，**第一段**讲了AlexNet这个模型的结构是有八个含有权重的层，
由五层卷积层和三层全连接层构成。**第二段**讲了卷积核、卷积层、池化层和全连接层的关系。**第三段**讲
了每一层的输入、输出等。具体的实现在[pytorch learning](https://github.com/chairc/daily-learning/blob/main/pytorch-learning/004_pytorch_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/00_AlexNet_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88AlexNet%EF%BC%89/00_AlexNet%E7%9A%84%E5%AE%9E%E7%8E%B0.py)
中。
