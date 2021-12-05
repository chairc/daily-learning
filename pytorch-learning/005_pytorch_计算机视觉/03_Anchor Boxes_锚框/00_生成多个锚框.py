#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from d2lzh_pytorch import torch_d2l as d2l

"""
    s:锚框面积与输入图像面积之比，例如s = 0.5，则锚框面积就是图像的s² = 0.25倍，且s∈(0,1]
    r:宽高比（w / h），且r>0
    锚框的宽度和高度分别是w * s * √r和h * s / √r 
    以同一像素为中心的锚框的数量是n + m − 1，对于整个输入图像，我们将共生成w * h * (n+m−1)个锚框
"""

torch.set_printoptions(2)


# 生成以每个像素为中心具有不同形状的锚框
def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成boxes_per_pixel个高和宽，
    # 之后用于创建锚框的四角坐标 (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    # 每个中心点都将有boxes_per_pixel个锚框，
    # 所以生成含所有锚框中心的网格，重复了boxes_per_pixel次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


# 绘制多个边框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


# 显示图片
image = d2l.plt.imread('../../Datasets/img/catdog.jpg')
h, w = image.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# Y.shape生成的形状为（批量大小，锚框数量，锚框的四角坐标）
print(f'Y.shape  ->  {Y.shape}')

# 将锚框变量Y的形状更改为（图像高度、图像宽度、以同一像素为中心生成的锚框的数量，4）
boxes = Y.reshape(h, w, 5, 4)
# 访问以(250,250)为中心的第一个锚框，输出左上角和右下角的(x,y)坐标，输出的值为两个轴的坐标分别除以图像的宽度和高度后结果
print(f'boxes[250, 250, 0, :]  ->  {boxes[250, 250, 0, :]}')

d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(image)
# 将(250,250)为中心的所有锚框画出来
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
