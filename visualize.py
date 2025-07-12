import torch
import torch.nn as nn
from torchviz import make_dot
from model import UNet1D  # 你的模型定义

# 模型实例化
model = UNet1D()

# 构造输入（注意要加 batch 维度）
example_input = torch.randn(1, 1, 200)  # shape: [batch, channel, length]

# 前向传播获取输出
output = model(example_input)

# 可视化计算图
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("unet1d_graph", format="png")  # 生成 unet1d_graph.png 文件
