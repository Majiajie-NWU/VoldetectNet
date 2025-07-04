
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossTemporalAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadCrossTemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = in_channels // num_heads

        # 定义键、查询、值的线性变换
        self.key = nn.Linear(in_channels, in_channels)
        self.query = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        # 输出的全连接层
        self.dense = nn.Linear(in_channels, in_channels)

        # 归一化和激活
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # x 的形状为 [batch_size, num_time_points, channels, height, width]
        batch_size, num_time_points, channels, height, width = x.size()
        x_flat = x.view(batch_size, num_time_points, channels * height * width).transpose(1, 2)  # [B, C*H*W, T]

        # 应用线性变换
        keys = self.key(x_flat)  # [B, C*H*W, T]
        queries = self.query(x_flat)  # [B, C*H*W, T]
        values = self.value(x_flat)  # [B, C*H*W, T]

        # 计算注意力得分
        attention_scores = torch.matmul(queries.transpose(-2, -1), keys) / math.sqrt(self.depth)  # [B, T, T]
        attention = F.softmax(attention_scores, dim=-1)  # Softmax 沿时间点
        attention = self.dropout(attention)

        # 应用注意力机制到值上
        weighted = torch.matmul(attention, values.transpose(-2, -1))  # [B, T, C*H*W]
        weighted_combined = torch.sum(weighted, dim=1)  # [B, C*H*W]

        # 重组回原始的特征图形状
        weighted_combined = weighted_combined.view(batch_size, channels, height, width)

        # 应用输出层并进行归一化
        output = self.dense(weighted_combined.view(batch_size, -1)).view(batch_size, channels, height, width)
        output = self.norm(output + x.view(batch_size, num_time_points, channels, height, width).mean(dim=1))

        return output