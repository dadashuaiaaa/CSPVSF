from typing import Tuple, Union, List, Any
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# Attention Fusion Layer
class AttentionFusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        input_dim: 输入每个token的特征维度
        output_dim: 输出每个token的特征维度
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(input_dim, output_dim)  # 将输出特征映射到64维

    def forward(self, x: List[Tensor]) -> Tensor:
        # x 是一个列表，包含多个张量，形状分别为 [B, Seq_len, Features]
        combined = torch.cat(x, dim=1)  # 沿着序列维度拼接，形状变为 [B, 48, Features]
        attn_output, _ = self.attention(combined, combined, combined)
        return self.fc(attn_output)  # 映射特征维度到64

# Text Adapter Module
class TextAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        lstm_hidden_size: int,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect = skip_connect

        # 分支1: 1x1 卷积
        self.dense_branch1 = BasicConv1d(in_channels, ch1x1, kernel_size=1)

        # 分支2: 1x1 降维 + 3x3 卷积
        self.dense_branch2 = nn.Sequential(
            BasicConv1d(in_channels + ch1x1, ch3x3red, kernel_size=1),
            BasicConv1d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 分支3: BiLSTM 处理全局特征
        self.bilstm = nn.LSTM(input_size=in_channels + ch1x1 + ch3x3, hidden_size=8, batch_first=True, bidirectional=True)
        
        # Attention融合模块，输入维度为16，输出映射到64维
        self.attention_fusion = AttentionFusion(input_dim=16, output_dim=64)

        # 全连接层
        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)
        self.proj_dense_branch1 = nn.Linear(32, 16)  # 将dense_branch1映射到16维
     

    def forward(self, x: Tensor) -> Tensor:
        # x 形状：[B, P, D]
        x0 = self.D_fc1(x)
        B, P, D = x0.shape
        x0 = F.relu(x0, inplace=True)

        # 取除CLS以外的部分，并转换成适用于卷积的形状
        xs = x0[:, 1:, :].permute(0, 2, 1)  # (B, in_channels, Seq_len)；此时Seq_len为 P-1

        # 分支1: 1x1卷积
        dense_branch1 = self.dense_branch1(xs)  # 形状：[B, ch1x1, Seq_len]
        
        # 分支2: 1x1降维 + 3x3卷积
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))  # 形状：[B, ch3x3, Seq_len]

        # 分支3: BiLSTM处理全局特征
        bilstm_input = torch.cat([xs, dense_branch1, dense_branch2], dim=1).permute(0, 2, 1)  # (B, Seq_len, Features)
        bilstm_output, _ = self.bilstm(bilstm_input)  # 形状：[B, Seq_len, 16]（因为hidden_size=8且bidirectional）

        # 对 dense_branch1 进行投影，将其特征数从 ch1x1（这里为32）映射到16
        dense_branch1 = dense_branch1.permute(0, 2, 1)  # (B, Seq_len, ch1x1)
        dense_branch1 = self.proj_dense_branch1(dense_branch1)  # 形状：[B, Seq_len, 16]

        # 将三个分支输出沿序列维度拼接
        # 假设每个分支的Seq_len都是16（这里 P-1 = 16），拼接后得到形状为 [B, 48, 16]
        fusion_input = [dense_branch1, dense_branch2.permute(0, 2, 1), bilstm_output]
        outputs = self.attention_fusion(fusion_input)  # 此时输出形状为 [B, 48, 64]（经过fc映射）

        # 如果想要最终序列长度为16，则需要对序列维度进行压缩
        # 可以使用自适应平均池化，将序列长度从48压缩到16：
        outputs = outputs.permute(0, 2, 1)  # 变为 [B, 64, 48]
        outputs = F.adaptive_avg_pool1d(outputs, output_size=16)  # 变为 [B, 64, 16]
        outputs = outputs.permute(0, 2, 1)  # 变回 [B, 16, 64]


        # 后续投影操作（可选）：比如这里使用 proj2 将特征再做一次映射（如果需要）
        #outputs = self.proj2(outputs)

        # 添加 CLS Token 并进行残差连接
        clstoken = x0[:, 0:1, :]  # 假设CLS token放在第一位
        outputs = torch.cat([clstoken, outputs], dim=1)  # 此时形状变为 [B, 1+16, 64]，即 [B, 17, 64]

        # 如果需要与 x0 做残差相加，则 x0 的形状也必须对应，示例代码中 x0 原本形状为 [B, P, D]（P=17），可以相加
        outputs += x0

        # 恢复原始维度
        outputs = self.D_fc2(outputs)

        if self.skip_connect:
            outputs += x

        return outputs
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class CrossModalAttention(nn.Module):
    def __init__(self, image_dim, text_dim, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        self.back_proj = nn.Linear(embed_dim, image_dim)
        
    def forward(self, image_features, text_features, attention_mask=None):


        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        query = image_features.permute(1, 0, 2)
        key = text_features.permute(1, 0, 2)
        value = text_features.permute(1, 0, 2)
        
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attention_mask)
        
        attn_output = self.back_proj(attn_output)
        
        return attn_output.permute(1, 0, 2)


class DynamicGating(nn.Module):
    def __init__(self, text_dim, num_branches):
        super(DynamicGating, self).__init__()
        self.gate_fc = nn.Linear(text_dim, num_branches)

    def forward(self, state):
        # Compute gate weights based on text semantics
        gates = self.gate_fc(state) # (B, num_branches)
        gates = torch.sigmoid(gates)  # Sigmoid to constrain between 0 and 1
        return gates

class DenseAligner(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        skip_connect=False,
    ):
        super().__init__()
        self.skip_connect = skip_connect

        self.dense_branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.dense_branch2 = nn.Sequential(
            nn.Conv2d(in_channels + ch1x1, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.dense_branch3 = nn.Sequential(
            nn.Conv2d(in_channels + ch1x1 + ch3x3, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

        self.downsample = nn.Conv2d(144, in_channels, kernel_size=1)

    def forward(self, x, i_enc4, split_token=5):
        x0 = F.relu(self.D_fc1(x), inplace=True)
        B, P, D = x0.shape
        W = H = int(math.sqrt(P - 1))
        xs = x0[:, split_token:, :].reshape(B, W, H, D).permute(0, 3, 1, 2)
        #print(i_enc4.shape)
        # 下采样 i_enc4 到匹配当前 patch 尺寸
        i_enc4_down = self.downsample(i_enc4)
        #print(i_enc4_down.shape)
        #print(xs.shape)
        # 将 i_enc4 加入分支结构（可选）
        xs = xs + i_enc4_down  # 融合来自 encoder 的视觉信息
        

        dense_branch1 = self.dense_branch1(xs)
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))

        outputs = torch.cat([dense_branch1, dense_branch2, dense_branch3], dim=1) + xs
        outputs = outputs.reshape(B, D, W * H).permute(0, 2, 1)

        clstoken = x0[:, 0:split_token, :]
        outputs = torch.cat([clstoken, outputs], dim=1)

        outputs += x0
        outputs = self.D_fc2(outputs)

        if self.skip_connect:
            outputs += x

        return outputs
