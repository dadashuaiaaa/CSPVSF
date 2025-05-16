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


import torch.nn.functional as F
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q, kv):
        B, C, H, W = q.shape
        N = H * W

        # reshape: [B, C, H, W] -> [B, N, C]
        q = q.flatten(2).transpose(1, 2)  # [B, N, C]
        kv = kv.flatten(2).transpose(1, 2)  # [B, N, C]

        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        # 多头注意力分头
        Q = Q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        # reshape 回 [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


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
        self.cross_attn = CrossAttentionFusion(in_channels)

        self.downsample = nn.Conv2d(288, in_channels, kernel_size=1)  # 假设 i_enc4/hv_4 通道数为 144

    def forward(self, x, i_enc4, hv_4, split_token=5):
        x0 = F.relu(self.D_fc1(x), inplace=True)  # [B, N, D]
        B, P, D = x0.shape
        W = H = int(math.sqrt(P - 1))
        xs = x0[:, split_token:, :].reshape(B, W, H, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        # ===> Step 1: 多分支卷积
        dense_branch1 = self.dense_branch1(xs)
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))
        xs = torch.cat([dense_branch1, dense_branch2, dense_branch3], dim=1) + xs  # [B, 144, H, W]

        
        fused = torch.cat([i_enc4, hv_4], dim=1)
        fused_down = self.downsample(fused) 
        xs = self.cross_attn(xs, fused_down) 
        


        
        # ===> Step 2: 注意力引导
        #i_enc4_down = self.downsample(i_enc4)
        #hv_4_down = self.downsample(hv_4)
        #xs1 = self.cross_attn(xs, i_enc4_down)
        #xs2 = self.cross_attn(xs, hv_4_down)
       # xs = xs1 + xs2  # [B, in_channels, H, W]

        # ===> Step 3: 输出重构回 patch 序列
        xs = xs.reshape(B, D, W * H).permute(0, 2, 1)  # [B, HW, D]
        clstoken = x0[:, 0:split_token, :]             # [B, split_token, D]
        outputs = torch.cat([clstoken, xs], dim=1)     # [B, N, D]
        outputs += x0
        outputs = self.D_fc2(outputs)

        if self.skip_connect:
            outputs += x

        return outputs
