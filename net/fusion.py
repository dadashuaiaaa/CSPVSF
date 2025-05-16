import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


class Adapter(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        x = self.conv(x)  # -> [B, d_model, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # -> [B, H*W, d_model]
        x = self.proj(x)
        return x  # [B, N, d_model]


class Fusion(nn.Module):
    def __init__(self,
                 d_img=[768, 768, 768],
                 d_model=64,
                 num_stages=3,
                 strides=[1, 1, 1],
                 num_layers=12,
                 shared_weights=False,
                 dino_layers=12,
                 output_dinov2=[4, 8]):
        super().__init__()

        self.d_img = d_img
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0

        self.adapter = Adapter(in_channels=144, d_model=768)  # match to dino embed dim

        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, img, i_enc4, hv_4, dino):
        B = img.shape[0]
        img = img.type(dino.patch_embed.proj.weight.dtype)
        vis_outs = []
        features_dino = []

        # ---- Dino Patch Embedding ----
        net_input = img.clone()
        B, nc, w, h = net_input.shape
        dino_f = dino.patch_embed(net_input)
        dino_f = torch.cat((dino.cls_token.expand(B, -1, -1), dino_f), dim=1)
        dino_f = dino_f + dino.interpolate_pos_encoding(dino_f, w, h)
        dino_f = torch.cat(
            (
                dino_f[:, :1],
                dino.register_tokens.expand(B, -1, -1),
                dino_f[:, 1:],
            ),
            dim=1,
        )

        # ---- Prepare i_enc4 ----
        #print(i_enc4.shape)
        #print("-----")
        #print(f"Before interpolation: i_enc4 - max: {i_enc4.max()}, min: {i_enc4.min()}, mean: {i_enc4.mean()}, std: {i_enc4.std()}")
        i_enc4_resized = F.interpolate(i_enc4, size=(20, 20), mode='bilinear', align_corners=False)  # Match Dino patch tokens

        hv_4_resized = F.interpolate(hv_4, size=(20, 20), mode='bilinear', align_corners=False)  # Match Dino patch tokens
        #print(f"After interpolation: i_enc4_resized - max: {i_enc4_resized.max()}, min: {i_enc4_resized.min()}, mean: {i_enc4_resized.mean()}, std: {i_enc4_resized.std()}")
        #print(i_enc4_resized.shape)
        adapter_feati_enc4 = i_enc4_resized

        adapter_feati_hv_4 = hv_4_resized
        #adapter_feat = self.adapter(i_enc4_resized)  # [B, N, 768]
        #print(dino_f.shape)
        #adapter_layers = [4, 8]
        # ---- Dino Blocks ----
        for i in range(self.dino_layers):
            #use_adapter = True if i in [4, 8] else False  # 只在第 4 和第 8 层使用适配器
            dino_f = dino.blocks[i](dino_f, adapter_feati_enc4, adapter_feati_hv_4)
            
            #if i in adapter_layers:
                #dino_f = dino.blocks[i](dino_f, adapter_feati_enc4, adapter_feati_hv_4)
            #else:
                #dino_f = dino.blocks[i](dino_f, None, None)
            if i in self.output_dinov2:
                features_dino.append(dino_f)

        dino_f = dino.norm(dino_f)
        features_dino.append(dino_f)

        for feature_dino in features_dino:
            feature_dino = feature_dino[:, 4 + 1:]  # skip cls + 4 register tokens
            B, L, C = feature_dino.shape
            H = int(L ** 0.5)
            W = L // H
            feature_dino = feature_dino.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            vis_outs.append(feature_dino)

        return vis_outs
