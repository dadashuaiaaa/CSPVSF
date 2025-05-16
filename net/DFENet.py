import torch
import torch.nn as nn
import torch.nn.functional as F
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
from model.dinov2.models.vision_transformer import vit_base
from .fusion import Fusion


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def deconv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))
class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x
class Neck(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=144):
        super(Neck, self).__init__()

        self.target_size = (35, 35)

        self.f3_proj = conv_layer(in_channels[0], out_channels, kernel_size=1, padding=0, stride=1)
        self.f4_proj = conv_layer(in_channels[1], out_channels, kernel_size=1, padding=0, stride=1)
        self.f5_proj = conv_layer(in_channels[2], out_channels, kernel_size=1, padding=0, stride=1)

        self.aggr = conv_layer(out_channels * 3, out_channels, kernel_size=1, padding=0, stride=1)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels, out_channels, kernel_size=3, padding=1),
            conv_layer(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, features):
        v3, v4, v5 = features  # each: [B, 768, H, W]

        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)

        v3 = F.interpolate(v3, size=self.target_size, mode='bilinear', align_corners=False)
        v4 = F.interpolate(v4, size=self.target_size, mode='bilinear', align_corners=False)
        v5 = F.interpolate(v5, size=self.target_size, mode='bilinear', align_corners=False)

        fq = torch.cat([v3, v4, v5], dim=1)  # [B, 3*C, 35, 35]
        fq = self.aggr(fq)                  # [B, C, 35, 35]
        fq1 = self.coordconv(fq)

        return fq + fq1                     # [B, C, 35, 35]


class Neckj(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=256):
        super(Neckj, self).__init__()

        self.target_size = (280, 280)

        self.f3_proj = conv_layer(in_channels[0], out_channels, kernel_size=1, padding=0, stride=1)
        self.f4_proj = conv_layer(in_channels[1], out_channels, kernel_size=1, padding=0, stride=1)
        self.f5_proj = conv_layer(in_channels[2], out_channels, kernel_size=1, padding=0, stride=1)

        self.aggr = conv_layer(out_channels * 3, out_channels, kernel_size=1, padding=0, stride=1)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels, out_channels, kernel_size=3, padding=1),
            conv_layer(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, features):
        v3, v4, v5 = features  # each: [B, 768, H, W]

        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)

        v3 = F.interpolate(v3, size=self.target_size, mode='bilinear', align_corners=False)
        v4 = F.interpolate(v4, size=self.target_size, mode='bilinear', align_corners=False)
        v5 = F.interpolate(v5, size=self.target_size, mode='bilinear', align_corners=False)

        fq = torch.cat([v3, v4, v5], dim=1)  # [B, 3*C, 35, 35]
        fq = self.aggr(fq)                  # [B, C, 35, 35]
        fq1 = self.coordconv(fq)

        return fq + fq1                     # [B, C, 35, 35]


class DINOCrossAttention(nn.Module):
    def __init__(self, dino_dim=768, inner_dim=256, out_dim=144, target_size=(20, 20)):
        super(DINOCrossAttention, self).__init__()
        self.q_proj = nn.Conv2d(out_dim, inner_dim, 1)
        self.k_proj = nn.Conv2d(out_dim, inner_dim, 1)
        self.v_proj = nn.Conv2d(144, inner_dim, 1)
        self.out_proj = nn.Conv2d(inner_dim, out_dim, 1)
        self.target_size = target_size

    def forward(self, q_feat, k_feat, dino_feat):
        q = F.interpolate(self.q_proj(q_feat), size=self.target_size, mode='bilinear')
        k = F.interpolate(self.k_proj(k_feat), size=self.target_size, mode='bilinear')
        v = F.interpolate(self.v_proj(dino_feat), size=self.target_size, mode='bilinear')
        #v = self.v_proj(v)

        B, C, H, W = q.shape
        Q = q.flatten(2).transpose(1, 2)
        K = k.flatten(2).transpose(1, 2)
        V = v.flatten(2).transpose(1, 2)

        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (C ** 0.5), dim=-1)
        out = torch.bmm(attn, V).transpose(1, 2).view(B, C, H, W)
        out = self.out_proj(out)
        return out


class DFENet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 vit_pretrain_path="pretrain/dinov2_vitb14_reg4_pretrain.pth"):
        super(DFENet, self).__init__()

        self.dinov2 = vit_base(
            patch_size=14,
            num_register_tokens=4,
            img_size=526,
            init_values=1.0,
            block_chunks=0,
        )

        self.fusion = Fusion(
            d_model=64,
            dino_layers=12,
            output_dinov2=[4, 8]
        )


        self.vit_proj = nn.Conv2d(768, channels[-1], kernel_size=1)
        self.neck = Neck()
        self.neck2 = Neckj()

        state_dict = torch.load(vit_pretrain_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.dinov2.load_state_dict(state_dict, strict=False)

        for param_name, param in self.dinov2.named_parameters():
            if 'adapter' not in param_name:
                param.requires_grad = False

        #for name, param in self.dinov2.named_parameters():
           # if param.requires_grad:
               # print(f"✅ {name} is trainable.")


        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # Encoder
        self.HVE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(3, ch1, 3, 1, 0, bias=False))
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 3, 3, 1, 0, bias=False))

        self.IE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(1, ch1, 3, 1, 0, bias=False))
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 3, 3, 1, 0, bias=False))

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        self.cross_attention_I = DINOCrossAttention(
    dino_dim=768, inner_dim=256, out_dim=channels[-1], target_size=(35, 35)
)
        self.cross_attention_HV = DINOCrossAttention(
    dino_dim=768, inner_dim=256, out_dim=channels[-1], target_size=(35, 35)
)
        self.cross_attention_HV = DINOCrossAttention(
    dino_dim=768, inner_dim=256, out_dim=channels[-1], target_size=(35, 35)
)
        self.hviconv_layer = nn.Conv2d(in_channels=288, out_channels=144, kernel_size=1)
        self.hviconv_layer2 = nn.Conv2d(in_channels=144, out_channels=3, kernel_size=1)

        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2:3, :, :].to(dtypes)

        i_enc0 = self.IE_block0(i)
       # print("----------")
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)

        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        #print(f"Data type of i_enc4: {type(i_enc4)}")
        

        x_resized = F.interpolate(x, size=(252, 252), mode='bilinear', align_corners=False)
        vis_I = self.fusion(x_resized, i_enc4, hv_4, self.dinov2)

        
        fused_feats = self.neck(vis_I) 
        
        
        kv_feats = torch.cat([i_enc4, hv_4], dim=1)  # B, C1+C2, H, W
        kv_feats = self.hviconv_layer(kv_feats) 


        fused_feats = F.interpolate(fused_feats, size=i_enc4.shape[2:], mode='bilinear', align_corners=False)

        i_dec4 = self.I_LCA4(fused_feats, kv_feats)
        hv_4 = self.HV_LCA4(kv_feats, fused_feats)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = i_dec0 + hv_0 
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb, output_hvi

    def HVIT(self, x):
        return self.trans.HVIT(x)
