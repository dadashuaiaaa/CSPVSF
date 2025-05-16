import torch
import torch.nn as nn
import torch.nn.functional as F
from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.LCA import HV_LCA, I_LCA
from huggingface_hub import PyTorchModelHubMixin
from model.dinov2.models.vision_transformer import vit_base


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 vit_pretrain_path="pretrain/dinov2_vitb14_reg4_pretrain.pth"):
        super(CIDNet, self).__init__()
        ch1, ch2, ch3, ch4 = channels
        head2, head3, head4 = heads[1], heads[2], heads[3]

        # --------------------- ViT 提取器 ---------------------
        self.dinov2 = vit_base(
            patch_size=14,
            num_register_tokens=4,
            img_size=526,
            init_values=1.0,
            block_chunks=0,
        )
        state_dict = torch.load(vit_pretrain_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.dinov2.load_state_dict(state_dict, strict=False)
        for name, param in self.dinov2.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        self.vit_channel_align = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),  # 从768通道映射到512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # 继续映射到256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=1)  # 最终映射到3通道
        )
        self.fusion_to_rgb = nn.Conv2d(6, 3, kernel_size=1)

        # --------------------- HVI 通路 ---------------------
        self.trans = RGB_HVI()

        # HV Path
        self.HVE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(3, ch1, 3, bias=False))
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 2, 3, bias=False))

        # I Path
        self.IE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(1, ch1, 3, bias=False))
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 1, 3, bias=False))

        # LCA 融合模块
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

    def extract_vit_features(self, x, out_size):
        x_resized = F.interpolate(x, size=(280, 280), mode='bilinear', align_corners=False)
        vit_output = self.dinov2.forward_features(x_resized)
        x_prenorm = vit_output['x_prenorm'][:, 1:401, :]  # 去除cls/reg

        B, N, C = x_prenorm.shape
        H = W = int(N**0.5)
        feat = x_prenorm.permute(0, 2, 1).contiguous().view(B, C, H, W)
        feat = torch.clamp(feat, -1e3, 1e3)
        feat = (feat - feat.mean(dim=(2, 3), keepdim=True)) / (feat.std(dim=(2, 3), keepdim=True) + 1e-4)
        feat = F.interpolate(feat, size=out_size, mode='bilinear', align_corners=False)
        return feat

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2:3, :, :].to(dtypes)

        # ----------- I & HV 编码部分 -----------
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0, hv_jump0 = i_enc0, hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1, hv_jump1 = i_enc2, hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2, hv_jump2 = i_enc3, hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        # ----------- 解码融合部分 -----------
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

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

        # ----------- HVI 输出 -----------
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        hvi_rgb = self.trans.PHVIT(output_hvi)

        # ----------- ViT 特征提取 -----------
        vit_feat = self.extract_vit_features(x, out_size=hvi_rgb.shape[-2:])
        vit_feat = self.vit_channel_align(vit_feat)

        # ----------- 融合输出 -----------
        fused = torch.cat([vit_feat, hvi_rgb], dim=1)
        output = self.fusion_to_rgb(fused)

        return output, output_hvi
        
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
        
     
        
        
