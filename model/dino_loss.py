import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dinov2loss.models.vision_transformer import vit_base
class DinoV2Loss(nn.Module):
    def __init__(self, vit_pretrain_path, use_cosine=False, device="cuda"):
        super(DinoV2Loss, self).__init__()
        self.use_cosine = use_cosine
        self.device = device

        # 初始化 ViT 模型（DINOV2）
        self.dinov2loss = vit_base(
            patch_size=14,
            num_register_tokens=4,
            img_size=526,
            init_values=1.0,
            block_chunks=0,
        )

        # 加载预训练权重
        state_dict = torch.load(vit_pretrain_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.dinov2loss.load_state_dict(state_dict, strict=False)

        # 冻结除 adapter 外的参数
        for name, param in self.dinov2loss.named_parameters():
            param.requires_grad = 'adapter' in name

        self.dinov2loss.eval()
        self.dinov2loss.to(device)

        if self.use_cosine:
            self.cosine_loss = nn.CosineEmbeddingLoss()

    def extract_features(self, images):
        """
        Args:
            images: [B, C, H, W], float, range [0, 1] or [0, 255]
        Returns:
            CLS token or averaged patch token features
        """
        if images.max() <= 1.0:
            images = images * 255.0
        images = images.clamp(0, 255)

        with torch.no_grad():
            outputs = self.dinov2loss(images)  # 假设输出为 dict，包含 last_hidden_state

            if isinstance(outputs, dict) and "last_hidden_state" in outputs:
                features = outputs["last_hidden_state"][:, 0]  # CLS token
            else:
                features = outputs[:, 0]  # Fallback

        return features  # [B, D]

    def forward(self, gen_images, ref_images):
        """
        Args:
            gen_images: [B, C, H, W]
            ref_images: [B, C, H, W]
        """
        x_resized = F.interpolate(gen_images, size=(252, 252), mode='bilinear', align_corners=False)
        ref_images = F.interpolate(ref_images, size=(252, 252), mode='bilinear', align_corners=False)  #392
        gen_feats = self.extract_features(x_resized.to(self.device))
        ref_feats = self.extract_features(ref_images.to(self.device))

        if self.use_cosine:
            targets = torch.ones(gen_feats.size(0)).to(self.device)
            loss = self.cosine_loss(gen_feats, ref_feats, targets)
        else:
            loss = F.mse_loss(gen_feats, ref_feats)

        return loss
