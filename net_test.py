from thop import profile
import torch
import time
from net.CIDNet import CIDNet

# 初始化模型与输入
model = CIDNet().to('cuda')
input = torch.rand(1, 3, 256, 256).to('cuda')

# 模型评估模式 & 同步 CUDA 时间
model.eval()
torch.cuda.synchronize()
time_start = time.time()
_ = model(input)
torch.cuda.synchronize()
time_end = time.time()
time_sum = time_end - time_start

print(f"\n⏱ 推理时间: {time_sum:.4f} 秒")

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 参数统计:")
print(f"  🔹 总参数量         : {total_params / 2**20:.2f} M")
print(f"  🔸 可训练参数量     : {trainable_params / 2**20:.2f} M")

# FLOPs 统计
macs, params = profile(model, inputs=(input,))
print(f"\n🧮 推理计算量 (FLOPs): {macs / 2**30:.2f} G")
