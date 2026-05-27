import torch
import torch.nn.functional as F


torch.manual_seed(42)
N,D_in,D_out=8,16,32

W=torch.randn(D_in,D_out)
b_conv=conv=torch.randn(D_out)
gamma=torch.randn(D_out)
beta=torch.randn(D_out)
mu=torch.randn(D_out)             # 假装是训练好的 running_mean
sigma2=torch.rand(D_out)+0.5
eps=1e-5

x=torch.randn(N,D_in)

# 路径 A：Linear → BN（推理模式）
out_linear=x@W+b_conv
out_bn=gamma*(out_linear-mu)/torch.sqrt(sigma2+eps)+beta

# 路径 B：Fused Linear（W' 和 b'）
scale=gamma/torch.sqrt(sigma2+eps)
w_fused=W*scale
b_fused=scale*(b_conv-mu)+beta
out_fused=x@w_fused+b_fused

# 对比
max_diff = (out_bn - out_fused).abs().max().item()
print(f"两条路径最大差异: {max_diff:.2e}")           # 应该 ≈ 1e-6 (浮点误差)
assert max_diff < 1e-5
print("Fused 等价")












