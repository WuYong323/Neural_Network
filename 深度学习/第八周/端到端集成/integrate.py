# integrate.py —— 把模型里所有 RMSNorm 换成指定实现
import torch
import torch.nn as nn
from rmsnorm_triton import triton_rmsnorm,_torch_rmsnorm


class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6,backend="torch"):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(dim))
        self.eps=eps
        self.backend=backend

    def forward(self,x):
        if self.backend=="triton" and x.is_cuda:
            try:
                return triton_rmsnorm(x,self.weight,self.eps)
            except Exception as e:
                import warnings
                warnings.warn(f"Triton 回退:{e}")

        return _torch_rmsnorm(x,self.weight,self.eps)

def swap_norm_backend(model:nn.Module,backend:str):
    """
    原地把 model 里所有 RMSNorm 的 backend 切换掉。
    为什么用遍历替换：不碰模型定义、随时可逆、保证对标双方结构 100% 一致。
    """
    for module in model.modules():
        if isinstance(module,RMSNorm):
            module.backend=backend
    return model






































