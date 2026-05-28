from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def states_bytes(opt:torch.optim.Optimizer)->int:
    total=0
    for group in opt.state.values():
        for s in group.values():
            if isinstance(s,Tensor):
                total+=s.numel()*s.element_size()

    return total


def main()->None:
    model=nn.Sequential(nn.Linear(1000,1000),nn.Linear(1000,1000))
    n_params=sum(p.numel() for p in model.parameters())
    param_mb=n_params*4/1024/1024
    print(f"参数量          : {n_params:,}")
    print(f"参数 FP32 显存  : {param_mb:.2f} MB\n")

    for p in model.parameters():
        p.grad=torch.rand_like(p)

    cases = [
        ("SGD          ", torch.optim.SGD(model.parameters(), lr=0.01)),
        ("SGD+Momentum ", torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)),
        ("RMSProp      ", torch.optim.RMSprop(model.parameters(), lr=0.001)),
        ("Adam         ", torch.optim.Adam(model.parameters(), lr=0.001)),
        ("AdamW        ", torch.optim.AdamW(model.parameters(), lr=0.001)),
    ]

    print(f"{'优化器':<14} {'state 显存':>12}   {'倍数 (× 参数)':>14}")
    print("-" * 50)
    for name,opt in cases:
        opt.step()
        sz=states_bytes(opt)/1024/1024
        ratio=sz/param_mb if param_mb>0 else 0
        print(f"{name:<14} {sz:>9.2f} MB   {ratio:>10.2f}×")

    print("\n结论：Adam/AdamW 的 state ≈ 2× 参数大小（m + v）")
    print("推论：7B FP32 训练显存 = 28(W) + 28(grad) + 56(Adam) = 112 GB")




if __name__=="__main__":
    main()





















