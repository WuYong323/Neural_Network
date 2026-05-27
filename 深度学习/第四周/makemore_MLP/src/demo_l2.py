from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



def main()->None:
    torch.manual_seed(0)

    model_a=nn.Linear(8,4)
    model_b=nn.Linear(8,4)
    model_b.load_state_dict(model_a.state_dict())

    lr,lam=0.1,1e-2
    opt_a=torch.optim.SGD(model_a.parameters(),lr=lr)                  # 手动加 L2
    opt_b=torch.optim.SGD(model_b.parameters(),lr=lr,weight_decay=lam)  # 内建

    x=torch.randn(16,8)
    y=torch.randint(0,4,(16,))

    # 方式 A：把 L2 加到 loss 里
    loss_a=F.cross_entropy(model_a(x),y)+0.5*lam*sum((p**2).sum() for p in model_a.parameters())
    opt_a.zero_grad()
    loss_a.backward()
    opt_a.step()

    # 方式 B：用 weight_decay
    loss_b=F.cross_entropy(model_b(x),y)
    opt_b.zero_grad()
    loss_b.backward()
    opt_b.step()


    diff=(model_a.weight-model_b.weight).abs().max().item()
    print(f"两种 L2 实现的最大参数差: {diff:.2e}  (应 < 1e-6)")



if __name__=="__main__":
    main()































