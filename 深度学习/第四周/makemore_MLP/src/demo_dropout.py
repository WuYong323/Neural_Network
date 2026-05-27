from __future__ import annotations

import torch
from torch import Tensor


def dropout_train(x:Tensor,p:float=0.5)->Tensor:
    mask=(torch.rand_like(x)>p).float()
    return x*mask/(1-p)

def dropout_eval(x:Tensor,p:float=0.5)->Tensor:
    return x


def main()->None:
    torch.manual_seed(0)
    x=torch.ones(10000)*1.0
    p=0.3

    out_train=dropout_train(x,p)
    out_eval=dropout_eval(x,p)

    print(f"输入均值        : {x.mean().item():.4f}")
    print(f"训练态输出均值  : {out_train.mean().item():.4f}  (≈ 1.0，inverse scaling 补偿)")
    print(f"推理态输出均值  : {out_eval.mean().item():.4f}  (= 1.0，identity)")

    # 与 PyTorch 内建 nn.Dropout 对比
    layer=torch.nn.Dropout(p=p)
    layer.train()
    pyt_train=layer(x).mean().item()
    layer.eval()
    pyt_eval=layer(x).mean().item()
    print(f"\nnn.Dropout train: {pyt_train:.4f}")
    print(f"nn.Dropout eval : {pyt_eval:.4f}")



if __name__=="__main__":
    main()




















