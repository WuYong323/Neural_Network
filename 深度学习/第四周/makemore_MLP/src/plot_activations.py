from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import torch
from pygments.styles import default
from torch import Tensor

from model import MakemoreMLP
from train import load_data



def hidden_preactivation(model:MakemoreMLP,X:Tensor)->Tensor:
    emb=model.C[X]
    flat=emb.view(emb.size(0),-1)
    pre=flat@model.W1+model.b1
    return pre


def main(scale_W1:float=1.0,tag:str="default")->None:
    torch.manual_seed(42)
    Xtr,Ytr,_,_,_,_,_,_=load_data(block_size=3)
    model=MakemoreMLP(block_size=3,embed_dim=10,hidden_size=200,seed=42)

    with torch.no_grad():
        model.W1.mul_(scale_W1)

    N=1024
    idx=torch.randperm(Xtr.size(0))[:N]
    pre=hidden_preactivation(model,Xtr[idx])
    h=torch.tanh(pre)

    sat_rate=(h.abs()>0.99).float().mean().item()
    std_pre=pre.std().item()
    print(f"[{tag}] pre-act std = {std_pre:.3f}   tanh 饱和率 = {sat_rate * 100:.1f}%")

    fig,axes=plt.subplots(1,2,figsize=(12,8))
    axes[0].hist(pre.flatten().detach().numpy(),bins=60,color="tab:blue",alpha=0.7)
    axes[0].set_title(f"pre-activation  (std={std_pre:.2f})")
    axes[0].axvline(-2,ls="--",c="r")
    axes[0].axvline(2, ls="--", c="r")
    axes[1].hist(h.flatten().detach().numpy(),bins=60,color="tab:orange",alpha=0.7)
    axes[1].set_title(f"tanh output  (sat={sat_rate*100:.1f}%)")
    axes[1].axvline(-0.99,ls="--",c="r")
    axes[1].axvline(0.99, ls="--", c="r")

    Path("logs/activation_hist").mkdir(exist_ok=True)
    out=f"logs/activation_hist/{tag}.png"
    plt.tight_layout()
    plt.savefig(out,dpi=120)
    plt.show()



if __name__=="__main__":
    main(scale_W1=2.0,tag="too_large")
    main(scale_W1=1.0,tag="default")
    main(scale_W1=0.1,tag="small")





















