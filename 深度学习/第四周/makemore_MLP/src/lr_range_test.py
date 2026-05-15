from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

import matplotlib.pyplot as plt

from model import MakemoreMLP
from train import load_data


def lr_range_test(
        model:MakemoreMLP,
        Xtr:Tensor,
        Ytr:Tensor,
        lr_start:float=1e-4,
        lr_end:float=10.0,
        num_steps:int=1000,
        batch_size:int=32,
        seed:int=42
)->tuple[list[float],list[float]]:
    g=torch.Generator().manual_seed(seed)
    N=Xtr.shape[0]

    lrs=torch.logspace(
        torch.log10(torch.tensor(lr_start)),
        torch.log10(torch.tensor(lr_end)),
        num_steps
    ).tolist()

    losses:list[float]=[]

    for step,lr in enumerate(lrs):
        ix=torch.randint(0,N,(batch_size,),generator=g)
        Xb,Yb=Xtr[ix],Ytr[ix]

        logits=model.forward(Xb)
        loss=F.cross_entropy(logits,Yb)

        for p in model.parameters():
            p.grad=None
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p.data-=lr*p.grad

        losses.append(loss.item())

        if loss>50 or torch.isnan(loss):
            print(f"diverged at step {step}, lr={lr:.4f}")
            return lrs[: step + 1], losses

    return lrs,losses


def plot(lrs:list[float],losses:list[float],save_path:str="logs/lr_range_test.png")->None:
    fig,ax=plt.subplots(figsize=(12,8))

    ax.plot(lrs,losses,lw=1)

    ax.set_xscale("log")

    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("loss")
    ax.set_title("LR Range Test")

    ax.grid(True,which="both",ls=":",alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path,dpi=120)
    plt.show()




def main()->None:
    Xtr,Ytr,_,_,_,_,_,_=load_data(block_size=3)
    model=MakemoreMLP(block_size=3,embed_dim=2,hidden_size=100)
    lrs,losses=lr_range_test(model,Xtr,Ytr,lr_start=1e-4,lr_end=10.0,num_steps=1000)
    plot(lrs,losses)


if __name__=="__main__":
    main()













