from __future__ import annotations

import time

import torch
from torch import Tensor
import torch.nn.functional as F

from dataset import build_dataset
from vocab import build_vocab
from model import MakemoreMLP

def load_data(
        path:str="names.txt",
        block_size:int=3,
        seed:int=42
)->tuple[Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,dict,dict]:
    with open(path,"r",encoding="utf-8") as f:
        words=[w.strip() for w in f if w.strip()]

    g=torch.Generator().manual_seed(seed)
    perm:list[int]=torch.randperm(len(words),generator=g).tolist()
    words=[words[i] for i in perm]

    n1=int(0.8*len(words))
    n2=int(0.9*len(words))
    train_words,dev_words,test_words=words[:n1],words[n1:n2],words[n2:]

    stoi,itos=build_vocab(words)
    Xtr,Ytr=build_dataset(train_words,stoi,block_size)
    Xdv, Ydv = build_dataset(dev_words, stoi, block_size)
    Xte, Yte = build_dataset(test_words, stoi, block_size)
    return Xtr,Ytr,Xdv,Ydv,Xte,Yte,stoi,itos


@torch.no_grad()
def eval_loss(model:MakemoreMLP,X:Tensor,Y:Tensor,batch_size:int=1024)->float:
    losses=[]
    for i in range(0,X.shape[0],batch_size):
        logits=model.forward(X[i:i+batch_size])
        losses.append(F.cross_entropy(logits,Y[i:i+batch_size]).item())
    return sum(losses)/len(losses)


def train(
        model:MakemoreMLP,
        Xtr:Tensor,
        Ytr:Tensor,
        Xdv:Tensor,
        Ydv:Tensor,
        steps:int=50000,
        batch_size:int=32,
        lr:float=0.1,
        eval_every:int=5000,
        seed:int=42
)->list[tuple[int,float,float]]:
    g=torch.Generator().manual_seed(seed)
    history:list[tuple[int,float,float]]=[]
    N=Xtr.shape[0]
    t0=time.perf_counter()

    for step in range(1,steps+1):
        ix=torch.randint(0,N,(batch_size,),generator=g)
        Xb,Yb=Xtr[ix],Ytr[ix]

        logits=model.forward(Xb)
        loss=F.cross_entropy(logits,Yb)

        for p in model.parameters():
            p.grad=None
        loss.backward()

        """
        手写SGD
        with torch.no_grad():
            for p in model.parameters():
                p.data-=lr*p.grad
        """

        # 优化器内建 weight_decay（工业默认）
        # 等价于"每步把 w 乘以 (1 - lr * weight_decay)"
        # PyTorch 的 weight_decay = Andrew Ng 视频里的 λ，但已经吸收了 lr，所以数值上对不上
        optimizer=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=1e-4)

        if step%eval_every==0 or step==steps or step==1:
            dev_loss=eval_loss(model,Xdv,Ydv)
            history.append((step,loss.item(),dev_loss))
            elapsed=time.perf_counter()-t0
            print(
                f"step {step:6d} | train {loss.item():.4f} | dev {dev_loss:.4f} | "
                f"{elapsed:.1f}s ({step / elapsed:.0f} it/s)"
            )
    return history


def main():
    steps=50000
    batch_size=32
    lr=0.1
    block_size=3
    embed_dim=10
    hidden_size=100

    Xtr,Ytr,Xdv,Ydv,Xte,Yte,_,_=load_data(block_size=block_size)
    print(f"train={Xtr.shape[0]:,} dev={Xdv.shape[0]:,} test={Xte.shape[0]:,}")

    model=MakemoreMLP(block_size=block_size,embed_dim=embed_dim,hidden_size=hidden_size)
    print(f"params: {model.num_params():,}")

    train(model,Xtr,Ytr,Xdv,Ydv,steps=steps,batch_size=batch_size,lr=lr)

    test_loss = eval_loss(model, Xte, Yte)
    print(f"\nfinal test loss: {test_loss:.4f}")



if __name__=="__main__":
    main()























