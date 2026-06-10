import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self,n_embd,head_size,block_size,dropout=0.0):
        super().__init__()
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)
        self.head_size=head_size

    def forward(self,x):
        B,T,C=x.shape
        q,k,v=self.query(x),self.key(x),self.value(x)
        wei=q@k.transpose(-2,-1)*self.head_size**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei=F.softmax(wei)
        wei=self.dropout(wei)
        return wei@v



class MultiHeadAttentionLoop(nn.Module):
    def __init__(self,n_head,n_embd,block_size,dropout=0.0):
        super().__init__()
        assert n_embd%n_head==0
        head_size=n_embd//n_head
        self.heads=nn.ModuleList([
            Head(n_embd,head_size,block_size,dropout) for _ in range(n_head)
        ])
        self.proj=nn.Linear(n_embd,n_embd,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads])
        return self.dropout(out)



class CausalSelfAttention(nn.Module):
    def __init__(self,n_head,n_embd,block_size,dropout=0.0):
        super().__init__()
        assert n_embd%n_head==0
        self.n_head,self.n_embd=n_head,n_embd
        self.c_attn=nn.Linear(n_embd,n_embd*3,bias=False)
        self.proj=nn.Linear(n_embd,n_embd,bias=False)
        self.attn_dropout=nn.Dropout(dropout)
        self.resid_dropout=nn.Dropout(dropout)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C=x.shape
        q,k,v=self.c_attn(x).split(self.n_head,dim=-1)
        hs=self.n_embd//self.n_head
        q=q.view(B,T,self.n_head,hs).transpose(1,2)
        k=k.view(B,T,self.n_head,hs).transpose(1,2)
        v=v.view(B,T,self.n_head,hs).transpose(1,2)
        wei=q@k.transpose(-2,-1)*hs**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei=F.softmax(wei,dim=-1)
        wei=self.attn_dropout(wei)
        out=wei@v
        out=out.tranpose(1,2).contiguous().view(B,T,C)
        return self.resid_dropout(out)


if __name__=="__main__":
    import time

    device="cuda" if torch.cuda.is_available() else "cpu"
    B,T,n_embd,n_head,block_size=8,256,384,6,256
    x=torch.randn(B,T,n_embd,device=device)

    # ---- 1. 形状自检：输入输出形状必须一致 (B,T,n_embd) ----
    mha=CausalSelfAttention(n_head,n_embd,block_size).to(device)
    out=mha(x)
    print("输入形状:", tuple(x.shape), "→ 输出形状:", tuple(out.shape))
    assert out.shape == (B, T, n_embd), "多头不应改变 (B,T,C) 形状"

    # ---- 2. 加速实测：fused（工业版）vs loop（教学版）----
    # 注意：CPU 上差距偏小；GPU 上 kernel 启动 + 带宽优势才充分显现
    loop=MultiHeadAttentionLoop(n_head,n_embd,block_size).to(device).eval()
    fused=CausalSelfAttention(n_head,n_embd,block_size).to(device).eval()

    def bench(model,x,iters=200):
        with torch.no_grad():
            for _ in range(20):
                model(x)
            if device=="cuda":
                torch.cuda.synchronize()
            t0=time.perf_counter()
            for _ in range(iters):
                model(x)
            if device=="cuda":
                torch.cuda.synchronize()
            return (time.perf_counter()-t0)/iters*1e3

    t_loop = bench(loop, x)
    t_fused = bench(fused, x)
    print(f"\n设备: {device}")
    print(f"教学版(循环) : {t_loop:.3f} ms/次")
    print(f"工业版(fused): {t_fused:.3f} ms/次")
    print(f"加速比       : {t_loop / t_fused:.2f}x  ← fused QKV 的实测收益")










































