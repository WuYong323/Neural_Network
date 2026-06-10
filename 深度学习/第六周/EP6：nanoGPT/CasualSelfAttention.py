import torch
import torch.nn as nn
import torch.nn.functional as F



class CasualSelfAttention(nn.Module):
    def __init__(self,n_head,n_embd,block_size,dropout=0.0):
        super().__init__()
        assert n_embd%n_head==0
        self.n_head=n_head
        self.n_embd=n_embd

        self.c_atten=nn.Linear(n_embd,n_embd*3,bias=False)
        self.c_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.atten_dropout=nn.Dropout(dropout)
        self.resid_dropout=nn.Dropout(dropout)
        self.dropout=dropout
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C=x.shape
        # ① 一次大 GEMM 算出 Q/K/V 三件套，再切成三块
        qkv=self.c_atten(x)
        q,k,v=qkv.split(self.n_embd,dim=-1)
        # ② reshape 出头维：(B,T,C) → (B,T,nh,hs) → (B,nh,T,hs)
        hs=C//self.n_head
        q=q.view(B,T,self.n_head,hs).transpose(1,2)
        k=k.view(B,T,self.n_head,hs).transpose(1,2)
        v=v.view(B,T,self.n_head,hs).transpose(1,2)
        # ③ 批量 attention：B×nh 个头全部并行（@ 自动把前两维当批量维）
        wei=(q@k.transpose(-2,-1))*hs**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei=F.softmax(wei,dim=-1)
        wei=self.atten_dropout(wei)
        out=wei@v                                       # (B, nh, T, hs)
        # ④ 合头：把头维 transpose 回去再 view 合并 → (B,T,C)
        out=out.transpose(1,2).contigous().view(B,T,C)
        # ⑤ 输出投影
        out=self.resid_dropout(self.c_proj(out))
        return out









































