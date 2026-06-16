import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)


class CasualSelfAttention(nn.Module):
    def __init__(self,n_head,n_embd,block_size,dropout=0.0):
        super().__init__()
        assert n_embd%n_head==0
        self.n_head,self.n_embd=n_head,n_embd
        self.c_attn=nn.Linear(n_embd,3*n_embd,bias=False)
        self.c_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x,past_kv=None):
        """
        x:        (B, T, C)。prefill 时 T = prompt 长度；decode 时 T = 1（只有新 token）。
        past_kv:  None（首步）或 (past_k, past_v)，形状各 (B, nh, T_past, hs)。
        返回:     (out, (k, v))。(k, v) 是"拼上本步后"的新缓存，喂给下一步。
        """
        B,T,C=x.shape
        hs=C//self.n_head

        # ① 只为"当前输入的 T 个 token"算 Q/K/V（decode 时 T=1，这就是省下重算的地方）
        q,k,v=self.c_attn(x).split(self.n_embd,dim=-1)
        q=q.view(B,T,self.n_head,hs).transpose(1,2)
        k=k.view(B,T,self.n_head,hs).transpose(1,2)
        v=v.view(B,T,self.n_head,hs).transpose(1,2)

        # ② 把历史缓存拼到前面：这是 KV Cache 的灵魂一步
        if past_kv is not None:
            past_k,past_v=past_kv
            k=torch.cat([past_k,k],dim=2)
            v=torch.cat([past_v,v],dim=2)

        T_full=k.size(2)

        # ③ attention：当前 q（长 T）对完整 k（长 T_full）打分
        wei=(q@k.transpose(-2,-1))*hs**-0.5

        # ④ causal mask：当前这 T 个 token 在完整序列里的"绝对位置"是 [T_full-T, T_full)
        mask=self.tril[T_full-T:T_full,:T_full]
        wei=wei.masked_fill(mask==0,float("-inf"))
        wei=F.softmax(wei,dim=-1)

        out=wei@v
        out=out.transpose(1,2).contiguous().view(B,T,C)
        out=self.c_proj(out)
        return out,(k,v)


class Block(nn.Module):
    def __init__(self,n_head,n_embd,block_size,dropout=0.0):
        super().__init__()
        self.ln1=nn.LayerNorm(n_embd)
        self.attn=CasualSelfAttention(n_head,n_embd,block_size,dropout)
        self.ln2=nn.LayerNorm(n_embd)
        self.ffn=nn.Sequential(
            nn.Linear(n_embd,n_embd*4,bias=False),
            nn.GELU(),
            nn.Linear(n_embd*4,n_embd,bias=False)
        )

    def forward(self,x,past_kv):
        attn_out,new_kv=self.attn(self.ln1(x),past_kv)
        x=x+attn_out
        x=x+self.ffn(self.ln2(x))
        return x,new_kv


class miniGPT(nn.Module):
    """
    最小 GPT：token+pos embedding → N 个 Block → LN → 输出头。
    generate 时支持两种模式：朴素（每步全量 forward）和 KV Cache。
    """
    def __init__(self,vocab_size,n_layer,n_head,n_embd,block_size,dropout=0.0):
        super().__init__()
        self.block_size=block_size
        self.tok_emb=nn.Embedding(vocab_size,n_embd)
        self.pos_emb=nn.Embedding(block_size,n_embd)
        self.blocks=nn.ModuleList([
            Block(n_head,n_embd,block_size,dropout) for _ in range(n_layer)
        ])
        self.ln_f=nn.LayerNorm(n_embd)
        self.head=nn.Linear(n_embd,vocab_size,bias=False)


    def forward(self,idx,past_kvs=None,pos_offset=0):
        """
        idx:        (B, T) 的 token id。
        past_kvs:   None 或 长度为 n_layer 的列表，每项是该层的 (k, v) 缓存。
        pos_offset: 当前这批 token 在完整序列里的起始位置（decode 时 = 已生成长度）。
        """
        B,T=idx.shape
        pos=torch.arange(pos_offset,pos_offset+T,device=idx.device)
        x=self.tok_emb(idx)+self.pos_emb(pos)

        new_kvs=[]
        for i,block in enumerate(self.blocks):
            past=None if past_kvs is None else past_kvs[i]
            x,new_kv=block(x,past)
            new_kvs.append(new_kv)
        x=self.ln_f(x)
        logits=self.head(x)
        return logits,new_kvs


@torch.no_grad()
def generate_naive(model,idx,max_new_tokens):
    """朴素自回归：每一步把'当前完整序列'重新 forward 一遍（O(n²) 浪费）。"""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-model.block_size:]
        logits,_=model(idx_cond)
        logits=logits[:,-1,:]
        next_id=logits.argmax(dim=-1,keepdims=True)         # 贪心解码（为了可复现对比）
        idx=torch.cat([idx,next_id],dim=1)
    return idx


@torch.no_grad()
def generate_kv(model,idx,max_new_tokens):
    """KV Cache 生成：先 prefill 整个 prompt，再每步只 forward 1 个新 token。"""
    model.eval()
    B,T0=idx.shape
    # ---- prefill 阶段：一次性处理整个 prompt，建立初始缓存 ----
    logits,past_kvs=model(idx,past_kvs=None,pos_offset=0)
    next_id=logits[:,-1,:].argmax(dim=-1,keepdims=True)
    out=torch.cat([idx,next_id],dim=1)
    # ---- decode 阶段：每步只喂'刚生成的那 1 个 token' ----
    for step in range(max_new_tokens-1):
        pos=T0+step
        logits,past_kvs=model(next_id,past_kvs,pos)
        next_id=logits[:,-1,:].argmax(dim=-1,keepdims=True)
        out=torch.cat([out,next_id],dim=1)
    return out


if __name__=="__main__":
    import time

    device='cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size,n_layer,n_head,n_embd,block_size=65,4,4,128,512
    model=miniGPT(vocab_size,n_layer,n_head,n_embd,block_size).to(device).eval()

    B,T_prompt,max_new=1,8,256
    idx=torch.randint(0,vocab_size,(B,T_prompt),device=device)

    # ---- 1. 正确性断言：KV Cache 结果必须和朴素逐 token 一致（最重要的一步）----
    out_naive=generate_naive(model,idx.clone(),max_new)
    out_kv=generate_kv(model,idx.clone(),max_new)
    assert torch.equal(out_naive,out_kv),"KV Cache 生成结果和朴素不一致。大概率是 mask 切片或 pos_offset 写错"
    print("正确性通过：KV Cache 与朴素生成逐 token 完全一致")

    # ---- 2. 测速：各生成 256 token，对比延迟 ----
    def bench(fn,model,idx,max_new,iters=10):
        with torch.no_grad():
            for _ in range(3):
                fn(model,idx.clone(),max_new)
            if device=="cuda":
                torch.cuda.synchronize()
            t0=time.perf_counter()
            for _ in range(iters):
                fn(model,idx.clone(),max_new)
            if device=="cuda":
                torch.cuda.synchronize()
            return (time.perf_counter()-t0)/iters*1e3

    t_naive=bench(generate_naive,model,idx,max_new)
    t_kv=bench(generate_kv,model,idx,max_new)
    print(f"\n设备: {device}  |  生成 {max_new} token")
    print(f"朴素生成   : {t_naive:8.1f} ms")
    print(f"KV Cache   : {t_kv:8.1f} ms")
    print(f"加速比     : {t_naive / t_kv:.2f}x  ← 生成越长，加速比越大")
















































