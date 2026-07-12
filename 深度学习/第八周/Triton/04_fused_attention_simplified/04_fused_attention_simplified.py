import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# ---------- 阶梯一 naive fused(无 online softmax,需 BLOCK_N >= T)----------

@triton.jit
def fused_attn_naive_kernel(
        Q,K,V,O,
        T,
        scale,
        D:tl.constexpr,
        BLOCK_M:tl.constexpr,
        BLOCK_N:tl.constexpr,
        IS_CAUSAL:tl.constexpr
):
    pid=tl.program_id(0)
    offs_m=pid*BLOCK_M+tl.arange(0,BLOCK_M)
    offs_n=tl.arange(0,BLOCK_N)
    offs_d=tl.arange(0,D)

    q=tl.load(Q+offs_m[:,None]*D+offs_d[None,:],mask=offs_m[:,None]<T,other=0.0)
    k=tl.load(K+offs_n[:,None]*D+offs_d[None,:],mask=offs_n[:,None]<T,other=0.0)
    v=tl.load(V+offs_n[:,None]*D+offs_d[None,:],mask=offs_n[:,None]<T,other=0.0)

    s=tl.dot(q,tl.trans(k))*scale
    s=tl.where(offs_n[None,:]<T,s,float('-inf'))
    if IS_CAUSAL:
        s=tl.where(offs_m[:,None]>=offs_n[None,:],s,float('-inf'))

    m=tl.max(s,axis=1)
    p=tl.exp(s-m[:,None])
    l=tl.sum(p,axis=1)
    p=p/l[:,None]

    o=tl.dot(p.to(v.dtype),v)
    tl.store(O+offs_m[:,None]*D+offs_d[None,:],o,mask=offs_m[:,None]<T)


def fused_attn_naive(q,k,v,causal=True):
    T, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = triton.next_power_of_2(T)  # 整块覆盖:小 T 才行
    grid = (triton.cdiv(T, BLOCK_M),)
    fused_attn_naive_kernel[grid](
        q, k, v, o, T, D ** -0.5,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IS_CAUSAL=causal,
    )
    return o


# ---------- 阶梯二 online softmax flash(真·FlashAttention 前向)----------
@triton.jit
def flash_attn_kernel(
        Q,K,V,O,
        T,scale,
        D:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,
        IS_CAUSAL:tl.constexpr
):
    pid=tl.program_id(0)
    offs_m=pid*BLOCK_M+tl.arange(0,BLOCK_M)
    offs_d=tl.arange(0,D)
    q=tl.load(Q+offs_m[:,None]*D+offs_d[None,:],mask=offs_m[:,None]<T,other=0.0)

    m_i=tl.full((BLOCK_M,),float("-inf"),dtype=tl.float32)
    l_i=tl.zeros((BLOCK_M,),dtype=tl.float32)
    acc=tl.zeros((BLOCK_M,D),dtype=tl.float32)

    for start_n in range(0,T,BLOCK_N):
        offs_n=start_n+tl.arange(0,BLOCK_N)
        k=tl.load(K+offs_n[:,None]*D+offs_d[None,:],mask=offs_n[:,None]<T,other=0.0)
        v=tl.load(V+offs_n[:,None]*D+offs_d[None,:],mask=offs_n[:,None]<T,other=0.0)
        s=tl.dot(q,tl.trans(k))*scale
        s=tl.where(offs_n[None,:]<T,s,float('-inf'))
        if IS_CAUSAL:
            s=tl.where(offs_m[:,None]>=offs_n[None,:],s,float("-inf"))
        m_ij=tl.max(s,axis=1)
        m_new=tl.maximum(m_i,m_ij)
        alpha=tl.exp(m_i-m_new)
        p=tl.exp(s-m_new[:,None])
        l_i=l_i*alpha+tl.sum(p,axis=1)
        acc=acc*alpha[:,None]+tl.dot(p.to(v.dtype),v)
        m_i=m_new

    acc=acc/l_i[:,None]
    tl.store(O+offs_m[:,None]*D+offs_d[None,:],acc,mask=offs_m[:,None]<T)


def flash_attn(q,k,v,causal=True,BLOCK_M=64,BLOCK_N=64):
    T,D=q.shape
    o=torch.empty_like(q)
    grid=(triton.cdiv(T,BLOCK_M),)
    flash_attn_kernel[grid](q,k,v,o,T,D**-0.5,D=D,BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,IS_CAUSAL=causal)
    return o



# ---------- 对标官方 + cosine / top-1 误差 ----------
def cosine_sim(a,b):
    a,b=a.flatten().float(),b.flatten().float()
    return F.cosine_similarity(a,b,dim=0).item()

def top1_match(a,b):
     return (a.argmax(dim=-1)==b.argmax(dim=-1)).float().mean().item()

def benchmark(fn,q,k,v,causal,iters=100):
    for _ in range(10):
        fn(q,k,v,causal)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q,k,v,causal)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/iters


if __name__=="__main__":
    torch.manual_seed(0)
    T,D=1024,64
    device="cuda"
    dtype=torch.float16
    q=torch.randn(T,D,device=device,dtype=dtype)
    k=torch.randn(T,D,device=device,dtype=dtype)
    v=torch.randn(T,D,device=device,dtype=dtype)

    out_mine=flash_attn(q,k,v,causal=True)

    # 官方内置 FlashAttention:F.scaled_dot_product_attention
    # 需要 (batch, heads, T, D) 形状,补上 batch/head 维
    out_ref=F.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None],is_causal=True)[0,0]

    print(f"cosine 相似度 = {cosine_sim(out_mine, out_ref):.6f}   (目标 > 0.999)")
    print(f"top-1 一致率  = {top1_match(out_mine, out_ref):.4f}")
    print(f"我的  kernel : {benchmark(flash_attn, q, k, v, True):.4f} ms")
    print(f"官方  SDPA   : {benchmark(lambda *a: F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], is_causal=True), q, k, v, True):.4f} ms")


































