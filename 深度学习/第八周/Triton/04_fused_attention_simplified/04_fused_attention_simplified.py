import torch
import triton
import triton.language as tl

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










































