import torch
import triton
import triton.language as tl

# ---------- ① autotune：给几组候选，让 Triton 自动选最快 ----------
@triton.autotune(
    configs=[
        # 每个 Config 是一套"配方"：BLOCK_SIZE + num_warps
        # BLOCK_SIZE 是每个program处理的元素
        # num_warps 是每个program里面的线程束(32个线程)
        triton.Config({"BLOCK_SIZE":1024},num_warps=4),
        triton.Config({"BLOCK_SIZE":1024},num_warps=8),
        triton.Config({"BLOCK_SIZE":2048},num_warps=8),
        triton.Config({"BLOCK_SIZE":4096},num_warps=8),
        triton.Config({"BLOCK_SIZE":4096},num_warps=16),
    ],
    key=['N']       # 当 N（行宽）变化时，重新跑一遍 autotune 选最优；N 不变就复用缓存
)

@triton.jit
def rmsnorm_kernel(x_ptr,w_ptr,out_ptr,
                   stride_row,N,eps,
                   BLOCK_SIZE:tl.constexpr
):
    row=tl.program_id(0)
    cols=tl.arange(0,BLOCK_SIZE)
    mask=cols<N
    x=tl.load(x_ptr+row*stride_row+cols,mask=mask,other=0.0)
    x_f32=x.to(tl.float32)
    mean_sq=tl.sum(x_f32*x_f32,axis=0)/N
    rstd=1.0/tl.sqrt(mean_sq+eps)
    w=tl.load(w_ptr+cols,mask=mask,other=0.0)
    out=(x_f32*rstd)*w.to(tl.float32)
    tl.store(out_ptr+row*stride_row+cols,out.to(x.dtype),mask=mask)


# ---------- ② Python 封装：外部像调普通函数一样调它 ----------
def rmsnorm_triton(x:torch.Tensor,weight:torch.Tensor,eps:float=1e-6):
    assert x.is_cuda and weight.is_cuda, "需要在GPU上"
    x=x.contiguous()
    M,N=x.shape
    out=torch.empty_like(x)
    grid=(M,)
    rmsnorm_kernel[grid](x,weight,out,x.stride(0),N,eps)
    return out


# ---------- ③ 参考实现（PyTorch eager，用来验证正确性） ----------
def rmsnorm_torch(x,weight,eps=1e-6):
    xf=x.float()
    ms=xf.pow(2).mean(dim=-1,keepdim=True)
    return (xf*torch.rsqrt(ms+eps)).to(x.dtype)*weight



# ---------- ④ 正确性检查 ----------
def correct():
    M, N = 4096, 4096
    x = torch.randn(M, N, device=device, dtype=torch.float16)
    w = torch.randn(N, device=device, dtype=torch.float16)
    y_triton = rmsnorm_triton(x, w)
    y_torch = rmsnorm_torch(x, w)

    ok = torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2)
    cos = torch.nn.functional.cosine_similarity(y_triton.flatten().float(), y_torch.flatten().float(), dim=0)
    print(f"allclose(rtol=1e-2): {ok}")
    print(f"cosine similarity  : {cos.item():.6f}")  # 期望 > 0.9999



# ---------- ⑤ 测三方，量带宽利用率 ----------
def benchmark():
    M,N=4096,4096
    dtype=torch.float16
    x=torch.randn(M,N,device=device,dtype=dtype)
    w=torch.randn(N,device=device,dtype=dtype)

    torch_compiled=torch.compile(rmsnorm_torch,mode="max-autotune")

    fns={
        "torch eager    ": lambda :rmsnorm_torch(x,w),
        "torch.compile  ": lambda :torch_compiled(x,w),
        "triton         ": lambda :rmsnorm_triton(x,w)
    }
    # RMSNorm 的访存量：读 x + 写 out ≈ 2 * M * N * 每元素字节数
    bytes_moved=2*M*N*x.element_size()
    for name,fn in fns.items():
        ms=triton.testing.do_bench(fn)
        gbps=bytes_moved/(ms*1e-3)/1e9
        print(f"{name}: {ms:.4f} ms | {gbps:6.1f} GB/s")


# ---------- ⑤ 正确性检查 ----------
if __name__=="__main__":
    torch.manual_seed(0)
    device="cuda"
    correct()
    benchmark()





































