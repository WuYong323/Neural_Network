import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import triton
import triton.language as tl

# ======================================================================
# 1) 融合 kernel:一次读、寄存器里连算、一次写
# ======================================================================
@triton.jit
def fused_mul_add_relu_kernel(
        x_ptr,a_ptr,b_ptr,out_ptr,
        n_elements,
        BLOCK_SIZE:tl.constexpr
):
    pid=tl.program_id(axis=0)
    offs=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    mask=offs<n_elements

    x=tl.load(x_ptr+offs,mask=mask)
    a=tl.load(a_ptr+offs,mask=mask)
    b=tl.load(b_ptr+offs,mask=mask)

    y=x*a+b
    y=tl.maximum(y,0.0)

    tl.store(out_ptr+offs,y,mask=mask)


def fused_triton(x,a,b):
    out=torch.empty_like(x)
    n=out.numel()
    grid=lambda meta:(triton.cdiv(n,meta["BLOCK_SIZE"]),)
    fused_mul_add_relu_kernel[grid](x,a,b,out,n,BLOCK_SIZE=1024)
    return out



# ======================================================================
# 2) 三个对标对象
# ======================================================================
def navie_torch(x,a,b):
    t1=x*a
    t2=t1+b
    return torch.relu(t2)

compiled_fused=torch.compile(navie_torch,mode="max-autotune")



# ======================================================================
# 3) 计时:torch.cuda.Event(接 W7 Day2 / Day1 的计时法)
# ======================================================================
def bench(fn,*args,warmup=30,iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/iters

def eff_bandwidth_gbps(ms,n,dtype_bytes=4):
    return (4*n*dtype_bytes)/(ms*1e-3)/1e9


# ======================================================================
# 4) 正确性 + 单点三方对标(取一个足够大的 memory-bound 尺寸)
# ======================================================================
def correctness_and_headline():
    torch.manual_seed(0)
    device="cuda"
    n=1<<24
    x=torch.randn(n,device=device)
    a=torch.randn(n,device=device)
    b=torch.randn(n,device=device)

    ref=navie_torch(x,a,b)
    out=fused_triton(x,a,b)
    assert torch.allclose(out,ref,rtol=1e-4,atol=1e-5), "融合结果不对"
    max_err=(out-ref).abs().max().item()
    print(f"[正确性] allclose 通过,最大逐元素误差 = {max_err:.2e}\n")

    t_naive=bench(navie_torch,x,a,b)
    t_compile=bench(compiled_fused,x,a,b)
    t_triton=bench(fused_triton,x,a,b)

    print(f"张量规模 n = {n:,}(fp32,memory-bound 区间)")
    print(f"{'方案':<22}{'耗时(ms)':>12}{'带宽(GB/s)':>14}{'相对朴素加速':>14}")
    print("-" * 62)
    for name, t in [("① 朴素 3 算子", t_naive),
                    ("② torch.compile", t_compile),
                    ("③ 手写 Triton 融合", t_triton)]:
        print(f"{name:<22}{t:>12.4f}{eff_bandwidth_gbps(t, n):>14.1f}"
              f"{t_naive / t:>14.2f}x")
    print("\n[预期] ②③ 都把访存从 8N 砍到 4N,加速比都逼近 2×,且彼此接近")
    print("       —— 因为都撞到同一堵 HBM 带宽墙,手写不会显著超过编译器。\n")



# ======================================================================
# 5) 关键实验:加速比 vs 张量大小(小 → 大),画曲线
# ======================================================================
def sweep_and_plot():
    torch.manual_seed(0)
    device="cuda"
    # 从 16K 扫到 64M;显存够的话可加 1<<27(128M)
    exps=[14,16,18,20,22,23,24,25,26]
    sizes,sp_triton,sp_compile=[],[],[]

    for e in exps:
        n=1<<e
        x = torch.randn(n, device="cuda")
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")

        t_naive=bench(navie_torch,x,a,b)
        t_triton=bench(fused_triton,x,a,b)
        t_compile=bench(compiled_fused,x,a,b)

        sizes.append(n)
        sp_triton.append(t_naive/t_triton)
        sp_compile.append(t_naive/t_compile)
        print(f"n=2^{e:<2}={n:>12,}  手写={t_naive / t_triton:5.2f}x  compile={t_naive / t_compile:5.2f}x")

        del x,a,b
        torch.cuda.empty_cache()

    plt.figure(figsize=(8,5))
    plt.plot(sizes,sp_triton,"o-",label="hand Triton fusion")
    plt.plot(sizes,sp_compile,"s--",label="torch.compile")
    plt.axhline(2.0,color="gray",ls=":",label="ceiling 2x (8N->4N)")
    plt.axhline(1.0,color="red",ls=":",label="1x (None)")
    plt.xscale("log")
    plt.xlabel("tensor number: N (log)")
    plt.ylabel("Speedup of relatively simple 3 operators")
    plt.title("Fusion speedup vs tensor size: small tensors hardly notice, big tensors get close to 2x")
    plt.legend()
    plt.grid(True,which="both",alpha=0.3)
    plt.tight_layout()
    plt.savefig("fusion_speedup_vs_size.png", dpi=150)
    print("\n[已保存] fusion_speedup_vs_size.png")



if __name__=="__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    correctness_and_headline()
    sweep_and_plot()










































