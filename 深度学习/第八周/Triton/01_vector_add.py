import torch
import triton
import triton.language as tl
from torch.fx.experimental.migrate_gradual_types.constraint_generator import torch_dim_inference_rule


# ======================================================================
# 1) Triton kernel —— program视角
# ======================================================================
@triton.jit
def add_kernel(a_ptr,b_ptr,c_ptr,   # 三个张量在显存里的首地址(设备指针)
               n_elements,                                      # 元素总数(运行时值,不是 constexpr)
               BLOCK_SIZE:tl.constexpr                          # 每个 program 处理多少元素(编译期常量)
               ):
    pid=tl.program_id(axis=0)
    block_start=pid*BLOCK_SIZE
    offsets=block_start+tl.arange(0,BLOCK_SIZE)
    mask=offsets<n_elements
    a=tl.load(a_ptr+offsets,mask=mask,other=0.0)
    b=tl.load(b_ptr+offsets,mask=mask,other=0.0)
    c=a+b
    tl.store(c_ptr+offsets,c,mask=mask)



# ======================================================================
# 2) Python 封装 —— 在 CPU 端定义 grid 并 launch
# ======================================================================
def triton_add(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    assert a.is_cuda and b.is_cuda, "输入必须在GPU上"
    assert a.is_contiguous() and b.is_contiguous(), "本 kernel 假设内存连续"
    c=torch.empty_like(a)
    n_elements=c.numel()

    grid=lambda meta:(triton.cdiv(n_elements,meta["BLOCK_SIZE"]),)

    add_kernel[grid](a,b,c,n_elements,BLOCK_SIZE=1024)
    return c



# ======================================================================
# 3) 计时工具 —— 用 torch.cuda.Event(W7 Day2 计时法)
# ======================================================================
def bench(fn,warmup:int=30,iters:int=100)->float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/iters

def gbps(ms:float,n:int,dtype_bytes:int=4)->float:
    moved_bytes=3*n*dtype_bytes
    return moved_bytes/(ms*1e-3)/1e9



# ======================================================================
# 4) 主程序:正确性验证 + 三方 benchmark
# ======================================================================
def main():
    torch.manual_seed(0)
    device="cuda"
    n=1000000
    a=torch.randn(n,device=device,dtype=torch.float32)
    b=torch.randn(n,device=device,dtype=torch.float32)

    # ---- 三个待测对象 ----
    # 1.PyTorch 原生逐元素加
    torch_add=lambda :a+b
    # 2.# torch.compile(底层也会生成 Triton)
    compiled=torch.compile(lambda x,y:x+y)
    compiled_add=lambda :compiled(a,b)
    # 3.手写Triton kernel
    my_triton_add=lambda :triton_add(a,b)

    # ---- 正确性(尺子①:elementwise 应逐元素几乎相等)----
    out_triton=triton_add(a,b)
    out_ref=a+b
    assert torch.allclose(out_triton, out_ref, rtol=1e-5, atol=1e-6), "结果不对!"
    max_err=(out_triton-out_ref).abs().max().item()
    print(f"[正确性] torch.allclose 通过,最大逐元素误差 = {max_err:.2e}")

    # ---- 三方计时 ----
    t_torch=bench(torch_add)
    t_compile=bench(compiled_add)
    t_triton=bench(my_triton_add)

    print(f"\n{'方案':<18}{'耗时(ms)':>12}{'有效带宽(GB/s)':>18}")
    print("-" * 48)
    for name, t in [("torch 原生", t_torch),("torch.compile", t_compile),("我的 Triton", t_triton)]:
        print(f"{name:<18}{t:>12.4f}{gbps(t, n):>18.1f}")
    print("\n[结论提示] 1M 元素太小,大概率三者持平、且远未跑满 HBM 带宽")
    print("           —— 因为此时是 launch/延迟受限,不是带宽受限。")
    print("           想看到带宽 roofline,把 n 调到 64M+(如 n = 1 << 26)再跑。")



if __name__=="__main__":
    main()

















































