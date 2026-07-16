# benchmark.py —— 四组端到端对标：eager / compile / +手写 / compile+手写
import torch
import time
import statistics

from generate import generate
from integrate import swap_norm_backend

from nanoGPT import GPT,vocab_size



def measure_ttft(model,x,top_k=50):
    """首 token 延迟：只生成 1 个 token 的端到端时间。"""
    torch.cuda.synchronize()
    t0=time.perf_counter()
    _=generate(model,x,max_new_tokens=1,top_k=top_k)
    torch.cuda.synchronize()
    return time.perf_counter()-t0

def measure_tokps(model,x,n_tokens=100,top_k=50):
    """稳定吞吐：生成 n_tokens 个 token，算 tok/s。"""
    torch.cuda.synchronize()
    t0=time.perf_counter()
    _=generate(model,x,max_new_tokens=n_tokens,top_k=top_k)
    torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    return n_tokens/dt

def bench_one(name,model,x,warmup=3,iters=10):
    """对一个配置做完整测量：预热 + 多次取中位数。"""
    for _ in range(warmup):
        _=generate(model,x,max_new_tokens=8,top_k=50)
    torch.cuda.synchronize()

    ttfts,tokps=[],[]
    for _ in range(iters):
        ttfts.append(measure_ttft(model,x))
        tokps.append(measure_tokps(model,x))
    return {
        "name":name,
        "ttft_ms":statistics.median(ttfts)*1000,
        "tok_s":statistics.median(tokps),
    }

def run_showdown(build_model_fn,x):
    """
    build_model_fn(): 每次返回一个全新加载的模型（避免四组互相污染状态）。
    """
    results=[]

    # ① eager + torch RMSNorm（纯基准线）
    m=swap_norm_backend(build_model_fn(),"torch").cuda().to(torch.bfloat16).eval()
    results.append(bench_one("① eager (baseline)", m, x))
    del m
    torch.cuda.empty_cache()

    # ② torch.compile(max-autotune) + torch RMSNorm（最强对手）
    m=swap_norm_backend(build_model_fn(),"torch").cuda().to(torch.bfloat16).eval()
    m=torch.compile(m,mode="max-autotune")    #或许不加会快一点
    results.append(bench_one("② compile(max-autotune)",m,x))
    del m
    torch.cuda.empty_cache()

    # ③ eager + 手写 Triton kernel
    m=swap_norm_backend(build_model_fn(),"triton").cuda().to(torch.bfloat16).eval()
    results.append(bench_one("③ eager + 手写kernel", m, x))
    del m
    torch.cuda.empty_cache()

    # ④ compile + 手写 kernel（graph break 陷阱）
    m=swap_norm_backend(build_model_fn(),"triton").cuda().to(torch.bfloat16).eval()
    m=torch.compile(m,mode="max-autotune")
    results.append(bench_one("④ compile + 手写kernel", m, x))
    del m
    torch.cuda.empty_cache()

    return results

def print_table(results):
    base=results[0]["tok_s"]
    print(f"{'配置':<26}{'TTFT(ms)':>12}{'tok/s':>12}{'相对eager':>12}")
    print("-" * 62)
    for r in results:
        print(f"{r['name']:<26}{r['ttft_ms']:>12.2f}"f"{r['tok_s']:>12.1f}{r['tok_s']/base:>11.2f}×")


if __name__=="__main__":
    device="cuda"

    x=torch.randint(0,vocab_size,(1,64),device=device)

    results=run_showdown(lambda :GPT(),x)
    print_table(results)






















