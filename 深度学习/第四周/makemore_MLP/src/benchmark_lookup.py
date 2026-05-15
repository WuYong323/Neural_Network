import time
import torch
import torch.nn.functional as F


def benchmark(device:str="cpu",n:int=10_000,vocab:int=27,dim:int=64)->None:
    device_t=torch.device(device)
    X=torch.randint(0,vocab,(n,),device=device_t)
    C=torch.randn(vocab,dim,device=device_t)

    for _ in range(3):
        _=F.one_hot(X,vocab).float()@C
        _=C[X]
    if device=="cuda":
        torch.cuda.synchronize()

    iters=100


    #方法A
    t0=time.perf_counter()
    for _ in range(iters):
        out_a=F.one_hot(X,vocab).float()@C
    if device == "cuda":
        torch.cuda.synchronize()
    t_a=(time.perf_counter()-t0)/iters*1e6  #微秒


    #方法B
    t0 = time.perf_counter()
    for _ in range(iters):
        out_a = C[X]
    if device == "cuda":
        torch.cuda.synchronize()
    t_b = (time.perf_counter() - t0) / iters * 1e6  # 微秒

    print(f"[{device}] N={n}, V={vocab}, d={dim}")
    print(f"  one_hot @ C : {t_a:8.1f} µs / call")
    print(f"  C[X]        : {t_b:8.1f} µs / call")
    print(f"  speedup     : {t_a / t_b:6.2f}×")


if __name__=="__main__":
    benchmark("cpu",n=10_000,vocab=27,dim=64)
    if torch.cuda.is_available():
        benchmark("cuda", n=10_000, vocab=27, dim=64)


