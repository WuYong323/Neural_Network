import torch

import time



def bench(conv,x,iters:int=100):
    conv,x=conv.cuda(),x.cuda()

    for _ in range(10):
        conv(x)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters):
        conv(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000



def main():
    x = torch.randn(32, 256, 56, 56)
    conv1x1 = torch.nn.Conv2d(256, 256, kernel_size=1)
    conv3x3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
    if torch.cuda.is_available():
        t1 = bench(conv1x1, x)
        t3 = bench(conv3x3, x)
        print(f"1x1: {t1:.3f} ms   3x3: {t3:.3f} ms   3x3/1x1 = {t3 / t1:.1f}x")
    else:
        print("无 GPU；带宽瓶颈现象在 CPU 上不明显，建议在 GPU 上跑")


if __name__=="__main__":
    main()


































