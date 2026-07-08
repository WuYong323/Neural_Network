import torch
import torch.nn as nn

torch.manual_seed(0)
assert torch.cuda.is_available(), "这个实验要 GPU; Windows 上没有 Triton, 去 H100 跑"
device="cuda"


class FFNBlock(nn.Module):
    def __init__(self,d=1024,hidden=4096):
        super().__init__()
        self.ln=nn.LayerNorm(d)
        self.fc1=nn.Linear(d,hidden)
        self.fc2=nn.Linear(hidden,d)
        self.act=nn.GELU()

    def forward(self,x):
        h=self.ln(x)
        h=self.act(self.fc1(h))
        h=self.fc2(h)
        return x+h

model=FFNBlock().to(device).eval()

x=torch.randn(1,1,1024,device=device)

def bench(fn,x,warmup=30,iters=100):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/iters

with torch.no_grad():
    # baseline: 原生 eager
    t_eager=bench(model,x)

    # after: 默认 compile(Dynamo 抓图 + Inductor 融合生成 Triton)
    compiled=torch.compile(model)
    t_default=bench(compiled,x)

    compiled_ro=torch.compile(model,mode="reduce_overhead")
    t_ro=bench(compiled_ro,x)

print(f"eager                : {t_eager:.4f} ms/iter  (baseline)")
print(f"compile default      : {t_default:.4f} ms/iter  "
      f"({t_eager/t_default:.2f}x)")
print(f"compile reduce-overhd : {t_ro:.4f} ms/iter  "
      f"({t_eager/t_ro:.2f}x)  ← 加了自动 CUDA Graph")










































