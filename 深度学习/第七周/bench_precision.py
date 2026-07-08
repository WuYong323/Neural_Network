import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
assert torch.cuda.is_available(), "要 GPU 才有 FP16 加速; 纯 CPU 上 FP16 反而可能更慢"
device="cuda"

class Block(nn.Module):
    def __init__(self,d=1024,heads=16,hidden=4096):
        super().__init__()
        self.ln1=nn.LayerNorm(d)
        self.ln2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d)
        self.proj=nn.LayerNorm(d,d)
        self.fc1=nn.Linear(d,hidden)
        self.fc2=nn.Linear(hidden,d)
        self.heads=heads

    def forward(self,x):
        B,T,C=x.shape
        q,k,v=self.qkv(self.ln1(x)).chunk(3,dim=-1)
        q,k,v=[t.view(B,T,self.heads,C//self.heads).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.proj(a.transpose(1,2).reshape(B,T,C))
        x=x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

model=Block().to(device).eval()
x_fp32=torch.randn(1,512,1024,device=device)

def bench(model,x,warmup=30,iters=100):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True)
    e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        model(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

with torch.no_grad():
    # --- baseline: FP32 ---
    t_fp32=bench(model,x_fp32)
    out_fp32=model(x_fp32)

    # --- FP16: 权重和输入都转半精度 ---
    model_fp16=Block().to(device).eval()
    model_fp16.load_state_dict(model.state_dict())
    model_fp16=model_fp16.half()
    x_fp16=x_fp32.half()
    t_fp16=bench(model_fp16,x_fp16)
    out_fp16=model_fp16(x_fp16).float()


# ---- 速度 ----
print(f"FP32: {t_fp32:.4f} ms/iter  (baseline)")
print(f"FP16: {t_fp16:.4f} ms/iter  ({t_fp32/t_fp16:.2f}x faster)")
# ---- 误差(复用 §6.3 的三把尺子) ----
diff = (out_fp32 - out_fp16).abs()
cos = F.cosine_similarity(out_fp32.flatten(1), out_fp16.flatten(1), dim=-1).mean()
print(f"最大逐元素误差: {diff.max().item():.4e}")
print(f"平均逐元素误差: {diff.mean().item():.4e}")
print(f"输出 cosine   : {cos.item():.6f}  (>0.999 说明 FP16 几乎无损)")







































