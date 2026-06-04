import torch
from model import build_cifar_resnet18



assert torch.cuda.is_available(), "本实验需要 GPU 才能测显存"
DEVICE = "cuda"
BATCH = 128


def mb(x):
    return x/1024/1024


model=build_cifar_resnet18(num_classes=10).to(DEVICE)
torch.cuda.synchronize()
param_mem=torch.cuda.memory_allocated()
print(f"[1] 仅参数:            {mb(param_mem):8.1f} MB")

n_params=sum(p.numel() for p in model.parameters())
print(f"    参数量 = {n_params/1e6:.2f} M, 公式估算 = {mb(n_params*4):.1f} MB")


X=torch.randn(BATCH,3,32,32,device=DEVICE)
y=torch.randint(0,10,(BATCH,),device=DEVICE)
torch.cuda.reset_peak_memory_stats()
model.train()
out=model(X)
loss=torch.nn.functional.cross_entropy(out,y)
torch.cuda.synchronize()
fwd_peak=torch.cuda.max_memory_allocated()
print(f"[2] 参数+激活(前向峰值): {mb(fwd_peak):8.1f} MB  -> 激活≈ {mb(fwd_peak-param_mem):.1f} MB")

loss.backward()
torch.cuda.synchronize()
grad_mem=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)
print(f"[3] 梯度:              {mb(grad_mem):8.1f} MB  （应≈参数大小，验证 梯度=参数×1）")

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
optimizer.step()
torch.cuda.synchronize()
opt_state_mem=0
for st in optimizer.state.values():
    for v in st.values():
        if torch.is_tensor(v):
            opt_state_mem+=v.numel()*v.element_size()
print(f"[4] 优化器状态(Adam):   {mb(opt_state_mem):8.1f} MB  （应≈参数×2: m 和 v 各一份）")

print("-"*50)
print(f"总峰值: {mb(torch.cuda.max_memory_allocated()):.1f} MB")








































