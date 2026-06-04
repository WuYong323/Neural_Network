import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile,ProfilerActivity,schedule,tensorboard_trace_handler


torch.manual_seed(42)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] running on: {DEVICE}")


model=models.resnet18(num_classes=10).to(DEVICE)
model.train()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.05)
criterion=nn.CrossEntropyLoss()


BATCH=64
x=torch.randn(BATCH,3,224,224,device=DEVICE)
y=torch.randint(0,10,(BATCH,),device=DEVICE)


def train_step():
    optimizer.zero_grad(set_to_none=True)
    out=model(x)
    loss=criterion(out,y)
    loss.backward()
    optimizer.step()


LOG_DIR=os.path.join(str(os.path.dirname(__file__)),"logs")
os.makedirs(LOG_DIR,exist_ok=True)


activities=[ProfilerActivity.CPU]
if DEVICE=="cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    schedule=schedule(wait=1,warmup=2,active=3,repeat=1),
    on_trace_ready=tensorboard_trace_handler(LOG_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(6):
        train_step()
        prof.step()

trace_path=os.path.join(LOG_DIR,"resnet18_trace.json")
#prof.export_chrome_trace(trace_path)   #两个只能选一个
print(f"[ok] tensorboard 日志已写入: {LOG_DIR}")
#print(f"[ok] chrome trace 已导出: {trace_path}")
print("[tip] 看 trace 两种方式:")
print("      1) tensorboard --logdir logs  (推荐, 不会崩)")
print(f"      2) 把 {os.path.basename(trace_path)} 拖进 chrome://tracing (别用 ui.perfetto.dev, 会触发其 SliceMipmapOperator bug)")

key_sort="cuda_time_total" if torch.cuda.is_available() else"cpu_time_total"
print(prof.key_averages().table(sort_by=key_sort,row_limit=15))









































