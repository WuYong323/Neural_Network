import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity,schedule,tensorboard_trace_handler



class MakemoreMLP(torch.nn.Module):
    def __init__(self,vocab_size=27,emb_dim=10,block_size=3,hidden=200):
        super().__init__()
        self.C=nn.Embedding(vocab_size,emb_dim)
        self.fc1=nn.Linear(emb_dim*block_size,hidden)
        self.bn=nn.BatchNorm1d(hidden)
        self.fc2=nn.Linear(hidden,vocab_size)

    def forward(self,X):
        emb=self.C(X).view(X.size(0),-1)
        h=torch.tanh(self.bn(self.fc1(emb)))
        return self.fc2(h)


device='cuda' if torch.cuda.is_available() else 'cpu'
model=MakemoreMLP().to(device)
optim=torch.optim.Adam(model.parameters(),lr=1e-3)
batch_size=128
X=torch.randint(0,27,(batch_size,3),device=device)
Y=torch.randint(0,27,(batch_size,),device=device)



prof_schedule=schedule(wait=1,warmup=2,active=3,repeat=1)
activity=[ProfilerActivity.CPU]
if device=="cuda":
    activity.append(ProfilerActivity.CUDA)

with profile(
    activities=activity,
    schedule=prof_schedule,
    record_shapes=True,
    profile_memory=True,
    with_stack=False
) as prof:
    for step in range(10):  # 1 wait + 2 warmup + 3 active = 6，跑 10 步留余量
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        prof.step()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total",
    row_limit=15,
))
prof.export_chrome_trace("logs/makemore_trace.json")




































