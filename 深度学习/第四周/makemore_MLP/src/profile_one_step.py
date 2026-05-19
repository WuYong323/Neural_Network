from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity,profile,record_function

from model import MakemoreMLP
from train import load_data



def main():
    Xtr,Ytr,_,_,_,_,_,_=load_data(block_size=3)
    model=MakemoreMLP(block_size=3,embed_dim=10,hidden_size=200)

    Xb,Yb=Xtr[:32],Ytr[:32]
    for _ in range(3):
        logits=model.forward(Xb)
        loss=F.cross_entropy(logits,Yb)
        for p in model.parameters():
            p.grad=None
        loss.backward()

    Path("logs").mkdir(exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        for step in range(10):
            with record_function("step"):
                with record_function("forward"):
                    logits=model.forward(Xb)
                    loss=F.cross_entropy(logits,Yb)
                with record_function("backward"):
                    for p in model.parameters():
                        p.grad=None
                    loss.backward()
                with record_function("update"):
                    with torch.no_grad():
                        for p in model.parameters():
                            p.data-=0.1*p.grad

    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20
    ))

    trace_path="logs/profiler_trace.json"
    prof.export_chrome_trace(trace_path)



if __name__=="__main__":
    main()



















