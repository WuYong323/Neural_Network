from __future__ import annotations

import time
import csv
from pathlib import Path

from torch import Tensor

from model import MakemoreMLP
from train import train,load_data,eval_loss




EXPERIMENTS:list[tuple[str,int,int,int,float]]=[
    ("W4D3_00_baseline", 3, 2, 100, 0.10),
    ("W4D3_01_emb10", 3, 10, 100, 0.10),
    ("W4D3_02_h200", 3, 10, 200, 0.10),
    ("W4D3_03_h300", 3, 10, 300, 0.05),
    ("W4D3_04_b5", 5, 10, 200, 0.05),
    ("W4D3_05_b8", 8, 10, 200, 0.05),
]


def run_one(
        exp_id:str,
        B:int,
        E:int,
        H:int,
        lr:float,
        Xtr:Tensor,
        Ytr:Tensor,
        Xdv:Tensor,
        Ydv:Tensor,
        steps:int=50000,
        seed:int=42
):
    model=MakemoreMLP(block_size=B,embed_dim=E,hidden_size=H,seed=seed)
    t0=time.perf_counter()
    history=train(model,Xtr,Ytr,Xdv,Ydv,steps=steps,batch_size=32,lr=lr,eval_every=steps,seed=seed)
    wall=time.perf_counter()-t0
    final_step,train_loss,dev_loss=history[-1]
    return {
        "exp_id": exp_id, "seed": seed,
        "block_size": B, "embed_dim": E, "hidden_size": H,
        "lr": lr, "batch_size": 32, "steps": steps,
        "num_params": model.num_params(),
        "train_loss_final": round(train_loss, 4),
        "dev_loss_final": round(dev_loss, 4),
        "wall_time_sec": round(wall, 1),
    }


def main()->None:
    Path("logs").mkdir(exist_ok=True)
    rows=[]
    for exp_id,B,E,H,lr in EXPERIMENTS:
        print(f"\n=== {exp_id}: B={B} E={E} H={H} lr={lr} ===")
        Xtr,Ytr,Xdv,Ydv,_,_,_,_=load_data(block_size=B)
        row=run_one(exp_id=exp_id,B=B,E=E,H=H,lr=lr,Xtr=Xtr,Ytr=Ytr,Xdv=Xdv,Ydv=Ydv)
        rows.append(row)
        print(f"→ dev_loss={row['dev_loss_final']:.4f}, "
              f"params={row['num_params']:,}, wall={row['wall_time_sec']}s")


    csv_path="logs/exp_compare.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



if __name__=="__main__":
    main()



























