from __future__ import annotations

import math

import matplotlib.pyplot as plt



def get_lr(step:int,warmup_steps:int=2000,total_steps:int=600000,max_lr:float=6e-4,min_lr:float=6e-5)->float:
    if step<warmup_steps:
        return max_lr*step/warmup_steps

    if step>total_steps:
        return min_lr

    decay_ratio=(step-warmup_steps)/(total_steps-warmup_steps)
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))

    return min_lr+coeff*(max_lr-min_lr)


def main()->None:
    total=600000
    steps=list(range(0,total+1,1000))
    lrs=[get_lr(step,warmup_steps=2000,total_steps=total) for step in steps]

    print(f"step      0 -> lr = {lrs[0]:.2e}")
    print(f"step  2,000 -> lr = {get_lr(2000):.2e}  (warmup 顶点)")
    print(f"step 300,000 -> lr = {get_lr(300_000):.2e}  (中段)")
    print(f"step 600,000 -> lr = {get_lr(600_000):.2e}  (终点 = min_lr)")

    plt.figure(figsize=(12,8))
    plt.plot(steps,lrs)
    plt.axvline(2000,color="red",linestyle="--",alpha=0.5,label="warmup end")
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.legend()

    plt.title("GPT/LLaMA warmup + cosine schedule")
    plt.tight_layout()

    plt.savefig("logs/lr_schedule.png",dpi=120)
    plt.show()



if __name__=="__main__":
    main()



































