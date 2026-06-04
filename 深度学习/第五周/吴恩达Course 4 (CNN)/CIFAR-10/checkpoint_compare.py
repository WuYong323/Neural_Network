import torch,time
import torch.nn as nn
from torch.nn import grad
from torch.utils.checkpoint import checkpoint_sequential
from model import build_cifar_resnet18



DEVICE="cuda"
BATCH=512
def mb(x): return x/1024/1024


def run(use_ckpt:bool):
    model=build_cifar_resnet18(num_classes=10).to(DEVICE).train()
    X=torch.randn(BATCH,3,32,32,device=DEVICE,requires_grad=True)
    y=torch.randint(0,10,(BATCH,),device=DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)

    body=nn.Sequential(model.conv1,model.bn1,model.relu,model.layer1,model.layer2,model.layer3,model.layer4)

    def step():
        opt.zero_grad(set_to_none=True)
        if use_ckpt:
            feat=checkpoint_sequential(body,segments=len(body),input=X,use_reentrant=False)
        else:
            feat=body(X)
        out=model.fc(model.avgpool(feat).flatten(1))
        loss=nn.functional.cross_entropy(out,y)
        loss.backward()
        opt.step()

    for _ in range(10):
        step()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()
    dt=(time.time()-t0)/20
    tag = "开启 checkpointing" if use_ckpt else "正常训练       "
    print(f"{tag} | 峰值显存 {mb(peak):7.1f} MB | 单步耗时 {dt * 1000:6.1f} ms")
    return peak, dt


if __name__=="__main__":
    print("=" * 60)
    p0, t0 = run(False)
    p1, t1 = run(True)
    print("=" * 60)
    print(f"显存节省: {(1 - p1 / p0) * 100:.1f}%   时间代价: +{(t1 / t0 - 1) * 100:.1f}%")
































