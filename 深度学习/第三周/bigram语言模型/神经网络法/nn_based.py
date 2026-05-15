from pathlib import Path
import torch
import torch.nn.functional as F

DATA_PATH=Path(__file__).parent.parent/"names.txt"
words=DATA_PATH.read_text(encoding="utf-8").splitlines()
words=[w.strip() for w in words if w.strip()]

chars=sorted(set(''.join(words)))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
V=len(stoi)
assert V==27

xs,ys=[],[]
for w in words:
    chs=["."]+list(w)+["."]
    for c1,c2 in zip(chs,chs[1:]):
        xs.append(stoi[c1])
        ys.append(stoi[c2])

xs=torch.tensor(xs,dtype=torch.long)
ys=torch.tensor(ys,dtype=torch.long)
N=xs.numel()        #样本量

g=torch.Generator().manual_seed(2147483647)
w=torch.randn((V,V),generator=g,requires_grad=True)


lr=50.0     # 学习率: bigram 这种简单模型可以用很大的 lr
epochs=200
l2=0.01

print(f"\n开始训练: lr={lr}, epochs={epochs}, lambda={l2}")

xenc=F.one_hot(xs,num_classes=V).float()
for epoch in range(epochs):
    logits=xenc@w
    log_probs=F.log_softmax(logits,dim=1)
    nll=-log_probs[torch.arange(N),ys].mean()
    reg=l2*(w**2).mean()
    loss=nll+reg

    w.grad=None
    loss.backward()

    with torch.no_grad():
        w-=lr*w.grad

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  epoch {epoch + 1:>3} | nll = {nll.item():.4f} | reg = {reg.item():.4f}")


with torch.no_grad():
    p_nn=F.softmax(w,dim=1)

g_sample = torch.Generator().manual_seed(2147483647)
generate=[]

for _ in range(20):
    out=[]
    ix=0
    while True:
        p_row=p_nn[ix]
        ix=torch.multinomial(p_row,num_samples=1,replacement=True,generator=g_sample).item()

        if ix==0:
            break
        out.append(itos[ix])
    generate.append("".join(out))

for name in generate:
    print(name)













