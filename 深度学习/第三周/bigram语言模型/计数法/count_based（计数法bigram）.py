from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import path

DATA_PATH=Path(__file__).parent.parent/"names.txt"
words=DATA_PATH.read_text(encoding='utf-8').splitlines()
words=[w.strip() for w in words if w.strip()]
print(f"训练集大小：{len(words)}个名字")
print(f"前5个示例{words[:5]}")
print(f"最短{min(len(w) for w in words)}")
print(f"最长{max(len(w) for w in words)}")


chars=sorted(set("".join(words)))
stoi={c:i+1 for i,c in enumerate(chars)}
stoi["."]=0
itos={i:c for c,i in stoi.items()}
V=len(stoi)
assert V==27,f"字符大小应该为27，实际为{V}"


N=torch.zeros((V,V),dtype=torch.int64)

for w in words:
    chs=['.']+list(w)+['.']
    for c1,c2 in zip(chs,chs[1:]):
        i,j=stoi[c1],stoi[c2]
        N[i,j]+=1

print(f"总的bigram对数：{N.sum().item()}")

# 可视化
plt.figure(figsize=(12,12))
plt.imshow(N,cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr=itos[i]+itos[j]
        plt.text(j,i,chstr,ha='center',va='bottom',color='gray')
        plt.text(j,i,N[i,j].item(),ha='center',va='top',color='gray')
plt.axis('off')
#plt.show()


N_smoothed=(N+1).float()
P=N_smoothed/N_smoothed.sum(dim=1,keepdim=True)
assert torch.allclose(P.sum(dim=1),torch.ones(V),atol=1e-6),"归一化失败"

#用 torch.multinomial 采样生成 20 个名字
g=torch.Generator().manual_seed(2147483647)

generated=[]
for _ in range(20):
    out=[]
    ix=0
    while True:
        p_row=P[ix]
        ix=torch.multinomial(p_row,num_samples=1,replacement=True,generator=g).item()
        if ix==0:
            break
        out.append(itos[ix])
    generated.append("".join(out))

for name in generated:
    print(name)


log_likelihood=0.0
n_pairs=0

for w in words:
    chs=["."]+list(w)+["."]
    for c1,c2 in zip(chs,chs[1:]):
        i,j=stoi[c1],stoi[c2]
        log_likelihood+=torch.log(P[i,j])
        n_pairs+=1

nll=-log_likelihood/n_pairs
print(f"{nll=}")


























