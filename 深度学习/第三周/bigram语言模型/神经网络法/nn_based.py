from pathlib import Path
import torch
import torch.nn.functional as F

DATA_PATH=Path(__file__).parent.parent/"names.txt"
words=DATA_PATH.read_text(encoding="utf-8").splitlines()
words=[w.strip() for w in words if w.strip()]

chars=sorted(set(''.join(words)))
stoi={s:i for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
V=len(stoi)
assert V==27

xs,ys=[],[]
for w in words:
    chs=["."]+list(w)+["."]
    for c1,c2 in zip(chs,chs[1:]):
        xs.append(c1)
        ys.append(c2)
        