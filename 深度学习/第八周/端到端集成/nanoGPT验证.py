import torch
from nanoGPT import GPT,get_batch
from integrate import swap_norm_backend
from generate import generate



torch.manual_seed(1337)

batch_size=16
block_size=128
n_embd=256
n_head=4
n_layer=4
dropout=0.0
device="cuda" if torch.cuda.is_available() else "cpu"

with open("斗破苍穹.txt",'r',encoding='gbk') as f:   #utf-8是中文训练集
    words=f.read()        #加splitlines（）就是逐句token，不加就是逐字母token（中文不加才训练的好）
chars=sorted(list(set(words)))
vocab_size=len(chars)
stoi={s:i for i,s in enumerate(chars)}
itos={i:s for s,i in stoi.items()}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:''.join(itos[i] for i in l)
data=torch.tensor(encode(words),dtype=torch.long)

model=GPT().to(device)
model = swap_norm_backend(model, "triton")
model = model.cuda().to(torch.bfloat16)

for block in model.transformer.h:
    block.attn.c_proj._is_residual_proj = True
    block.mlp.c_proj._is_residual_proj = True
model.apply(model._init_weights)

xb, yb = get_batch()
_, loss = model(xb, yb)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW + 3e-4:LLM 默认起手式
for step in range(2000):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # set_to_none 比置零省一次显存写,工业惯例
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

start_ids = encode("萧炎")
x = torch.tensor(start_ids, dtype=torch.long, device="cuda")[None,:]
out = generate(model, x, max_new_tokens=50, top_k=50)
print(decode(out[0].tolist()))





