import math
import torch
import torch.nn as nn
from torch.nn import functional as F


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


def get_batch():
    ix=torch.randint(len(data)-1-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device),y.to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn=nn.Linear(n_embd,3*n_embd,bias=False)         #进食后人：bias=False
        self.c_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(n_embd,dim=2)
        q=q.view(B,T,n_head,C//n_head).transpose(1,2)
        k=k.view(B,T,n_head,C//n_head).transpose(1,2)
        v=v.view(B,T,n_head,C//n_head).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc=nn.Linear(n_embd,n_embd*4,bias=False)
        self.gelu=nn.GELU()
        self.c_proj=nn.Linear(n_embd*4,n_embd,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))



class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1=nn.LayerNorm(n_embd)
        self.attn=CausalSelfAttention()
        self.ln_2=nn.LayerNorm(n_embd)
        self.mlp=MLP()

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))  # 通信：token交换信息
        x=x+self.mlp(self.ln_2(x))  # 计算：逐token独立加工
        return x



class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size,n_embd),  #token查表
            wpe=nn.Embedding(block_size,n_embd),  #位置查表
            h=nn.ModuleList([Block() for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd)
        ))
        self.lm_head=nn.Linear(n_embd,vocab_size,bias=False)
        self.transformer.wte.weight=self.lm_head.weight  #共用矩阵
        self.apply(self._init_weights)
        self.config={"batch_size":16,
                    "block_size":128,
                    "n_embd":256,
                    "n_head":4,
                    "n_layer":4,
                    "dropout":0.0
        }

    def _init_weights(self,module):
        std = 0.02
        if isinstance(module,nn.Linear):
            if hasattr(module,"_is_residual_proj"):
                std*=(2*n_layer)**-0.5
            nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:   #好习惯
                nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=std)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_emb=self.transformer.wte(idx)
        pos_emb=self.transformer.wpe(torch.arange(T,device=idx.device))
        x=tok_emb+pos_emb
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,vocab_size),targets.view(-1))
        return logits,loss

    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        self.eval()   #好习惯
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits,_=self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx


if __name__=="__main__":
    model=GPT().to(device)
    for block in model.transformer.h:
        block.attn.c_proj._is_residual_proj = True
        block.mlp.c_proj._is_residual_proj = True
    model.apply(model._init_weights)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab_size = {vocab_size}, 参数量 = {n_params / 1e3:.0f}K, device = {device}")

    xb, yb = get_batch()
    _, loss = model(xb, yb)
    expected = math.log(vocab_size)
    print(f"初始 loss = {loss.item():.4f}, 理论值 -log(1/{vocab_size}) = {expected:.4f}")
    assert abs(loss.item() - expected) < 0.3, "初始 loss 偏离理论值,检查初始化/最后一层!"

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW + 3e-4:LLM 默认起手式
    for step in range(2000):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)  # set_to_none 比置零省一次显存写,工业惯例
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- 生成样例 ---")
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))






















































