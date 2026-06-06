import torch
import torch.nn.functional as F

import random



torch.manual_seed(42)

words=open("names.txt").read().splitlines()
chars=sorted(list(set("".join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size=len(stoi)

block_size=8
def build_dataset(words):
    X,Y=[],[]
    for w in words:
        context=[0]*block_size
        for ch in w+'.':
            ix=stoi[ch]
            X.append(context)
            Y.append(ix)
            context=context[1:]+[ix]
    return torch.tensor(X),torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1,n2=int(0.8*len(words)),int(0.9*len(words))
Xtr,Ytr=build_dataset(words[:n1])
Xdev,Ydev=build_dataset(words[n1:n2])


class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight=torch.randn((fan_in,fan_out))/fan_in**0.5
        self.bias=torch.zeros(fan_out) if bias else None

    def __call__(self,x):
        self.out=x@self.weight
        if self.bias is not None:
            self.out=self.out+self.bias
        return self.out

    def parameters(self):
        return [self.weight]+([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        self.eps=eps
        self.momentum=momentum
        self.train=True
        self.gamma,self.beta=torch.ones(dim),torch.zeros(dim)
        self.running_mean,self.running_var=torch.zeros(dim),torch.ones(dim)

    def __call__(self,x):
        if self.train:
            dim=0 if x.ndim==2 else (0,1)
            mean,var=x.mean(dim,keepdim=True),x.var(dim,keepdim=True)
            with torch.no_grad():
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*var
        else:
            mean,var=self.running_mean,self.running_var

        self.out=self.gamma*(x-mean)/torch.sqrt(self.eps+var)+self.beta
        return self.out

    def parameters(self):
        return [self.beta,self.gamma]


class Tanh:
    def __call__(self,x):
        self.out=torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self,num,dim):
        self.weight=torch.randn((num,dim))

    def __call__(self,ix):
        self.out=self.weight[ix]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self,n):
        self.n=n

    def __call__(self,x):
        B,T,C=x.shape
        x=x.view(B,T//self.n,C*self.n)
        if x.shape[1]==1:
            x=x.squeeze(1)
        self.out=x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self,layers):
        self.layers=layers

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        self.out=x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



n_embd,n_hidden=10,68
model=Sequential([
    Embedding(vocab_size,n_embd),
    FlattenConsecutive(2),Linear(n_embd*2,n_hidden,False),BatchNorm1d(n_hidden),Tanh(),
    FlattenConsecutive(2),Linear(n_hidden*2,n_hidden,False),BatchNorm1d(n_hidden),Tanh(),
    FlattenConsecutive(2),Linear(n_hidden*2,n_hidden,False),BatchNorm1d(n_hidden),Tanh(),
    Linear(n_hidden,vocab_size)
])
with torch.no_grad():
    model.layers[-1].weight*=0.1
parameters=model.parameters()
for p in parameters:
    p.requires_grad=True
print("参数量：",sum(p.nelement() for p in parameters))


for i in range(20000):
    ix=torch.randint(0,Xtr.shape[0],(32,))
    logits=model(Xtr[ix])
    loss=F.cross_entropy(logits,Ytr[ix])
    for p in parameters:
        p.grad=None
    loss.backward()
    lr=0.1 if i<15000 else 0.01
    for p in parameters:
        p.data-=lr*p.grad
    if i%5000==0:
        print(f"{i:5d} | loss {loss.item():.4f}")


for layer in model.layers:
    if isinstance(layer,BatchNorm1d):
        layer.train=False


with torch.no_grad():
    dev_loss=F.cross_entropy(model(Xdev),Ydev)
    print(f"dev loss: {dev_loss.item():.4f}")


"""
  ┌───────────────────────┬────────────────────────────────────┬──────────────┐                            
  │          层            |                操作                 │   输出形状    │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ 输入                   │ —                                  │ (32, 8)      │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ Embedding(27,10)      │ 每个索引查表成 10 维向量                │ (32, 8, 10)  │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ FlattenConsecutive(2) │ 把相邻 2 个时间步拼到通道维              │ (32, 4, 20)  │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ Linear(20,68)         │ 20→68                              │ (32, 4, 68)  │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ BatchNorm1d(68)       │ 归一化（按 (0,1) 维统计）              │ (32, 4, 68)  │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ Tanh                  │ 逐元素                              │ (32, 4, 68)  │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ FlattenConsecutive(2) │ 相邻 2 步拼接                        │ (32, 2, 136) │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ Linear(136,68)        │ 136→68                             │ (32, 2, 68)  │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ BatchNorm1d + Tanh    │ —                                  │ (32, 2, 68)  │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ FlattenConsecutive(2) │ 相邻 2 步拼接，T 变 1 → squeeze(1)    │ (32, 136)    │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ Linear(136,68)        │ 136→68                             │ (32, 68)     │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ BatchNorm1d + Tanh    │ 2D 输入，按 0 维统计                  │ (32, 68)     │
  ├───────────────────────┼────────────────────────────────────┼──────────────┤
  │ Linear(68,27)         │ 输出 logits                         │ (32, 27)     │
  └───────────────────────┴────────────────────────────────────┴──────────────┘
"""



























































