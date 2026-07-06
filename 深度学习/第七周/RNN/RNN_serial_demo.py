import torch

torch.manual_seed(0)
T,d_in,d_h=5,4,8

#一个朴素的RNN cell：ht=tan(xt@wxh+ht-1@whh +b)
wxh=torch.randn(d_in,d_h)*0.1
whh=torch.randn(d_h,d_h)*0.1
b=torch.zeros(d_h)

def rnn_step(x_t,h_prev):
    return torch.tanh(x_t@wxh+h_prev@whh+b)

xs=torch.randn(T,d_in)
h=torch.zeros(d_h)

hs=[]
for t in range(T):
    h=rnn_step(xs[t],h)
    hs.append(h)
hs=torch.stack(hs)    #必须创建一个新维度，把所有张量“码整齐”放在这个新维度上。

print("每步 hidden state 形状:", hs.shape)   # 预期输出: torch.Size([5, 8])
print("注意: 算 hs[3] 之前,hs[0..2] 必须已经算完——这是物理上的强制顺序")








































