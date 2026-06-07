import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F



class Head(nn.Module):
    def __init__(self,n_embd:int,head_size:int,block_size:int,dropout:float=0.0):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)

        """
        将这个下三角矩阵注册为模块的一个缓冲区（buffer），命名为 "tril"。
        缓冲区 不是可训练参数（不会在 model.parameters() 中，也不会被优化器更新），但它会：
            随模型一起保存到 state_dict 中，加载时自动恢复；
            当调用 model.to(device) 时，会和参数一起自动移动到对应设备（CPU/GPU）；
            在计算图中不会被追踪梯度，节省显存。
        """
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)
        self.head_size=head_size


    def forward(self,x:Tensor)->Tensor:
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)

        # ① 匹配度表 + ② 缩放（1/√d，防 softmax 饱和)
        wei=q@k.transpose(-2,-1)*self.head_size**-0.5
        # ③ causal mask：上三角（未来）填 -inf。只取 [:T,:T] 以兼容比 block_size 短的序列
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        # ④ softmax 沿最后一维：把每一行变成"加起来=1 的注意力比例"
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        # ⑤ 加权汇总 V
        out=wei@v
        return out


if __name__=="__main__":
    B,T,C,head_size,block_size=2,8,32,16,8
    x=torch.randn(B,T,C)
    head=Head(n_embd=C,head_size=head_size,block_size=block_size)
    out=head(x)
    print("输入  x  形状:", tuple(x.shape))  # (2, 8, 32)
    print("输出 out 形状:", tuple(out.shape))  # (2, 8, 16) —— 每个token聚合后的新表示
    assert out.shape == (B, T, head_size)

    q,k=head.query(x),head.key(x)
    wei=q@k.transpose(-2,-1)*head_size**-0.5
    wei=wei.masked_fill(head.tril[:T,:T]==0,float("-inf"))
    wei=F.softmax(wei,dim=-1)
    print("\n第 0 句话的注意力权重矩阵（应是下三角，上三角=0）:")
    print(wei[0].round(decimals=2))
    print("每行之和（应全为 1）:", wei[0].sum(dim=-1).round(decimals=3).tolist())

    x2=x.clone()
    x2[:,1:,:]=torch.randn(B,T-1,C)
    out2=head(x2)
    diff=(out[:,0,:]-out2[:,0,:]).abs().max().item()
    print(f"\n改掉未来 token 后，第0个token输出的最大变化: {diff:.2e}  (应≈0，证明 causal mask 生效)")
    assert diff < 1e-6, "causal mask 失效！第0个token偷看了未来"
    print("causal mask 验证通过：信息只从过去流向未来")











































