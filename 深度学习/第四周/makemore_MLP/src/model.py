from __future__ import  annotations
import torch
from torch import Tensor

from vocab import vocab_size

class MakemoreMLP:
    def __init__(
            self,
            block_size:int=3,
            embed_dim:int=2,
            hidden_size:int=100,
            vocab_size:int=vocab_size,
            seed:int=2147483647,
    )->None:
        g=torch.Generator().manual_seed(seed)
        self.block_size=block_size
        self.embed_dim=embed_dim
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size

        # 这里先用最朴素的 randn，Day 4 (EP4) 会改成 kaiming
        self.C=torch.randn((vocab_size,embed_dim),generator=g,requires_grad=True)
        self.W1=torch.randn((block_size*embed_dim,hidden_size),generator=g,requires_grad=True)
        self.b1=torch.randn(hidden_size,generator=g,requires_grad=True)
        self.W2=torch.randn((hidden_size,vocab_size),generator=g,requires_grad=True)
        self.b2=torch.randn(vocab_size,generator=g,requires_grad=True)


    def parameters(self) ->list[Tensor]:
        return [self.C,self.W1,self.b1,self.W2,self.b2]

    def num_params(self)->int:
        return sum(p.numel() for p in self.parameters())

    def forward(self,X:Tensor)->Tensor:
        emb=self.C[X]
        flat=emb.view(emb.shape[0],-1)
        hidden=torch.tanh(flat@self.W1+self.b1)
        logits=hidden@self.W2+self.b2
        return logits


def _self_test()->None:
    N,B,E,H,V=32,3,2,100,27
    model=MakemoreMLP(block_size=B,embed_dim=E,hidden_size=H,vocab_size=V)

    X_fake=torch.randint(0,V,(N,B))
    logits=model.forward(X_fake)

    assert logits.shape == (N, V), f"got {tuple(logits.shape)}"
    assert logits.dtype == torch.float32
    print(f"forward OK; params = {model.num_params():,}")


if __name__=="__main__":
    _self_test()




















