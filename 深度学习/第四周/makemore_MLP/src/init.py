from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def init_last_layer(W2:Tensor,b2:Tensor,scale:float=0.01)->None:
    with torch.no_grad():
        W2.mul_(scale)
        b2.zero_()



def expected_initial_loss(num_classes:int)->float:
    return math.log(num_classes)



def _selfcheck()->None:
    torch.manual_seed(42)
    N,V,H=32,27,200
    hidden=torch.randn(N,H)
    W2=torch.randn(H,V)
    b2=torch.randn(V)
    Y=torch.randint(0,V,(N,))

    logits_bad=hidden@W2+b2
    loss_bad=F.cross_entropy(logits_bad,Y).item()

    init_last_layer(W2,b2,scale=0.01)
    logits_good=hidden@W2+b2
    loss_good=F.cross_entropy(logits_good,Y).item()

    print(f"修复前初始 loss = {loss_bad:.3f}   ← 远高于 {expected_initial_loss(V):.3f}")
    print(f"修复后初始 loss = {loss_good:.3f}   ← 接近 log(27) = {expected_initial_loss(V):.3f}")



def kaiming_init_(W:Tensor,fan_in:int,gain:float=math.sqrt(2))->None:
    std=gain/math.sqrt(fan_in)
    with torch.no_grad():
        W.normal_(mean=0,std=std)



def init_Makemore_mlp(model,gain_hidden:float=5/3,scale_last:float=0.01)->None:
    B,E=model.block_size,model.C.shape[1]
    H=model.W1.shape[1]
    fan_in_W1=E*B
    kaiming_init_(model.W1,fan_in_W1,gain=gain_hidden)
    with torch.no_grad():
        model.b1.zero_()
    init_last_layer(model.W2,model.b2,scale=scale_last)






if __name__=="__main__":
    _selfcheck()





































