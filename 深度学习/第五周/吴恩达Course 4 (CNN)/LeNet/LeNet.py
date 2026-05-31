from __future__ import annotations

import torch
from torch import nn,Tensor



class LeNet5(nn.Module):
    def __init__(self,num_classes:int=10,act:str="tanh"):
        super().__init__()
        Act={"tanh":nn.Tanh,"relu":nn.ReLU}[act]

        self.features=nn.Sequential(
            nn.Conv2d(1,6,5),
            Act(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(6,16,5),
            Act(),
            nn.AvgPool2d(2,2)
        )

        self.classifier=nn.Sequential(
            nn.Linear(16*5*5,120),
            Act(),
            nn.Linear(120,84),
            Act(),
            nn.Linear(84,num_classes)
        )

    def forward(self,x:Tensor)->Tensor:
        x=self.features(x)
        x=x.view(x.size(0),-1)
        return self.classifier(x)


def _self_test()->None:
    model=LeNet5(num_classes=10,act="tanh")
    x=torch.randn(8,1,32,32)
    y=model(x)
    assert y.shape == (8, 10), f"got {tuple(y.shape)}"
    n_params=sum(p.numel() for p in model.parameters())
    print(f"forward OK; logits shape = {tuple(y.shape)}; params = {n_params:,}")



if __name__=="__main__":
    _self_test()






































