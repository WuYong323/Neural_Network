from __future__ import annotations

import torch
from torch import Tensor
from torch._higher_order_ops import invoke_leaf_function


class BatchNorm1d:
    def __init__(self,dim:int,momentum:float=0.1,esp:float=1e-5):
        self.gamma=torch.ones(dim,requires_grad=True)
        self.beta=torch.zeros(dim,requires_grad=True)

        self.running_mean=torch.zeros(dim)
        self.running_var=torch.ones(dim)

        self.momentum=momentum
        self.esp=esp
        self.training=True


    def parameters(self):
        return [self.gamma,self.beta]


    def __call__(self,x:Tensor)->Tensor:
        if self.training:
            mean=x.mean(0,keepdim=True)
            var=x.var(0,keepdim=True,unbiased=False)

            with torch.no_grad():
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean.squeeze(0)
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*var.squeeze(0)

        else:
            mean=self.running_mean.unsqueeze(0)
            var=self.running_var.unsqueeze(0)

        x_hat=(x-mean)/torch.sqrt(var+self.esp)
        return self.gamma*x_hat+self.beta


    def eval(self):
        self.training=False
        return self


    def train(self):
        self.training=True
        return self



def _selfcheck()->None:
    torch.manual_seed(42)
    bn=BatchNorm1d(dim=4)

    x=torch.randn(32,4)*3.0+5.0
    y=bn(x)
    print(f"训练输入  mean={x.mean(0).tolist()}")
    print(f"训练输出  mean={y.mean(0).tolist()}  std={y.std(0).tolist()}")
    print(f"          (应该近似 mean≈0, std≈1，因为 γ=1, β=0)")

    for _ in range(200):
        x_batch=torch.randn(32,4)*3.0+5.0
        bn(x_batch)
    print(f"\n200 步后 running_mean = {bn.running_mean.tolist()}  (应≈ 5)")
    print(f"200 步后 running_var  = {bn.running_var.tolist()}  (应≈ 9)")

    bn.eval()
    x_single=torch.randn(1,4)*3.0+5.0
    y_single=bn(x_single)

    print(f"\neval 单样本输入 = {x_single.tolist()}")
    print(f"eval 单样本输出 = {y_single.tolist()}  (不应该是 inf/nan)")

    assert not torch.isnan(y_single).any(), "单样本推理出了 NaN！"
    assert not torch.isinf(y_single).any(), "单样本推理出了 Inf！"
    print("\n 自检全部通过")



if __name__=="__main__":
    _selfcheck()
























