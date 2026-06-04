import torch
import torch.nn as nn


torch.manual_seed(0)

DEPTH=50
WIDTH=64
USE_RELU=True



class PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(
            [nn.Linear(WIDTH,WIDTH) for _ in range(DEPTH)]
        )
        self.act=nn.ReLU() if USE_RELU else nn.Tanh()

    def forward(self,x):
        for layer in self.layers:
            x=self.act(layer(x))
        return x



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(
            [nn.Linear(WIDTH,WIDTH) for _ in range(DEPTH)]
        )
        self.act=nn.ReLU() if USE_RELU else nn.Tanh()

    def forward(self,x):
        for layer in self.layers:
            x=self.act(layer(x))+x
        return x



def run(model:nn.Module,name):
    model.zero_grad()
    x=torch.randn(16,WIDTH)
    out=model(x)
    loss=out.pow(2).mean()
    loss.backward()

    grad_norms=[layer.weight.grad.norm().item() for layer in model.layers]

    print(f"\n===== {name} =====")
    print(f"第 1 层(最浅)梯度范数 : {grad_norms[0]:.3e}")
    print(f"第 25 层(中间)梯度范数: {grad_norms[24]:.3e}")
    print(f"第 50 层(最深)梯度范数: {grad_norms[-1]:.3e}")
    # 浅层/深层比值越小，说明梯度衰减越严重
    ratio = grad_norms[0] / (grad_norms[-1] + 1e-12)
    print(f"浅层/深层 比值        : {ratio:.3e}  （越接近1越健康）")



if __name__=="__main__":
    run(PlainNet(),"PlainNet  不加 skip")
    run(ResNet(), "ResNet  加 skip")









































