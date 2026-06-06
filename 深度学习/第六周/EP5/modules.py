import torch


class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight= torch.randn(fan_in,fan_out)/fan_in**0.5
        self.bias=torch.zeros(fan_out) if bias else None

    def __call__(self,x):
        self.out=self.weight@x
        if self.bias is not None:
            self.out=self.out+self.bias
        return self.out

    def parameters(self):
        return [self.weight]+([] if self.bias is None else [self.bias])


class Tanh:
    def __call__(self,x):
        self.out=torch.tanh(x)
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
        return [p for layer in self.layers for p in layer]


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








































