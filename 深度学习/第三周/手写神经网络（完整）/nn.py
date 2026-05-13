import random
from engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self,n_in,nonlin=True):
        self.w=[Value(random.uniform(-1,1))for _ in range(n_in)]
        self.b=Value(0.0)
        self.nonlin=nonlin

    def __call__(self, x):
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w+[self.b]

    def __repr__(self):
        tag='tanh' if self.nonlin else 'linear'
        return f"{tag}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self,n_in,n_out,nonlin=True):
        self.neurons=[Neuron(n_in,nonlin=nonlin) for _ in range(n_out)]

    def __call__(self,x):
        outs=[n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer[{','.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self,n_in,n_outs):
        sizes=[n_in]+n_outs
        self.layers=[Layer(sizes[i],sizes[i+1],nonlin=(i!=len(n_outs)-1 )) for i in range(len(n_outs))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{','.join(str(l) for l in self.layers)}]"



if __name__=="__main__":
    random.seed(0)

    model=MLP(2,[4,4,1])
    print(model)
    print(f"参数总数{len(model.parameters())}")

    x=[Value(1.0),Value(-2.0)]
    y=model(x)
    print(f"y={y}")
    








