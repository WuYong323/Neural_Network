import torch
from value import Value

x1=torch.Tensor([2.0]).double();x1.requires_grad=True
x2=torch.Tensor([0.0]).double();x2.requires_grad=True
w1=torch.Tensor([-3.0]).double();w1.requires_grad=True
w2=torch.Tensor([1.0]).double();w2.requires_grad=True
b=torch.Tensor([6.88137358782]).double();b.requires_grad=True
n=x1*w1+x2*w2+b
o=torch.tanh(n)

print(o.data.item())
o.backward()
print("---------")
print("x2",x2.grad.item())
print("w2",w2.grad.item())
print("x1",x1.grad.item())
print("w1",w1.grad.item())


print("--------------------------")


x1=Value(2.0)
x2=Value(0.0)
w1=Value(-3.0)
w2=Value(1.0)
b=Value(6.88137358782)
n=x1*w1+x2*w2+b
o=n.tanh()

print(o.data)
o.backward()
print("---------")
print("x2",x2.grad)
print("w2",w2.grad)
print("x1",x1.grad)
print("w1",w1.grad)
