import math

class Value:
    def __init__(self,data,_children=(),op='',label=''):
        self.data=data
        self._prev=set(_children)
        self.op=op
        self.label=label
        self.grad=0.0
        self._backward=lambda :None

    def __repr__(self):
        if self.label:
            return f"Value(label='{self.label}',data={self.data})"
        return f"Value(data={self.data})"

    def __add__(self, other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out

    def __mul__(self, other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out

    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self,),'tanh')
        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        return out

    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),'exp')
        def _backward():
            self.grad+=math.exp(x)*out.grad
        out._backward=_backward
        return out

    def __pow__(self, power):
        assert isinstance(power,(int,float)) ,"only supporting int/float for now"
        x=self.data
        out=Value(x**pow,(self,power),f'*{power}')
        def _backward():
            self.grad+=out.grad*power*x**(power-1)
        out._backward=_backward
        return out

    def __truediv__(self, other):
        return self*other**-1

    def __radd__(self, other):
        return self+other

    def __rmul__(self, other):
        return self*other

    def __neg__(self):
        return self*(-1)

    def __sub__(self, other):
        return self+(-other)

    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad=1.0
        for node in reversed(topo):
            node._backward()




