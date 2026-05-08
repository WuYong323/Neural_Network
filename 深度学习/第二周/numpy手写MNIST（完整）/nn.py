import numpy as np


def initialize_parameters_xavier(layer_dims,seed=42):
    np.random.randint(seed)

    parameters={}
    L=len(layer_dims)

    for l in range(1,L):
        n_in=layer_dims[l-1]
        n_out=layer_dims[l]

        std=np.sqrt(2/(n_in+n_out))

        parameters[f"W{l}"]=np.random.randn(n_in,n_out)*std
        parameters[f"b{l}"]=np.zeros(n_out)

    return parameters


def relu(Z):
    return np.maximum(0,Z)


def softmax(Z):
    Z_shift=Z-np.max(Z,axis=1,keepdims=True)
    exp_Z=np.exp(Z_shift)
    return exp_Z/np.sum(exp_Z,axis=1,keepdims=True)


def cross_entropy(A2,Y,eps=1e-12):
    N=Y.shape[0]
    return -np.sum(Y*np.log(A2+eps))/N


def forward(X,parameters):
    cache={"A0":X}
    A=X
    L=len(parameters)//2

    for l in range(1,L):
        W=parameters[f"W{l}"]
        b=parameters[f"b{l}"]
        Z=A@W+b
        A=relu(Z)
        cache[f"A{l}"]=A
        cache[f"Z{l}"]=Z

    W=parameters[f"W{L}"]
    b=parameters[f"b{L}"]
    Z=A@W+b
    A_final=softmax(Z)
    cache[f"A{L}"]=A_final
    cache[f"Z{L}"]=Z

    return A_final,cache


def computer_loss():
    pass


def backward(Y,cache,params,l2=0.0):
    X,Z1,A1,A2=cache["A0"],cache["Z1"],cache["A1"],cache["A2"]
    W2=params["W2"]
    N=Y.shape[0]

    dZ2=(A2-Y)/N
    dW2=A1.T@dZ2
    db2=np.sum(dZ2,axis=0)
    dA1=dZ2@W2.T
    dZ1=dA1*(Z1>0)
    dW1=X.T@dZ1
    db1=np.sum(dZ1,axis=0)

    if l2>0:
        dW1+=l2*params["W1"]
        dW2+=l2*params["W2"]

    return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}


def update_params(params,grads,lr):
    for k in params:
        params[k]-=lr*grads[k]

    return params



def shape_check(params,X,Y):
    A2,cache=forward(X,params)

    assert A2.shape == Y.shape, f"A2 {A2.shape} vs Y {Y.shape}"
    grads=backward(Y,cache,params)

    for k in params:
        assert params[k].shape == grads[k].shape, \
            f"{k} shape mismatch: {params[k].shape} vs {grads[k].shape}"
    print("[shape_check] OK")


if __name__ == "__main__":
    np.random.seed(42)
    N,n_features=64,784
    X=np.random.randn(N,n_features)
    y_indices=np.random.randint(0,10,N)
    Y=np.eye(10)[y_indices,:]

    params=initialize_parameters_xavier([784,128,10])

    A_final,cache=forward(X,params)

    loss=cross_entropy(A_final,Y)
    print(f"A_final shape: {A_final.shape}")
    print(f"列和（应该全是 1）: {A_final.sum(axis=1)[:5]}")
    print(f"全部非负? {(A_final >= 0).all()}")
    print(f"初始 loss: {loss:.4f}")
    print(f"理论值 log(10): {np.log(10):.4f}")
    print(f"cache keys: {list(cache.keys())}")












