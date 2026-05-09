import numpy as np

from nn import initialize_parameters_xavier,backward,forward,compute_loss



def numerical_gradient(params,X,Y,param_name,eps=1e-5,n_samples=20,l2=0.0):
    rng=np.random.default_rng(42)
    W=params[param_name]
    idxs=rng.choice(W.size,size=min(n_samples,W.size),replace=False)

    num_grad=np.zeros(len(idxs))

    for i,idx in enumerate(idxs):
        coords=np.unravel_index(idx,W.shape)

        orig=W[coords]
        W[coords]=orig+eps
        A2_p,_=forward(X,params)
        loss_p=compute_loss(A2_p,Y,params,l2)

        W[coords]=orig-eps
        A2_m,_=forward(X,params)
        loss_m=compute_loss(A2_m,Y,params,l2)

        W[coords]=orig

        num_grad[i]=(loss_p-loss_m)/(2*eps)

    return idxs,num_grad


def check(param_name,l2=0.0):
    rng=np.random.default_rng(42)
    X=rng.standard_normal((8,784))
    y_idx=rng.integers(0,10,8)
    Y=np.eye(10)[y_idx,:]

    params=initialize_parameters_xavier([784,32,10])
    _,cache=forward(X,params)
    grade=backward(Y,cache,params,l2)

    idxs,num_g=numerical_gradient(params,X,Y,param_name, l2=l2)
    ang_g=grade[param_name].ravel()[idxs]

    rel_err=np.abs(num_g-ang_g)/np.maximum(np.maximum(np.abs(num_g),np.abs(ang_g)),1e-12)

    ok = "OK" if rel_err.max() < 1e-5 else "FAIL"

    print(f"[{param_name}] max rel_err = {rel_err.max():.3e}, "
          f"mean = {rel_err.mean():.3e}  {ok}")

    return rel_err.max()



if __name__=="__main__":
    print("== without L2 ==")
    for name in ["W1","b1","W2","b2"]:
        check(name,l2=0.0)

    print("\n== with L2 = 1e-3 ==")
    for name in ["W1", "b1", "W2", "b2"]:
        check(name, l2=1e-3)















