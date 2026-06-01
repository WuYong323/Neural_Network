import numpy as np


def conv2d_maive(X,W,stride=1,pad=0):
    N,C_in,H,wd=X.shape
    C_out,_,kh,kw=W.shape
    Xp=np.pad(X,((0,0),(0,0),(pad,pad),(pad,pad)))
    H_out=(H+2*pad-kh)//stride+1
    W_out=(W+2*pad-kw)//stride+1
    out=np.zeros((N,C_out,H_out,W_out))

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h0,w0=stride*i,stride*j
                    region=Xp[n,:,h0:h0+kh,w0:w0+kw]
                    out[n,co,i,j]=np.sum(region*W[co])
    return out

































