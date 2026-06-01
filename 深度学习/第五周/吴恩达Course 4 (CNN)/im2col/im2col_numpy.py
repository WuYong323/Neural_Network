import numpy as np

import torch
import torch.nn.functional as F



def get_im2col_indices(C_in,H_out,W_out,kh,kw,stride):
    # k: 每个展开元素来自哪个输入通道，形状 (C_in*kh*kw, 1)
    k=np.repeat(np.arange(C_in),kh*kw).reshape(-1,1)

    # i: 每个展开元素在原图中的行坐标
    i0=np.repeat(np.arange(kh),kw)
    i0=np.tile(i0,C_in)
    i1=stride*np.repeat(np.arange(H_out),W_out)
    i=i0.reshape(-1,1)+i1.reshape(1,-1)

    # j: 每个展开元素在原图中的列坐标（同理）
    j0=np.tile(np.arange(kw),kh*C_in)
    j1=stride*np.tile(np.arange(W_out),H_out)
    j=j0.reshape(-1,1)+j1.reshape(1,-1)
    return k,i,j



def im2col(X,kh,kw,stride=1,pad=0):
    """X: (N, C_in, H, W) → cols: (C_in*kh*kw, H_out*W_out*N)"""
    N,C_in,H,W=X.shape
    H_out=(H-kh+2*pad)//stride+1
    W_out=(W-kw+2*pad)//stride+1
    Xp=np.pad(X,((0,0),(0,0),(pad,pad),(pad,pad)))

    k,i,j=get_im2col_indices(C_in,H_out,W_out,kh,kw,stride)
    cols=Xp[:,k,i,j]
    cols=cols.transpose(1,2,0).reshape(C_in * kh * kw, -1)
    return cols,H_out,W_out



def conv2d_im2col(X,W,b=None,stride=1,pad=1):
    N=X.shape[0]
    C_out,C_in,kh,kw=W.shape
    cols,H_out,W_out=im2col(X,kh,kw,stride,pad)

    W_row=W.reshape(C_out,-1)
    out=W_row@cols
    if b is not None:
        out+=b.reshape(-1,1)

    out=out.reshape(C_out,H_out,W_out,N).transpose(3,0,1,2)
    return out



def main()->None:
    np.random.seed(0)
    X=np.random.randn(2,3,8,8).astype(np.float64)
    W = np.random.randn(4, 3, 3, 3).astype(np.float64)
    b = np.random.randn(4).astype(np.float64)

    out_mine = conv2d_im2col(X, W, b, stride=1, pad=1)
    out_torch = F.conv2d(torch.tensor(X), torch.tensor(W),torch.tensor(b), stride=1, padding=1).numpy()

    err = np.abs(out_mine - out_torch).max()
    print(f"shape: {out_mine.shape}, max abs error: {err:.2e}")



if __name__=="__main__":
    main()










































