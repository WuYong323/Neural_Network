import numpy as np



def zero_pad(X,pad):
    X_pad=np.pad(
        X,
        pad_width=((0,0),(pad,pad),(pad,pad),(0,0)),
        mode="constant",
        constant_values=0
    )
    return X_pad



def conv_single_step(a_slice_prev,W,b):
    s=np.multiply(a_slice_prev,W)
    Z=float(np.sum(s))
    Z=Z+b
    return Z



def conv_forward(A_prev,W,b,hparameters):
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (f,f,n_C_prev,n_C)=W.shape
    stride=hparameters["stride"]
    pad=hparameters["pad"]

    n_H=int((n_H_prev-f+2*pad)/stride+1)
    n_W=int((n_W_prev-f+2*pad)/stride+1)

    Z=np.zeros((m,n_H,n_W,n_C))

    A_prev_pad=zero_pad(A_prev,pad)

    for i in range(m):
        a_prev_pad=A_prev_pad[i]
        for h in range(n_H):
            vert_start=h*stride
            vert_end=vert_start+f
            for w in range(n_W):
                horiz_start=w*stride
                horiz_end=horiz_start+f
                for c in range(n_C):
                    a_slice_prev=a_prev_pad[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ]
                    Z[i,h,w,c]=conv_single_step(
                        a_slice_prev,W[:,:,:,c],b[:,:,:,c]
                    )
    cache=(A_prev,W,b,hparameters)
    return Z,cache



def pool_forward(A_prev,hparameters,mode="max"):
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    f=hparameters["f"]
    stride=hparameters["stride"]

    n_H=int((n_H_prev-f)/stride+1)
    n_W=int((n_W_prev-f)/stride+1)
    n_C=n_C_prev

    A=np.zeros((m,n_H,n_W,n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vs,ve=h*stride,h*stride+f
                    hs,he=w*stride,w*stride+f
                    a_slice=A_prev[i,vs:ve,hs:he,c]

                    if mode=="max":
                        A[i,h,w,c]=np.max(a_slice)
                    elif mode=="average":
                        A[i,h,w,c]=np.mean(a_slice)
    cache=(A_prev,hparameters)
    return A,cache



def main():
    np.random.seed(1)
    X=np.random.randn(4,3,3,2)
    X_pad=zero_pad(X,2)
    print(X.shape, '->', X_pad.shape)
    print(X)
    print(X_pad)



if __name__=="__main__":
    main()



























