import numpy as np
from nn import forward
from data_loader import load_mnist

def load_checkpoint(path):
    d=np.load(path)
    return {"W1":d["W1"],"b1":d["b1"],"W2":d["W2"],"b2":d["b2"]}


if __name__=="__main__":
    path="checkpoint/final_best.npz"
    params=load_checkpoint(path)
    _,_,X_test,Y_test=load_mnist()
    A2,_=forward(X_test,params)
    acc=(np.argmax(A2,axis=1)==np.argmax(Y_test,axis=1)).mean()
    print(f"{path}  test_acc = {acc:.4f}")