import os
import numpy as np


HERE=os.path.dirname(os.path.abspath(__file__))
HERE=os.path.join(HERE,"MNIST数据集")
TRAIN_CSV=os.path.join(HERE,"mnist_train.csv")
TEST_CSV=os.path.join(HERE,"mnist_test.csv")
CACHE_NPZ=os.path.join(HERE,"mnist_cache.npz")


def load_csv(path):
    data=np.loadtxt(path,delimiter=",",skiprows=1,dtype=np.int64)
    X=data[:,1:].astype(np.float32)
    y=data[:,0].astype(np.int64)
    return X,y


def one_hot(y,C=10):
    Y=np.zeros((len(y),C),dtype=np.float32)
    Y[np.arange(len(y)),y]=1.0
    return Y



def load_mnist(normalize=True,use_cache=True):
    if use_cache and os.path.exists(CACHE_NPZ):
        d=np.load(CACHE_NPZ)
        X_train,Y_train=d["X_train"],d["Y_train"]
        X_test,Y_test=d["X_test"],d["Y_test"]

        if not normalize:
            X_train=(X_train*255.0).astype(np.float32)
            X_test=(X_test*255.0).astype(np.float32)

        return X_train,Y_train,X_test,Y_test

    X_train,y_train=load_csv(TRAIN_CSV)
    X_test,y_test=load_csv(TEST_CSV)

    X_train_n=X_train/255.0
    X_test_n=X_test/255.0
    Y_train,Y_test=one_hot(y_train),one_hot(y_test)

    if use_cache:
        np.savez_compressed(
            CACHE_NPZ,
            X_train=X_train_n,
            Y_train=Y_train,
            X_test=X_test_n,
            Y_test=Y_test
        )

    if normalize:
        X_train=X_train_n
        X_test=X_test_n

    return X_train,Y_train,X_test,Y_test


if __name__=="__main__":
    load_mnist()











