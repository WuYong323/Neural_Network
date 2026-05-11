import numpy as np
import time,os,csv

from nn import initialize_parameters_xavier,forward,backward,update_params,compute_loss
from data_loader import load_mnist
from config import config



def evaluate(params,X,Y,batch=1024):
    total_loss,total_correct,n=0.0,0,0

    for i in range(0,len(X),batch):
        xb,yb=X[i:i+batch],Y[i:i+batch]
        A2,_=forward(xb,params)
        total_loss+=compute_loss(A2,yb,params)*len(xb)
        total_correct+=(np.argmax(A2,1)==np.argmax(yb,1)).sum()
        n+=len(xb)

    return total_loss/n,total_correct/n



def train():
    X_train,Y_train,X_test,Y_test=load_mnist()
    params=initialize_parameters_xavier(config["layer_dims"],seed=config["seed"])

    os.makedirs("logs",exist_ok=True)
    os.makedirs("checkpoints",exist_ok=True)
    log_path=f"logs/{config["tag"]}.csv"
    with open(log_path,'w',newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch","train_loss","train_acc",
             "test_loss","test_acc","lr","times"]
        )

    rng=np.random.default_rng(config["seed"])
    N=len(X_train)
    best_test_acc,lr=0.0,config["lr"]
    t0=time.time()

    for epoch in range(1,config["epochs"]+1):
        idx=rng.permutation(N)
        Xs,Ys=X_train[idx],Y_train[idx]

        for i in range(0,N,config["batch"]):
            xb,yb=Xs[i:i+config["batch"]],Ys[i:i+config["batch"]]
            _,cache=forward(xb,params)
            grads=backward(yb,cache,params,config["l2"])
            update_params(params,grads,lr)

        train_loss,train_acc=evaluate(params,X_train,Y_train)
        test_loss,test_acc=evaluate(params,X_test,Y_test)

        if config["lr_decay"]>0 and epoch%config["lr_decay"]==0:
            lr*=0.5

        dt=time.time()-t0

        print(f"[{epoch:02d}/{config["epochs"]}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_acc={test_acc:.4f} lr={lr:.4f} time={dt:.1f}s")

        with open(log_path,'a',newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch,train_loss,train_acc,
                 test_loss,test_acc,lr,dt]
            )

        if test_acc>best_test_acc:
            best_test_acc=test_acc
            np.savez(f"checkpoints/final_best.npz",
                     W1=params["W1"],b1=params["b1"],
                     W2=params["W2"],b2=params["b2"],
                     epoch=epoch,test_acc=test_acc
                     )

    print(f"best test_acc = {best_test_acc:.4f}")
    return best_test_acc


if __name__=="__main__":
    train()







