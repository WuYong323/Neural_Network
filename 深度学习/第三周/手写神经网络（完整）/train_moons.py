import matplotlib.pyplot as plt
import matplotlib
import random
import time
import os
import numpy as np
from sklearn.datasets import make_moons

from engine import Value
from nn import MLP

SEED=1337
EPOCH=100
lr=1.0
random.seed(SEED)
np.random.seed(SEED)


X,y=make_moons(n_samples=100,noise=0.1,random_state=SEED)
y=y*2-1

model=MLP(2,[16,16,1])
print(model)
print(f"模型参数为: {len(model.parameters())}")


def loss_fn(model,X,y,alpha=1e-4):
    inputs=[[Value(xi) for xi in rows] for rows in X]
    scores=[model(x) for x in inputs]

    data_losses=[(1+-float(yi)*score).relu() for score,yi in zip(scores,y)]
    data_loss=sum(data_losses)*(1.0/len(data_losses))

    reg_loss=alpha*sum((p*p for p in model.parameters()),Value(0.0))
    total_loss=data_loss+reg_loss

    accuracy=sum((s.data>0)==(yi>0) for s,yi in zip(scores,y))/len(y)
    return total_loss,accuracy



t0=time.time()

for epoch in range(EPOCH):
    loss,acc=loss_fn(model,X,y)

    model.zero_grad()

    loss.backward()

    current_lr=lr-0.9*epoch/EPOCH
    for p in model.parameters():
        p.data-=lr*p.grad

    if epoch % 10 == 0 or epoch == EPOCH - 1:
        print(f"step {epoch:3d}  loss {loss.data:.4f}  acc {acc*100:.1f}%")


t1=time.time()-t0

print(f"训练完成，用时：{t1:.2f}s")


loss,acc=loss_fn(model,X,y,)


os.makedirs('logs',exist_ok=True)

h=0.05
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1

xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

Xmesh=np.c_[xx.ravel(),yy.ravel()]

inputs=[[Value(xi) for xi in rows] for rows in Xmesh]
scores=[model(x).data for x in inputs]
Z=(np.array(scores)>0).reshape(xx.shape)

plt.figure(figsize=(8,8))
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral,edgecolors='k')
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)

plt.title(f"Decision boundary  |  loss={loss.data:.3f}  acc={acc*100:.1f}%")

plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())

plt.tight_layout()
plt.savefig('logs/decision_boundary.png', dpi=120)



























