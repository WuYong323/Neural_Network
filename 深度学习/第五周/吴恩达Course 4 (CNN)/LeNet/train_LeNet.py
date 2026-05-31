from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,transforms

from LeNet import LeNet5


def set_seed(seed:int=42)->None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def build_loaders(batch_size:int=128)->tuple[DataLoader,DataLoader]:
    tf=transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set=datasets.MNIST("./data",train=True,download=True,transform=tf)
    test_set=datasets.MNIST("./data",train=False,download=True,transform=tf)

    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)
    test_loader=DataLoader(test_set,batch_size=256,shuffle=False,num_workers=4)

    return train_loader,test_loader



@torch.no_grad()
def evaluate(model:nn.Module,loader:DataLoader,device:str)->float:
    model.eval()
    correct=total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        pred=model(x).argmax(dim=1)
        correct+=(pred==y).sum().item()
        total+=y.size(0)
    return correct/total



def main()->None:
    epochs=5
    batch_size=128
    lr=1e-3
    act="relu"
    seed=42

    set_seed(seed)
    device="cuda" if torch.cuda.is_available() else "cpu"

    train_loader,test_loader=build_loaders(batch_size)
    model=LeNet5(num_classes=10,act=act).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)

    t0=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        running=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)
            loss.backward()
            optimizer.step()
            running+=loss.item()
        acc=evaluate(model,test_loader,device)
        print(f"epoch {epoch} | train_loss {running / len(train_loader):.4f} "
              f"| test_acc {acc * 100:.2f}% | {time.perf_counter() - t0:.1f}s")

    final=evaluate(model,test_loader,device)
    print(f"\nFINAL test acc = {final * 100:.2f}%  "
          f"({'PASS' if final >= 0.985 else 'FAIL没到 98.5%'})")



if __name__=="__main__":
    main()

























