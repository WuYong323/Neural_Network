import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

import h5py

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path



class SignsCNN(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()
        self.conv1=nn.Conv2d(3,8,4,padding="same")
        self.bn1=nn.BatchNorm2d(8)
        self.pool1=nn.MaxPool2d(8,8)

        self.conv2=nn.Conv2d(8,16,2,padding="same")
        self.bn2=nn.BatchNorm2d(16)
        self.pool2=nn.MaxPool2d(4,4)

        self.fc=nn.Linear(16*2*2,num_classes)

    def forward(self,X):
        X=self.pool1(torch.relu(self.bn1(self.conv1(X))))
        X=self.pool2(torch.relu(self.bn2(self.conv2(X))))
        X=torch.flatten(X,start_dim=1)
        return self.fc(X)



def load_dataset():
    train_dataset=h5py.File("datasets/train_signs.h5","r")
    train_set_x_orig=np.array(train_dataset["train_set_x"][:])
    train_set_y_orig=np.array(train_dataset["train_set_y"][:])
    test_dataset=h5py.File("datasets/test_signs.h5","r")
    test_set_x_orig=np.array(test_dataset["test_set_x"][:])
    test_set_y_orig=np.array(test_dataset["test_set_y"][:])
    classes=np.array(test_dataset["list_classes"][:])
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes



def main():
    lr=1e-3
    batch_size=64

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes=load_dataset()

    X_train_t=torch.from_numpy(train_set_x_orig).permute(0,3,1,2).float()/255.0
    Y_train_t=torch.from_numpy(train_set_y_orig).long()

    X_test_t = torch.from_numpy(test_set_x_orig).permute(0, 3, 1, 2).float() / 255.0
    Y_test_t = torch.from_numpy(test_set_y_orig).long()

    train_ds=TensorDataset(X_train_t,Y_train_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)

    train_loader=DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # ← 多进程加载，CPU 准备数据时 GPU 不闲（对应 TF 的 prefetch）
        pin_memory=True,  # ← 锁页内存，CPU→GPU 拷贝快一档
        persistent_workers=True,  # ← worker 不在每个 epoch 重启，省启动开销
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # ← 多进程加载，CPU 准备数据时 GPU 不闲（对应 TF 的 prefetch）
        pin_memory=True,  # ← 锁页内存，CPU→GPU 拷贝快一档
        persistent_workers=True,  # ← worker 不在每个 epoch 重启，省启动开销
    )

    device=torch.device('cude' if torch.cuda.is_available() else 'cpu')
    model=SignsCNN(num_classes=6).to(device)

    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    criterion=nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,factor=0.5,patience=3
    )

    best_val_acc=0.0
    patience_counter=0
    EARLY_STOP_PATIENCE=5

    for epoch in range(50):
        model.train()
        train_loss,train_correct,train_total=0.0,0,0
        for x,y in train_loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # set_to_none=True 比默认 zero_ 略快，工业默认开

            logits=model(x)
            loss=criterion(logits,y)

            loss.backward()

            optimizer.step()

            train_loss+=loss.item()*x.size(0)
            train_correct+=(logits.argmax(1)==y).sum().item()
            train_total+=x.size(0)

        # ============ 验证阶段 ============
        model.eval()
        val_loss,val_correct,val_total=0.0,0,0

        with torch.no_grad():
            for x,y in test_loader:
                x,y=x.to(device),y.to(device)
                logits=model(x)
                loss=criterion(logits,y)
                val_loss+=loss.item()*x.size(0)
                val_correct+=(logits.argmax(1)==y).sum().item()
                val_total+=x.size(0)

        val_acc=val_correct/val_total
        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss / train_total:.4f} acc={train_correct / train_total:.4f} | "
              f"val_loss={val_loss / val_total:.4f} acc={val_acc:.4f}")

        scheduler.step(val_loss/val_total)

        # 手写 EarlyStopping + best checkpoint(工业级)
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            patience_counter=0
            torch.save(model.state_dict(),'best_signs.pt')
        else:
            patience_counter+=1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break



if __name__=="__main__":
    torch.random.manual_seed(42)

    model_path=Path("best_signs.pt")
    if not model_path.is_file(): main()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SignsCNN(num_classes=6)
    model.load_state_dict(torch.load("best_signs.pt",map_location='cpu'))
    model.eval()
    model.to(device)

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

    indexes=torch.randint(0,100,(3,))

    for index in indexes:
        single_img=test_set_x_orig[index]
        x = torch.from_numpy(single_img).permute(2, 0, 1).float()/255.0  # HWC -> CHW
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(-1).item()

        plt.imshow(single_img)
        plt.title(f"number={pred}")
        plt.show()























































