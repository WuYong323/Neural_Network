import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from model import build_cifar_resnet18



torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BATCH=128
EPOCHS=15
LR=1e-3

def build_data():
    # ---- 数据增强 + 归一化（CIFAR-10 的标准均值/标准差，业界通用常数）----
    # RandomCrop + RandomHorizontalFlip 是 CIFAR 的"黄金组合"，没有它精度会掉 3~5 个点
    CIFAR_MEAN=(0.4914,0.4822,0.4465)       #三个标准通道，所以有三个数字
    CIFAR_STD=(0.2470,0.2435,0.2616)


    train_tf=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),       #后续 Normalize 需要输入是 Tensor，且数值在 [0, 1]，所以 ToTensor 必须在 Normalize 之前
        transforms.Normalize(CIFAR_MEAN,CIFAR_STD)
    ])


    test_tf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN,CIFAR_STD)
    ])

    train_set=datasets.CIFAR10("./data",train=True,download=True,transform=train_tf)
    test_set=datasets.CIFAR10("./data",train=False,download=True,transform=test_tf)

    train_loader=DataLoader(
        train_set,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader=DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader,test_loader



def evaluate():
    model.eval()
    total=correct=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            pred=model(x).argmax(1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)
    return 100.0*correct/total



def train():
    for epoch in range(EPOCHS):
        model.train()
        for x,y,in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss=criterion(model(x),y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        acc=evaluate()
        print(f"epoch {epoch + 1:2d}/{EPOCHS}  test_acc={acc:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), "resnet18_cifar.pth")



CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def show_random_predictions(n=3):
    # 随机抽 n 张测试集图片，跑一遍推理，把"真实标签 / 预测标签"画出来
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    # 反归一化用的均值/标准差，要和 build_data 里保持一致
    mean = np.array((0.4914, 0.4822, 0.4465))
    std = np.array((0.2470, 0.2435, 0.2616))

    test_set = test_loader.dataset
    idxs = random.sample(range(len(test_set)), n)

    model.eval()
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    with torch.no_grad():
        for ax, idx in zip(axes, idxs):
            x, y = test_set[idx]                       # x: 已归一化的 Tensor (3,32,32)
            logit = model(x.unsqueeze(0).to(DEVICE))   # 加 batch 维度送进模型
            pred = logit.argmax(1).item()

            # 把归一化的 Tensor 还原成可显示的 [0,1] 图片
            img = x.cpu().numpy().transpose(1, 2, 0) * std + mean
            img = img.clip(0, 1)

            ax.imshow(img)
            ax.axis("off")
            ok = (pred == y)
            ax.set_title(f"true: {CIFAR_CLASSES[y]}\npred: {CIFAR_CLASSES[pred]}",
                         color="green" if ok else "red", fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    train_loader,test_loader=build_data()

    model = build_cifar_resnet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    train()
    show_random_predictions(3)









































