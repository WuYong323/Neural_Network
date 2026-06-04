import torch

import random
import numpy as np
import matplotlib.pyplot as plt

from train_resnet import build_data
from model import build_cifar_resnet18


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = "resnet18_cifar.pth"
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def load_model():
    # 先按训练时的结构搭网络，再把训练好的参数灌进去
    model = build_cifar_resnet18(num_classes=10)
    state = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()                # 关掉 dropout / 固定 BN，推理必须做
    return model


@torch.no_grad()
def evaluate(model, test_loader):
    # 跑完整个测试集，算总体准确率
    total = correct = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def show_random_predictions(model, test_loader, n=3):
    # 随机抽 n 张测试集图片，跑一遍推理，把"真实标签 / 预测标签"画出来
    # 反归一化用的均值/标准差，要和 build_data 里保持一致
    mean = np.array((0.4914, 0.4822, 0.4465))
    std = np.array((0.2470, 0.2435, 0.2616))

    test_set = test_loader.dataset
    idxs = random.sample(range(len(test_set)), n)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
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


if __name__ == "__main__":
    _, test_loader = build_data()
    model = load_model()

    #acc = evaluate(model, test_loader)
    #print(f"test_acc = {acc:.2f}%")

    show_random_predictions(model, test_loader, n=3)
