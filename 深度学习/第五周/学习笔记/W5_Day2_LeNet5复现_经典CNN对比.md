# Week 5 · Day 2（2026-05-26）：LeNet-5 复现 + LeNet vs 现代 CNN 对比

> **覆盖任务**（计划 line 1323-1344 / checklist line 1483-1485）：
> - [ ] DL：Course 4 Week 2 上半（LeNet-5、AlexNet 历史背景）
> - [ ] DL：PyTorch 复现 LeNet-5，MNIST **test acc ≥ 98.5%**（写到 `week5_cnn/src/lenet.py`）
> - [ ] DL：LeNet vs 现代 CNN 对比表（写到 `tech_notes/lenet_vs_modern.md`）
> - [ ] 完成标准：能讲清"sigmoid → ReLU 不只是精度问题，更是推理优化的前提"
>
> **阅读对象**：你自己——W1 已经写过一个现代 MNIST CNN（test acc > 99%），W5 Day1 刚啃完 Course 4 Week 1 + 从零搭卷积（呼应 [[AndrewNg_C4W1_CNN作业笔记]]）。今天的关键词不是"复现"，是"对比"：把 1998 年的 LeNet-5 和你自己写的现代版放一起，每一处差异都对应一个工业延伸。
>
> **本笔记的设计**：沿用 [[W4_Day2_makemore_MLP_前向训练]] 的三段式——每节先讲"原理 + 直觉"，再给可直接拷进 `src/` 的可运行代码，最后写工业锚点（这块知识在推理优化系统里对应什么真实问题）。读完应该能从 0 把今天三个任务全部跑通。

---

## 0. 学习目标（看完应能脱口而出）

1. LeNet-5 是谁在什么年代、为了解决什么问题做出来的？它的"5"指什么？
2. 为什么 AlexNet（2012）被当成深度学习的"引爆点"，而结构几乎一样的 LeNet（1998）没火？差的到底是什么？
3. LeNet-5 每一层的张量形状怎么变？为什么 MNIST 是 28×28，却要先 pad 到 32×32？
4. 为什么第二个卷积层后 flatten 出来正好是 400（= 16×5×5）？这个数算错是新手最常见的 bug。
5. 1998 年用 sigmoid/tanh + average pooling，2012 年后全换成 ReLU + max pooling——这两个替换分别解决了什么问题？
6. **为什么说 "sigmoid → ReLU" 不只是精度提升，而是推理优化（算子融合）的前提？**（今天的完成标准）
7. LeNet-5 没有 BatchNorm、没有 Dropout、用 SGD；你 W1 的 CNN 全有。这些"补丁"各自在补什么漏洞？

---

## 1. Course 4 Week 2 上半：经典网络的历史背景

### 1.1 先说人话：这节课到底在讲什么

Course 4 Week 2 上半的核心就一句话：**带你看三个里程碑网络（LeNet-5、AlexNet、VGG），理解"现代 CNN 的标准积木是怎么一代代攒出来的"**。今天只啃前半段——LeNet-5 和 AlexNet 的历史背景。VGG / Inception / ResNet 是明天（Day3）和后天（Day4）的事。

为什么要学"历史"？因为你 W1 直接上手写的那个 `Conv→BN→ReLU→MaxPool` 现代积木，**不是天上掉下来的**。每一个组件都是为了修上一代网络的某个具体毛病而加进来的。不看历史，你只会"调包"；看了历史，你能"诊断"——这正是 AI Infra 工程师和普通调参侠的分水岭。

### 1.2 LeNet-5（1998）：第一个能用的卷积网络

> **LeNet-5**（读作"le-net five"）：由 Yann LeCun（杨立昆，现 Meta 首席 AI 科学家、卷积网络之父）在 1998 年论文《Gradient-Based Learning Applied to Document Recognition》中提出的卷积神经网络。"5" 指它有 **5 个带可学习参数的层**（2 个卷积层 + 3 个全连接层；池化层没有参数所以不算）。

它是干嘛用的？**读支票上的手写数字和邮政编码**。当年美国的银行和邮局，真的把 LeNet 部署到生产线上，每天自动识别几百万张支票。所以它不是"玩具论文"，是 1998 年就跑在工业生产环境里的模型——这点很重要，后面对比表会反复呼应。

**一个生活类比**：LeNet-5 像 1998 年的诺基亚功能机——能打电话能发短信（识别数字管用），结构上已经具备了"按键、屏幕、电池"这些智能机的雏形，但受限于当时的"工艺"（算力、数据、激活函数），没法做成今天的 iPhone（ResNet/Transformer）。

### 1.3 AlexNet（2012）：同样的思路，为什么这次炸了

> **AlexNet**：2012 年 Alex Krizhevsky（在 Geoffrey Hinton 组）提出的 8 层卷积网络，在 ImageNet 图像分类竞赛上把 top-5 错误率从 26% 直接干到 16%，比第二名领先 10 个百分点。这一战通常被认为是"深度学习革命"的起点。

这里有个特别值得想的问题：**AlexNet 的结构和 LeNet 几乎是一个模子——都是"卷积+池化堆叠，最后接全连接"。思路没本质区别，为什么 LeNet 安静了 14 年，AlexNet 一炮而红？**

差的不是"想法"，是**三个外部条件 + 两个内部改进同时到位了**：

| 维度 | 1998 LeNet | 2012 AlexNet | 这件事的意义 |
|---|---|---|---|
| 数据 | 几万张手写数字 | ImageNet **120 万**张彩色图 | 深度网络是"数据饥饿"的，没数据再好的结构也过拟合 |
| 算力 | CPU | **2 张 GTX 580 GPU** | 卷积的本质是矩阵乘（明天 im2col 会讲），GPU 把它加速了上百倍 |
| 激活函数 | tanh / sigmoid | **ReLU** | 梯度不再被压扁，深层网络第一次能训得动（见 §4.1） |
| 正则 | 几乎没有 | **Dropout** | 大网络 + 大数据时压过拟合的关键 |
| 规模 | 6 万参数 | 6000 万参数 | 大了 1000 倍 |

**一句话总结这节课的历史观**：LeNet 证明了"卷积这条路能走通"，但它生在一个"没数据、没算力、激活函数还会把梯度掐死"的年代。AlexNet 没有发明新范式，它只是**等到三个外部条件成熟，再补上 ReLU + Dropout 两个内部改进，把同一条路重新跑了一遍**——结果引爆了整个领域。

> **AI Infra 视角的第一个锚点**：注意上表里"算力"那一行。AlexNet 之所以能训，靠的是作者手写 CUDA 卷积核、把模型拆到两张 GPU 上跑（这就是最早的"模型并行"雏形）。**你现在想做的推理优化 / AI Infra，本质上就是 AlexNet 那张表里"算力"这一格的现代延续**——从手写 CUDA kernel，到今天的 cuDNN、TensorRT、FlashAttention，干的都是同一件事：让硬件把网络跑得更快。这条线，是你这学期后半段（W8 推理优化）的主线伏笔。

---

## 2. LeNet-5 结构解剖：把每一层的形状算明白

### 2.1 先看全貌：一张图走完整个网络

LeNet-5 处理 MNIST 的标准流水线（输入是单通道灰度手写数字）：

```
输入        (B, 1, 32, 32)        ← 注意是 32×32，不是 28×28（见 §2.2）
  │
C1: Conv(1→6,  k=5, s=1, p=0)     → (B, 6, 28, 28)    6 个 5×5 卷积核
  │ tanh
S2: AvgPool(k=2, s=2)             → (B, 6, 14, 14)    尺寸减半
  │
C3: Conv(6→16, k=5, s=1, p=0)     → (B, 16, 10, 10)   16 个卷积核
  │ tanh
S4: AvgPool(k=2, s=2)             → (B, 16, 5, 5)      尺寸再减半
  │
flatten                           → (B, 400)           16 × 5 × 5 = 400 ★
  │
F5: Linear(400 → 120)             → (B, 120)
  │ tanh
F6: Linear(120 → 84)              → (B, 84)
  │ tanh
输出: Linear(84 → 10)             → (B, 10)            10 个类别的 logits
```

> 原论文里 `S2/S4` 是带可学习权重的"子采样层"，输出层是 RBF（径向基），这些都是 1998 年的历史包袱。**现代复现统一简化为 AvgPool + 标准 Linear**——这是社区公认的"教学标准版 LeNet-5"，d2l.ai、PyTorch 官方示例都这么写，你照着写不会错。

### 2.2 为什么要把 28×28 pad 到 32×32

这是新手第一个"为什么"。MNIST 原图是 28×28，但 LeNet-5 设计时输入是 32×32。原因很实在：

- **C1 用 5×5 卷积、无 padding**，输出尺寸 = `28 - 5 + 1 = 24`。如果直接喂 28×28，两次卷积+两次池化后，特征图会缩得太小，**边缘的笔画信息在第一层就被"啃掉"了**。
- LeCun 的做法是先把 28×28 的数字**居中放进 32×32 的画布**（四周补背景），这样 C1 输出正好是 28×28，后面尺寸链条对齐得很漂亮。

**怎么做**：用 `transforms.Pad(2)`（四周各补 2 像素，28+2+2=32），或者直接用 `transforms.Resize(32)`。本笔记用 Pad，因为它保持数字原始大小、只补背景，更贴近原论文。

### 2.3 那个 400 是怎么来的（最容易算错的地方）

`flatten` 后是 `16 × 5 × 5 = 400`。新手 90% 的报错都卡在这里——`Linear` 的输入维度写错，运行时报 `mat1 and mat2 shapes cannot be multiplied`。

记住一个**保命习惯**：永远不要硬编码 400，让代码自己算。两种工业做法：

1. **跑一次 dummy 前向**，`print(x.shape)` 看真实形状（调试时最快）。
2. **用 `nn.LazyLinear`** 或在 forward 里 `x.view(x.size(0), -1)` 后用 `x.shape[1]` 反推——让框架自动推断输入维度。

> 这个"形状对不上"的坑，本质和你 [[W4_Day2_makemore_MLP_前向训练]] §1.3 里"`W1` 的 6 = block_size × embed_dim 不要硬编码"是**同一个教训**：任何由上游超参算出来的维度，都不该手写死，要么让框架推断，要么显式算给注释看。

### 2.4 输出尺寸公式（贴在显示器上的那张）

卷积/池化后的边长，记这一个公式就够：

```
out = floor( (in + 2×padding − kernel) / stride ) + 1
```

拿 C1 验证：`(32 + 0 − 5) / 1 + 1 = 28` ✓
拿 S2 验证：`(28 + 0 − 2) / 2 + 1 = 14` ✓

> 这条公式呼应 [[AndrewNg_C4W1_CNN作业笔记]] §0 第 2 题——你 Day1 已经推过一遍，今天是在真实网络里用它逐层验算。养成"写完一层立刻心算输出尺寸"的习惯，能让你 80% 的 shape bug 在写代码时就被拦下，而不是运行时才崩。

---

## 3. LeNet-5 模型定义（`week5_cnn/src/lenet.py`）

### 3.1 设计决策（先讲为什么这么写）

**Q1：为什么用 `nn.Sequential` + 两段（features / classifier）拆分？**
这是 torchvision 里所有经典网络（AlexNet、VGG、ResNet）的标准写法：`features` 负责"提特征"（卷积部分），`classifier` 负责"做分类"（全连接部分）。这样拆有两个工业好处：迁移学习时可以只换 `classifier`、冻结 `features`；导出/量化时两段可以分别处理。**今天就按工业惯例写，养成肌肉记忆。**

**Q2：为什么提供一个 `act` 参数（tanh 还是 relu 可切换）？**
今天的任务核心是"对比"。把激活函数做成可切换的，你就能用**同一份代码**跑出"原版 tanh LeNet" vs "现代 relu LeNet"两组结果，亲眼看到 §4.1 说的差异。这就是"可复现对比实验"的工程做法——别复制两份代码，让差异变成一个开关。 

**Q3：输出层为什么不加激活、不加 softmax？**
和 [[W4_Day2_makemore_MLP_前向训练]] §2.2 Q3 一模一样的道理：`nn.CrossEntropyLoss` 内部自带 `log_softmax`，**数值稳定**。输出层永远是裸 `Linear`，吐 logits。

### 3.2 完整代码

```python
"""LeNet-5（教学标准版）模型定义。

环境：python>=3.9, torch>=2.0（CPU 即可，有 CUDA 更快）
运行自测：python -m src.lenet
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


class LeNet5(nn.Module):
    """LeNet-5（LeCun 1998）的现代教学复现。

    与原论文的差异（社区标准简化）：
      - 子采样层 S2/S4 用无参数的 AvgPool2d 代替（原版带可学习权重）
      - 输出层用标准 Linear 代替 RBF
      - 激活函数做成可切换（act='tanh' 还原 1998，'relu' 对标现代）

    输入约定: (B, 1, 32, 32)，MNIST 需先 Pad(2) 把 28×28 → 32×32。
    """

    def __init__(self, num_classes: int = 10, act: str = "tanh") -> None:
        super().__init__()
        # 选激活函数：tanh 还原原版，relu 模拟现代——这是今天对比实验的开关
        Act = {"tanh": nn.Tanh, "relu": nn.ReLU}[act]

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),   # C1: (B,1,32,32) -> (B,6,28,28)
            Act(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S2: -> (B,6,14,14)
            nn.Conv2d(6, 16, kernel_size=5),  # C3: -> (B,16,10,10)
            Act(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S4: -> (B,16,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # F5: 400 -> 120（400 = 16×5×5，别硬背，会算）
            Act(),
            nn.Linear(120, 84),          # F6: 120 -> 84
            Act(),
            nn.Linear(84, num_classes),  # 输出: 84 -> 10，裸 Linear 吐 logits
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # flatten：保留 batch 维，其余拉平。view 要求内存连续，
        # 这里 AvgPool 的输出是连续的，所以安全（同 W4D2 §2.2 Q1 的取舍）
        x = x.view(x.size(0), -1)        # (B,16,5,5) -> (B,400)
        return self.classifier(x)


def _self_test() -> None:
    """形状自测——不连数据集，纯随机张量验证 forward + 参数量。"""
    model = LeNet5(num_classes=10, act="tanh")
    x = torch.randn(8, 1, 32, 32)        # 假 batch
    y = model(x)
    assert y.shape == (8, 10), f"got {tuple(y.shape)}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ forward OK; logits shape = {tuple(y.shape)}; params = {n_params:,}")
    # 参考量级：约 6.2 万参数——和"AlexNet 6000 万"差整整 1000 倍，
    # 直观感受 §1.3 表格里"规模"那一行的含义。


if __name__ == "__main__":
    _self_test()
```

跑 `python -m src.lenet`，应输出：

```
✓ forward OK; logits shape = (8, 10); params = 61,706
```

> **6.2 万参数**——把这个数记住。等你 Day5 训 ResNet18（1100 万参数）、以后看 GPT-2（1.5 亿）时，LeNet 这个"6 万"是你心里的最小标尺。AI Infra 工程师对"参数量→显存→延迟"的数感，就是这样一个网络一个网络攒出来的。

---

## 4. 训练脚本：冲 test acc ≥ 98.5%（`week5_cnn/src/train_lenet.py`）

### 4.1 怎么稳稳达标（先讲策略）

98.5% 在 MNIST 上不算高门槛，但**用原版 tanh + SGD 想稳定达标其实有点费劲**——这恰好是今天对比的素材。达标配方：

- **优化器用 Adam（lr=1e-3, weight_decay=1e-4）**：计划里 W4 学的 `weight_decay` 派上用场。Adam 比 SGD 收敛快、对 lr 不敏感，5 个 epoch 就能上 99%。
- **激活函数其实 tanh/relu 都能过线**，但 relu 收敛明显更快（§5 会量化对比）。
- **数据要做标准化**：`Normalize((0.1307,), (0.3081,))` 是 MNIST 全局公认的均值/标准差。不做的话收敛会慢一截。

> **完全可复现的三件套**（呼应你计划 line 22"里程碑要可复现"）：固定 seed、写死环境、给运行命令。下面代码全做了。

### 4.2 完整训练脚本

```python
"""LeNet-5 在 MNIST 上训练，目标 test acc >= 98.5%。

环境：python>=3.9, torch>=2.0, torchvision>=0.15
运行：
    python -m src.train_lenet --epochs 5 --act relu
    python -m src.train_lenet --epochs 5 --act tanh   # 对比原版
首次运行会自动下载 MNIST 到 ./data（约 12MB）。
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.lenet import LeNet5


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证可复现（计划 line 22 的硬性要求）。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """MNIST DataLoader。Pad(2) 把 28×28 → 32×32 喂给 LeNet。"""
    tf = transforms.Compose([
        transforms.Pad(2),                       # 28×28 -> 32×32（见 §2.2）
        transforms.ToTensor(),                   # [0,255] uint8 -> [0,1] float
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 公认均值/方差
    ])
    train_set = datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    # num_workers=0 在 Windows 上最省心（多进程 DataLoader 在 win 上易踩坑）
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """返回 test accuracy。@no_grad 关掉 autograd：省显存、提速。"""
    model.eval()                                 # 切到推理模式（对 BN/Dropout 关键，LeNet 虽没有也应养成习惯）
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)            # logits 最大的那个类
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--act", choices=["tanh", "relu"], default="relu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} act={args.act}")

    train_loader, test_loader = build_loaders(args.batch_size)
    model = LeNet5(num_classes=10, act=args.act).to(device)
    # weight_decay=1e-4：W4 学的 L2 正则，防过拟合（LeNet 本身没正则层，靠它补一点）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()                            # 切回训练模式
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)  # 同 W4D2 §3.2：set_to_none 省一次写零
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch} | train_loss {running/len(train_loader):.4f} "
              f"| test_acc {acc*100:.2f}% | {time.perf_counter()-t0:.1f}s")

    final = evaluate(model, test_loader, device)
    print(f"\nFINAL test acc = {final*100:.2f}%  "
          f"({'PASS ✓' if final >= 0.985 else 'FAIL ✗ 没到 98.5%'})")


if __name__ == "__main__":
    main()
```

### 4.3 预期输出（CPU 约 1-2 分钟/epoch，GPU 几秒）

```
device=cpu act=relu
epoch 1 | train_loss 0.2856 | test_acc 97.84% | 65.3s
epoch 2 | train_loss 0.0712 | test_acc 98.61% | 130.1s
epoch 3 | train_loss 0.0503 | test_acc 98.83% | 195.0s
epoch 4 | train_loss 0.0389 | test_acc 99.02% | 260.4s
epoch 5 | train_loss 0.0310 | test_acc 99.08% | 325.7s

FINAL test acc = 99.08%  (PASS ✓)
```

> 第 2 个 epoch 就过 98.5% 了。如果你跑 `--act tanh`，会发现同样 5 epoch 大概在 98.7-98.9% 落地——**也能过线，但每个 epoch 的收敛都慢半拍**，第 1 个 epoch 往往只有 96% 左右。这个"慢半拍"就是 §5 要解释的 tanh 的代价。

### 4.4 常见错误与调试（贴近工业排错）

| 现象 | 根因 | 怎么修 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (8x576 and 400x120)` | 忘了 Pad，输入是 28×28，flatten 出来是 576 不是 400 | 加 `transforms.Pad(2)`，或按 §2.3 让维度自动推断 |
| test_acc 卡在 11% 左右不动 | 学习率太大梯度炸 / 忘了 Normalize | lr 降到 1e-3；加 `Normalize` |
| Windows 上 DataLoader 卡死或报 `BrokenPipe` | `num_workers>0` 在 win 上的多进程坑 | 设 `num_workers=0`（已在代码里） |
| 每次跑结果都不一样 | 没固定 seed | 调 `set_seed()`（已在代码里），但注意 GPU 上 cuDNN 仍有微小不确定性 |
| `model.eval()` 忘了调，acc 偏低 | 推理时 BN/Dropout 没切模式 | 评估前必须 `model.eval()`，训练前 `model.train()`——这是 [[W4_Day5_BatchNorm]] 训推双行为的延续，LeNet 没 BN 影响小，但养成习惯能救你未来的命 |

---

## 5. 核心任务：LeNet vs 现代 CNN 对比表（`tech_notes/lenet_vs_modern.md`）

这是今天产出的"主菜"。对比的两个对象：**LeNet-5（1998）** 和 **你 W1 写的现代 MNIST CNN**（test acc > 99%，结构是 `Conv→BN→ReLU→MaxPool→...→Dropout→FC`）。

### 5.1 总对比表

| 维度 | LeNet-5 (1998) | 你的 W1 现代 CNN | 这个差异解决了什么问题 |
|---|---|---|---|
| **激活函数** | tanh / sigmoid | **ReLU** | tanh 两端饱和→梯度消失；ReLU 正区间梯度恒为 1，深层网络才训得动（§5.2） |
| **池化** | average pooling | **max pooling** | max 保留最强响应（边缘/笔画），对分类更鲁棒；avg 会把强特征"稀释" |
| **正则化** | 几乎没有 | **BatchNorm + Dropout** | BN 稳定每层分布让训练更快更稳；Dropout 防过拟合（[[W4_Day5_BatchNorm]]） |
| **优化器** | SGD（+手调 lr） | **Adam** | Adam 自适应学习率，收敛快、对 lr 不敏感（[[W4_Day2_makemore_MLP_前向训练]] 的 lr-range 痛点的工业解法） |
| **权重初始化** | 朴素随机 / 高斯 | **Kaiming/He 初始化** | 配合 ReLU 让初始信号方差不爆不灭（呼应 W4 Kaiming，CNN 里 `fan_in = C_in×k×k`） |
| **参数规模** | ~6 万 | ~几十万到百万 | 数据多了，模型也能更大 |
| **输出层** | RBF（径向基） | 裸 Linear + CrossEntropy | log_softmax 数值稳定，是现代标准 |

### 5.2 为什么 tanh/sigmoid → ReLU 是关键一跃

> **sigmoid**（S 型函数）：把任意实数压到 (0,1)。**tanh**：压到 (−1,1)。两者都有"饱和区"——输入很大或很小时，曲线变平，**斜率（梯度）趋近 0**。

**问题出在哪**：反向传播时梯度是**逐层连乘**的。如果每一层的激活梯度都是个 < 1 的小数（sigmoid 最大斜率才 0.25），那 5 层连乘就是 0.25⁵ ≈ 0.001——**梯度传到前面几层时几乎归零，前层根本学不动**。这就是"梯度消失"。

**一个类比**：sigmoid 像一条层层抽成的传话游戏，每传一个人声音衰减到 1/4，传 5 个人后第一个人完全听不见后面的反馈。ReLU 像在正区间架了一条"直通管道"——梯度恒为 1，传多少层都不衰减。

**ReLU 的代价**：负区间梯度恒为 0，可能出现"神经元死亡"（Dead ReLU）。工业上的平衡选择是 **GELU / SiLU**（LLM 标配），它们在负区间留一点小梯度。但 ReLU 因为"简单 + 快 + 够用"，至今仍是 CNN 的默认。

### 5.3 ★ 今天的完成标准：为什么 sigmoid→ReLU 是"推理优化的前提"

> 这是计划 line 1341/1344 明确要求你"能讲清"的那句话。它比"ReLU 精度高"深一层——**ReLU 是算子融合（operator fusion）能成立的前提**。

**先讲什么是算子融合**：推理时，一个 `Conv → 激活` 的序列，naïve 实现是两步——先算完整个卷积、把结果（一大块特征图）写回显存，再读回来逐元素过激活函数。**两次显存读写**。

```
未融合：  Conv 计算 → 写显存 → 读显存 → 激活 → 写显存     （慢，卡在带宽）
融合后：  Conv 算出一个值，当场过激活，再写显存          （快，少一轮读写）
```

明天（Day3）你会学到，卷积是 **memory-bound（带宽受限）** 还是 compute-bound，往往卡在显存读写上。**省掉中间那次"写完再读"，就是实打实的提速。**

**关键来了——为什么必须是 ReLU 才能融合**：

- **ReLU = `max(x, 0)`**，是一个**逐元素（element-wise）、无参数、无状态**的操作。Conv 算出某个位置的值后，**立刻**就能对它取 `max(·,0)`，不需要等别的数据、不需要查任何表。所以编译器/推理引擎能把它"焊"进卷积 kernel 的最后一步，融合成**一个** GPU kernel。
- **sigmoid/tanh 含 `exp()`**：是超越函数，计算更贵，而且历史上常用查表近似。虽然理论上也能融合，但它"重"，融合收益和实现复杂度都不划算。ReLU 的 `max` 几乎免费，融合是纯赚。

**所以这句话的完整版是**：sigmoid → ReLU 这个替换，表面是"解决梯度消失、提升精度"，**深层是把激活函数从一个'笨重的超越函数'变成了一个'几乎免费的逐元素 max'，让推理引擎可以把 `Conv-ReLU`（甚至 `Conv-BN-ReLU`）融合成单个 kernel，省掉中间特征图的显存往返**。这就是为什么现代推理框架（TensorRT、ONNX Runtime）的图优化第一步几乎都是"找 Conv-BN-ReLU 模式做融合"——而这一切的前提，是 1998 年那个 tanh 在 2012 年被换成了 ReLU。

> **串联**：这正好把 [[W4_Day5_BatchNorm]] 的"fused BN"补全了——BN 折叠进 Conv（消掉 BN 的乘加），ReLU 焊在 Conv 输出（消掉激活的往返），两者合起来就是推理引擎里那个著名的 **`Conv-BN-ReLU` 三合一融合 kernel**。你 Day4 看 ResNet 源码、W8 学 TensorRT 时，会反复撞见它。这是你 AI Infra 主线上一个真正的"高频考点"。

---

## 6. 把今天三个任务串成可运行命令

```bash
# 0. 建项目结构（如果还没建）
#    week5_cnn/
#      ├─ src/lenet.py          ← §3
#      ├─ src/train_lenet.py    ← §4
#      ├─ tech_notes/lenet_vs_modern.md   ← §5 对比表
#      └─ logs/
cd week5_cnn

# 1. 模型形状自测（无需数据，秒级）
python -m src.lenet
# 期望：✓ forward OK; logits shape = (8, 10); params = 61,706

# 2. 训练达标（首次会下载 MNIST 到 ./data）
python -m src.train_lenet --epochs 5 --act relu
# 期望：FINAL test acc >= 98.5% (PASS ✓)，通常第 2 epoch 就过线

# 3. 对比实验：原版 tanh 跑一遍，对照收敛速度
python -m src.train_lenet --epochs 5 --act tanh
# 期望：也能过线，但每个 epoch 收敛慢半拍——这就是 §5.2 的活证据

# 4. 把 §5 的对比表 + §5.3 的标准答案抄进 tech_notes/lenet_vs_modern.md
```

---

## 7. 自测题（合上文档默答）

1. LeNet-5 的"5"指的是什么？池化层算不算在内？
2. MNIST 是 28×28，为什么 LeNet 要先 pad 到 32×32？不 pad 会怎样？
3. 第二个池化层后 flatten 出来是多少维？这个数怎么算？为什么不该硬编码？
4. 卷积输出尺寸公式是什么？用它验算 `Conv2d(6,16,k=5)` 输入 14×14 的输出边长。
5. AlexNet 和 LeNet 结构思路几乎一样，为什么 AlexNet 火了而 LeNet 没有？（至少说 3 点）
6. 为什么 sigmoid 会导致梯度消失，而 ReLU 不会？（从"梯度连乘"角度讲）
7. **★ 为什么说 sigmoid→ReLU 是推理优化（算子融合）的前提？ReLU 的什么特性让它能被"焊进"卷积 kernel？**
8. 训练前要 `model.train()`、评估前要 `model.eval()`，对 LeNet 影响小，但对哪些层是生死攸关的？

> 参考答案位置：1→§1.2；2→§2.2；3→§2.3；4→§2.4；5→§1.3 表格；6→§5.2；7→§5.3（今天的完成标准，必须能脱稿讲）；8→§4.4 末行 + [[W4_Day5_BatchNorm]]。

---

## 8. 与已有笔记的串联

| 今天的内容 | 关联点 |
|---|---|
| 卷积输出尺寸公式、im2col 预热 | [[AndrewNg_C4W1_CNN作业笔记]] §0 Q2 / Q4，今天在真实网络里逐层验算 |
| `view` flatten、裸 Linear 输出、`zero_grad(set_to_none)` | [[W4_Day2_makemore_MLP_前向训练]] §2.2 / §3.2 的 CNN 版复用 |
| `model.train()/eval()` 训推双行为、fused BN | [[W4_Day5_BatchNorm]]，今天用 `Conv-BN-ReLU` 融合把它补全 |
| Adam + weight_decay、Kaiming 初始化 | W4 优化器与初始化知识，今天首次用在 CNN 上 |
| sigmoid→ReLU 是算子融合前提 | 为 Day3（FLOPs/MAC、memory-bound）和 W8（TensorRT 图优化）埋的伏笔 |

**明天（Day3，缓冲日）**：Course 4 Week 2 下半（VGG/Inception/ResNet 概念预热）+ **手写 im2col numpy 版**（和 `F.conv2d` 误差 < 1e-5）+ FLOPs vs MAC vs latency 三角关系。今天 §5.3 说的"卷积卡在显存带宽上"，明天会用 arithmetic intensity 量化讲清楚——为什么 1×1 Conv 是 memory-bound。

**Day4（本周最重要的一天）**：ResNet 残差块。今天对比表里"现代 CNN 比 LeNet 多了什么"，到 Day4 会迎来最大的那个答案——**残差连接**，它是从 ResNet 到 Transformer 的通用补丁。

---

## 9. 完成标准 checklist（对齐计划 line 1483-1485）

- [ ] `src/lenet.py` 写完，`python -m src.lenet` 输出 `params = 61,706`
- [ ] `src/train_lenet.py` 写完，`--act relu` 跑出 **FINAL test acc ≥ 98.5%**
- [ ] 跑过 `--act tanh` 对照组，观察到收敛慢半拍
- [ ] `tech_notes/lenet_vs_modern.md` 对比表（§5.1）抄写完成
- [ ] 能脱稿讲清 §5.3：**sigmoid→ReLU 是算子融合的前提**（今天的硬指标）
- [ ] 固定 seed + 环境说明 + 运行命令齐全（可复现，计划 line 22）
- [ ] `W5_day2_log.md` 记录：初始 acc、达标 epoch、tanh vs relu 收敛差异

---

*笔记生成日期：2026-05-30（补记 W5 Day 2 任务）｜风格沿用 [[feedback-note-style]] 三段式*
