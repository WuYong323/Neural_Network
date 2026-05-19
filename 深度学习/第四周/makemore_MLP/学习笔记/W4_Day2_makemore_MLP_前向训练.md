# Week 4 · Day 2（2026-05-19）：MLP 前向 + mini-batch SGD + lr-range test

> **覆盖任务**（计划 line 1160-1167）：
> - [x] DL：看 EP3 第 0:40-1:00
> - [x] DL：完整 MLP 前向实现（emb → flatten → W1 b1 tanh → W2 b2）
> - [x] DL：mini-batch SGD 训练循环
> - [x] DL：lr-range test 跑通，画出 loss-vs-lr 曲线
> - [ ] DL：在 log 里分析"MLP 失去 Q9 三张优惠券里的哪几张"
>
> **阅读对象**：你自己——已完成 Day 1 的 dataset / vocab / embedding lookup，刚理解 `C[X]` 是查表的状态。
>
> **本笔记的设计**：每节先讲"原理 + 直觉"，再给可直接拷进 `src/` 的可运行代码，最后写工业锚点（推理优化视角下这块知识对应什么真实问题）。读完应该能从 0 把今天的 5 个任务全部跑通。

---

## 0. 学习目标（看完应能回答）

1. EP3 视频 0:40-1:00 这 20 分钟到底讲了什么？
2. `(N, 3, 2)` 的 embedding 张量怎么变成 `(N, 6)`？为什么用 `view` 而不是 `reshape`？
3. MLP 前向的每一步矩阵形状如何变化？为什么 `tanh` 放在 hidden 层后而不是 logits 后？
4. mini-batch SGD 比 full-batch 强在哪里？工业代码里 batch_size 怎么选？
5. lr-range test 是什么？为什么它比"拍脑袋设 lr=0.01"靠谱？
6. **bigram 训练为什么 lr=50 都不会炸？MLP 把哪几张"稳定性优惠券"收回去了？**
7. PyTorch 里 `loss.backward()` 之前为什么必须 `optimizer.zero_grad()`？

---

## 1. EP3 第 0:40-1:00 视频内容速览

这 20 分钟 Karpathy 做了两件事：

### 1.1 把 embedding 接到 MLP 上

Day 1 我们已经有了 `emb = C[X]`，形状 `(N, 3, 2)`。但 `nn.Linear` 只吃 2D 输入，所以要先 flatten：

```
emb       : (N, 3, 2)        ← Day 1 终点
flatten   : (N, 6)            ← 把 3 个字符的 2 维向量拼成一根 6 维向量
hidden    : (N, 100) = tanh(flatten @ W1 + b1)
logits    : (N, 27)  = hidden @ W2 + b2
loss      : scalar  = F.cross_entropy(logits, Y)
```

### 1.2 第一次完整前向 + 一次 backward

视频里 Karpathy 用一个 minibatch（32 个样本）跑了一次：
- 前向得到初始 loss（这里他遇到了"loss 太大"问题，这是 Day 4 EP4 才会修，今天先跑通就行）
- `loss.backward()` 触发 PyTorch autograd（**就是你第3周复现的那套机制**，PyTorch 版工业实现）
- 用最朴素的 SGD：`for p in params: p.data -= lr * p.grad`

> **关键差别**：第3周 micrograd 一次反向传播要遍历 Python 对象图，慢得离谱；PyTorch tensor.backward 走 C++ + CUDA，能直接跑大 batch。今天起所有训练都用 PyTorch autograd，micrograd 任务结束。

### 1.3 跟视频时的一个易错点

Karpathy 视频里 hidden 用 100 个神经元、embed_dim=2、block_size=3，所以 `W1.shape = (6, 100)`。**注意 6 是 `block_size * embed_dim`**——这个数随你超参变，写代码时不要硬编码。

---

## 2. MLP 前向：完整代码（`src/model.py`）

### 2.1 形状速查表（先看这个再读代码）

| 名字 | 形状 | 含义 |
|---|---|---|
| `X` | `(N, B)` | B = block_size，N 个样本，每行 B 个字符 id |
| `Y` | `(N,)` | 每个样本的目标字符 id |
| `C` | `(V, E)` | V=27 词表大小，E=embed_dim |
| `emb = C[X]` | `(N, B, E)` | fancy indexing |
| `flat = emb.view(N, B*E)` | `(N, B*E)` | flatten |
| `W1` | `(B*E, H)` | H = hidden_size，例如 100 |
| `b1` | `(H,)` | bias 广播 |
| `hidden = tanh(flat @ W1 + b1)` | `(N, H)` | |
| `W2` | `(H, V)` | |
| `b2` | `(V,)` | |
| `logits = hidden @ W2 + b2` | `(N, V)` | **不要再 softmax，CE loss 自己会做** |
| `loss = F.cross_entropy(logits, Y)` | scalar | |

### 2.2 关键设计决策

**Q1：为什么 `view` 不用 `reshape`？**
- `view` 要求张量内存连续（contiguous），强制你显式处理非连续情况——这是工程上的好习惯
- `reshape` 在不连续时会偷偷复制内存，性能问题被隐藏
- `emb = C[X]` 是 fancy indexing 出来的，PyTorch 保证它连续，所以 `view` 安全

**Q2：为什么 `tanh` 放在 hidden 后，不放在 logits 后？**
- logits 必须是无约束实数（softmax 之后才是概率）
- tanh 把输出压到 `[-1, 1]`，直接限制 logits 范围 → 模型表达能力被压死
- 工业模型（GPT-2、LLaMA）输出层全部是裸 Linear，没有任何激活

**Q3：为什么 `cross_entropy` 吃 logits 不吃概率？**
- `F.cross_entropy(logits, Y)` 内部是 `log_softmax + nll_loss`，**数值稳定**（用 log-sum-exp 技巧避免 exp 溢出）
- 你自己 `softmax → log → -mean(Y * log_p)` 那种写法在 fp16 / 大 logits 时会炸
- **所有工业代码都用 `cross_entropy(logits, ...)` 接口**，不要自己拼 softmax+log

### 2.3 完整代码

```python
"""makemore MLP 模型定义。

运行：python -m src.model  # 自测前向
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.vocab import VOCAB_SIZE


class MakemoreMLP:
    """Bengio 2003 风格的字符级 MLP。

    用 dataclass 风格的"权重容器 + 函数式 forward"实现，方便 Day 4 改初始化、
    Day 5 插 BatchNorm，而不用受 nn.Module 子类化的束缚。

    Attributes:
        C:  (vocab_size, embed_dim)  字符 embedding 表
        W1: (block_size * embed_dim, hidden_size)
        b1: (hidden_size,)
        W2: (hidden_size, vocab_size)
        b2: (vocab_size,)
    """

    def __init__(
        self,
        block_size: int = 3,
        embed_dim: int = 2,
        hidden_size: int = 100,
        vocab_size: int = VOCAB_SIZE,
        seed: int = 2147483647,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 这里先用最朴素的 randn，Day 4 (EP4) 会改成 kaiming
        self.C = torch.randn((vocab_size, embed_dim), generator=g, requires_grad=True)
        self.W1 = torch.randn((block_size * embed_dim, hidden_size), generator=g, requires_grad=True)
        self.b1 = torch.randn(hidden_size, generator=g, requires_grad=True)
        self.W2 = torch.randn((hidden_size, vocab_size), generator=g, requires_grad=True)
        self.b2 = torch.randn(vocab_size, generator=g, requires_grad=True)

    def parameters(self) -> list[Tensor]:
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, X: Tensor) -> Tensor:
        """前向：返回 logits，不做 softmax。

        Args:
            X: (N, block_size), dtype=long

        Returns:
            logits: (N, vocab_size)
        """
        emb = self.C[X]                                      # (N, B, E)
        flat = emb.view(emb.shape[0], -1)                    # (N, B*E)
        hidden = torch.tanh(flat @ self.W1 + self.b1)        # (N, H)
        logits = hidden @ self.W2 + self.b2                  # (N, V)
        return logits


def _self_test() -> None:
    """形状自测——不连数据集，纯随机张量验证 forward 正确性。"""
    N, B, E, H, V = 32, 3, 2, 100, 27
    model = MakemoreMLP(block_size=B, embed_dim=E, hidden_size=H, vocab_size=V)

    X_fake = torch.randint(0, V, (N, B))
    logits = model.forward(X_fake)

    assert logits.shape == (N, V), f"got {tuple(logits.shape)}"
    assert logits.dtype == torch.float32
    print(f"✓ forward OK; params = {model.num_params():,}")
    # 对照量级：B*E=6, H=100, V=27 →
    #   C: 27*2=54, W1: 6*100=600, b1:100, W2:100*27=2700, b2:27
    #   合计 3481


if __name__ == "__main__":
    _self_test()
```

跑一下 `python -m src.model`，应输出：
```
✓ forward OK; params = 3,481
```

---

## 3. mini-batch SGD：训练循环代码（`src/train.py`）

### 3.1 为什么要 mini-batch（不是 full-batch 也不是 SGD）

| 类型 | 每次更新看多少样本 | 优点 | 缺点 |
|---|---|---|---|
| full-batch | 全部 N 个 | 梯度方向最准 | 内存炸；N 大时一个 epoch 一次更新太慢 |
| **mini-batch** | 32-512 个 | **内存可控 + 梯度方向"够准" + GPU 并行充分** | 引入梯度噪声（其实是好事，能逃局部极小） |
| 纯 SGD（batch=1） | 1 个 | 内存最小 | 噪声太大，GPU 浪费 |

**工业默认**：训练大模型 batch_size = 1024-4M tokens（按 token 数算，不是样本数）。
**这里 makemore**：32 是 Karpathy 视频默认值——小数据集 + 字符级，足够。

### 3.2 PyTorch 训练循环的 5 步骨架

```python
for it in range(max_iters):
    # 1. 采样一个 batch
    ix = torch.randint(0, N, (batch_size,))
    Xb, Yb = X[ix], Y[ix]

    # 2. forward + loss
    logits = model.forward(Xb)
    loss = F.cross_entropy(logits, Yb)

    # 3. 清零梯度（关键！没有它梯度会累加上一轮的）
    for p in model.parameters():
        p.grad = None      # 比 zero_() 略快：直接释放显存，下次 backward 时重新分配

    # 4. backward：触发 autograd 把梯度填到 p.grad 里
    loss.backward()

    # 5. 更新参数
    with torch.no_grad():           # 更新本身不需要梯度，关掉 autograd 提速 + 防 graph 污染
        for p in model.parameters():
            p.data -= lr * p.grad
```

**Q：为什么 `p.grad = None` 而不是 `p.grad.zero_()`？**
两者效果都是"清零梯度"，但前者直接 `del` 掉那块显存，下次 backward 时 PyTorch 重新分配——**省一次写零的时间**，且能把"忘记清零"的 bug 直接变成 NoneType crash（更容易发现）。`torch.optim` 的 `zero_grad(set_to_none=True)` 默认就是这个行为。

**Q：为什么更新参数要 `torch.no_grad()`？**
`p.data -= lr * p.grad` 这一行如果不在 `no_grad` 里，PyTorch 会以为你在构建新的计算图，把 `p` 也标记为需要梯度的中间节点——下一轮 backward 时图会越积越大，最终 OOM。

### 3.3 完整训练脚本

```python
"""makemore MLP 训练脚本。

运行：
    python -m src.train --steps 50000 --lr 0.1 --batch_size 32
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from src.dataset import build_dataset
from src.model import MakemoreMLP
from src.vocab import build_vocab


def load_data(
    path: str = "data/names.txt",
    block_size: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, dict]:
    """读 names.txt，按 80/10/10 切 train/dev/test。"""
    with open(path, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(words), generator=g).tolist()
    words = [words[i] for i in perm]

    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    train_words, dev_words, test_words = words[:n1], words[n1:n2], words[n2:]

    stoi, itos = build_vocab(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size)
    Xdv, Ydv = build_dataset(dev_words, stoi, block_size)
    Xte, Yte = build_dataset(test_words, stoi, block_size)
    return Xtr, Ytr, Xdv, Ydv, Xte, Yte, stoi, itos


@torch.no_grad()
def eval_loss(model: MakemoreMLP, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 1024) -> float:
    """在整个 split 上算平均 loss（分批以防 OOM）。"""
    losses = []
    for i in range(0, X.shape[0], batch_size):
        logits = model.forward(X[i : i + batch_size])
        losses.append(F.cross_entropy(logits, Y[i : i + batch_size]).item())
    return sum(losses) / len(losses)


def train(
    model: MakemoreMLP,
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xdv: torch.Tensor,
    Ydv: torch.Tensor,
    steps: int = 50_000,
    batch_size: int = 32,
    lr: float = 0.1,
    eval_every: int = 5_000,
    seed: int = 42,
) -> list[tuple[int, float, float]]:
    """跑 mini-batch SGD，返回 [(step, train_loss, dev_loss), ...]。"""
    g = torch.Generator().manual_seed(seed)
    history: list[tuple[int, float, float]] = []
    N = Xtr.shape[0]
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        ix = torch.randint(0, N, (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = model.forward(Xb)
        loss = F.cross_entropy(logits, Yb)

        for p in model.parameters():
            p.grad = None
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p.data -= lr * p.grad

        if step % eval_every == 0 or step == steps:
            dev_loss = eval_loss(model, Xdv, Ydv)
            history.append((step, loss.item(), dev_loss))
            elapsed = time.perf_counter() - t0
            print(
                f"step {step:6d} | train {loss.item():.4f} | dev {dev_loss:.4f} | "
                f"{elapsed:.1f}s ({step / elapsed:.0f} it/s)"
            )

    return history


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--block_size", type=int, default=3)
    p.add_argument("--embed_dim", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=100)
    args = p.parse_args()

    Xtr, Ytr, Xdv, Ydv, Xte, Yte, _, _ = load_data(block_size=args.block_size)
    print(f"train={Xtr.shape[0]:,} dev={Xdv.shape[0]:,} test={Xte.shape[0]:,}")

    model = MakemoreMLP(
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
    )
    print(f"params: {model.num_params():,}")

    train(model, Xtr, Ytr, Xdv, Ydv, steps=args.steps, batch_size=args.batch_size, lr=args.lr)

    test_loss = eval_loss(model, Xte, Yte)
    print(f"\nfinal test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
```

**预期输出**（用默认超参跑 50k 步，CPU 约 2-3 分钟）：

```
train=182,580 dev=22,767 test=22,866
params: 3,481
step   5000 | train 2.4521 | dev 2.4982 | ...
step  50000 | train 2.2810 | dev 2.3104 | ...
final test loss: 2.31
```

> dev loss 2.3 左右 = 这周还没加 BN、kaiming 初始化时的合理水平。Day 3 调超参冲 2.2，Day 5 加 BN 冲 2.1。

---

## 4. lr-range test：从"拍脑袋调 lr"升级为"画曲线找 lr"

### 4.1 问题：lr 应该设多少？

随便选 lr 的常见后果：
- 太大（如 lr=10）→ loss 一上来就 NaN，训练发散
- 太小（如 lr=0.0001）→ 5 万步还没怎么动，浪费时间
- "经验值 0.01" → 对你这个具体模型/初始化/batch_size **不一定合适**

### 4.2 lr-range test 思想（fast.ai / Smith 2017 的 Cyclical LR 论文推广）

**用一次"指数递增 lr 的训练"，画 loss 随 lr 变化的曲线**，曲线会呈现典型的三段式：

```
loss
 │
 │ ████                                 ← 平台期：lr 太小，没学
 │     ████
 │         ████
 │             ████                     ← 下降期：lr 合适，loss 在降
 │                 ████
 │                     ██
 │                       █
 │                        █             ← 拐点：最佳 lr 大约在这里
 │                         ██
 │                           ████████   ← 发散期：lr 太大，loss 反弹
 └─────────────────────────────────── log(lr)
   1e-4    1e-3    1e-2    1e-1    1     10
```

**经验法则**：取**最陡下降段中点**对应的 lr，或**发散点的 1/10**。

### 4.3 实现（`src/lr_range_test.py`）

```python
"""lr-range test：跑一次 lr 指数递增的小训练，画 loss-vs-lr 曲线。

运行：python -m src.lr_range_test
输出：logs/lr_range_test.png
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.dataset import build_dataset
from src.model import MakemoreMLP
from src.train import load_data


def lr_range_test(
    model: MakemoreMLP,
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    lr_start: float = 1e-4,
    lr_end: float = 10.0,
    num_steps: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """跑 num_steps 步，每步用指数递增的 lr，记录 loss。

    Returns:
        (lrs, losses) —— 一一对应，长度都是 num_steps。
    """
    g = torch.Generator().manual_seed(seed)
    N = Xtr.shape[0]

    # lr 在 log 空间均匀
    lrs = torch.logspace(
        torch.log10(torch.tensor(lr_start)),
        torch.log10(torch.tensor(lr_end)),
        num_steps,
    ).tolist()

    losses: list[float] = []
    for step, lr in enumerate(lrs):
        ix = torch.randint(0, N, (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = model.forward(Xb)
        loss = F.cross_entropy(logits, Yb)

        for p in model.parameters():
            p.grad = None
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p.data -= lr * p.grad

        losses.append(loss.item())

        # 安全阀：loss 已经爆炸就停
        if loss.item() > 20 or torch.isnan(loss):
            print(f"diverged at step {step}, lr={lr:.4f}")
            return lrs[: step + 1], losses

    return lrs, losses


def plot(lrs: list[float], losses: list[float], save_path: str = "logs/lr_range_test.png") -> None:
    """画 loss-vs-lr 曲线（log x 轴）。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("loss")
    ax.set_title("LR Range Test")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"saved {save_path}")


def main() -> None:
    Xtr, Ytr, _, _, _, _, _, _ = load_data(block_size=3)
    model = MakemoreMLP(block_size=3, embed_dim=2, hidden_size=100)
    lrs, losses = lr_range_test(model, Xtr, Ytr, lr_start=1e-4, lr_end=10.0, num_steps=1000)
    plot(lrs, losses)


if __name__ == "__main__":
    main()
```

**怎么读这张图**：
- 找 loss 开始**显著下降**的 lr → 这是最小可用 lr
- 找 loss **开始反弹**的 lr → 这是发散点
- **正式训练 lr ≈ 发散点 / 10**，或下降段中点

makemore MLP 跑一次大概会发现：
- lr < 1e-3：基本不动
- lr ∈ [1e-2, 0.5]：loss 稳定下降
- lr > 1：开始抖
- lr > 5：直接发散

→ 选 lr=0.1 或 0.2 都合理。

### 4.4 工业延伸

- **fast.ai 的 `lr_finder()`** 就是这个思路的工业实现，每次开新模型都先跑一遍
- **OneCycleLR / CosineAnnealing** 是把 lr-range test 找到的"最佳 lr"作为峰值的调度策略
- **大模型预训练**：通常用 warmup + cosine decay，warmup 阶段就是变相的 lr-range test 防止初期发散
- **AdamW + transformer 的"魔法常数 lr=3e-4"**：是社区在 transformer 上做了无数次 lr-range test 的统计众数，并不是先验

---

## 5. 工业接轨：MLP 把 bigram 的"三张稳定性优惠券"收回去了哪几张

> 这一节回答今天最深的一个任务：**为什么 bigram 训练时 lr=50 都不会炸，MLP 却必须用 lr=0.1？**
> 这不是"调参经验"问题，是数学上 MLP 比 bigram 失去了几个稳定性结构。

### 5.1 三张优惠券是什么（bigram 享有的稳定性"福利"）

回看你第3周复现的 bigram 神经网络版：`logits = X_onehot @ W` + `cross_entropy`。它享受三张"优惠券"：

| 券 | 内容 | bigram 为什么享有 |
|---|---|---|
| **券 1：全局凸优化** | loss 只有一个最优解，任何下山方向都对 | 模型是**线性 + softmax**，softmax+CE 在线性参数空间里**严格凸** |
| **券 2：梯度有上界** | 单步梯度大小不会爆炸 | softmax 输出 ∈ [0,1]，`dlogits = p - y` 每项 ∈ [-1,1]，传到 W 的梯度也有界 |
| **券 3：稀疏更新** | 每步只更新一行 W（被访问那个字符对应那行） | one-hot 输入让 `dW = X_onehot.T @ dlogits` 只有"被访问行"非零 |

**这三张券一起，给了 bigram "用 lr=50 都不发散"的奇迹**——你可以拿 lr=50 跑你第3周的 bigram，loss 依然在降。

### 5.2 MLP 收回了哪几张

| 券 | MLP 状态 | 原因 |
|---|---|---|
| **券 1：全局凸** | ❌ **完全收回** | hidden 层非线性（tanh）让 loss 对 W1 不再凸——存在大量局部极小、鞍点 |
| **券 2：梯度有界** | ⚠️ **部分收回** | `dlogits` 仍然有界，但反传到 `W1` 时要乘 `W2.T`：梯度幅值随 W2 大小放大；越深的网络放大越多（这就是梯度爆炸的根源） |
| **券 3：稀疏更新** | ✓ **保留**（仅在 embedding 层 `C`） | `C[X]` 仍然是 lookup，`dC` 仍然只更新被访问行；但 `W1, W2, b1, b2` 是稠密更新 |

### 5.3 工程后果

- **lr 范围从 [小, 50] 收缩到 [小, ~1]**：因为券 2 部分失效，大 lr 一步就把 W1 推到饱和区（tanh 输出 ±1，梯度 0），陷死
- **初始化突然变重要**：bigram 时初始化随便（W=randn 也能训）；MLP 时随便初始化，hidden 层一开始就饱和或激活全 0，根本训不动 → Day 4 EP4 整集都在修这个
- **要画 lr-range test 曲线了**：bigram 不需要这种工具，MLP 必须

### 5.4 工业延伸：这三张券在更大模型里如何继续被收走

| 模型 | 券 1（凸） | 券 2（梯度有界） | 券 3（稀疏） |
|---|---|---|---|
| bigram | ✓ | ✓ | ✓ |
| **makemore MLP**（本周） | ✗ | ⚠️ | 仅 embedding |
| ResNet50（下周） | ✗ | ⚠️ 靠 BN + 残差续命 | ✗ |
| GPT-2 | ✗ | ⚠️ 靠 LayerNorm + 残差 + Adam 续命 | ✗ |
| **每张券的工业补丁** | 用更好的优化器（Adam）+ 大量 init 调参 | **BatchNorm / LayerNorm**（Day 5 主题）+ gradient clipping + 残差连接 | **embedding 量化 + weight tying** |

→ 你 Day 5 学 BatchNorm 时再回看这张表，就知道 BN 不是"加进去能涨点"的玄学技巧——它是**显式给券 2 续命的工程补丁**。

### 5.5 写到 `W4_day2_log.md` 的标准答案

> bigram 享有"全局凸 + 梯度有界 + 稀疏更新"三张稳定性优惠券，所以 lr=50 也不发散。MLP 引入 tanh 非线性后**全局凸性彻底丢失**，引入隐层后**梯度幅值会被 W2.T 连乘放大**——这两张券的失效合力把可用 lr 范围从 [小, 50] 压缩到 [小, ~1]。第三张券"稀疏更新"在 embedding 层 `C` 上保留，因此 `C[X]` 仍是 lookup，但 `W1/W2` 变成稠密更新。这就是为什么今天必须做 lr-range test、Day 4 必须修初始化、Day 5 必须加 BatchNorm——它们都是给被收走的两张券打的工程补丁。

---

## 6. 把今天 5 个任务串成一条可运行的命令

```bash
# 0. 先确保 Day 1 的 dataset / vocab / embedding_demo 都跑通
pytest tests/ -v

# 1. 看 EP3 第 0:40-1:00（视频任务，无代码产出）

# 2. 模型自测
python -m src.model
# 期望：✓ forward OK; params = 3,481

# 3. 跑一次完整 mini-batch SGD 训练
python -m src.train --steps 50000 --lr 0.1 --batch_size 32
# 期望：final test loss ~ 2.3

# 4. 跑 lr-range test，画曲线
python -m src.lr_range_test
# 期望：logs/lr_range_test.png 生成，能看到典型三段式曲线

# 5. 把 §5.5 那段标准答案抄进 W4_day2_log.md
```

---

## 7. 自测题（合上文档默答）

1. `MakemoreMLP(B=3, E=2, H=100, V=27)` 的总参数量是多少？哪一层最大？
2. 训练循环里如果忘了 `p.grad = None`，会发生什么？
3. 为什么 `cross_entropy` 不要先 softmax？
4. lr-range test 曲线如果**没有出现下降段**（loss 从头到尾一条平线），可能是什么原因？
5. bigram 用 lr=50 不炸，MLP 用 lr=10 就炸——直接的数学原因是什么？
6. 如果把 `tanh` 换成 `relu`，三张优惠券的状态会变化吗？

> 参考答案位置：1→§2.3 `_self_test` 注释；2→§3.2 末尾；3→§2.2 Q3；4→§4.4（通常是初始化太小，权重几乎是 0，再大 lr 也不动；或 batch_size 太大平滑掉了）；5→§5.2 券 1+券 2；6→relu 让饱和问题消失（券 2 在前向更平稳）但死亡 ReLU 现象是另一种"券 2 失效"，工业上 GELU/SiLU 是平衡选择。

---

## 8. 与已有笔记的串联

| 今天的内容 | 关联点 |
|---|---|
| MLP forward 矩阵形状 | Day 1 §3.4 的形状速查表，今天扩展到完整网络 |
| `loss.backward()` | 第3周 `autograd_explained.md` §1-2 闭包机制的 PyTorch C++ 工业版 |
| mini-batch SGD | 第2周 numpy 网络的训练循环升级版（PyTorch tensor + autograd 替代手写 backward） |
| lr-range test | Day 1 提到的"参数共享 → 训练稳定性"的具体诊断工具 |
| 三张优惠券分析 | 第3周 bigram 训练为什么 lr=50 不炸的数学解释；为 Day 4 (kaiming init) 和 Day 5 (BatchNorm) 铺垫 |

**明天（Day 3，缓冲日）的内容**：把 train/dev/test split 跑通，三轴调参（embed_dim / hidden / block_size），第一次 `torch.profiler` 上手，dev loss 冲 2.2。

**Day 4 EP4 上半（最关键的初始化诊断）的入口**：今天你训练时如果 step 1 的 loss 是 27 而不是 ≈3.3（=log(27)），那就是 EP4 第一个要修的"初始 logits 不均匀"问题——记下你今天看到的初始 loss 数值，Day 4 用它当锚点。

---

*笔记生成日期：2026-05-19（W4 Day 2）*
