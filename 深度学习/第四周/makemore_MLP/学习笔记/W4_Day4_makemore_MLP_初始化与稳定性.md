# Week 4 · Day 4（2026-05-21，周四）：EP4 上半 · 初始 loss 诊断 + tanh 饱和 + Kaiming 初始化

> **覆盖任务**（计划 line 1181-1186）：
> - [ ] DL：看 EP4 第 0:00-1:00
> - [ ] DL：诊断并修复初始 loss 异常（从 27 降到 ≈3.3）
> - [ ] DL：画 tanh 激活直方图，饱和率 < 20%
> - [ ] DL：实现 kaiming 初始化
> - [ ] DL：完成 `tech_notes/init_and_stability.md`（含 FP16/INT8 延伸）
>
> **阅读对象**：你自己——已完成 Day 3 的 train/dev/test split + 三轴消融 + 首次 profiler。MLP 能跑、loss 能降到 2.2 以下，但你**还没有问过一个问题**：第一步 loss 为什么是 27，不是别的数字？
>
> **本笔记的设计**：每节"现象 → 直觉 → 数学 → 代码 → 工业锚点"五段式。读完应该能：（1）解释清楚为什么初始 loss 在"理论下限 3.3"和"灾难值 27"之间随机晃；（2）画出 tanh 激活直方图并读懂"饱和率"；（3）写出 Kaiming 初始化的公式且能说清 √(2/fan_in) 里 2 是怎么来的；（4）知道这套初始化在 FP16/INT8 真实工业训练里**绝对不是可选项**。

---

## 0. 学习目标（看完应能回答）

1. 一个 V=27 类分类问题，刚初始化的网络应该输出什么样的概率分布？对应的交叉熵 loss 是多少？这个数字（≈3.3）是怎么算出来的？
2. 跑出来初始 loss 是 27，不是 3.3——这意味着什么？模型在第一步**自信地猜错了**还是**犹豫地猜错了**？哪个更糟糕？
3. 怎么用一行代码把 W2 / b2 改一下就能让初始 loss 立刻回到 3.3？这个修改到底改了什么物理量？
4. 什么叫"tanh 神经元饱和"？饱和的神经元为什么是"训练里的死人"？饱和率 50% 和 5% 在训练曲线上看起来一样吗？
5. Kaiming 初始化里的 √(2/fan_in) 是怎么推出来的？为什么 ReLU 是 √2 而 tanh 是 5/3？
6. **为什么 FP16 训练前，初始化系数稍微大一点点就会让训练在第 3 步就 NaN？这跟 loss scaling 是什么关系？**
7. INT8 量化为什么对"权重分布是否接近 0 均值正态"特别敏感？跟今天的初始化是同一个故事吗？
8. 今天修好的"初始 loss = 3.3 + 饱和率 < 20%"，到底从 Day 3 的三张优惠券表里**收回了哪张券**？还差哪张？（明天 BN 出场）

---

## 1. EP4 第 0:00-1:00 视频内容速览

这 1 小时 Karpathy 把"训练能不能开始"和"训练有没有效率"两件事掰开讲——这是整个系列里**第一次把"调试一个能跑的网络"当成研究对象**。

### 1.1 三个连续诊断动作

| 顺序 | 现象 | 诊断 | 修复 |
|---|---|---|---|
| ① | 初始 loss = 27（理论应是 3.3） | logits 初始化太分散，softmax 极度不均 | 把最后一层 `W2 *= 0.01, b2 *= 0`，让初始 logits ≈ 0 |
| ② | tanh 输出直方图：大量在 ±1 处 | 隐层输入太大，进入 tanh 饱和区 → 反向梯度 ≈ 0 | 把 `W1 *= 0.2`（粗暴版） |
| ③ | 凭手感乘 0.2 不可移植到大网络 | 没有理论支持，靠"试" | 引入 Kaiming 初始化：`W = randn * gain / √fan_in` |

**视频核心叙事**：从"试出来的 0.01 和 0.2"到"有数学依据的 √(2/fan_in)"——这是工业训练里**初始化从经验变成科学**的关键一步。

### 1.2 三个图的演变

Karpathy 一直在看三种直方图：
- **激活分布**（每层 tanh 输出的值分布）——理想：以 0 为中心、方差稳定、几乎没有 ±1 附近的点
- **梯度分布**（每个参数 `.grad` 的值分布）——理想：方差大致跨层一致
- **更新比例**（`update / weight`，每步参数变化 / 自身大小）——理想：在 1e-3 量级，太大说明 lr 大或梯度爆，太小说明梯度消失

> 这三张图是工业训练里"看一眼就知道初始化对不对"的硬技能，今天先掌握第一张（激活分布），后两张明天 BN + 后续课会用到。

---

## 2. 初始 loss = 27 的"自信地猜错"诊断

### 2.1 一个直觉问题：你刚生下来一只小狗，让它猜你左手还是右手有零食

```
情况 A：小狗左右各 50% 看一眼，犹豫着猜（每次都吃到一半）
情况 B：小狗"啪"一下扑左手——但其实右手里有零食
```

哪个表现更糟糕？**B 更糟**——它不是不会，它是**自信地搞错了方向**。神经网络初始 loss 异常就是 B。

### 2.2 数学上："理论下限 loss"是什么

对一个 V=27 类的分类问题，**一个完全随机的模型**（不偷看任何信息）应该输出什么？
- 每一类概率：1/27 ≈ 0.037
- 正确类的预测概率也是 1/27
- 单样本交叉熵：`-log(1/27) = log(27) ≈ 3.296`

```python
import math
math.log(27)   # → 3.2958...
```

> **这个 3.3 不是"训练到最后的目标"，是"什么都没学时的合理起点"**。低于它说明在学，高于它说明**比随机更糟**。

### 2.3 为什么会跳到 27？

假设最后一层算出的 logits 是：

```
logits = [..., 8.7, ..., -12.3, ...]  ← 数值跨度大，比如 ±20 量级
```

经过 softmax，**最大那一项的概率会被推到接近 1，其他几乎是 0**。如果"接近 1"的那一项**恰好不是正确答案**，loss 就是：

```
-log(P_正确)  =  -log(很小的数)  =  很大的数
```

比如 P_正确 = 1e-12，loss = 27.6。**这就是 27 的来源**：模型"信心满满地说错了类"。

> **本质**：刚初始化的网络，logits 不应该有"信心"——它应该均匀（≈ 0），让 softmax 输出接近 1/V。

### 2.4 用一行代码"压平"信心

```python
# 原始（坏）
W2 = torch.randn(H, V) * 1.0
b2 = torch.randn(V) * 1.0

# 修复（好）
W2 = torch.randn(H, V) * 0.01   # 缩 100×，hidden @ W2 输出 → 接近 0
b2 = torch.zeros(V)              # 偏置归零
```

物理含义：

| 修改 | 改变了什么 | 效果 |
|---|---|---|
| `W2 *= 0.01` | 第二层权重的标准差从 1.0 → 0.01 | `hidden @ W2` 输出从 ±量级 1 → 量级 0.01 |
| `b2 = 0` | 不让偏置一开始就给某一类"先天优势" | softmax 起点真正均匀 |

跑一下，**初始 loss 立刻从 27 掉到 3.3 附近**。

### 2.5 为什么不直接把所有层都 `*= 0.01`？

学过线代的本能反应："那我把所有权重都缩小不就行了？"——不行。

```
W1 *= 0.01 → hidden 也变 0.01 量级 → tanh(0.01) ≈ 0.01 → 输出几乎是直线
→ 整个网络退化成线性，连 MLP 都不算了
```

> **关键直觉**：初始化的目标不是"让所有数字变小"，而是**让每一层的"信号"既不爆炸也不消失**。只有最后一层（直接喂给 loss 的那层）才适合大幅压缩，因为它的输出是 logits，**logits 不需要表示信息，只需要表示"还没有偏好"**。

---

## 3. 修复初始 loss 的可运行代码

放在 `src/init.py`（今天的小产出之一）：

```python
"""初始化策略：分两部分——
1) 最后一层（输出 logits）：W *= 0.01, b = 0 → 让初始 loss 接近 log(V)
2) 隐层：交给 Day 4 §5 的 Kaiming 初始化

运行：python -m src.init  # 跑自检
"""
from __future__ import annotations

import math

import torch


def init_last_layer(W2: torch.Tensor, b2: torch.Tensor, scale: float = 0.01) -> None:
    """In-place：压扁最后一层，让初始 logits ≈ 0。"""
    with torch.no_grad():
        W2.mul_(scale)
        b2.zero_()


def expected_initial_loss(num_classes: int) -> float:
    """完全随机分类器的交叉熵下限。"""
    return math.log(num_classes)


def _selfcheck() -> None:
    """跑一个 minibatch 验证：修复后初始 loss ≈ log(27)。"""
    import torch.nn.functional as F

    torch.manual_seed(42)
    N, V, H = 32, 27, 200
    hidden = torch.randn(N, H)          # 假装是 tanh 出来的
    W2 = torch.randn(H, V)              # 默认 N(0,1)，会造成 loss 爆炸
    b2 = torch.randn(V)
    Y = torch.randint(0, V, (N,))

    logits_bad = hidden @ W2 + b2
    loss_bad = F.cross_entropy(logits_bad, Y).item()

    init_last_layer(W2, b2, scale=0.01)
    logits_good = hidden @ W2 + b2
    loss_good = F.cross_entropy(logits_good, Y).item()

    print(f"修复前初始 loss = {loss_bad:.3f}   ← 远高于 {expected_initial_loss(V):.3f}")
    print(f"修复后初始 loss = {loss_good:.3f}   ← 接近 log(27) = {expected_initial_loss(V):.3f}")


if __name__ == "__main__":
    _selfcheck()
```

预期输出（seed=42）：

```
修复前初始 loss = 17.xxx   ← 远高于 3.296
修复后初始 loss = 3.31x    ← 接近 log(27) = 3.296
```

> **注意**：视频里 Karpathy 第一次跑出来是 27，我们 seed=42 跑出来可能是 17 或 12——具体数字跟随机数有关，但**只要远大于 log(V)，就是同一个病**。

---

## 4. tanh 饱和：训练里的"死神经元"

### 4.1 一个比喻：满分 100 分的学生考了 100 分

你考了 100 分，老师再多给你 1 道难题——分数还是 100。你的"分数"对"难题数量"的导数 = 0：**你不再被任何外部刺激改变**。

tanh 神经元也是这样：

```
tanh(x) =   1, 当 x → +∞
tanh(x) =  -1, 当 x → -∞
tanh(x) ≈   x, 当 x ≈ 0
tanh'(x) = 1 - tanh²(x)   ← 这是导数
```

- 当 tanh 输出在 ±1 附近，导数 ≈ 0
- 反向传播时，**梯度乘上这个 0**，往前传的梯度也 ≈ 0
- 这个神经元前面的所有权重在这一步**完全收不到信号**——它"死了"

### 4.2 什么叫"饱和率"

定义：一个 batch 里 |tanh 输出| > 0.99 的神经元占比。

```python
h = torch.tanh(pre_activation)   # shape (N, H)
saturation_rate = (h.abs() > 0.99).float().mean().item()
```

| 饱和率 | 训练表现 | 大概原因 |
|---|---|---|
| < 5% | 健康 | 初始化合理，激活分布近似 N(0, ~0.5²) |
| 20%-50% | 慢 | W1 偏大，hidden pre-activation 太"凸出" |
| > 80% | 几乎学不动 | 初始化严重过大；要么训不动要么需要超小 lr |

### 4.3 为什么"饱和率 50% 但 loss 还在降"是陷阱

很多新手看到 loss 在降就以为没事——其实只是**剩下 50% 的"活神经元"在勉强干活**。这意味着：

1. **你买了 200 个神经元的容量，实际只用 100 个**——参数浪费 50%
2. 一旦数据稍微变难，这 100 个不够用，你以为加 hidden 没效果，其实是你新加的也被饱和掉了
3. 在 ResNet 这种几十层的网络里，**逐层饱和率叠加**——10 层各 50% 饱和，第 10 层有效信号 ≈ 0

> **工业法则**：在 PR review 里看到模型不收敛，第一件事就是让作者把激活直方图发出来——这是 Karpathy 在 OpenAI / Tesla 反复布道的工业习惯。

### 4.4 画饱和率直方图（`src/plot_activations.py`）

```python
"""画 tanh hidden 层的激活分布 + 饱和率。

运行：python -m src.plot_activations
输出：logs/activation_hist_<config>.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.model import MakemoreMLP
from src.train import load_data


def hidden_preactivation(model: MakemoreMLP, X: torch.Tensor) -> torch.Tensor:
    """返回 tanh 之前的 pre-activation，shape (N, H)。"""
    emb = model.C[X]                                 # (N, B, E)
    flat = emb.view(emb.size(0), -1)                 # (N, B*E)
    pre = flat @ model.W1 + model.b1                 # (N, H)
    return pre


def main(scale_W1: float = 1.0, tag: str = "default") -> None:
    torch.manual_seed(42)
    Xtr, Ytr, _, _, _, _, _, _ = load_data(block_size=3)
    model = MakemoreMLP(block_size=3, embed_dim=10, hidden_size=200, seed=42)

    # 手动 scale 模拟"初始化的好坏"
    with torch.no_grad():
        model.W1.mul_(scale_W1)

    # 取一个 batch 看激活
    N = 1024
    idx = torch.randperm(Xtr.size(0))[:N]
    pre = hidden_preactivation(model, Xtr[idx])
    h = torch.tanh(pre)

    sat_rate = (h.abs() > 0.99).float().mean().item()
    std_pre = pre.std().item()
    print(f"[{tag}] pre-act std = {std_pre:.3f}   tanh 饱和率 = {sat_rate*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pre.flatten().detach().numpy(), bins=60, color="tab:blue", alpha=0.7)
    axes[0].set_title(f"pre-activation  (std={std_pre:.2f})")
    axes[0].axvline(-2, ls="--", c="r"); axes[0].axvline(2, ls="--", c="r")
    axes[1].hist(h.flatten().detach().numpy(), bins=60, color="tab:orange", alpha=0.7)
    axes[1].set_title(f"tanh output  (sat={sat_rate*100:.1f}%)")
    axes[1].axvline(-0.99, ls="--", c="r"); axes[1].axvline(0.99, ls="--", c="r")

    Path("logs").mkdir(exist_ok=True)
    out = f"logs/activation_hist_{tag}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    # 跑三组对比：太大 / 默认 / Kaiming 推荐
    main(scale_W1=3.0, tag="too_large")
    main(scale_W1=1.0, tag="default")
    main(scale_W1=0.3, tag="small")
```

预期：
- `too_large`：pre-act std ~6，tanh 饱和率 70%+，直方图大量集中在 ±1
- `default`：std ~2，饱和率 25%-40%
- `small`：std ~0.6，饱和率 < 10%，但**也别太小**（小了变线性）——这是 §5 Kaiming 要解决的"恰到好处"问题

---

## 5. Kaiming 初始化：从"试出来 0.2"到"有理由的 √(2/fan_in)"

### 5.1 直觉：为什么权重要 ÷ √(fan_in)

设想一个 hidden 层神经元，输入 `x` 有 `fan_in = 200` 维：

```
y = w1*x1 + w2*x2 + ... + w200*x200
```

如果每个 `x_i ~ N(0, 1)`，每个 `w_i ~ N(0, 1)`，那么 `y` 的方差是：

```
Var(y) = Σ Var(w_i * x_i) = fan_in × 1 × 1 = 200
→ std(y) = √200 ≈ 14
```

`y = 14`这种量级喂给 tanh，**直接饱和到 ±1**。每多一层，方差再乘 fan_in 一次——10 层后方差 200^10，纯爆炸。

**解决方案**：把权重的标准差除以 √fan_in：

```
w_i ~ N(0, 1/fan_in)   ← std(w_i) = 1/√fan_in
→ Var(y) = fan_in × (1/fan_in) × 1 = 1   ✓
→ std(y) = 1，喂给 tanh 不饱和
```

> **这就是 Xavier/Glorot 初始化的核心**：让每一层的输出方差 = 输入方差 = 1。

### 5.2 ReLU 的特殊性：为什么是 √2 不是 √1

ReLU 把负数砍掉一半：

```
ReLU(x) = max(0, x)
→ 若 x ~ N(0, 1)，ReLU(x) 的方差 ≈ 1/2（一半被砍）
```

所以 ReLU 后面的层，输入方差只有"线性版本"的一半——补偿回来要把权重方差再放大 2 倍：

```
ReLU 版本：std(w) = √(2/fan_in)   ← 这里的 2 来源于"砍掉一半"
线性/tanh 版本：std(w) = √(1/fan_in) 或者带 gain
```

### 5.3 gain：不同激活函数的修正系数

PyTorch 官方给出的 gain 表（来源：`torch.nn.init.calculate_gain`）：

| 激活函数 | gain | 公式 |
|---|---|---|
| linear / Identity | 1.0 | `std(w) = 1/√fan_in` |
| sigmoid | 1.0 | 同上 |
| tanh | **5/3 ≈ 1.667** | `std(w) = (5/3) / √fan_in` |
| ReLU | **√2 ≈ 1.414** | `std(w) = √2 / √fan_in` |
| LeakyReLU(α) | √(2/(1+α²)) | 一般取 √2 |

> **为什么 tanh 是 5/3**：tanh 在 0 附近导数是 1（同线性），但稍微远一点点就压缩——为了让"前向输出方差"和"反向梯度方差"都保持稳定，要稍微补偿一下"压缩损失"。5/3 是 Glorot 论文里的经验值，He 论文里推广到 ReLU 拿到 √2。

### 5.4 完整初始化函数（`src/init.py` 补充）

```python
def kaiming_init_(W: torch.Tensor, fan_in: int, gain: float = math.sqrt(2)) -> None:
    """In-place Kaiming 初始化：std = gain / sqrt(fan_in)。

    Args:
        W: 权重张量（shape 任意，但 fan_in 维由用户指定）
        fan_in: 输入维度
        gain: 激活函数系数。ReLU = √2，tanh = 5/3，linear = 1.
    """
    std = gain / math.sqrt(fan_in)
    with torch.no_grad():
        W.normal_(mean=0.0, std=std)


def init_makemore_mlp(model, gain_hidden: float = 5/3, scale_last: float = 0.01) -> None:
    """对 MakemoreMLP 做完整初始化：
       - W1 用 Kaiming（tanh gain=5/3）
       - W2 / b2 用 §3 的压扁
       - C（embedding）保持默认 randn（embedding 不接激活函数，无需 fan_in 校正）
    """
    B, E = model.block_size, model.C.shape[1]
    H = model.W1.shape[1]
    fan_in_W1 = B * E
    kaiming_init_(model.W1, fan_in_W1, gain=gain_hidden)
    with torch.no_grad():
        model.b1.zero_()
    init_last_layer(model.W2, model.b2, scale=scale_last)
```

### 5.5 跑前后对比（写到 W4_day4_log.md）

```python
# 三组初始化下，初始 loss + 饱和率 + 训练 1000 步后的 dev loss
| 配置                 | init loss | tanh 饱和率 | dev@1000步 |
|---------------------|-----------|------------|------------|
| 朴素 randn(N(0,1))   |  17.xx    |  ~70%      |  3.1       |  ← Day 2 真实状态
| 仅修最后层 (§3)      |   3.31    |  ~50%      |  2.7       |  ← 修了"自信猜错"但 hidden 还饱和
| Kaiming + 修最后层   |   3.30    |  ~10%      |  2.4       |  ← 今天的目标
```

> 数字会随 seed 有 ±0.05 波动，但**三档之间的差距非常稳定**——这是 Karpathy 视频里要传达的"初始化决定能不能训"。

---

## 6. 工业锚点：FP16/INT8 时代，初始化变成了"红线"

写到 `tech_notes/init_and_stability.md`，这是今天的核心交付物之一。

### 6.1 §1 FP32 时代的初始化：错了只是慢

FP32 动态范围：约 ±3.4 × 10³⁸。初始化乘错一个 10× 系数？激活变 10×，还在 FP32 能表示范围内，只是收敛慢——**能训，就是慢**。

### 6.2 §2 FP16 时代：错了直接 NaN

FP16 动态范围：±65504（约 ±6.5 × 10⁴）。如果 hidden 层 pre-activation 的方差是 100，单个值很可能超过 1000——平方/累加几步**直接溢出成 inf → backward 全 NaN → 训练废了**。

**真实工业现象**：
- 同样的代码、同样的数据，FP32 跑得好好的，开了 `model.half()` 跑 3 步 NaN
- 不是 PyTorch bug，是初始化没考虑 FP16 的动态范围

**工业应对**：
1. **Loss scaling**（NVIDIA Apex / PyTorch AMP）：在 backward 前把 loss 乘 1024×，避免梯度太小被 FP16 round 到 0；update 时再除回去
2. **BF16 替代 FP16**：BF16 牺牲精度换动态范围（指数位多 3 位，能表示到 10³⁸），对初始化不敏感——这就是为什么现在 LLM 训练几乎全部用 BF16
3. **Mixed precision**：参数用 FP32 master copy，前向/反向用 FP16，更新时用 FP32 累加——这是 GPT/LLaMA 训练的标配

> **关键认知**：你今天调初始化让 loss 从 27 降到 3.3，看起来是"小事"。**在 FP16 时代，这个调整就是"能不能跑 1 个 epoch"和"第 3 步 NaN"的区别**。

### 6.3 §3 INT8 量化对初始化的隐藏依赖

INT8 量化的本质：把 FP32 的权重映射到 [-128, 127] 的整数。

```
W_int8 = round(W_fp32 / scale)
scale = max(|W_fp32|) / 127     ← 量化粒度
```

如果权重分布**不**接近 0 均值正态：
- 长尾分布（一个 outlier = 100，其他都是 0.1）→ scale 被 outlier 拉大 → 大部分权重 round 后变 0 → 模型死掉
- 均匀分布的尾巴 → 同样问题

Kaiming 初始化的核心副产品：**让权重一开始就是 0 均值、std = gain/√fan_in 的正态分布**。训练过程会大致保持这个形态——**自带"量化友好"属性**。

**真实工业现象**：
- 同样架构的两个模型，A 用 Kaiming 训练，B 用 default randn 训练
- FP32 精度差不多
- 量化到 INT8 后，A 几乎无损，B 掉点 5%+
- 原因：B 的权重分布有更多 outlier，量化误差大

### 6.4 §4 与 BatchNorm 的关系（Day 5 伏笔）

Kaiming 让"**初始**激活分布稳定"——但训练 1000 步后，分布会**drift**（漂移），可能又变成大量饱和。

BatchNorm 做的事：**每一步都强制把激活归一化到 0 均值、单位方差**——相当于"动态版 Kaiming"，每步都把 §5 重新做一遍。

```
Kaiming 初始化  =  打开门让模型能进屋
BatchNorm       =  屋里装空调，温度永远稳定
```

> 明天 Day 5 就是 BN 上场——你今天调好的初始 loss = 3.3 + 饱和率 10%，明天会看到 BN 如何**让这个状态一直保持下去**（而不是只在第一步成立）。

### 6.5 §5 与 §Q9 / Q4 的串联

回顾 Day 2 末尾 "MLP 失去了 bigram 的哪几张优惠券"：

| 优惠券 | bigram | MLP（Day 2-3） | MLP + 修最后层 + Kaiming（今天） |
|---|---|---|---|
| 全局凸 | ✓ | ✗（已失） | ✗（仍失，结构问题） |
| softmax+CE 梯度有界 | ✓ | ✓（保留） | ✓（保留） |
| 稀疏更新（one-hot） | ✓ | ✗（dense embedding） | ✗ |
| **初始 loss 合理** | ✓（自然就 log V） | ✗（27 灾难） | ✅（今天修好） |
| **激活无饱和** | n/a（无 hidden） | ✗（50% 饱和） | ✅（今天修好） |
| **训练中分布稳定** | n/a | ✗ | ✗（明天 BN 修） |

---

## 7. 串联起来：今天的完整工作流

```bash
# 1. 看 EP4 第 0:00-1:00（视频任务，约 1h）

# 2. 跑诊断（自检 Day 3 模型的初始 loss）
python -m src.train --steps 50   # 看前 5 步 loss，应该 27/17 之类的大数

# 3. 实现 src/init.py（init_last_layer + kaiming_init_）
python -m src.init               # 跑自检：loss_bad >> log(27) > loss_good ≈ log(27)

# 4. 在 src/model.py 的 __init__ 末尾调用 init_makemore_mlp(self)
#    再跑一次 src/train.py，看初始 loss 是否 ≈ 3.3

# 5. 跑 src/plot_activations.py（三档：too_large / default / small）
#    确认 small (Kaiming 等价) 的饱和率 < 20%

# 6. 写 tech_notes/init_and_stability.md（按 §6 的 §1-§5 五节）

# 7. 在 W4_day4_log.md 记录三档对比表（§5.5）+ 明天 BN 的入口数值
```

---

## 8. 自测题（合上文档默答）

1. 一个 V=10000 的 LLM 词表，理论初始 loss 应该多少？跑出来 5.0 / 9.2 / 15 哪个正常？
2. 如果**只**把 `W2 *= 0.01`，不把 `b2 = 0`，初始 loss 会怎样？
3. 为什么 §3 修复后 dev loss 反而下降（从 2.7 → 2.4 而不是 2.5）？仅仅是"训得快"还是"训得更好"？
4. 你的 hidden_size = 200，但激活直方图显示 100 个神经元长期处于 ±1 附近——这意味着模型"实际容量"是多少？为什么 hidden=500 也救不了？
5. 把 `gain = 5/3` 改成 `gain = 1.0`（线性版本）来初始化 tanh 网络，会发生什么？激活会偏大还是偏小？
6. **FP16 训练时为什么必须 loss scaling？跟今天的 Kaiming 是配合还是替代关系？**
7. INT8 量化前，是用 Kaiming 训练好的模型量化容易，还是用 randn 训练好的模型量化容易？为什么？
8. 第 3 周 micrograd 的 make_moons 训练你没做 Kaiming——为什么也能跑通？（提示：网络层数和宽度都小）

> 参考答案位置：1→ log(10000)≈9.21，所以 9.2 正常，5 说明已经在学（不可能初始就这么低，可能是测错位置），15 说明 logits 太分散；2→ 还是会爆，因为 b2 也贡献一个独立的偏置项；3→ 训得更好——前 N 步浪费在"修复方向"上，等于实际有效训练步数变多 + 饱和神经元变少 = 真实容量翻倍；4→ 实际容量 = 100；hidden=500 → 250 个饱和 + 250 个活，比例不变，不解决问题；5→ tanh 激活会偏小（因为没补偿 tanh 的压缩），网络近似线性，dev loss 收敛慢且 plateau 在更高位置；6→ 配合：Kaiming 让前向不爆，loss scaling 让梯度不消失——两者解决 FP16 动态范围的两端；7→ Kaiming 训练的模型容易，因为权重分布更接近 0 均值正态，量化 scale 不会被 outlier 拉偏；8→ 1 层 hidden + 16 个神经元 + tanh，randn 的方差也只让 pre-act 在 ±4 量级，饱和但还能训；几十层 ResNet 同样配置直接死。

---

## 9. 与已有笔记的串联

| 今天的内容 | 关联点 |
|---|---|
| 初始 loss = log(V) | Day 2 §Q9 中提到"bigram 配 lr=50 不炸"的根本原因之一就是 softmax+CE 在 logits=0 时本来就有最自然的初始点；MLP 要靠人工调出来 |
| 修 W2 *= 0.01 | Day 2 §5 三张优惠券里的"softmax+CE 梯度有界"——今天这张券终于真正生效（之前 logits 太分散，梯度被 softmax 推到极端） |
| tanh 饱和 = 死神经元 | Week 3 micrograd `_backward` 闭包里 `tanh.grad = (1 - tanh²) * out.grad`——今天的"饱和率"就是这个 (1-tanh²) 接近 0 的比例统计 |
| Kaiming 数学推导 | 第2周 numpy 网络你用的 Xavier `* √(2/fan_in)` 其实就是 ReLU 版的 Kaiming——今天给它配上"为什么是 2"的理论 |
| FP16 训练初始化敏感 | `autograd_explained.md §5.2`（训练显存 = 参数 + 梯度 + Adam state + 激活）；FP16 把这四块都减半，但 numerical stability 边界也变窄 |
| INT8 量化对权重分布敏感 | `tech_notes/embedding_as_lookup.md`（embedding 是 memory-bound）——量化后 embedding table 从 GB 降到几百 MB，但前提是分布 friendly |
| §6.4 BN 是动态 Kaiming | 明天 Day 5 `tech_notes/batchnorm_inference.md` 直接接今天 §6.4 |

**明天（Day 5）的入口数值**：你今天能跑到"初始 loss=3.3 + 饱和率 10%"，但训练 1000 步后再看激活直方图——可能又有 30%-50% 饱和。这就是 BN 出场的"现象证据"。明天 BN 一上，**全程饱和率应能压在 5% 以下**。

**Day 7 元笔记 §3 伏笔**：今天的初始化代价是 0（推理时和训练时同一份权重，没有额外状态），但**只在初始时刻成立**——这是为什么后续要 BN 这种"运行时维持"的工具，BN 就要付出训推双行为的代价（running_mean/var）。这两段一起串成 `week4_industrial_view.md` 第 4 节"初始化的隐藏角色"。

---

## 10. 完成标准检查清单

- [ ] EP4 第 0:00-1:00 看完，能复述"Karpathy 是怎么从 27 一路诊断到 3.3 的"
- [ ] `src/init.py` 完成，含 `init_last_layer` + `kaiming_init_` + `init_makemore_mlp`，`python -m src.init` 自检通过
- [ ] `src/model.py` 接入新初始化，重新训练初始 loss ≈ 3.3 ± 0.1
- [ ] `src/plot_activations.py` 完成，跑出三档对比图（too_large / default / small），small 档饱和率 < 20%
- [ ] **`tech_notes/init_and_stability.md`** 完成，按 §6.1-§6.5 五节（FP32→FP16→INT8→BN→优惠券串联）
- [ ] `W4_day4_log.md` 记录：
  - 三档初始化对比表（init loss / 饱和率 / dev@1000）
  - 今天最大发现 + 卡壳点
  - 明天 BN 的入口数值：训练 1000 步后激活直方图的"漂移程度"
- [ ] **能口头解释**："为什么 FP16 训练里 Kaiming 初始化不是 nice-to-have 而是 must-have？"
  （提示：FP16 动态范围 ±65504，初始化偏 10× 一层就溢出 → NaN）

---

*笔记生成日期：2026-05-21（W4 Day 4，周四）*
