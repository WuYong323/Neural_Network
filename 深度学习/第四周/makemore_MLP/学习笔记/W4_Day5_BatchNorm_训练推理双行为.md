# Week 4 · Day 5（2026-05-22，周五）：EP4 下半 · BatchNorm 训练/推理双行为 + Fused BN-Conv

> **覆盖任务**（计划 line 1193-1197）：
>
> - [ ] DL：看 EP4 第 1:00-end
> - [ ] DL：手写 BatchNorm1d 类，含 training / eval 双行为
> - [ ] DL：把 BN 插入 MLP，对比训练曲线 with/without BN
> - [ ] DL：**完成 `tech_notes/batchnorm_inference.md`**（含 fused BN-Conv 数学推导）
> - [ ] DL：浏览 PyTorch `torch/nn/modules/batchnorm.py` 源码 `if self.training` 分支
>
> **阅读对象**：你自己——Day 4 已经把"初始 loss=3.3 + 饱和率 10%"调出来了，但**只在第 0 步成立**。今天要解决的问题是："训练 1000 步以后，激活分布会漂移回去吗？怎么让它一直稳？"——这就是 BatchNorm 的舞台。
>
> **本笔记的设计**：每节"现象 → 直觉 → 数学 → 代码 → 工业锚点"五段式。读完应该能：（1）口述 BN 训练/推理为什么必须两套行为；（2）手写出带 running stats 的 BN1d；（3）从数学上推出 fused BN-Conv 的合并公式；（4）解释清楚为什么 TensorRT/ONNX/PyTorch 推理时 BN 算子"完全消失"；（5）把 BN 和 Day 4 的 Kaiming、`autograd_explained.md §5.1` 的 `model.eval()` 串成一条完整因果链。

---

## 0. 学习目标（看完应能回答）

1. Day 4 调好的"初始 loss=3.3 + 饱和率 10%"为什么训练 1000 步后会失效？什么叫"内部协变量偏移"（Internal Covariate Shift）？
2. BN 在训练时到底在做什么？为什么用 batch 内统计而不是全局统计？
3. 为什么推理时**不能**继续用当前 batch 的统计？换种问法：推理时 batch_size=1 用 batch 内 mean/var 会发生什么数学灾难？
4. running_mean / running_var 是什么？为什么用 EMA（指数移动平均）而不是简单平均？momentum=0.1 是大还是小？
5. γ（gamma）和 β（beta）是干什么的？BN 既然要"归一化到 0/1"，为什么又给两个可学习参数把它放缩回去？这不是白做了吗？
6. **Fused BN-Conv：推理时 BN 算子为什么可以"消失"？合并到前面 Conv 的 W 和 b 里的数学公式是什么？**
7. 为什么 Frozen BN 是 fine-tune 大模型的标配？detection / segmentation 任务为什么 batch 一小就必须冻 BN？
8. INT8 量化时为什么必须先 fuse Conv-BN-ReLU？不 fuse 会发生什么？
9. **`model.eval()` 关掉的两个最重要的算子是什么？为什么 BN 排第一**（呼应 `autograd_explained.md §5.1`）？

---

## 1. 为什么需要 BN：Day 4 留下的"漂移问题"

### 1.1 一个比喻：你早上调好了空调 22℃，但屋里有 100 个人陆续进来

```
8:00  屋里 0 人，温度 22℃（Day 4 你调好的初始状态）
9:00  屋里 50 人，体温辐射叠加，温度 28℃（训练 500 步，激活分布右移）
10:00 屋里 100 人，温度 32℃（训练 1000 步，饱和率回到 40%）
```

Day 4 的 Kaiming 像是"早上把空调温度调到 22℃"——**只在 8:00 那一刻是对的**。一旦训练进行，每一层的权重 W 都在变，导致下一层看到的输入分布也在变。等到第 1000 步，你的 hidden pre-activation 可能又跑到方差 4-5，重新进入 tanh 饱和区。

**学术名称**：Internal Covariate Shift（内部协变量偏移）。Sergey Ioffe 和 Christian Szegedy 在 BatchNorm 原始论文（2015）就是用这个词命名问题的。

> 译注："covariate shift"原本是统计学概念——训练分布和测试分布不一致。"internal"是说同样的问题发生在**网络内部层与层之间**：第 5 层早期看到的输入分布，到训练后期已经完全变了。

### 1.2 朴素解法："屋里温度高了就开空调"——但每一步都开

最直觉的想法：每一步 forward 时，把每一层的 pre-activation **强制归一化**到 0 均值、单位方差。

```python
# 朴素版（不带学习参数）
mean = x.mean(0)          # 沿 batch 维算每个 feature 的均值
var = x.var(0)            # 沿 batch 维算每个 feature 的方差
x_normalized = (x - mean) / (var + eps).sqrt()
```

这个"每步都归一化"的操作就是 BN 的核心动作。它把 Day 4 静态的 Kaiming 升级成了**动态的 Kaiming**——每一步都把激活"拉回" 0 均值/单位方差。

> **直觉锚点**：
> ```
> Kaiming 初始化  =  打开门让模型能进屋（只在第 0 步生效）
> BatchNorm       =  屋里装空调，温度永远稳定（每一步生效）
> ```

### 1.3 为什么是 batch 内统计，不是全局统计？

合理的反问："为什么不用整个训练集的 mean/var？那样不是更稳定吗？"

两个原因：

1. **算不动**：训练集有 60000 张图，每步前向都重新算全集统计，等价于训练慢 60000 倍
2. **梯度断了**：mean/var 如果只是"查表得到的常数"，反向传播时 `(x - mean) / std` 这个变换对 x 的梯度就只有 `1/std`——和后面所有层接不上信息流
3. **核心**：用 batch 内统计意味着 mean/var **是 x 本身的函数**——反向传播会"穿过"它们，让 BN 真正参与到 chain rule 里

> 这也是为什么 BN 对 batch size 敏感：batch 太小（< 8），mean/var 估计噪声很大；batch 太大（> 1024），mean/var 接近全集统计，BN 几乎没作用。**主流训练 batch_size 32-256 是有道理的**。

---

## 2. 训练时 BN 在做什么：完整数学

### 2.1 公式（4 行，必须背下来）

设一个 batch 输入 `x` shape `(N, D)`，N 个样本、D 个特征：

```
1) μ_B  = (1/N) Σᵢ xᵢ                              ← 沿 batch 维的均值，shape (D,)
2) σ²_B = (1/N) Σᵢ (xᵢ - μ_B)²                      ← 沿 batch 维的方差，shape (D,)
3) x̂ᵢ   = (xᵢ - μ_B) / √(σ²_B + ε)                  ← 归一化，shape (N, D)
4) yᵢ   = γ · x̂ᵢ + β                                ← 仿射变换，shape (N, D)
```

`ε`（一般 1e-5）是数值稳定项，防止方差为 0 时除零。

### 2.2 γ 和 β 为什么不是"白做了"

新手看到 BN 公式的第一反应：

> 你先 `(x-μ)/σ` 归一化到 N(0,1)，然后又 `γ·x̂+β` 把它放缩回任意分布——这不是白做了一遍吗？

**关键差别**：γ 和 β 是**可学习参数**（参与梯度更新），μ 和 σ 是**统计量**（从数据算出来的）。

```
原始版本：x 直接喂 tanh，分布完全由前面的 W 决定，不可控
归一化后再仿射：x 先被强制拉到 N(0,1)，再让模型"自己学"它需要多大的方差和均值

→ 最坏情况：γ = σ, β = μ，BN 学成了恒等变换（不影响）
→ 最好情况：γ = 0.5, β = 0，BN 把激活压到 ±0.5 区间，完全避开 tanh 饱和
```

> **直觉锚点**：γβ 是给模型一个"反悔权"——"你不一定要严格 0 均值/单位方差，但起点必须是这个标准化形态，你自己学怎么调整"。这是 **expressivity（表达能力）和 stability（稳定性）的双赢**。

### 2.3 一个工业上常见的误解

很多博客把 BN 描述成"减均值除方差"，**忽略了 γβ**。这导致一些代码实现里把 γβ 写死成 `1` 和 `0`，跑出来 loss 不降。

> **工业法则**：任何 BN 实现，γβ 必须是 `nn.Parameter`（参与梯度），不是 buffer（只是缓存）。这是 PyTorch 官方 BN 的设计：γβ 是参数，running_mean/running_var 是 buffer。

---

## 3. 推理时为什么不能用 batch 内统计：除零灾难

### 3.1 一个真实场景

你的模型上线了，部署成 HTTP 服务，每次请求**一个用户的一张图**：

```python
def predict(image):
    x = preprocess(image)        # shape (1, 784)
    logits = model(x)            # batch_size = 1
    return softmax(logits)
```

如果 BN 还按训练逻辑跑：

```python
mean = x.mean(0)   # (1,784) 沿 batch 维取均值 → 还是 x 本身
var  = x.var(0)    # (1,784) 单样本方差 → 全是 0！
x_hat = (x - mean) / (var + eps).sqrt()   # 0 / sqrt(eps) → 数值崩溃
```

**单样本的方差是 0**，归一化后整个张量变成 `0/sqrt(eps)`，等价于一个巨大的常数张量——**整个网络的输出和输入无关**，等于胡乱预测。

### 3.2 为什么 batch_size=8 也不行

你说："那我推理时把请求攒到 batch_size=8 再算？"

问题更深：
- **同一个 batch 里的 8 张图可能来自完全不同的用户/场景**——它们的统计量没有任何"分布意义"
- **同一个用户连续发 2 张图，第 1 张和第 8 张的预测结果不一样**——因为它们在不同 batch 里被不同的 mean/var 归一化了
- **不可复现**：同样的输入，因为 batch 里其他样本不同，输出不同——这在生产环境是灾难

> **核心**：训练时 batch 内统计是"对训练分布的无偏估计"；推理时 batch 是"随机请求堆出来的临时集合"——它**不代表数据分布**。

### 3.3 解法：训练时偷偷记下"全集统计"，推理时用

```
训练阶段：每个 batch 算出 μ_B 和 σ²_B，同时维护两个滚动平均：
  running_mean ← (1 - momentum) × running_mean + momentum × μ_B
  running_var  ← (1 - momentum) × running_var  + momentum × σ²_B

推理阶段：完全不算 batch 统计，直接用 running_mean / running_var
  x_hat = (x - running_mean) / √(running_var + ε)
  y     = γ · x_hat + β
```

momentum=0.1 的含义：每次新 batch 只贡献 10%，旧值保留 90%。这是 **EMA（指数移动平均）**——比简单平均更"重视近期"，因为训练后期的统计才反映真实分布。

> **注意**：PyTorch 的 `momentum` 定义和 SGD 的 momentum 含义**相反**。SGD 的 momentum=0.9 意思是"保留 90% 旧动量"，PyTorch BN 的 momentum=0.1 意思是"用 10% 新值替换旧值"——同名不同物，每次都要查文档。

---

## 4. 手写 BatchNorm1d：完整可运行代码

### 4.1 核心类（`src/batchnorm.py`）

```python
"""手写 BatchNorm1d，含 training / eval 双行为。

运行：python -m src.batchnorm  # 跑自检
"""
from __future__ import annotations

import torch


class BatchNorm1d:
    """对应 PyTorch nn.BatchNorm1d 的 from-scratch 版本。

    输入 shape: (N, D)
    - N: batch size
    - D: feature dimension (per-feature normalization)
    """

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        # 可学习参数：γ 初始为 1，β 初始为 0 → 起步是恒等变换
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # buffer（统计量，不参与梯度）：用 EMA 累积全集统计
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

        self.momentum = momentum
        self.eps = eps
        self.training = True  # 关键状态位

    def parameters(self):
        """只返回可学习参数（γ, β），不包括 buffer。"""
        return [self.gamma, self.beta]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 训练分支：用 batch 内统计 + 更新 running stats
            mean = x.mean(0, keepdim=True)            # (1, D)
            var = x.var(0, keepdim=True, unbiased=False)  # (1, D)，用有偏估计跟 PyTorch 一致

            # 用 torch.no_grad 更新 running stats（这是 buffer，不要梯度）
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze(0)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze(0)
        else:
            # 推理分支：用累积的 running stats
            mean = self.running_mean.unsqueeze(0)     # (1, D)
            var = self.running_var.unsqueeze(0)       # (1, D)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


def _selfcheck() -> None:
    """三步自检：
    1) 训练模式下，输出近似 N(0,1) 然后被 γβ 仿射
    2) 训练若干步后 running stats 接近真实统计
    3) eval 模式下，单样本推理不崩溃
    """
    torch.manual_seed(42)
    bn = BatchNorm1d(dim=4)

    # ① 训练模式
    x = torch.randn(32, 4) * 3.0 + 5.0   # 故意造一个 N(5, 3²) 的输入
    y = bn(x)
    print(f"训练输入  mean={x.mean(0).tolist()}")
    print(f"训练输出  mean={y.mean(0).tolist()}  std={y.std(0).tolist()}")
    print(f"          (应该近似 mean≈0, std≈1，因为 γ=1, β=0)")

    # ② 多步训练后查看 running stats
    for _ in range(200):
        x_batch = torch.randn(32, 4) * 3.0 + 5.0
        bn(x_batch)
    print(f"\n200 步后 running_mean = {bn.running_mean.tolist()}  (应≈ 5)")
    print(f"200 步后 running_var  = {bn.running_var.tolist()}  (应≈ 9)")

    # ③ 切到 eval，单样本推理
    bn.eval()
    x_single = torch.randn(1, 4) * 3.0 + 5.0
    y_single = bn(x_single)
    print(f"\neval 单样本输入 = {x_single.tolist()}")
    print(f"eval 单样本输出 = {y_single.tolist()}  (不应该是 inf/nan)")

    assert not torch.isnan(y_single).any(), "单样本推理出了 NaN！"
    assert not torch.isinf(y_single).any(), "单样本推理出了 Inf！"
    print("\n✅ 自检全部通过")


if __name__ == "__main__":
    _selfcheck()
```

### 4.2 预期输出

```
训练输入  mean=[5.0xx, 5.0xx, 5.0xx, 5.0xx]
训练输出  mean=[0.000, 0.000, 0.000, 0.000]   ← 被归一化到 0
          std=[0.984, 0.984, 0.984, 0.984]    ← 接近 1（不是严格 1，因为是有偏估计 + ε）

200 步后 running_mean = [4.9xx, 5.0xx, 5.0xx, 4.9xx]   ← 收敛到真实均值 5
200 步后 running_var  = [8.9xx, 9.1xx, 8.8xx, 9.0xx]   ← 收敛到真实方差 9

eval 单样本输入 = [[6.7, 3.2, 5.1, 4.8]]
eval 单样本输出 = [[0.57, -0.59, 0.04, -0.07]]   ← 用 running stats，没崩
✅ 自检全部通过
```

> **关键观察**：训练模式下 `y.std()` 接近但不严格等于 1——这是因为 PyTorch 默认用**有偏估计**（除以 N），以及 ε 的稳定项。如果你用 `unbiased=True`（除以 N-1），数字会更接近 1，但会和 PyTorch 官方实现对不上。

### 4.3 把 BN 插入 MakemoreMLP

修改 `src/model.py`（伪代码示意）：

```python
class MakemoreMLP:
    def __init__(self, ..., use_bn: bool = False):
        # ... 原有的 C, W1, b1, W2, b2
        self.use_bn = use_bn
        if use_bn:
            self.bn = BatchNorm1d(hidden_size)

    def __call__(self, X):
        emb = self.C[X]
        flat = emb.view(emb.size(0), -1)
        pre = flat @ self.W1 + self.b1          # (N, H)
        if self.use_bn:
            pre = self.bn(pre)                  # ← 插在 tanh 之前
        h = torch.tanh(pre)
        logits = h @ self.W2 + self.b2
        return logits

    def parameters(self):
        ps = [self.C, self.W1, self.b1, self.W2, self.b2]
        if self.use_bn:
            ps += self.bn.parameters()           # 加入 γ, β
        return ps

    def eval(self):
        if self.use_bn:
            self.bn.eval()

    def train(self):
        if self.use_bn:
            self.bn.train()
```

> **细节 1**：BN 插在 `tanh` 之前，不是之后。原始 BN 论文是这样设计的，因为 BN 要解决的是"喂给非线性激活前的分布漂移"。
>
> **细节 2**：因为 BN 已经会加 β（偏置），`b1` 实际上是冗余的——加进去的常数会被 BN 减掉。工业代码里有 BN 的 Conv/Linear 一般会设 `bias=False`，省一点参数。这里教学版保留 `b1` 不影响正确性。

### 4.4 with BN vs without BN 对比实验

```python
# 跑两次训练，记录 loss 曲线
torch.manual_seed(42)
m1 = MakemoreMLP(..., use_bn=False)  # baseline
m2 = MakemoreMLP(..., use_bn=True)   # with BN

# 训练 2000 步，每 100 步记一个 dev loss，画图
```

**预期现象**（保存到 `logs/bn_compare.png`）：

| 步数 | baseline dev loss | with BN dev loss | 说明 |
|---|---|---|---|
| 0 | 3.30 | 3.30 | 起点一致（Day 4 调好的初始 loss） |
| 200 | 2.85 | 2.55 | BN 让前期收敛快很多 |
| 1000 | 2.40 | 2.18 | baseline 开始放缓，BN 仍在降 |
| 2000 | 2.30 | 2.10 | BN 终值更低（敏感于 lr，可能 ±0.05） |

并且：
- 用 `plot_activations.py` 跑 1000 步后的激活：baseline 饱和率回到 35%-40%，with BN 全程稳定在 5%-10%
- BN 允许用更大的 lr（baseline 最优 0.1，with BN 可以 0.3-0.5）

---

## 5. Fused BN-Conv：推理优化的"消失术"

> 这一节是今天工业含金量最高的部分，写到 `tech_notes/batchnorm_inference.md`。

### 5.1 一个比喻：两个连续的"按比例缩放"可以合并成一个

```
你有一杯 100ml 的水：
  动作 A：先稀释一半 → 50ml 水 + 50ml 稀释液（总 100ml）
  动作 B：再倒掉 30%  → 70ml

合并：直接"取 70% 的稀释一半" → 等价的单次操作
```

数学上：两个线性变换的复合还是线性变换——**可以合并**。

### 5.2 BN 在推理时是什么？一个仿射变换

eval 模式下，BN 完整公式：

```
y = γ · (x - μ) / √(σ² + ε) + β
  = γ/√(σ²+ε) · x  +  (β - γμ/√(σ²+ε))
  = a · x + b              ← 这就是 y = ax + b 的仿射！
```

其中：
```
a = γ / √(σ² + ε)              ← 标量系数（per-channel）
b = β - γμ / √(σ² + ε)         ← 偏置项（per-channel）
```

> 推理时 BN 已经**退化成最简单的仿射变换**——没有训练分支，没有 batch 统计，就是 `ax+b`。

### 5.3 前面是 Conv（或 Linear）：又一个仿射

```
Conv 输出：x = W * input + b_conv          （W 是 4D 卷积核，但每个 output channel 在 BN 看来是个 scalar）
```

对单个 output channel c：

```
x_c = W_c * input + b_conv_c
```

### 5.4 复合两个仿射：BN 算子消失

把 Conv 输出代入 BN：

```
y_c = a_c · x_c + b_c
    = a_c · (W_c * input + b_conv_c) + b_c
    = (a_c · W_c) * input + (a_c · b_conv_c + b_c)
    = W'_c        * input + b'_c
```

其中：

```
W'_c = a_c · W_c = γ_c · W_c / √(σ²_c + ε)
b'_c = a_c · b_conv_c + b_c = γ_c · (b_conv_c - μ_c) / √(σ²_c + ε) + β_c
```

**这就是 Fused BN-Conv 的核心公式**。

> **效果**：原本 Conv→BN 两个算子两次显存读写、两次计算；fuse 之后**变成一个新的 Conv**（W' 和 b'），BN 在推理图里**完全消失**。

### 5.5 工业意义

| 维度 | 不 fuse | fuse 后 |
|---|---|---|
| 算子数 | 2（Conv + BN） | 1（Conv） |
| 显存读写 | 每次 BN 都要读写一次特征图 | 省掉 BN 那次 |
| 推理延迟 | baseline | 减少 5%-15%（取决于网络） |
| 量化友好 | BN 在中间会破坏量化范围 | 量化只针对 Conv 输出 |
| 工具支持 | PyTorch 默认不 fuse | TensorRT/ONNX/TorchScript/MLIR 自动 fuse |

### 5.6 实际代码：手动验证 fuse 等价性

```python
"""验证 Conv-BN fuse 数学等价（教学版，用 Linear 代替 Conv 简化）"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)
N, D_in, D_out = 8, 16, 32

# 原始：Linear + BN
W = torch.randn(D_in, D_out)
b_conv = torch.randn(D_out)
gamma = torch.randn(D_out)
beta = torch.randn(D_out)
mu = torch.randn(D_out)             # 假装是训练好的 running_mean
sigma2 = torch.rand(D_out) + 0.5    # 训练好的 running_var
eps = 1e-5

x = torch.randn(N, D_in)

# 路径 A：Linear → BN（推理模式）
out_linear = x @ W + b_conv                          # (N, D_out)
out_bn = gamma * (out_linear - mu) / torch.sqrt(sigma2 + eps) + beta

# 路径 B：Fused Linear（W' 和 b'）
scale = gamma / torch.sqrt(sigma2 + eps)             # a_c
W_fused = W * scale                                  # (D_in, D_out)，每个 column 乘 scale_c
b_fused = scale * (b_conv - mu) + beta               # (D_out,)
out_fused = x @ W_fused + b_fused

# 对比
max_diff = (out_bn - out_fused).abs().max().item()
print(f"两条路径最大差异: {max_diff:.2e}")           # 应该 ≈ 1e-6 (浮点误差)
assert max_diff < 1e-5
print("✅ Fused 等价")
```

> 真实的 Conv-BN fuse 多一步"把 W 的输出维度对齐到 BN 的 per-channel"——但数学完全一样。

### 5.7 PyTorch 官方工具

PyTorch 自带 `torch.nn.utils.fusion.fuse_conv_bn_eval`：

```python
import torch.nn.utils.fusion as fusion

model.eval()
fused_conv = fusion.fuse_conv_bn_eval(conv_layer, bn_layer)
# fused_conv 是一个新的 Conv，BN 算子在推理图里不再存在
```

ONNX/TensorRT 在导出/编译时会**自动**做这件事——你不需要手动调用。

---

## 6. Frozen BN：fine-tune 的标配

### 6.1 场景：在 ImageNet 上预训练的 ResNet 改做你的 100 张图像分类任务

你的 batch_size 可能只有 8（GPU 内存有限），如果 BN 还在 training 模式：

- 每个 batch 8 个样本，mean/var 估计噪声极大
- running_mean/running_var 被你的小数据**污染**——原本在 ImageNet 上学到的稳定统计被几百步训练就破坏了
- 推理时 BN 用的是被污染的统计，性能崩盘

### 6.2 解决：Frozen BN

```python
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()                          # 切到 eval 模式
            for p in module.parameters():
                p.requires_grad = False            # γ, β 也冻住
```

- `module.eval()`：BN 使用 running_mean/running_var（来自预训练），不再更新
- `requires_grad = False`：γβ 不参与梯度，不被你的小数据 overfit

### 6.3 工业现象

- **目标检测（Detection）**：Mask R-CNN / Faster R-CNN 几乎全部 frozen BN，因为 detection 训练 batch 经常 1-2 张（高分辨率图占显存）
- **语义分割（Segmentation）**：同样问题，SyncBN（跨多卡同步统计）或 frozen BN 是必选
- **大模型 fine-tune**：LLaMA / GPT 这种没有 BN 的模型用 LayerNorm（下周可能会看），但所有 norm 类算子在 fine-tune 时一般都设 `eval`

> **工业法则**："小 batch + BN training mode" 是 fine-tune 三大死法之一（另外两个是 lr 没调小、梯度爆炸没 clip）。

---

## 7. INT8 量化为什么必须先 fuse Conv-BN-ReLU

### 7.1 量化的基本动作

把 FP32 的张量映射到 INT8（范围 [-128, 127]）：

```
x_int8 = round(x_fp32 / scale)
scale = max(|x_fp32|) / 127
```

一次量化 → 反量化的损失：约等于 `scale/2` 的量级。

### 7.2 不 fuse：连续三次量化损失累加

```
input → [Conv (FP32→INT8→FP32)] → [BN (FP32→INT8→FP32)] → [ReLU (FP32→INT8→FP32)] → output
        ↑                          ↑                       ↑
        损失 1                      损失 2                    损失 3
```

三次损失累加，模型精度可能掉 3%-5%。

### 7.3 fuse 后：一次量化

```
input → [Fused Conv-BN-ReLU (FP32→INT8→FP32)] → output
                                  ↑
                                 仅 1 次损失
```

工业实测：ResNet50 量化到 INT8：
- 不 fuse：top-1 acc 掉 4-5%
- fuse 后：top-1 acc 掉 0.5-1%

**fuse 不仅快，还更准**——这是 Conv-BN-ReLU 几乎成了"工业标配三连"的根本原因。

> **TensorRT / ONNX Runtime / TVM** 等推理引擎都会自动识别 Conv→BN→ReLU 模式并 fuse 成一个算子。**模型导出时 BN 必须在 eval 模式**，否则导出图里还带着 batch 统计的训练逻辑，引擎认不出 fuse 模式。

---

## 8. 浏览 PyTorch 源码：`torch/nn/modules/batchnorm.py`

> 任务清单第 5 条：浏览源码 `if self.training` 分支。读源码不需要全看懂，只看**核心控制流**。

### 8.1 关键代码位置

PyTorch 2.x 的源码大致结构（`torch/nn/modules/batchnorm.py`）：

```python
class _BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, ...):
        # ...
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = Parameter(torch.empty(num_features))   # ← γ
        self.bias = Parameter(torch.empty(num_features))     # ← β

    def forward(self, input):
        # 关键控制流
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,                # ← 决定走训练还是推理路径
            exponential_average_factor,
            self.eps,
        )
```

### 8.2 关注三个事实

1. **`weight` 和 `bias` 是 Parameter**（γβ 参与梯度），**`running_mean/var` 是 buffer**（不参与梯度，但要随模型 save/load）。这和我们 §4 的实现一致。
2. **`if self.training` 控制 `bn_training` 标志**，最终传给底层 `F.batch_norm`（C++ 实现）。这个标志决定底层走哪条路径。
3. **`track_running_stats`**：这是一个额外开关，如果设 `False`，BN 即使在 eval 模式也用 batch 内统计（极少使用，但 SyncBN 等场景会用到）。

### 8.3 跟我们手写版的差异

| 特性 | 手写版（§4.1） | PyTorch 官方 |
|---|---|---|
| 训练/推理双分支 | ✅ | ✅ |
| running stats EMA | ✅ | ✅（还有更精细的 `track_running_stats`/`exponential_average_factor`） |
| γβ 参数 | ✅ | ✅ |
| C++ 加速底层 | ❌（纯 Python） | ✅（`F.batch_norm` 走 ATen） |
| 多卡同步（SyncBN） | ❌ | ✅（`SyncBatchNorm` 子类，用于分布式训练） |
| 通道顺序（NCHW vs NHWC） | 不支持 | ✅ |

### 8.4 SyncBN：分布式训练的扩展（了解即可）

多卡训练时，每张卡只看到 batch 的一部分。如果各自算 mean/var，**统计精度不够**，BN 效果变差。**SyncBN 让所有卡先聚合一次 mean/var，再做归一化**——本质是分布式版的 BN。

> 这是 Detection / Segmentation / LLM 训练的标配。下周看 ResNet 时如果用到多卡训练，就会用 SyncBN。

---

## 9. 与已有笔记的串联

| 今天的内容 | 关联点 |
|---|---|
| BN 训练时归一化激活 | Day 4 §1.3（"BN 是动态 Kaiming"）——今天给那句话配上了完整数学和代码 |
| running stats EMA | 类似 SGD Momentum 的"动态平均"概念，但更新公式相反方向（Day 6 Course 2 Week 2 会展开） |
| 推理时 BN 用 running stats | `autograd_explained.md §5.1`（`model.eval()` 关掉的最重要算子之一就是 BN）——今天看清了"关掉"到底关了什么 |
| Fused BN-Conv | `embedding_as_lookup.md` 是"训练 vs 推理算子形态变化"的第一例（embedding 在推理时是 lookup），今天 fused BN 是第二例（BN 在推理时彻底消失）——**形成 AI Infra 主线"算子在训练推理时长不同样"的第二个工业证据** |
| Frozen BN | Day 7 元笔记 §3 "推理优化的隐藏维度"——fine-tune 时 BN 必须冻，这是大多数人踩坑的第一个 pitfall |
| INT8 量化前先 fuse | `init_and_stability.md §6.3`（INT8 对权重分布敏感）——今天补上"为什么量化前必须 fuse"，把 §6.3 那张故事讲全 |
| γβ 让 BN 可恢复成恒等 | 第 3 周 micrograd 里 `_backward` 闭包的设计哲学："给模型自由度，让它自己学需要的变换"——γβ 是这个哲学的具体例子 |

**回顾 Day 2 §Q9 三张优惠券表 的更新版**：

| 优惠券 | bigram | Day 2 MLP | Day 4 (Kaiming) | **Day 5 (+BN)** |
|---|---|---|---|---|
| 全局凸 | ✓ | ✗ | ✗ | ✗（结构问题，BN 救不了） |
| softmax+CE 梯度有界 | ✓ | ✓ | ✓ | ✓ |
| 稀疏更新 | ✓ | ✗ | ✗ | ✗ |
| 初始 loss 合理 | ✓ | ✗ | ✅ | ✅ |
| 激活无饱和 | n/a | ✗ | ✅（仅初始） | ✅（**全程**） |
| 训练中分布稳定 | n/a | ✗ | ✗ | ✅（BN 主战场） |

**这张表更新到第 5 行 "训练中分布稳定" → ✅，是今天最大的认知收获**。明天 Day 6 是 Course 2 收尾（Adam state 显存），优惠券表暂时不会再变。

---

## 10. 自测题（合上文档默答）

1. 为什么 BN 训练用 `var(0)` 而不是 `var(1)`？换句话说，为什么沿 batch 维归一化，不是沿 feature 维？
2. momentum=0.1 和 momentum=0.9 哪个让 running stats 收敛更快？为什么？
3. 训练时 `model.train()` 跟在 forward 之前调，eval 时 `model.eval()` 也必须调。如果忘了调 `eval()` 就推理，会发生什么？（具体到 BN）
4. γβ 既然可学习，会不会被学成 `γ = 1, β = 0`（即恒等变换）？这种情况下 BN 等于啥都没干，模型还训得好吗？
5. **Fused BN-Conv 后，新的 Conv 权重 W' 和原 W 的 shape 一样吗？为什么？**
6. INT8 量化时，如果**不** fuse Conv-BN-ReLU，量化误差为什么不只是"3 倍单算子误差"，而可能更糟？
7. SyncBN 跨多卡同步统计——同步的是 batch 内统计 μ_B / σ²_B，还是 running stats？为什么？
8. **batch_size=1 时，能不能训练带 BN 的网络？如果不能，有哪两种工业替代方案？**
9. 为什么 LLM（GPT/LLaMA）用 LayerNorm 不用 BatchNorm？提示：LLM 训练时 batch 里每个样本是不等长序列。
10. Day 4 你的 dev loss 调到 2.2，今天加 BN 后降到 2.1——这 0.1 的提升主要来自"训得更快"还是"训得更好"？怎么判断？

> 参考答案位置（合上文档前先答）：
> 1→ var(0) 沿 batch 维，每个 feature 独立归一化，符合 "feature 间相互独立学习的尺度"假设；var(1) 沿 feature 维会把不同语义的特征混在一起（loss 和 acc 是两个不同维度，混着归一化没意义）；
> 2→ momentum=0.9 收敛快（每次新 batch 替换 90% 旧值），但抖动大；0.1 慢但稳，是 PyTorch 默认；
> 3→ 用当前 batch 统计而非 running stats，等价于"每个推理 batch 用一份临时不同的归一化"——单样本会直接除零，多样本结果不可复现；
> 4→ 会，理论上最优解可以是 γ=σ, β=μ（学成恒等）；但实践中通常学到 γ<σ 让激活方差变小，避开非线性饱和——所以"能学成恒等"是表达力保证，不是实际行为；
> 5→ shape 完全一样。fuse 只是"把 W 的每个 output channel 乘一个标量 a_c"，矩阵形状不变；
> 6→ 三次量化损失会**累积**到一个值上，而不是独立分布。每次 round 都把误差信号放大一点，三次后误差可能不是 3 倍而是 5-10 倍——这是非线性累积，不是简单相加；
> 7→ 同步 batch 内统计 μ_B / σ²_B。running stats 是各卡分别维护的（最后取平均或主卡的），同步实时 batch 统计才能让 BN 等价于"在一个大 batch 上算"；
> 8→ 不能（var=0 除零）。替代：（a）GroupNorm：沿 channel group 归一化，不依赖 batch；（b）LayerNorm：沿 feature 维归一化。Transformer/LLM 用 LN 就是这个理由；
> 9→ 序列变长 + 大 batch 不切实际（OOM）+ LM 任务每个 token 独立预测——LN 沿 hidden 维归一化，与 batch / seq_len 无关；
> 10→ 看"前 500 步 loss"——baseline 在 500 步 ≈ 2.85，with BN 在 500 步 ≈ 2.55。BN 主要是"训得更快"，dev loss 终值 ±0.05 的差距说明"更好"的贡献小；真实工业上更看重的是"收敛速度 × wall-clock time 节省"——这是 BN 的核心商业价值，不是 acc 那 0.1。

---

## 11. 完成标准检查清单

- [ ] EP4 第 1:00-end 看完，能复述"Karpathy 是怎么从手写归一化一路演进到带 γβ 和 running stats 的完整 BN 的"
- [ ] `src/batchnorm.py` 完成，含 `BatchNorm1d` 类 + `_selfcheck()`，`python -m src.batchnorm` 自检通过
- [ ] `src/model.py` 接入 `use_bn=True` 选项，能切换两种模式重新训练
- [ ] 跑出 with BN vs without BN 训练曲线对比图，存到 `logs/bn_compare.png`，dev loss 终值差距 ≥ 0.05
- [ ] 跑 `plot_activations.py`（with BN 模式），确认训练 1000 步后饱和率仍 < 10%
- [ ] **`tech_notes/batchnorm_inference.md`** 完成，按 §1-§5 五节（训练/推理双行为 + 必须双行为的原因 + Fused BN-Conv 数学推导 + Frozen BN + 与 `autograd_explained.md §5.1` 的连接）
- [ ] 浏览 PyTorch 源码 `torch/nn/modules/batchnorm.py`，至少能说出三个观察点（§8.2）
- [ ] `W4_day5_log.md` 记录：
  - with/without BN 对比表（前 500 步收敛速度 + 终值 dev loss + 饱和率）
  - Fused BN 数学推导自验证代码的输出截图
  - 今天最大发现 + 卡壳点
  - 明天 Day 6 的入口数值：你目前 MLP 的参数量 + Adam state 显存预估
- [ ] **能脱稿口述**：
  - "BN 训练/推理为什么必须两套行为"（核心：batch_size=1 单样本方差为 0）
  - "Fused BN-Conv 推理时怎么变成零开销"（核心：两个仿射变换合并）
  - "Frozen BN 为什么是 fine-tune 标配"（核心：小 batch 污染 running stats）

---

*笔记生成日期：2026-05-22（W4 Day 5，周五）*
*下一篇：W4 Day 6 — Andrew Ng Course 2 收尾 + Adam state 显存代价*
