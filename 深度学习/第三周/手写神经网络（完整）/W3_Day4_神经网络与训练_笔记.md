# W3 Day4 学习笔记：从自动微分到完整神经网络——Neuron / Layer / MLP + make_moons

> 日期：2026-05-14（周四）
> 前置：你已完成 Day2，`Value` 类能正向构图，也能反向传梯度（三个 demo 全过）
> 视频范围：Karpathy "Neural Networks: Zero to Hero" EP1 第 2:25 – 2:25 结束
> 今日核心：**把昨天做好的 `Value` 当乐高积木，拼出第一个能学习的神经网络**——从单神经元到 MLP，再拿 `make_moons` 训到 loss < 0.1，最后画决策边界、对比 numpy 版本的性能。

---

## 0. 先看清楚今天要拿到什么

完成这份笔记 + 代码后，你应该能：

1. 用一句话回答：**一个"神经元"本质上是什么？一层网络和一个 MLP 又是什么？为什么它们能复用 `Value` 的反向传播？**
2. 手画一个 `[2, 4, 1]` 的 MLP 结构图，数出它有多少个参数（**权重 + 偏置**），并和 `len(model.parameters())` 对上。
3. 不看代码写出训练循环的 4 个关键步骤（forward → loss → backward → update），并解释为什么每步之前要 `zero_grad`。
4. 把 `make_moons` 的输出 `X, y` 描述清楚：形状、含义、为什么说它是"非线性可分"。
5. 跑通 `train_moons.py`，在训练集上做到 **loss < 0.1**，保存决策边界图 `logs/decision_boundary.png`。
6. 在当日 log 里用**数量级估算 + 代码依据**解释：**为什么 micrograd 比 numpy 网络慢大约 100 倍？** 不要只说"因为是 Python" 。

如果第 3、6 条做不到，不要进 Day5——训练循环和性能直觉是后面一切 DL 工程的地基。

---

## 1. 从 Value 到神经元：数学层面的桥

### 1.1 昨天你在干什么？今天要干什么？

Day2 你让 `Value` 会了两件事：
- 正向：`c = a*b + d` 自动建图
- 反向：`L.backward()` 把梯度一路传回 `a, b, d`

但那只是**在一个标量表达式里求梯度**。真正的神经网络里，一个预测值 $\hat{y}$ 是由**成百上千个参数**一起算出来的——而且这些参数会被**反复复用**（同一个权重服务多个样本）。

今天你要证明一件事：**只要正向表达式能用 `Value` 写出来，反向传播就白送**——不管表达式里套了多少层、有多少参数。

### 1.2 单神经元的数学长什么样

一个神经元接收 $n$ 个输入 $x_1, x_2, \dots, x_n$，内部持有 $n$ 个权重 $w_1, \dots, w_n$ 和一个偏置 $b$，输出：

$$
y = \phi\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

其中 $\phi$ 是非线性激活函数（今天用 `tanh`）。

**注意看：**
- $w_i$ 和 $b$ 是**参数**（要被优化的）
- $x_i$ 是**输入**（来自上一层或数据）
- 整个表达式只用到了 `+ * tanh`——Day2 你全实现过了

所以：**把 $w_i, b$ 都声明成 `Value` 对象，直接用 `+ * tanh` 算出 $y$，`y.backward()` 就会把梯度传回每个 $w_i$ 和 $b$**。这是全部的魔法。

### 1.3 为什么要激活函数？

去掉 $\phi$，神经元变成 $y = \sum w_i x_i + b$——一个线性函数。再多层**线性函数的复合还是线性函数**（$A(Bx) = ABx$），整个网络就坍缩成一个单层的线性模型，连"异或"都学不会。

`tanh` 把输出掰成 S 形，引入非线性，**多层堆叠后才能逼近任意复杂的决策边界**。今天的 `make_moons` 就是故意设计成线性不可分的——为的就是让你亲眼看到非线性的必要性。

### 1.4 三层抽象：Neuron → Layer → MLP

| 抽象 | 是什么 | 参数数量 |
|---|---|---|
| **Neuron** | 一个加权求和 + 激活，**输出 1 个标量** | `n_in + 1`（$n$ 个权重 + 1 个偏置） |
| **Layer** | 并列一堆 Neuron，**共享同一个输入向量**，输出一个向量 | `n_out × (n_in + 1)` |
| **MLP** | 多个 Layer 串联，**上一层的输出当下一层的输入** | 各层之和 |

记住这三件事，你就能徒手数出 `MLP([2, 4, 4, 1])` 有多少个参数：

- Layer1：2→4，每个 neuron 有 `2+1=3` 个参数，共 `4*3=12`
- Layer2：4→4，每个 neuron 有 `4+1=5` 个参数，共 `4*5=20`
- Layer3：4→1，每个 neuron 有 `4+1=5` 个参数，共 `1*5=5`
- 合计 **37 个参数**

训练的时候，`model.parameters()` 返回的就是这 37 个 `Value`，每次 `backward()` 后全部拿到梯度。

---

## 2. 目录结构与文件分工

今天在 `week3_karpathy/micrograd_redo/` 下重新开一个干净目录（`_redo` 是"再做一遍、巩固"的意思——和 Day2 的 `micrograd_follow` 平行），结构：

```
week3_karpathy/
└── micrograd_redo/
    ├── engine.py          # Value 类（从 Day2 搬来，补两个方法）
    ├── nn.py              # 今天的主角：Neuron / Layer / MLP
    ├── train_moons.py     # 今天的训练脚本
    └── logs/
        ├── decision_boundary.png   # 今天要产出的图
        └── W3_day4_log.md          # 今天的训练日志 + 性能分析
```

**为什么要拆成三个文件？**
- `engine.py` 管"怎么求导"（底层）
- `nn.py` 管"怎么组装网络"（中层）
- `train_moons.py` 管"怎么跑一次实验"（上层）

这是 PyTorch 的分层思路：`torch.autograd` ≈ `engine.py`，`torch.nn` ≈ `nn.py`，你的训练脚本 ≈ `train_moons.py`。今天亲手做一遍分层，以后看 PyTorch 代码会非常舒服。

---

## 3. 先给 engine.py 打两个补丁

Day2 的 `Value` 只有 `+ * tanh`。今天为了写 MLP + MSE，要补两样：

### 3.1 补丁 A：`__pow__`（计算 `(y_pred - y_true)**2`）

局部导数：若 $f = x^k$（$k$ 为常数），则 $\dfrac{df}{dx} = k x^{k-1}$。

```python
def __pow__(self, other):
    # 只支持常数指数（int / float），不支持 Value ** Value
    assert isinstance(other, (int, float)), "只允许 int/float 做指数"
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out
```

### 3.2 补丁 B：`relu`（备用激活，Day5 会用）

局部导数：`1 if x > 0 else 0`。

```python
def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
```

> 今天只需要 `tanh`，`relu` 先写上占位，Day5 会切换对比。

### 3.3 还要补一个 `__truediv__`（可选，训练里用不到但推荐）

```python
def __truediv__(self, other):   # a / b  等价于  a * b**-1
    return self * other ** -1
```

把这三块粘进 Day2 那份 `Value` 类里，保存为 `micrograd_redo/engine.py`。**原来的 3 个 demo 仍然应该能过**——这是你确认补丁没破坏旧功能的最快办法。

---

## 4. 核心代码：`micrograd_redo/nn.py`

下面是完整的 `nn.py`，直接粘贴运行即可（依赖 `engine.py`）。**每一段都配注释，读的时候把它当"每行为什么要这样写"的答疑对照**。

```python
"""
W3 Day4: 在 Value 之上搭神经网络。
三层抽象：Neuron（单元）→ Layer（并列一堆 Neuron）→ MLP（串联一堆 Layer）。
整个文件没有一个地方"手写梯度"——全部交给 engine.py 的 Value.backward()。
"""

import random
from engine import Value


# ======================================================================
# 抽象 0：所有网络模块的祖宗
# ======================================================================
class Module:
    """类比 torch.nn.Module，只提供两件事：zero_grad + parameters。"""

    def zero_grad(self):
        """把所有参数的梯度清零。训练循环每一轮开头必须调用。"""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """子类必须实现：返回自己持有的所有 Value 参数（weights + biases）。"""
        return []


# ======================================================================
# 抽象 1：单个神经元
# ======================================================================
class Neuron(Module):
    """
    y = phi(  sum_i(w_i * x_i) + b  )

    - n_in:    输入维度
    - nonlin:  True 时套一层 tanh；False 时输出纯线性（最后一层常用）
    """

    def __init__(self, n_in, nonlin=True):
        # 权重初始化：均匀分布 [-1, 1]
        # Karpathy 视频里用的就是这个简单版；Day5 会换成 Xavier/He
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)               # 偏置从 0 起步是常见做法
        self.nonlin = nonlin

    def __call__(self, x):
        """前向：x 是一个长度为 n_in 的 Value 列表（或能被加法/乘法吸收的 python 数）。"""
        # sum + 初始值 self.b 避免从 0 起累加产生多余节点
        # zip(self.w, x) 把第 i 个权重配到第 i 个输入上
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        tag = 'tanh' if self.nonlin else 'linear'
        return f"{tag}Neuron({len(self.w)})"


# ======================================================================
# 抽象 2：一层（多个 Neuron 并列）
# ======================================================================
class Layer(Module):
    """同一个输入喂给 n_out 个 Neuron，得到长度 n_out 的输出列表。"""

    def __init__(self, n_in, n_out, nonlin=True):
        self.neurons = [Neuron(n_in, nonlin=nonlin) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # 单输出的层直接把 list 拆成一个 Value，省去调用端 outs[0] 的繁琐
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer[{', '.join(str(n) for n in self.neurons)}]"


# ======================================================================
# 抽象 3：多层感知机
# ======================================================================
class MLP(Module):
    """
    sizes: [n_in, h1, h2, ..., n_out]
    例：MLP(2, [4, 4, 1]) 表示 2 维输入，两个 4 维隐藏层，1 维输出
    最后一层默认线性（nonlin=False），便于回归；分类时外层再接 sigmoid 即可。
    """

    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(n_outs) - 1))
            for i in range(len(n_outs))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"


# ======================================================================
# 自测：结构 + 参数个数
# ======================================================================
if __name__ == "__main__":
    random.seed(0)

    model = MLP(2, [4, 4, 1])
    print(model)
    print(f"参数总数: {len(model.parameters())}")
    # 手算：Layer1 4*(2+1)=12，Layer2 4*(4+1)=20，Layer3 1*(4+1)=5，合计 37
    assert len(model.parameters()) == 37, "参数数目应为 37，请检查 Layer/Neuron"

    # 单样本前向
    x = [Value(1.0), Value(-2.0)]
    y = model(x)
    print(f"y = {y}")
    assert hasattr(y, 'backward'), "单输出 MLP 的输出应是 Value"
    print("OK nn.py 自测通过")
```

### 4.1 逐段复盘（这是今天最重要的一节，必看）

- **`Module` 基类**：只管 `zero_grad` 和 `parameters`——这两件事所有模块都要做。PyTorch 的 `nn.Module` 做的事多得多（`.to(device)`、钩子、缓冲区……），但核心还是这两个。**你今天手搓一次，以后看 PyTorch 源码就不会慌**。

- **`Neuron.__call__` 里的 `sum(..., self.b)`**：Python 内置 `sum` 的第二个参数是"起始值"。这里用 `self.b` 当起始值，等价于 `self.b + w1*x1 + w2*x2 + …`。写成 `sum(...) + self.b` 功能一样，但会多一个中间 Value 节点；**能合并就合并，减少图上的节点数 = 反向传播少走几步**。

- **`nonlin=False` 只在最后一层**：如果最后一层也套 tanh，输出被压进 (-1, 1) 区间，回归任务就废了。分类任务最后一层一般也不套激活，而是把 logits 交给外层的 sigmoid/softmax。**今天因为 make_moons 标签是 -1/+1，tanh 输出正好落在同样区间，所以最后一层是不是线性影响不大**——但养成"最后一层线性"的习惯对后面一切任务都对。

- **`parameters()` 的两层推导**：`[p for layer in ... for p in layer.parameters()]`——这种"嵌套推导"在 Python 里等价于"外 for 先，内 for 后"。不熟的话就拆成两层循环写，功能一样。

---

## 5. 数据集：`make_moons` 到底是什么？

### 5.1 一句话描述

`sklearn.datasets.make_moons` 生成**两个交错的半月形**点云，一个标记为 0，另一个标记为 1。**用一条直线切不开**——所以它是验证"非线性网络是否真的能学到非线性边界"的经典玩具数据集。

### 5.2 形状和含义

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.1, random_state=1)
# X.shape == (100, 2)     每行一个样本，两列是 (x1, x2) 平面坐标
# y.shape == (100,)       每个样本的类别，取值 {0, 1}
```

训练前的标准操作：**把 y 从 {0, 1} 映射到 {-1, +1}**，这样就可以直接用 MSE 损失 + tanh/线性输出，无需额外的 sigmoid。

```python
y = y * 2 - 1      # 0 -> -1, 1 -> +1
```

### 5.3 为什么用 MSE 而不是交叉熵？

严格说分类任务更适合交叉熵，但 Karpathy 教学版里为了 **micrograd 实现简单**，用的是 MSE（或 hinge loss）：

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

你已经有 `+ - * ** tanh`，MSE 写一行就能搭出来；交叉熵需要 `log` 和 `exp`，Day5 再补。**不用纠结，今天的目标是跑通，不是追求最优解**。

---

## 6. 训练脚本：`micrograd_redo/train_moons.py`

```python
"""
W3 Day4 训练脚本：用 MLP(2, [16, 16, 1]) 拟合 make_moons，目标 loss < 0.1。
训练完保存决策边界图到 logs/decision_boundary.png。
"""

import os
import random
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 无需弹窗，直接存图
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from engine import Value
from nn import MLP


# ----------------------------------------------------------------------
# 0. 复现性：固定随机种子
# ----------------------------------------------------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# 1. 数据
# ----------------------------------------------------------------------
X, y = make_moons(n_samples=100, noise=0.1, random_state=SEED)
y = y * 2 - 1          # {0,1} -> {-1,+1}
# X shape: (100, 2),  y shape: (100,)

# ----------------------------------------------------------------------
# 2. 模型
# ----------------------------------------------------------------------
model = MLP(2, [16, 16, 1])
print(model)
print(f"参数总数: {len(model.parameters())}")

# ----------------------------------------------------------------------
# 3. 损失函数：MSE + L2 正则
# ----------------------------------------------------------------------
def loss_fn(model, X, y, alpha=1e-4):
    """
    - MSE：对每个样本 (y_pred - y_true)^2，取平均
    - L2 正则：对所有参数的平方和乘 alpha，避免权重爆炸
    """
    # 把每个样本的特征包成 Value 列表
    inputs = [[Value(xi) for xi in row] for row in X]
    scores = [model(x) for x in inputs]       # 每个 score 是一个 Value

    # 数据损失：逐样本 MSE，再求平均
    data_losses = [(score - yi) ** 2 for score, yi in zip(scores, y)]
    data_loss = sum(data_losses) * (1.0 / len(data_losses))

    # L2 正则
    reg_loss = alpha * sum((p * p for p in model.parameters()), Value(0.0))
    total = data_loss + reg_loss

    # 顺手算训练准确率
    accuracy = sum((s.data > 0) == (yi > 0) for s, yi in zip(scores, y)) / len(y)
    return total, accuracy


# ----------------------------------------------------------------------
# 4. 训练循环
# ----------------------------------------------------------------------
EPOCHS = 100
t0 = time.time()

for epoch in range(EPOCHS):
    # forward + loss
    loss, acc = loss_fn(model, X, y)

    # zero_grad：每轮必须，否则梯度会累加到上一轮
    model.zero_grad()

    # backward：梯度填回所有 Value 参数
    loss.backward()

    # SGD 更新：学习率随 epoch 线性衰减（简易 scheduler）
    lr = 1.0 - 0.9 * epoch / EPOCHS
    for p in model.parameters():
        p.data -= lr * p.grad

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"step {epoch:3d}  loss {loss.data:.4f}  acc {acc*100:.1f}%")

elapsed = time.time() - t0
print(f"\n训练完成，用时 {elapsed:.2f}s")
assert loss.data < 0.1, f"loss 未降到 0.1 以下（当前 {loss.data:.4f}），检查学习率/轮数"

# ----------------------------------------------------------------------
# 5. 保存决策边界图
# ----------------------------------------------------------------------
os.makedirs('logs', exist_ok=True)

# 生成覆盖数据范围的网格
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]

# 对每个网格点推理（MLP 对单样本前向，批量靠循环）
inputs = [[Value(xi) for xi in row] for row in Xmesh]
scores = [model(x).data for x in inputs]
Z = (np.array(scores) > 0).reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
plt.title(f"Decision boundary  |  loss={loss.data:.3f}  acc={acc*100:.1f}%")
plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())
plt.tight_layout()
plt.savefig('logs/decision_boundary.png', dpi=120)
print("决策边界已保存到 logs/decision_boundary.png")
```

### 6.1 训练循环的 4 个步骤（背下来）

```
for epoch in range(EPOCHS):
    loss = loss_fn(...)     # 1. forward
    model.zero_grad()       # 2. 清旧梯度（极易忘）
    loss.backward()         # 3. 反向传播填梯度
    for p in params:        # 4. 用梯度更新参数
        p.data -= lr * p.grad
```

**为什么 `zero_grad` 不能忘？** 因为 Day2 你亲手写的 `_backward` 都是 `self.grad += ...` 的累加语义。如果不清零，第 2 轮的梯度会叠在第 1 轮上，方向完全错乱，loss 会开始乱跳。

**为什么学习率要衰减？** 前期大步快走，后期小步精调。不衰减的结果是**终局在最优点附近来回震荡**，loss 卡在 0.2 左右下不去——你今天如果卡在这里，八成就是 lr 没衰减。

---

## 7. 预期运行输出（拿去对答案）

```
MLP of [Layer[tanhNeuron(2), tanhNeuron(2), tanhNeuron(2), ...]],
 Layer[tanhNeuron(16) ...],
 Layer[linearNeuron(16)]]
参数总数: 337
step   0  loss 0.85xx  acc 52.0%
step  10  loss 0.31xx  acc 84.0%
step  20  loss 0.15xx  acc 92.0%
step  30  loss 0.09xx  acc 96.0%
...
step  99  loss 0.03xx  acc 100.0%

训练完成，用时 15-40s（取决于你的机器）
决策边界已保存到 logs/decision_boundary.png
```

参数总数 337 怎么来的？`MLP(2, [16, 16, 1])`：
- L1: 16×(2+1) = 48
- L2: 16×(16+1) = 272
- L3: 1×(16+1) = 17
- 合计 **337**（自己验一遍，算得对说明 §1.4 真学会了）

图片应该长成这样：**两条互相缠绕的半月形**被一条**弯弯曲曲的决策边界**分开，几乎所有点都被正确着色。如果边界是条直线——说明网络坍缩成线性了，检查是不是激活函数没用上。

---

## 8. micrograd 为什么比 numpy 网络慢大约 100 倍？（今天的分析题）

这是今天最重要的"工程直觉题"。**把下面的推理写进 `logs/W3_day4_log.md`**。

### 8.1 先量化

上周你写过一个纯 numpy 的 2 层 MLP 跑 MNIST，batch=64 的一次前向 + 反向大约 **几毫秒**级。今天这个 micrograd 的 MLP(2,[16,16,1]) 对 100 个样本跑一个 epoch 是 **几百毫秒**级。

**参数量上差几十倍（numpy 版几万，micrograd 版几百），但时间反而长，说明单位算力的差距极大**。粗估 micrograd 每个浮点乘法的有效吞吐比 numpy 慢了约 **100 倍**。这个数量级来自三处：

### 8.2 原因 1：每个标量一个 Python 对象，每次运算一个函数调用（主因）

numpy 版一次 `W @ x` 是一个 C 函数调用，内部循环用 SIMD 指令**一条一条向量加乘**。

micrograd 版一次 `W @ x` 是 `n_out × n_in` 个 `Value.__mul__` 调用，每个调用：
- 创建一个新的 `Value` 对象（Python 对象分配）
- 构造一个 `_backward` 闭包（闭包捕获外层变量）
- 更新 `_prev` 集合
- 往结果上附加 `_op` 字符串

**一次乘法本来是 1 条 CPU 指令的事，在 micrograd 里变成了几十上百条 Python 字节码**。这就是 100 倍的主力来源。

### 8.3 原因 2：反向时每个节点又调一次闭包 + 字典查找

`backward()` 里的 `for v in reversed(topo): v._backward()`——每个节点一次函数调用、一次属性查找（`self.grad`、`out.grad`）、一次加法赋值。`topo` 有几千个节点时，这几千次 Python 层的解释开销加起来就很可观。

numpy 反向传播靠**一次矩阵乘法**（`dW = dZ @ X.T`）就把一整层的梯度算完了——**同样是"每个参数拿到一个梯度"，但 C 底层的循环快了好几个数量级**。

### 8.4 原因 3：每轮都重建计算图

micrograd 版每个 epoch 都把所有 `Value` 对象**重新创建一遍**（因为 forward 一遍就产生一遍新图）。Python 对象分配 + GC 压力不可忽略。

numpy 版不建图，参数就是 ndarray，每轮都是**原地复用**同一块内存。

### 8.5 一句话结论

**micrograd 的慢是 Python 对象模型的慢，不是算法的慢。** 把一张图上几千个标量操作换成**几个大张量操作**（numpy / PyTorch 的路线），单次内核的固定开销被摊薄，总时间立刻回到几毫秒级。

换个说法：**micrograd 教你"怎么做对"，PyTorch 教你"怎么做快"。** 两条路径都走一遍，你就真的懂自动微分框架了。

### 8.6 想验证？做这个小实验

在训练脚本里加：

```python
import time
t0 = time.time()
_ = loss_fn(model, X, y)
print(f"单次 forward: {(time.time()-t0)*1000:.1f} ms")
```

然后再用 numpy 实现一个等价结构的 MLP，跑一次 forward 计时。**实测比一下**，把两个数字写进 log。**你亲眼量过的数字比我说的任何话都更有说服力**。

---

## 9. 当日 log 模板：`logs/W3_day4_log.md`

把下面这段复制进 `W3_day4_log.md`，填空就行：

```markdown
# W3 Day4 训练日志（2026-05-14）

## 产出
- micrograd_redo/engine.py   : 在 Day2 基础上加 __pow__ / relu / __truediv__
- micrograd_redo/nn.py       : Neuron / Layer / MLP + Module
- micrograd_redo/train_moons.py : 训练脚本
- logs/decision_boundary.png : 决策边界图

## 关键指标
- 模型：MLP(2, [16, 16, 1]),  参数 337 个
- 数据：make_moons,  n=100, noise=0.1
- 训练：100 个 epoch,  lr 从 1.0 线性衰减到 0.1
- 最终：loss = ____,   训练集 acc = ____%
- 耗时：____ s

## 遇到的坑
1. ____（举例：zero_grad 忘了，loss 不降反升）
2. ____
3. ____

## micrograd vs numpy 网络的速度对比
- micrograd 单次 forward: ____ ms
- numpy    单次 forward: ____ ms（若今天没做，留到 Day5）
- 核心原因（按重要度排列）：
  1. Python 对象开销：每个标量一个 Value 对象，每次运算一个函数调用 + 闭包构造
  2. 解释器开销：backward 逐节点调用 _backward，几千次 Python 字节码解释
  3. 图重建：每 epoch 重新创建所有 Value 对象，分配 + GC 压力
- 一句话：micrograd 慢的是 Python 对象模型，不是算法。张量化是唯一出路。

## 明日 Day5 想验证/尝试的
- ____
```

---

## 10. 常见坑总结（踩过的人都说对）

| 症状 | 原因 | 修法 |
|---|---|---|
| loss 在 0.5 附近反复震荡 | 学习率太大，或没衰减 | 把 lr scheduler 加上，或初始 lr 从 1.0 改 0.3 |
| loss 一直是常数不动 | zero_grad 忘了，梯度累加到爆、然后 NaN；或 forward 里把 Value 误写成了 python 数 | 检查循环里有没有 `model.zero_grad()`；打印 `params[0].grad` 看是否 0 |
| 决策边界是直线 | 最后一层 `nonlin` 没设对，或所有层都用线性 | `MLP` 的初始化逻辑里只有最后一层 `nonlin=False` |
| acc 涨到 92% 就卡住 | 模型太小 | 把隐藏层从 [4, 4] 改 [16, 16] |
| `nn.py` 单测失败："参数数目应为 37" | `Neuron.__init__` 里忘记 `+ 1` 的 bias | 每个 Neuron 的参数数 = `n_in + 1`，不是 `n_in` |
| 程序跑 10 秒没动静 | 没输出 flush，或陷入构图循环 | 把 `print` 放进循环里每轮都打 |

---

## 11. 自测题（做完再睡）

1. **不看代码**，写出 `Neuron.__call__` 的三行核心代码。
2. `MLP(3, [5, 5, 2])` 有多少个参数？
3. 为什么 `loss.backward()` 之前必须 `zero_grad`？如果忘了，loss 会怎样变化？
4. make_moons 的 y 从 {0, 1} 映射到 {-1, +1} 后，为什么可以直接用 MSE？
5. 如果我把最后一层也加上 tanh，训练会怎样？（预测 + 实验验证）
6. 用不超过 3 句话向别人解释"micrograd 比 numpy 网络慢一百倍的主要原因"。

六道题都能脱稿答上来，就可以安心进 Day5 了。

---

## 12. 今日一句话总结

> **自动微分 = 正向建图 + 反向按局部导数累加**；**神经网络 = 若干 `Value` 运算的有组织堆叠**。
> 今天你证明了：**只要底层 `Value` 够对、够通用，上层 MLP 就是几十行胶水代码**。这正是 PyTorch 的哲学——把导数的事彻底交给引擎，把架构的事留给人类。

做完作业打勾，睡个好觉，明天见。
