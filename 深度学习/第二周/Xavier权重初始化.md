# Xavier 权重初始化学习笔记

> 配套：Andrew Ng DL Course 1 Week 3
> 阅读建议：从头到尾按顺序看，每一节都建立在上一节的结论上。带★的部分是核心，必须真正理解，不能只是背公式。

---

## 一、问题的起点：为什么权重初始化重要？

神经网络的训练流程是：**前向传播算损失 → 反向传播算梯度 → 用梯度更新参数**。
但所有权重 $W$ 在训练开始前必须先有一个"初值"。这个初值不是随便选的——选不好，网络根本训练不起来。

### 1.1 几种"看起来很合理但其实是灾难"的初始化

| 初始化方式 | 后果 | 原因 |
|---|---|---|
| 全部初始化为 0 | 网络永远学不到东西 | 同一层所有神经元输出相同、梯度相同，相当于只有一个神经元（**对称性破坏失败**）|
| 全部初始化为同一常数 | 同上 | 同上 |
| 用很大的随机数（比如 $\mathcal{N}(0, 1)$）| 激活值爆炸 / 梯度爆炸 | 每经过一层，数值的方差越来越大|
| 用很小的随机数（比如 $\mathcal{N}(0, 0.0001)$）| 激活值衰减到 0 / 梯度消失 | 每经过一层，数值的方差越来越小|

### 1.2 我们真正想要的是什么？★

**核心目标**：让信号在网络中"稳定地"流动。

更具体一点：

- 前向传播时，希望**每一层激活值的方差大致相等**（不爆炸、不消失）
- 反向传播时，希望**每一层梯度的方差大致相等**（不爆炸、不消失）

这就是 Xavier 初始化要解决的问题。

---

## 二、核心思想：方差守恒 ★

把神经网络想象成一根"水管"，信号是流过的水。如果水管每节越来越粗，水会越来越缓（方差消失）；越来越细，水会越来越急（方差爆炸）。我们希望**水管粗细均匀**——也就是每层的方差都差不多。

那能不能找到一个 $\mathrm{Var}(W)$，让这件事自动成立？

**可以**。下面用数学推一遍。

---

## 三、前向传播的方差推导 ★

考虑某一层：

$$
z_j = \sum_{i=1}^{n_{in}} W_{ji} \cdot x_i
$$

其中：
- $x_i$ 是上一层的输入（$n_{in}$ 个）
- $W_{ji}$ 是该层的权重
- $z_j$ 是该层线性变换后的输出（暂不考虑激活函数）

### 3.1 设定假设（必须先约定清楚）

为了能推下去，我们做几个理想化假设：

1. $x_i$ 之间相互独立，均值为 0，方差都等于 $\mathrm{Var}(x)$
2. $W_{ji}$ 之间相互独立，均值为 0，方差都等于 $\mathrm{Var}(W)$
3. $W$ 和 $x$ 相互独立

> 直觉：如果 $W$ 是我们随机初始化的，$x$ 是上一层的输出，它们之间确实相互独立。均值为 0 是我们主动选的（对称分布更稳）。

### 3.2 推导 $\mathrm{Var}(z_j)$

**第一步**：先看单项 $W_{ji} \cdot x_i$ 的方差。

利用恒等式 $\mathrm{Var}(Y) = E[Y^2] - (E[Y])^2$：

$$
\mathrm{Var}(W_{ji} \cdot x_i) = E[(W_{ji} x_i)^2] - (E[W_{ji} x_i])^2
$$

因为 $W$ 和 $x$ 独立：

$$
= E[W_{ji}^2] \cdot E[x_i^2] - (E[W_{ji}] \cdot E[x_i])^2
$$

又因为 $E[W] = 0, E[x] = 0$，所以第二项是 0：

$$
= E[W_{ji}^2] \cdot E[x_i^2]
$$

而 $E[W^2] = \mathrm{Var}(W) + (E[W])^2 = \mathrm{Var}(W)$（因为均值为 0），同理 $E[x^2] = \mathrm{Var}(x)$。所以：

$$
\boxed{\mathrm{Var}(W_{ji} \cdot x_i) = \mathrm{Var}(W) \cdot \mathrm{Var}(x)}
$$

**第二步**：累加 $n_{in}$ 个独立的项。

独立随机变量之和的方差等于方差之和：

$$
\mathrm{Var}(z_j) = \sum_{i=1}^{n_{in}} \mathrm{Var}(W_{ji} x_i) = n_{in} \cdot \mathrm{Var}(W) \cdot \mathrm{Var}(x)
$$

### 3.3 让方差守恒：得到第一个条件

我们希望 $\mathrm{Var}(z_j) = \mathrm{Var}(x)$，于是：

$$
n_{in} \cdot \mathrm{Var}(W) \cdot \mathrm{Var}(x) = \mathrm{Var}(x)
$$

$$
\boxed{\mathrm{Var}(W) = \frac{1}{n_{in}} \quad \text{（前向条件）}}
$$

> **直觉解读**：扇入 $n_{in}$ 越大（神经元接收的输入越多），权重就要越小，否则把这么多东西加起来会爆。

---

## 四、反向传播的方差推导 ★

只满足前向还不够。反向传播时梯度也在层与层之间流动，同样可能爆炸或消失。

反向传播时，梯度通过 $W^T$ 传递（这是链式法则的结果）：

$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^{n_{out}} W_{ji} \cdot \frac{\partial L}{\partial z_j}
$$

注意这里求和的上限是 $n_{out}$（这一层有多少个输出，反向时就要从多少个梯度汇总过来）。

用完全相同的方差推导逻辑：

$$
\mathrm{Var}\left(\frac{\partial L}{\partial x_i}\right) = n_{out} \cdot \mathrm{Var}(W) \cdot \mathrm{Var}\left(\frac{\partial L}{\partial z_j}\right)
$$

希望梯度方差守恒，于是：

$$
\boxed{\mathrm{Var}(W) = \frac{1}{n_{out}} \quad \text{（反向条件）}}
$$

> **直觉解读**：扇出 $n_{out}$ 越大（这层影响越多下游神经元），权重也要越小，否则反向时梯度会爆。

---

## 五、Xavier 的折中：调和平均 ★

我们得到两个相互冲突的条件：

- 前向要求：$\mathrm{Var}(W) = 1 / n_{in}$
- 反向要求：$\mathrm{Var}(W) = 1 / n_{out}$

只有当 $n_{in} = n_{out}$ 时它们才能同时满足。一般情况下两者不等，怎么办？

**Xavier 的做法是取两者的"平均"作为折中**：

$$
\boxed{\mathrm{Var}(W) = \frac{2}{n_{in} + n_{out}}}
$$

> **为什么是这个形式？**
> $\frac{2}{n_{in}+n_{out}}$ 实际上是 $\frac{1}{n_{in}}$ 和 $\frac{1}{n_{out}}$ 的**调和平均**。
> 在两个条件都希望满足时，调和平均是一个数学上自然的折中——它是平均"倒数水平"，不会偏向任何一方。

这就是 Xavier 初始化的核心结论。剩下的都是怎么用一个**具体的随机分布**来实现这个方差。

---

## 六、两种具体实现：正态版 vs 均匀版

### 6.1 正态分布版本

直接采样：

$$
W \sim \mathcal{N}\left(0,\ \frac{2}{n_{in} + n_{out}}\right)
$$

也就是均值 0，方差就是上面那个值。

### 6.2 均匀分布版本

如果想从均匀分布 $U[-a, a]$ 采样，怎么定 $a$？

均匀分布 $U[-a, a]$ 的方差是 $\dfrac{a^2}{3}$（这是标准结论，可以查或自己积分推一下）。

让它等于目标方差：

$$
\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}}
\quad\Longrightarrow\quad
a = \sqrt{\frac{6}{n_{in} + n_{out}}}
$$

所以：

$$
W \sim U\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},\ \sqrt{\frac{6}{n_{in}+n_{out}}}\right]
$$

> 两种版本本质等价，方差一样；实践中可以任选一个。原论文用的是均匀版。

---

## 七、Numpy 实现

放到你的 `week2_numpy_nn/src/nn.py` 里。

```python
import numpy as np

def initialize_parameters_xavier(layer_dims, seed=1):
    """
    Xavier 初始化（正态版）。

    Args:
        layer_dims: list[int]，每层维度。
                    例如 [2, 4, 1] 表示输入 2 维、一个 4 神经元的隐藏层、输出 1 维。
        seed: 随机种子，方便复现。

    Returns:
        parameters: dict，包含 W1, b1, W2, b2, ...
                    W_l 形状为 (layer_dims[l], layer_dims[l-1])
                    b_l 形状为 (layer_dims[l], 1)
    """
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)  # 包含输入层在内的层数

    for l in range(1, L):
        n_in = layer_dims[l - 1]
        n_out = layer_dims[l]

        # 标准差 = sqrt(方差) = sqrt(2 / (n_in + n_out))
        std = np.sqrt(2.0 / (n_in + n_out))

        parameters[f"W{l}"] = np.random.randn(n_out, n_in) * std
        parameters[f"b{l}"] = np.zeros((n_out, 1))

    return parameters
```

### 几个实现细节注意点

1. **`W` 的形状是 `(n_out, n_in)`**：因为前向传播是 $Z = W X$，$X$ 形状 `(n_in, m)`，所以 $W$ 必须是 `(n_out, n_in)`。
2. **`np.random.randn(...) * std` 的来历**：`randn` 默认采样自 $\mathcal{N}(0, 1)$，乘以 `std` 就把方差缩放到 `std**2`。这是把"标准正态"变成"任意方差正态"的标准技巧。
3. **`b` 初始化为 0**：偏置不存在对称性问题（对应不同神经元的 $W$ 已经是随机的，足够打破对称），全 0 是惯例做法。

### 想要均匀版？

把核心两行换成：

```python
limit = np.sqrt(6.0 / (n_in + n_out))
parameters[f"W{l}"] = np.random.uniform(-limit, limit, size=(n_out, n_in))
```

---

## 八、验证你写对了

写完后跑这段，做三项检查：

```python
params = initialize_parameters_xavier([784, 128, 64, 10])

for k, v in params.items():
    if k.startswith("W"):
        l = int(k[1:])
        n_out, n_in = v.shape
        expected_std = np.sqrt(2.0 / (n_in + n_out))
        print(f"{k}: shape={v.shape}, "
              f"mean={v.mean():+.4f}, "
              f"std={v.std():.4f} (期望≈{expected_std:.4f})")
    else:
        print(f"{k}: shape={v.shape}, all_zero={(v == 0).all()}")
```

**通过标准**：
- 形状对：`W1: (128, 784)`、`b1: (128, 1)`、`W2: (64, 128)`、`b2: (64, 1)`、`W3: (10, 64)`、`b3: (10, 1)`
- 每个 `W` 的实际 std 应该接近期望值（误差 1% 以内）
- 每个 `W` 的均值应该接近 0
- 每个 `b` 应该全为 0

---

## 九、什么时候不该用 Xavier？—— 与 He 初始化的对比 ★

Xavier 推导时假设激活函数是**关于 0 对称的线性近似**——这对 `tanh`、`sigmoid` 在 0 附近基本成立。

但对 `ReLU`，激活函数会把一半的负数砍成 0，相当于**信号方差被砍了一半**。这种情况下 Xavier 会让信号每层衰减一半，深层网络仍然学不动。

**He 初始化**（也叫 Kaiming 初始化）就是为 ReLU 设计的，把方差翻倍补偿：

| 激活函数 | 推荐初始化 | 方差公式 |
|---|---|---|
| tanh, sigmoid | **Xavier** | $\mathrm{Var}(W) = \dfrac{2}{n_{in} + n_{out}}$ |
| ReLU, Leaky ReLU | **He** | $\mathrm{Var}(W) = \dfrac{2}{n_{in}}$ |

> Course 1 Week 3 用的激活是 tanh，所以用 Xavier 完全正确。等到了用 ReLU 的网络（后面的 Course 会讲），切换成 He。

---

## 十、常见疑问 FAQ

**Q1：Xavier 是"必须"用吗？不能就用 `randn * 0.01` 吗？**
浅层网络（比如 2~3 层）用 `randn * 0.01` 也能跑起来，因为问题还没暴露。但只要层数稍多，就会看到训练曲线长时间不动——这就是方差消失。Xavier 让你不用调这个常数，自动适配每层的 `n_in, n_out`。

**Q2：为什么要均值为 0？**
非零均值会让激活值整体偏移，深层叠加后偏移会放大；同时反向传播也会引入额外偏置项。均值为 0 是让方差推导成立的前提，也是数值稳定性的需要。

**Q3：偏置 $b$ 为什么不用 Xavier？**
$b$ 不参与"信号放大"，它只是平移。对称性已经被随机的 $W$ 打破了，$b$ 全 0 完全 OK，而且更简单。

**Q4：原论文里"Xavier"是谁？**
2010 年 Xavier Glorot 和 Yoshua Bengio 的论文《Understanding the difficulty of training deep feedforward neural networks》。后来大家用作者名字称呼这个方法，所以叫 Xavier 初始化或 Glorot 初始化。

**Q5：现代框架（PyTorch / TensorFlow）默认用 Xavier 吗？**
PyTorch 的 `nn.Linear` 默认是一种基于均匀分布、依赖 $n_{in}$ 的方案（接近但不完全是 Xavier）。要用标准 Xavier 调 `torch.nn.init.xavier_normal_` 或 `xavier_uniform_`。

---

## 十一、一句话总结

> Xavier 初始化让权重的方差等于 $\dfrac{2}{n_{in}+n_{out}}$，这样信号在前向和反向传播中都能近似保持方差不变，从而避免深层网络的梯度爆炸和消失。**它适用于 tanh / sigmoid；ReLU 用 He。**

---

## 任务自检清单

完成今天的 Xavier 任务，应该能做到以下几件事：

- [ ] 不看笔记能解释为什么 $W$ 不能全 0、不能太大、不能太小
- [ ] 能口述前向方差推导的关键三步：单项方差 → 累加 → 让等于 Var(x)
- [ ] 知道 $\dfrac{2}{n_{in}+n_{out}}$ 是怎么来的（前向、反向两个条件的折中）
- [ ] 能在 `nn.py` 里独立写出 `initialize_parameters_xavier` 函数
- [ ] 跑通验证脚本，三项检查全部通过
- [ ] 知道什么时候应该改用 He 初始化
