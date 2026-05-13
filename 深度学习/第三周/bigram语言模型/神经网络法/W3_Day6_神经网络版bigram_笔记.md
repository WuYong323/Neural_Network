# W3 Day6 学习笔记：神经网络版 bigram——梯度下降是怎么"重新发明"计数法的

> 日期：2026-05-16（周六）
> 前置：你已完成 Day5，跑通了计数法 bigram，得到 NLL ≈ 2.45，会用 `torch.multinomial` 采样
> 视频范围：Karpathy "Neural Networks: Zero to Hero" EP2 第 1:00 – 1:57
> 今日核心：**用一个 27→27 的全连接 + softmax 训练同一个 bigram 任务**，然后惊讶地发现：训练完的 `softmax(W)` 几乎等于昨天用纯统计算出的 `P` 矩阵。这不是巧合，背后是"最大似然估计 = 计数归一化"这一深层等价。理解这件事，你才算真正打通"统计语言模型"和"神经网络语言模型"两条线。

---

## 0. 先看清楚今天要拿到什么

完成这份笔记 + 代码后，你应该能：

1. 用一句话回答：**为什么神经网络版 bigram 和昨天的计数法 bigram 几乎给出相同的解？** 不能只说"巧合"或"梯度下降很牛"——要能说出"最大似然 = 频率"这层数学关系。
2. 解释 **one-hot 编码 + 矩阵乘法**等价于"按下标查表"——这是为什么神经网络版能学到和计数法一样的东西的第一步。
3. 默写 softmax 公式：$\text{softmax}(z)_j = \dfrac{e^{z_j}}{\sum_k e^{z_k}}$，并解释为什么需要它（让任意实数变成合法概率分布）。
4. 不看代码徒手写出交叉熵损失：$L = -\frac{1}{N}\sum_i \log P_i(y_i^{\text{true}})$，并解释为什么"最小化交叉熵 = 最大化训练集对数似然"。
5. 把 L2 正则项 $\lambda \cdot \frac{1}{|W|}\sum W_{ij}^2$ 加进 loss，并解释**正则项越大 → 生成的分布越接近均匀 → NLL 上升**——把这个现象变成你能预测的规律。
6. 跑通 `bigram/nn_based.py`，得到 NLL ≈ 2.45（与 Day5 计数法误差 < 0.05），并对比 `softmax(W)` 与 `P` 的元素差异（最大值应 < 0.05）。
7. 写一段 200 字以内的"为什么两个版本会收敛到几乎相同结果"的解释段落，存到 `W3_day6_log.md`。

如果第 1、4 条做不到，不要进 Day7——明天的 `tech_notes/autograd_explained.md` 会用到这两个理解。

---

## 1. 知识点串讲：从计数到神经网络的概念桥

### 1.1 昨天的 bigram 在做什么？用神经网络的语言重新表述

昨天计数法：

```
当前字符 c_t  →  查 P[c_t] 行  →  得到 27 维概率分布  →  采样
```

整个过程没有"参数"——P 是数完直接除出来的。

今天要做的事：**把"查表"换成"矩阵乘法"，让 P 变成可学习的参数 W**。

```
当前字符 c_t  →  one-hot 成 27 维向量 x  →  W·x = logits  →  softmax(logits) = 27 维分布  →  采样
```

每多一个步骤：
- **one-hot**：把整数下标变成一个除一个位置是 1 之外全是 0 的向量。比如字符 `a` (id=1) 变成 `[0,1,0,0,...,0]`。
- **W (27×27)**：可学习的权重矩阵。$W_{ij}$ 直觉上代表"如果当前字符是 i，给字符 j 多大的'分数'"。
- **softmax**：把 logits（任意实数）压成合法的概率分布（非负、和为 1）。
- **交叉熵 + 梯度下降**：通过最小化 NLL 来调 W。

### 1.2 关键洞察 1：one-hot · W = W 的某一行

设 $x$ 是字符 $i$ 的 one-hot 向量（第 $i$ 位是 1，其余 0），那么：

$$
\text{logits} = W^\top x = W[i, :]
$$

也就是说，**用 one-hot 做矩阵乘法，本质上就是在选 W 的第 i 行**。这是为什么 27 维 one-hot 输入 + 27 维输出的网络能学到和计数法一样的东西——结构上 W 的每一行就对应"以字符 i 为前提时的下一字符分数"，跟计数法里的 N[i, :] 直接对应。

> ⚠️ 注意维度约定：如果你写 `logits = x @ W`（x 是行向量），那 W 的形状是 (27, 27)，logits = W 的第 i 行。Karpathy 视频里就是这种约定，今天的代码也跟着走。

### 1.3 关键洞察 2：softmax 是把"分数"变"概率"的标准操作

W 的元素可以是任何实数（正、负、很大、很小）。直接拿来当概率显然不行——可能为负、可能不归一。

**softmax 解决两件事：**

$$
\text{softmax}(z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

- 指数 $e^{z_j}$ 把任何实数变成正数 → 解决"概率非负"。
- 除以 $\sum_k e^{z_k}$ → 解决"概率和为 1"。

**直觉**：softmax 像一个"加权放大器"——分数大的位置概率被指数地放大，分数小的位置被压扁。如果两个 logits 相差 1，对应的概率比是 $e \approx 2.7$ 倍。

**数值稳定性的小坑**（昨天讲过，今天再强调）：直接算 $e^{z_j}$ 在 $z_j$ 大时会溢出。标准做法是先减 max：

$$
\text{softmax}(z) = \text{softmax}(z - \max(z))
$$

这个变换在数学上不改变结果（分子分母同除一个常数），但避免了 $e^{1000}$ 这种数值灾难。PyTorch 的 `F.softmax` 内部已经做了这个，自己写 numpy 时记得加。

### 1.4 关键洞察 3：交叉熵损失 = 负对数似然的 minibatch 版本

对一个样本（前一字符 $i$，真实下一字符 $j$），它的损失是：

$$
\ell = -\log P(j | i) = -\log \text{softmax}(W[i,:])_j
$$

对 N 个样本求平均，就是交叉熵：

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \log P(j_n | i_n)
$$

**这就是昨天 NLL 公式的一字不差的复制**。区别仅在于：今天 $P$ 是 $\text{softmax}(W)$，参数是 W；昨天 $P$ 是计数除以行和，没有参数。

**最小化 L 等价于什么？** 等价于最大化 $\prod_n P(j_n | i_n)$——也就是让模型对训练集里所有真实 bigram 对都分配尽可能大的概率。这正是**最大似然估计（MLE）**的定义。

### 1.5 为什么两个版本最终会几乎相等？（今天最重要的一段）

这是今天笔记的灵魂。请慢慢读。

**事实**：在 27 维 one-hot 输入 + softmax + 交叉熵的设定下，最大似然解的解析形式就是计数归一化。

**简化的证明思路**（不严格但抓住关键）：

设训练集里 bigram $(i, j)$ 出现了 $N_{ij}$ 次。损失函数可以写成：

$$
L = -\frac{1}{N} \sum_{i,j} N_{ij} \log P_{ij}, \quad P_{ij} = \frac{e^{W_{ij}}}{\sum_k e^{W_{ik}}}
$$

对 $W_{ij}$ 求偏导（用 softmax + 交叉熵的标准结果，推导用拉格朗日乘子法即可）：

$$
\frac{\partial L}{\partial W_{ij}} = \frac{1}{N}\left( N_i \cdot P_{ij} - N_{ij} \right)
$$

其中 $N_i = \sum_j N_{ij}$ 是字符 $i$ 作为前驱出现的总次数。

令导数为 0（取最优解的条件）：

$$
P_{ij} = \frac{N_{ij}}{N_i}
$$

**这正是昨天计数归一化得到的 $P$！** 也就是说：**只要训练充分，神经网络版 bigram 必然收敛到与计数法相同的 P。**

> ⚠️ 实际有微小差异（< 0.05）来自三个地方：
> - 训练步数有限，没真的到极小值
> - 学习率不够小，最后在最优点附近震荡
> - L2 正则（如果加了）会让解轻微偏向"更均匀"

### 1.6 为什么我们还要费劲学神经网络版？

既然计数法已经给出了同样的最优解，**为什么不一直用计数法？**

答案是：**计数法在上下文变长后会爆炸。**

| 模型 | 上下文长度 | 表大小 |
|---|---|---|
| bigram（看前 1 字符） | 1 | $27^2 = 729$ |
| trigram（看前 2 字符） | 2 | $27^3 = 19683$ |
| 5-gram（看前 4 字符） | 4 | $27^5 \approx 1.4 \times 10^7$ |
| 10-gram（看前 9 字符） | 9 | $27^{10} \approx 2 \times 10^{14}$ |

到 5-gram 已经接近内存极限，10-gram 完全不可能。而且数据稀疏严重——绝大多数 5-gram 在训练集里根本没出现，计数法只能给它们 0 或者均匀的平滑值，毫无判断力。

**神经网络版的真正价值**：用 $O(\text{embedding\_dim} \times \text{vocab})$ 的参数共享，让相似上下文（比如 `the` 和 `a` 都是冠词）能复用学到的统计规律。这是下周 EP3（makemore MLP）和后面 nanoGPT 的全部技术含量。

**今天做这个看似多余的实验，是为了让你建立信心：神经网络的最优解和直觉一致**。后面看到 Transformer 给出"看起来很神秘"的结果时，你心里有个底——它仍然在做"对训练集做某种最大似然估计"，没有任何魔法。

### 1.7 L2 正则做了什么？

把 loss 改写成：

$$
L_{\text{reg}} = L + \lambda \cdot \frac{1}{|W|} \sum_{i,j} W_{ij}^2
$$

$\lambda$ 越大，模型越倾向于让 W 的元素接近 0；W 全 0 时 softmax 输出均匀分布。

**所以加大 $\lambda$ 的两个直观效果：**
1. NLL 上升（模型变笨，离训练集统计更远）
2. 生成结果更"杂乱"（每个字符都被采到的概率更接近 1/27）

**为什么还要正则？** 在 bigram 这种小模型上几乎没用——训练集太大，过拟合不严重。但在大模型（参数量 >> 数据量）上，L2 正则能防止 W 学到一些奇怪的极大值，提高泛化。**今天加 L2 主要是为了体验"正则强度 ↔ NLL"的可控关系**，这是个调参直觉。

---

## 2. 目录结构与文件分工

```
week3_karpathy/
└── bigram/
    ├── names.txt             # 训练数据 (Day5 已下载)
    ├── count_based.py        # Day5: 计数法 bigram
    ├── nn_based.py           # Day6 主角: 神经网络版 bigram
    ├── comparison.md         # Day6 交付: NLL 对比表 + 解释段落
    └── logs/
        ├── bigram_samples.txt        # Day5 输出
        ├── bigram_samples_nn.txt     # Day6 输出
        └── W3_day6_log.md            # 今日训练日志
```

---

## 3. 完整可运行代码：`bigram/nn_based.py`

```python
"""
W3 Day6: 神经网络版 bigram 语言模型
对应视频: Karpathy "Zero to Hero" EP2 第 1:00 - 1:57

整个文件做五件事:
  1. 读 names.txt, 构造字符表 (与 Day5 完全一致)
  2. 把训练集变成 (xs, ys) 张量: xs 是当前字符的 id, ys 是下一字符的 id
  3. 定义可学习参数 W (27, 27), 随机初始化
  4. 训练循环: 前向 (one-hot + W + softmax) -> 交叉熵 -> 反向 -> 梯度下降
  5. 训练完后: 采样生成名字, 算 NLL, 对比 softmax(W) 与 Day5 的 P

不用 torch.nn (也不用 torch.optim), 全部手写 — 这样你能看到每一步发生什么.
"""

from pathlib import Path
import torch
import torch.nn.functional as F


# ======================================================================
# 1. 读数据 + 构造字符表 (与 Day5 一致)
# ======================================================================
DATA_PATH = Path(__file__).parent / "names.txt"
words = DATA_PATH.read_text(encoding="utf-8").splitlines()
words = [w.strip().lower() for w in words if w.strip()]

chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
V = len(stoi)
assert V == 27


# ======================================================================
# 2. 构造训练集 (xs, ys)
# ======================================================================
# 把每个 (前驱, 后继) 拆成两个张量
# xs[k] = 第 k 个 bigram 的前一个字符 id
# ys[k] = 第 k 个 bigram 的后一个字符 id (= 模型要预测的目标)
xs_list, ys_list = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for c1, c2 in zip(chs, chs[1:]):
        xs_list.append(stoi[c1])
        ys_list.append(stoi[c2])

xs = torch.tensor(xs_list, dtype=torch.long)
ys = torch.tensor(ys_list, dtype=torch.long)
N = xs.numel()
print(f"训练样本数 N = {N}")
print(f"前 5 个 (xs, ys) = {list(zip(xs[:5].tolist(), ys[:5].tolist()))}")


# ======================================================================
# 3. 初始化可学习参数 W
# ======================================================================
g = torch.Generator().manual_seed(2147483647)   # 固定种子,保证可复现
W = torch.randn((V, V), generator=g, requires_grad=True)
# requires_grad=True 表示这个张量需要参与反向传播,梯度会被记录


# ======================================================================
# 4. 训练循环
# ======================================================================
LR = 50.0           # 学习率: bigram 这种简单模型可以用很大的 lr
EPOCHS = 200        # 训练步数
LAMBDA_REG = 0.01   # L2 正则强度,可以调大试试 (例如 0.1, 1.0)

print(f"\n开始训练: lr={LR}, epochs={EPOCHS}, lambda={LAMBDA_REG}")
for epoch in range(EPOCHS):
    # ---- 前向 ----
    # 1) one-hot 编码: 把 xs (N,) 变成 (N, 27) 的 one-hot 矩阵
    xenc = F.one_hot(xs, num_classes=V).float()
    # 2) 矩阵乘法得到 logits (N, 27)
    logits = xenc @ W
    # 3) softmax 得到概率分布 (N, 27)
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    # 4) 交叉熵 loss: 取出每个样本对真实标签的 log 概率, 取负平均
    #    probs[range(N), ys] 是花式索引: 第 k 个样本的第 ys[k] 列
    nll = -probs[torch.arange(N), ys].log().mean()
    reg = LAMBDA_REG * (W ** 2).mean()
    loss = nll + reg

    # ---- 反向 ----
    W.grad = None       # 等价于 W.grad.zero_(), 但更省内存
    loss.backward()

    # ---- 更新 ----
    with torch.no_grad():
        W -= LR * W.grad

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:>3} | nll = {nll.item():.4f} | reg = {reg.item():.4f}")

final_nll = nll.item()
print(f"\n训练后 NLL = {final_nll:.4f}")


# ======================================================================
# 5. 用训练后的 W 采样生成 20 个名字
# ======================================================================
# 把 W 转成最终的概率矩阵 P_nn (与 Day5 的 P 形状/含义完全一致)
with torch.no_grad():
    P_nn = F.softmax(W, dim=1)   # (27, 27), 每行和为 1

g_sample = torch.Generator().manual_seed(2147483647)
generated = []
for _ in range(20):
    out = []
    ix = 0
    while True:
        p_row = P_nn[ix]
        ix = torch.multinomial(p_row, num_samples=1, replacement=True,
                               generator=g_sample).item()
        if ix == 0:
            break
        out.append(itos[ix])
    generated.append("".join(out))

print(f"\n生成的 20 个名字:")
for name in generated:
    print(f"  {name}")

log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
(log_dir / "bigram_samples_nn.txt").write_text("\n".join(generated), encoding="utf-8")


# ======================================================================
# 6. 对比 softmax(W) 与 Day5 的计数归一化矩阵 P
# ======================================================================
# 重新构造 Day5 的 P (Laplace 平滑 +1, 与 count_based.py 保持一致)
N_count = torch.zeros((V, V), dtype=torch.int64)
for w in words:
    chs = ["."] + list(w) + ["."]
    for c1, c2 in zip(chs, chs[1:]):
        N_count[stoi[c1], stoi[c2]] += 1
N_smooth = (N_count + 1).float()
P_count = N_smooth / N_smooth.sum(dim=1, keepdim=True)

# 对比
diff = (P_nn - P_count).abs()
print(f"\n对比 softmax(W) 与计数法 P:")
print(f"  最大元素差异: {diff.max().item():.4f}")
print(f"  平均元素差异: {diff.mean().item():.4f}")
print(f"  相对差异(平均): {(diff / (P_count + 1e-8)).mean().item():.4f}")

# 抽几个具体位置看看
print(f"\n几个具体位置的对比:")
for i, j in [(0, stoi['a']), (stoi['e'], stoi['m']), (stoi['n'], 0)]:
    print(f"  P[{itos[i]!r:>3} -> {itos[j]!r:>3}]: "
          f"计数法={P_count[i,j]:.4f}, NN版={P_nn[i,j]:.4f}, "
          f"差={abs(P_count[i,j]-P_nn[i,j]):.4f}")
```

**预期运行结果（数值会因种子和训练步数微调）：**

```
训练样本数 N = 228146
前 5 个 (xs, ys) = [(0, 5), (5, 13), (13, 13), (13, 1), (1, 0)]

开始训练: lr=50.0, epochs=200, lambda=0.01
  epoch   1 | nll = 3.7686 | reg = 0.0102
  epoch  20 | nll = 2.6817 | reg = 0.0186
  epoch  40 | nll = 2.5504 | reg = 0.0223
  ...
  epoch 200 | nll = 2.4798 | reg = 0.0317

训练后 NLL = 2.4798

生成的 20 个名字:
  mor
  axx
  minaymoryles
  ...

对比 softmax(W) 与计数法 P:
  最大元素差异: 0.0234
  平均元素差异: 0.0042
  相对差异(平均): 0.0612

几个具体位置的对比:
  P[ '.' ->  'a']: 计数法=0.1377, NN版=0.1361, 差=0.0016
  P[ 'e' ->  'm']: 计数法=0.0383, NN版=0.0394, 差=0.0011
  P[ 'n' ->  '.']: 计数法=0.3658, NN版=0.3622, 差=0.0036
```

**关键观察：** 最大差异 < 0.05，平均差异 < 0.01——这就是"两版本几乎相等"的实证证据。

---

## 4. 把代码读进脑子里：6 个关键问答

### Q1：为什么训练样本要拆成 `(xs, ys)` 两个张量？不能直接用字符串吗？

PyTorch 的所有运算都在张量上做（这样才能 GPU 加速、自动求导）。`xs` 是 (N,) 的 LongTensor 表示"输入字符 id"，`ys` 是 (N,) 的 LongTensor 表示"目标字符 id"。一旦变成张量，整个训练就是矩阵运算，没有任何 Python 层的循环——这是为什么神经网络版能在几秒内跑完 20 万样本 × 200 epoch。

### Q2：`F.one_hot(xs, num_classes=V).float()` 在做什么？

`xs` 是 (N,) 的整数。`F.one_hot(xs, 27)` 把每个整数变成 27 维 one-hot 向量，结果是 (N, 27) 的 LongTensor。再 `.float()` 转成浮点（因为后面要做矩阵乘法和梯度，整数不行）。

这一步在大模型里其实会被替换成 **embedding 查表**（`nn.Embedding`）——后者在底层就是"先 one-hot 再矩阵乘"，但实现上跳过了 one-hot 的稀疏存储，直接按下标取行。所以我下周看到 `nn.Embedding` 时会立刻明白：**它就是这一步的高效版本**。

### Q3：为什么 `loss.backward()` 之前要 `W.grad = None`？

PyTorch 的梯度默认**累加**（这是为了支持 RNN 等需要跨步累积梯度的场景）。如果不清零，第二次 `backward()` 后 `W.grad` 是两次梯度之和，更新方向就错了。

`W.grad = None` 比 `W.grad.zero_()` 略快（直接释放内存而不是清零），是 PyTorch 标准最佳实践。

### Q4：为什么学习率 `lr=50` 这么大？深度学习里通常是 0.001 啊？

bigram 是个**线性模型 + softmax**，loss 表面非常平缓，没有深度网络那种"梯度爆炸/消失"的危险。实际有效学习率 ≈ `lr × 单个梯度的尺度`，bigram 梯度本身很小（因为 softmax 的梯度在远离最优时是 $O(1/V)$），所以 lr 可以大到 50。

如果你 lr=0.001 训练，会发现 NLL 200 epoch 后还在 3.0 附近——根本没学动。**这是个重要直觉：lr 必须和模型本身的梯度尺度配套，不存在"通用"的 lr**。

### Q5：`probs[torch.arange(N), ys]` 为什么能取出每个样本对真实标签的概率？

这是 PyTorch 的"花式索引"。`probs` 形状 (N, 27)：
- `torch.arange(N)` = [0, 1, 2, ..., N-1]
- `ys` 形状 (N,)，元素是真实标签 id

`probs[torch.arange(N), ys]` 等价于 `[probs[0, ys[0]], probs[1, ys[1]], ..., probs[N-1, ys[N-1]]]`——逐样本取出"模型对真实标签分配的概率"。一行代码完成 N 次查表。

### Q6：为什么 `with torch.no_grad():` 包住更新代码？

`W -= LR * W.grad` 这行如果不加 `no_grad`，PyTorch 会把这个减法记入计算图（因为 W 是 `requires_grad=True`），下一次 backward 时会出莫名其妙的错。

`with torch.no_grad():` 告诉 PyTorch："这块代码我不需要求导，别记。" 所有"参数更新"和"评估推理"代码都应该包在 `no_grad` 里。第8周会再讲一次：**推理时没有 `no_grad` 就是 2 倍内存浪费 + 不必要的计算图开销**——这是推理优化的第一刀。

---

## 5. L2 正则强度扫描实验

把 `LAMBDA_REG` 改成不同值，跑同样的 200 epoch，记录 NLL 与生成结果：

| `LAMBDA_REG` | 训练后 NLL | 生成名字示例 | 直观评价 |
|---|---|---|---|
| 0.0（无正则） | 2.45 | `mor`, `axx`, `minay` | 与计数法几乎相同 |
| 0.01 | 2.48 | `mor`, `axx`, `minay` | 几乎无差别 |
| 0.1 | 2.55 | `mone`, `aliv`, `kely` | 略微"更平均",字母分布开始接近均匀 |
| 1.0 | 2.85 | `oxnz`, `qbma`, `ldzx` | 已经很乱,但仍有元音辅音规律 |
| 10.0 | 3.20 | `qzpx`, `wkmf`, `bvjy` | 接近瞎猜（log27=3.30）,W 被压扁到接近 0 |

**结论**：lambda 是个"模型笨度"旋钮。lambda=0 让模型完全贴合训练集统计；lambda 越大模型越接近"无知识的均匀分布"。

**自己跑一遍把数据填到 `comparison.md`**——不要只看我列的，要看到你自己机器上的数字才有感觉。

---

## 6. 思考题（写到 `W3_day6_log.md`）

### 思考题 1（必答）：为什么两个版本会收敛到几乎相同的结果？

写 200 字以内的解释，必须包含三个要素：
- "最大似然估计"这个名词
- W 的最优解的解析形式（`P[i,j] = N[i,j] / N[i, :].sum()`）
- 微小差异的来源（训练步数有限、L2 正则）

参考开头：

> 神经网络版 bigram 训练的目标是最小化交叉熵损失，这等价于最大化训练集对数似然——也就是最大似然估计（MLE）。在 27 维 one-hot 输入 + softmax 这个特定结构下，MLE 的解析解可以通过对 W 求导并令导数为 0 推出，结果正好是 `P[i,j] = N[i,j] / N[i,:].sum()`，与昨天计数法 + 归一化得到的 P 完全一致。所以梯度下降只是用迭代的方式逼近这个早已被解析解给出的最优点。实测中两者差异在 0.01~0.05 之间，主要来自……

### 思考题 2：如果不用 softmax 而是直接归一化（`probs = relu(logits) / relu(logits).sum()`），会有什么问题？

提示：考虑两个角度——
- 训练时：relu 对负 logits 的导数是 0，那一部分参数会"死"在初始化值上，永远不更新。
- 测试时：极端情况下分母可能为 0（所有 logits 都 ≤ 0），整个分布算不出来。

### 思考题 3：把 lambda 调到 100，预测会发生什么？跑一下验证。

预测两个量：
- NLL 大致会到多少？（提示：如果 W 被完全压成 0，softmax 输出均匀分布，NLL = log(27) = 3.30）
- 生成的名字有什么特征？（提示：每个字符的概率都接近 1/27，很多名字会立刻终止——因为 `.` 的概率也是 1/27）

写下你的预测，再跑代码验证，对比预测和实际。**这种"先预测再验证"的训练习惯，是培养 DL 直觉最有效的方法**——比你单纯看 10 个实验都管用。

---

## 7. 当日交付清单

- [ ] `bigram/nn_based.py`（运行无报错，NLL ≈ 2.45）
- [ ] `bigram/logs/bigram_samples_nn.txt`（20 个生成名字）
- [ ] `bigram/comparison.md`（含：lambda 扫描表、softmax(W) vs P 的差异表、200 字解释段落）
- [ ] `bigram/logs/W3_day6_log.md`（含 3 道思考题的回答）
- [ ] 算法：KMP + Trie 各 1 道错题重做，写到 `tech_notes/string_review.md`
- [ ] 数学：第3章速查卡（期望/方差/协方差/相关系数 + 常见分布数字特征表）
- [ ] AI 工具：第 5 次核验练习 + 5 次练习的"应对策略"汇总

完成标准：
1. NLL ≈ 2.45，与 Day5 计数法差异 < 0.05
2. `softmax(W)` 与计数法 `P` 的最大元素差异 < 0.05
3. 能用 200 字解释"为什么两个版本收敛到相同解"，并能在 lambda 扫描表里讲出"为什么 lambda 越大 NLL 越高"

---

## 8. 与下周 Day7 + 第 4 周（EP3 makemore MLP）的衔接

### 周日（Day7）会做什么

- 把这周所有 micrograd 知识整理成 `tech_notes/autograd_explained.md`，特别强调"为什么推理时不需要梯度"——这是第 8 周推理优化的第一个伏笔
- 给 micrograd 加 relu 激活并验证（已在 Day3-4 留好接口）
- 周复盘 + GitHub commit

### 第 4 周（EP3）会怎么扩展今天的 bigram

下周的核心扩展：**从看前 1 个字符 → 看前 N 个字符**（context window 加大）。

但马上你会发现：N=3 时，one-hot 输入维度是 $27^3 \approx 2 \times 10^4$，参数 W 形状变成 $(20000, 27)$，计数法的 N 矩阵更是 $20000 \times 27$——开始爆炸。

EP3 引入两个救命的招：
1. **Embedding 表**：把每个字符映射到一个低维（比如 10 维）向量，N 个字符的输入只是 N×10 维而不是 N×27
2. **MLP 隐藏层**：用一个非线性层让"3 个字符的 embedding"压成"27 维 logits"

这两步合起来，参数从指数级降回线性级，模型也能学到"非平凡的非线性组合"。今天的神经网络版 bigram 是 N=1、embedding=one-hot、隐藏层=空的最简版本——明天会逐步加上去。

> ⚠️ **如果今天的代码没跑通，明天不要硬推进 EP3**。EP3 的代码默认你已经理解今天的"one-hot + W + softmax + 交叉熵 + 梯度下降"完整链路，缺一不可。
