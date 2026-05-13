# W3 Day5 学习笔记：bigram 语言模型——从计数到概率分布的第一次跨越

> 日期：2026-05-15（周五）
> 前置：你已完成 Day1-Day4，独立复现了 micrograd（含 relu），并用它训练 make_moons 二分类到 loss < 0.1
> 视频范围：Karpathy "Neural Networks: Zero to Hero" EP2 第 0:00 – 1:00
> 今日核心：**搞清楚"语言模型到底在建模什么"**——本质上就是 `P(下一个字符 | 当前上下文)` 这一个条件分布。今天用最朴素的方法（计数 + 归一化）把它做出来，明天再看神经网络版本是怎么收敛到几乎相同的解的。

---

## 0. 先看清楚今天要拿到什么

完成这份笔记 + 代码后，你应该能：

1. 用一句话回答：**什么是语言模型？bigram 是它最简化的形式，简化在哪？**
2. 解释 `names.txt`（约 32k 个英文人名）经过预处理后，每个名字会变成什么样的训练样本——为什么需要起止符 `.`？
3. 不看代码徒手写出 27×27 计数矩阵 `N` 的构造逻辑，并解释 `N[i, j]` 的物理含义。
4. 解释**为什么归一化时要按行（`axis=1`）而不是按列**——以及如果按列归一化会得到什么完全不同的东西（提示：那是 `P(前一个字符 | 后一个字符)`，毫无用处）。
5. 解释 `torch.multinomial` 在做什么：输入是什么、输出是什么、为什么采样不能用 `argmax`。
6. 默写出 NLL 公式 $\text{NLL} = -\frac{1}{N}\sum_i \log P(\text{真实下一个字符})$，并解释**为什么训练集上的 NLL 应该在 2.4 附近**——这个数字不是凭空冒出来的，它和"27 个字符随机猜"的 $\log 27 \approx 3.30$ 直接对比。
7. 跑通 `bigram/count_based.py`，生成 20 个名字到 `logs/bigram_samples.txt`，NLL 值在 `2.45 ± 0.05` 之间。

如果第 4、6 条做不到，不要进 Day 6——明天的神经网络版 bigram 就是为了"用梯度下降逼近今天这个 NLL"，没理解今天的目标，明天的对比就失去意义。

---

## 1. 知识点串讲：从"猜下一个字符"到语言模型

### 1.1 语言模型到底在建模什么？

一句话定义：**语言模型给定上下文 $c$，输出下一个 token 的概率分布 $P(\text{next} | c)$**。

不是"输出一个字符"，而是"输出一个分布"——所有可能字符上的概率加起来等于 1。这点非常关键，因为它告诉你两件事：

- **训练目标**：让模型对训练集里"实际出现的下一个字符"分配高概率。
- **生成方式**：从这个分布里**采样**，得到一个字符；再把它接到上下文末尾，重复这个过程。

**bigram** 是最简化的语言模型——它假设"上下文"只看**前 1 个字符**：

$$
P(\text{next} | \text{context}) \approx P(c_{t+1} | c_t)
$$

只看前 1 个字符当然丢掉了大量信息（人类预测下一个字母会看整个单词），但它够小够清楚，是建立"语言模型"直觉的最佳起点。下周的 makemore MLP（EP3）会扩展到看前 N 个字符 + embedding，nanoGPT（EP6）会扩展到看前几千个 token + Self-Attention——但**底层目标一直都是这同一件事**。

### 1.2 训练数据：`names.txt` 长什么样？

Karpathy 提供的 `names.txt` 是 32,033 个常见英文人名（每行一个，全小写）：

```
emma
olivia
ava
isabella
...
```

为什么用人名？因为它**短、纯字母、模式清楚**——元音辅音交替、有典型开头（`em-`、`av-`）和结尾（`-a`、`-ah`、`-en`）。是字符级语言模型的完美玩具数据。

**关键预处理：每个名字前后加起止符 `.`**

```
emma  →  .emma.
```

为什么要加？因为模型需要学两件额外的事：

- **从哪里开始？** 看到 `.` 之后，应该输出哪些字符的概率高？（应该是常见首字母 `a/e/m/k/...`）
- **什么时候停？** 名字结束的标志是什么？（学到 `.` 是终止符，采样到它就停）

如果不加 `.`，你只能学到中间的 bigram，永远不知道"开头长什么样、什么时候停"——生成时就只能死循环或随机截断。

### 1.3 字符表：27 个字符

```
26 个小写字母 a-z + 1 个起止符 .  =  27 个字符
```

我们建立两个映射：

- `stoi`（string-to-int）：`{'.':0, 'a':1, 'b':2, ..., 'z':26}`
- `itos`（int-to-string）：反过来

这 27 个 id 既是 `N` 矩阵的行号也是列号——`N[i, j]` 就表示"字符 `itos[i]` 后面紧跟字符 `itos[j]`"出现了多少次。

### 1.4 bigram 计数法：怎么从一堆名字得到一个分布？

**Step 1：扫所有训练样本，对每对相邻字符 +1**

```
.emma.  →  ('.','e'), ('e','m'), ('m','m'), ('m','a'), ('a','.')
.olivia. →  ('.','o'), ('o','l'), ('l','i'), ('i','v'), ('v','i'), ('i','a'), ('a','.')
...
```

每见到一对 `(c1, c2)`，就执行 `N[stoi[c1], stoi[c2]] += 1`。扫完整本数据后，`N` 是个 `(27, 27)` 的整数矩阵。

**Step 2：按行归一化得到条件概率 `P[i, j] = P(j | i)`**

```python
P = N.float()
P /= P.sum(dim=1, keepdim=True)   # 每行除以该行的总和
```

⚠️ 这里 `keepdim=True` 是关键。`N.sum(dim=1)` 默认会把 `dim=1` 这一维压掉，得到形状 `(27,)`；与 `(27, 27)` 矩阵相除时，PyTorch 会按"右对齐广播"——把 `(27,)` 解读成 `(1, 27)`，结果是**按列归一化**，含义完全错了。`keepdim=True` 让 sum 保留为 `(27, 1)`，按行广播才正确。这是 bigram 实现的头号 bug。

**为什么按行归一化？** 我们要的是 $P(\text{next}|\text{current}) = \frac{N[i,j]}{\sum_j N[i,j]}$——分母是"当前字符 $i$ 在训练集里**作为前一个字符**的总出现次数"，这正是第 $i$ 行的和。如果按列归一化，得到的是 $P(\text{prev}|\text{next})$——给定后一个字符，反推前一个字符的概率，对生成毫无意义。

### 1.5 平滑（model smoothing）：为什么不能让 0 出现在 `N` 里？

如果某对 bigram 在训练集里**完全没出现**（比如 `qz`），那么 `N[q,z] = 0`，归一化后 `P[q,z] = 0`。看似没问题，但：

**测试集里只要出现一次 `qz`，NLL 就会变成 $-\log 0 = +\infty$，模型直接崩盘**。

解决：**给 `N` 全体 +1（Laplace 平滑）** —— `N = N + 1`。这样所有概率都严格大于 0，最坏也是均匀分布。代价是稍微稀释了真实分布，但保证了数值稳定。今天就用 `+1`，简单粗暴够用。

### 1.6 采样：`torch.multinomial` 是什么？

直觉上，"按概率分布采样"就是：把分布看成一根分段染色的数轴，每段长度等于该字符的概率，然后均匀随机抽一个点，看它落在哪段——那段对应的字符就是采样结果。

`torch.multinomial(probs, num_samples=1, replacement=True)` 干的就是这件事：

- 输入 `probs`：一个长度 27 的概率向量（**必须非负**，不强制和为 1，函数内部会归一化）。
- 输出：一个 LongTensor，里面是被采到的下标。
- `replacement=True` 表示有放回采样——每次都独立从同一分布抽。

**为什么不用 `argmax`？** `argmax` 永远输出概率最大的那个字符——给定 `.` 永远输出最常见的首字母 `a`，那你生成的 32k 个名字会全部以 `a` 开头，毫无多样性。语言模型生成的灵魂就在于"概率高的常出现，概率低的偶尔出现"，这正是采样在做的事。

### 1.7 评估：负对数似然（NLL）

我们已经有了一个分布 $P$，怎么衡量它"好不好"？

**答案：在训练集（或验证集）上算它分配给"真实下一个字符"的概率，越高越好。**

但概率会非常小（小数连乘），数值上根本算不动；所以取对数变加法，再取负使其为正、越小越好：

$$
\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N} \log P(c_{i+1} | c_i)
$$

其中 $N$ 是训练集里所有 bigram 对的总数。

**几个关键的对照基线，记到肌肉记忆里：**

| 模型 | 期望 NLL | 解读 |
|---|---|---|
| 27 维均匀分布（瞎猜） | $\log 27 \approx 3.30$ | 任何模型都该比这个低，否则不如不学 |
| bigram 计数法 + Laplace 平滑 | **≈ 2.45** | 学到了"哪些字符更常衔接"的统计 |
| 神经网络版 bigram（明天 Day6） | **≈ 2.45**，几乎相等 | 因为最大似然解就是计数归一化 |
| MLP（看前 3 个字符，下周 EP3） | ≈ 2.20 | 上下文变长，不确定性下降 |
| nanoGPT（前几千 token，第6周 EP6） | ≈ 1.x | 看到完整上下文，逼近真实分布 |

**为什么是 2.45 不是更低？** 因为 bigram 太短视——只看前 1 个字符，根本没法区分"`em` 后面接 `m`（emma）"和"`em` 后面接 `e`（emery）"这种依赖更长上下文的差异。**这个 2.45 就是 bigram 模型本质能力的天花板**，再怎么调都突破不了。

> ⚠️ 视频里 Karpathy 早期版本（不加平滑、稍有不同实现）报告的 NLL 是 ≈ 2.45。具体数值在 2.40 ~ 2.50 之间都正常，差距来自是否平滑、是否包含 `.→.` 这种边界对。**只要不超过 3.30（瞎猜基线），方向就是对的。**

---

## 2. 目录结构与文件分工

今天在 `week3_karpathy/` 下新开 `bigram/` 子目录：

```
week3_karpathy/
└── bigram/
    ├── names.txt              # 训练数据（去 karpathy/makemore 仓库下载）
    ├── count_based.py         # 今天的主角：计数法 bigram
    └── logs/
        ├── bigram_samples.txt # 生成的 20 个名字
        └── W3_day5_log.md     # 今日训练日志
```

`count_based.py` 之所以这么命名，是为了和明天的 `nn_based.py`（神经网络版）形成对比——文件名一对照，你脑子里就立刻浮现"两种实现、几乎相同的 NLL"的关键洞察。

**怎么拿到 `names.txt`？** 从 Karpathy 的 makemore 仓库下载：

```bash
# 在 bigram/ 目录下执行
curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
# 或：wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

下载后用任何文本编辑器打开看一眼，确认是 32,033 行小写人名。

---

## 3. 完整可运行代码：`bigram/count_based.py`

```python
"""
W3 Day5: bigram 语言模型——计数法版本
对应视频：Karpathy "Zero to Hero" EP2 第 0:00 - 1:00

整个文件只做四件事：
  1. 读 names.txt，构造 27 个字符的字典
  2. 扫一遍数据，把每对相邻字符的计数填进 27×27 矩阵 N
  3. 行归一化得到概率矩阵 P
  4. 用 P 采样生成 20 个名字，并算训练集 NLL

不需要 torch.nn，只用 torch 的张量运算（也可以纯 numpy，
但用 torch 是为了和明天 nn_based.py 共用同一套 API）。
"""

from pathlib import Path
import torch


# ======================================================================
# 1. 读数据 + 构造字符表
# ======================================================================
DATA_PATH = Path(__file__).parent / "names.txt"
words = DATA_PATH.read_text(encoding="utf-8").splitlines()
words = [w.strip().lower() for w in words if w.strip()]
print(f"训练集大小: {len(words)} 个名字")
print(f"前 5 个示例: {words[:5]}")
print(f"最短/最长: {min(len(w) for w in words)} / {max(len(w) for w in words)}")

# 27 个字符: 26 字母 + 1 个起止符 '.'
chars = sorted(set("".join(words)))           # 26 个小写字母
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
V = len(stoi)                                  # vocab size = 27
assert V == 27, f"字符表大小应为 27，实际 {V}"


# ======================================================================
# 2. 构造 27×27 计数矩阵 N
# ======================================================================
N = torch.zeros((V, V), dtype=torch.int64)

for w in words:
    # 关键:首尾各加一个 '.',让模型学开头/结尾
    chs = ["."] + list(w) + ["."]
    for c1, c2 in zip(chs, chs[1:]):
        i, j = stoi[c1], stoi[c2]
        N[i, j] += 1

print(f"\n总 bigram 对数: {N.sum().item()}")
print(f"最常见的 5 对 bigram:")
flat = N.flatten()
top5_idx = torch.topk(flat, 5).indices
for idx in top5_idx.tolist():
    i, j = idx // V, idx % V
    print(f"  '{itos[i]}{itos[j]}' 出现 {N[i, j].item()} 次")


# ======================================================================
# 3. 归一化得到概率矩阵 P
# ======================================================================
# Laplace 平滑: 全体 +1, 防止训练集没出现的 bigram 在测试集触发 log(0)
N_smoothed = (N + 1).float()
P = N_smoothed / N_smoothed.sum(dim=1, keepdim=True)
#                                  ^^^^^^^^^^^^^^
# keepdim=True 保留 (27,1) 维度,确保按"行"广播除法
# 否则会按"列"广播,得到的是 P(prev|next) — 完全错误的方向

# 自检: 每行概率和应为 1
assert torch.allclose(P.sum(dim=1), torch.ones(V), atol=1e-6), "归一化失败"


# ======================================================================
# 4. 用 torch.multinomial 采样生成 20 个名字
# ======================================================================
g = torch.Generator().manual_seed(2147483647)   # 固定种子,保证可复现

generated = []
for _ in range(20):
    out = []
    ix = 0                                       # 从 '.' 开始
    while True:
        p_row = P[ix]                            # (27,) 当前字符之后的分布
        ix = torch.multinomial(
            p_row, num_samples=1, replacement=True, generator=g
        ).item()
        if ix == 0:                              # 采到 '.' 就停止
            break
        out.append(itos[ix])
    generated.append("".join(out))

print(f"\n生成的 20 个名字:")
for name in generated:
    print(f"  {name}")

# 写到日志文件
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
(log_dir / "bigram_samples.txt").write_text("\n".join(generated), encoding="utf-8")


# ======================================================================
# 5. 计算训练集 NLL (作为评估指标)
# ======================================================================
log_likelihood = 0.0
n_pairs = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for c1, c2 in zip(chs, chs[1:]):
        i, j = stoi[c1], stoi[c2]
        log_likelihood += torch.log(P[i, j])
        n_pairs += 1

nll = -log_likelihood / n_pairs
print(f"\n训练集 NLL = {nll.item():.4f}")
print(f"对照基线: 27 维均匀分布 NLL = log(27) = {torch.log(torch.tensor(27.0)).item():.4f}")
print(f"如果你的 NLL 在 2.40 ~ 2.50 之间, 说明模型正常工作")
```

**运行结果应该长这样：**

```
训练集大小: 32033 个名字
前 5 个示例: ['emma', 'olivia', 'ava', 'isabella', 'sophia']
最短/最长: 2 / 15

总 bigram 对数: 228146
最常见的 5 对 bigram:
  'n.' 出现 6763 次
  'a.' 出现 6640 次
  '.a' 出现 4410 次
  'an' 出现 5438 次
  'e.' 出现 3983 次

生成的 20 个名字:
  mor
  axx
  minaymoryles
  kondlaisah
  anchshizarie
  ...

训练集 NLL = 2.4541
对照基线: 27 维均匀分布 NLL = log(27) = 3.2958
如果你的 NLL 在 2.40 ~ 2.50 之间, 说明模型正常工作
```

具体名字会因 `manual_seed` 与平滑细节有微小差异，但 NLL 应该非常接近 2.45。

---

## 4. 把代码读进脑子里：5 个关键问答

读完代码后你应当能回答下面 5 题。如果某题答不上来，回去重读对应的代码段。

### Q1：为什么用 `Path(__file__).parent / "names.txt"` 而不是直接 `"names.txt"`？

因为相对路径依赖**当前工作目录**，从不同位置运行脚本会失败。`__file__` 永远指向脚本本身，`.parent` 拿到脚本所在目录，这样不管从哪 `python count_based.py`，都能找到同目录下的 `names.txt`。这是项目工程化的小习惯，但日积月累能省你大量调试时间。

### Q2：为什么字符表是 `{'.': 0, 'a': 1, ..., 'z': 26}` 而不是 `{'a': 0, ..., 'z': 25, '.': 26}`？

只是一个习惯——把特殊符号放 0 号位，让普通字符占连续段。后面 makemore 的代码里也是这套约定，跟着 Karpathy 走可以减少认知负担。**重要的是一致性**：`stoi` 和 `itos` 必须互逆，一旦定下来就别再换。

### Q3：`zip(chs, chs[1:])` 为什么能枚举相邻对？

`chs = ['.', 'e', 'm', 'm', 'a', '.']`
`chs[1:] = ['e', 'm', 'm', 'a', '.']`

`zip` 把它们按位置配对，输出 `('.','e'), ('e','m'), ('m','m'), ('m','a'), ('a','.')`——正好就是所有相邻对。这是 Python 里枚举相邻 pair 的标准写法，比 `for i in range(len(chs)-1)` 干净。

### Q4：`torch.multinomial(p_row, num_samples=1, replacement=True)` 为什么传 `replacement=True`？

我们每次只采 1 个，理论上 `replacement` 设啥都不影响这一次的结果。但 PyTorch 在某些版本里 `replacement=False` 会有更严格的检查（比如要求概率非零的位置数 ≥ num_samples），用 `True` 更稳妥。**记住这个习惯：单次采样总是 `replacement=True`**。

### Q5：为什么 NLL 用 `torch.log(P[i, j])` 而不是 `np.log(P[i, j])`？

因为 `P` 是 PyTorch tensor，对 tensor 用 `np.log` 会先把它转成 numpy（可能触发 GPU→CPU 同步），慢且容易出 bug。**统一在一个数值框架里完成所有运算**，是减少隐藏 bug 的好习惯。

---

## 5. 思考题（写到 `W3_day5_log.md`）

这三题都没有标准答案，但能写出自己的回答说明你理解到位了。

### 思考题 1：如果不加 `.` 起止符会发生什么？

具体描述两件事：
- 训练上少学了什么模式？
- 生成时怎么决定"开头"和"结束"？分别用什么策略也凑合能跑？（例如随机选一个起始字符、固定生成长度 8）和加 `.` 相比，缺点在哪？

### 思考题 2：为什么 bigram 的 NLL 不可能比 2.4 低很多？

提示从两个角度想：
- 信息论角度：bigram 只用了**前 1 个字符**作为上下文，丢掉了人名里"位置 / 长度 / 跨字符模式"的所有信息。这部分丢失的信息会以"残余熵"的形式留在 NLL 里。
- 实证角度：如果你尝试不平滑、调整 `N` 的初始值（比如全体 +0.5 而不是 +1），NLL 会有微小变化但永远在 2.4 附近——这说明这是模型结构本身的天花板，不是参数问题。

### 思考题 3：从 `argmax` 切换到 `multinomial` 会怎么改变生成结果？

把代码里的：

```python
ix = torch.multinomial(p_row, num_samples=1, replacement=True, generator=g).item()
```

改成：

```python
ix = p_row.argmax().item()
```

跑一遍。**预测：会陷入死循环或得到全部相同的输出**。在你的 log 里写出实际现象 + 解释为什么——这是理解"采样 vs 贪心"差异的最佳实操。

---

## 6. 常见 bug 排查表

| 现象 | 大概率原因 |
|---|---|
| 报错 `FileNotFoundError: names.txt` | 没下载数据，或脚本不在 `bigram/` 目录下，或没用 `Path(__file__).parent` |
| `V != 27` | 数据里有非小写字母字符（数字、空格、奇怪的 unicode），用 `chars = sorted(set(...))` 后 print 一下 |
| NLL 算出来是 `nan` 或 `inf` | 没做 Laplace 平滑，训练集里某个 bigram 没出现导致 `log(0)` |
| NLL ≈ 3.3（接近瞎猜） | 多半是归一化方向错了——`keepdim=False` 或者忘了 `dim=1`，得到的是按列归一化或全局归一化 |
| 生成的全部以 'a' 开头 | 用了 `argmax` 不是 `multinomial`，或者 `manual_seed` 写在了 `for` 循环内每次重置 |
| 生成的名字都很长 / 不停 | 没把 `ix == 0` 作为终止条件，或起止符的 id 不是 0（检查 `stoi['.']`） |
| NLL ≈ 2.7（比 2.45 高一档） | 平滑太狠——比如 `N + 100` 而不是 `N + 1`，把分布稀释成接近均匀了 |

---

## 7. 与 Day6 的衔接（明天看 EP2 后半段会做的事）

明天 Day6 的核心实验：**用神经网络（一层 27→27 的全连接 + softmax）训练同一个 bigram 任务**，然后比较：

- 训练完后 `softmax(W)` 是否近似等于今天的 `P`？
- 神经网络版的 NLL 是否也收敛到 ≈ 2.45？

**预期答案：是的，两者会非常接近。** 这一惊人的等价是 EP2 的高潮——它告诉你：神经网络在这个简单任务上学到的，**和你今天用计数 + 归一化算出来的解几乎完全一致**。原因是：

> 在 27 维 one-hot 输入 + softmax + 交叉熵损失的设定下，最大似然解的解析形式就是计数归一化。

也就是说，**梯度下降在这里是在用一种"绕远路"的方式做你今天直接做的事**。这不是说神经网络多余——它的真正价值在你扩展到看更多字符（MLP）、更长上下文（Transformer）时才显现出来；那时候计数法就因为组合爆炸（27 个字符的 5-gram 就要 27⁵ ≈ 1430 万个表项）而完全失效，只有神经网络的参数共享才能 scale。

**今天的 bigram 是 makemore 系列的"基线"。** 把今天的 NLL = 2.45 这个数字写到 log 最显眼的地方——它会成为后面所有更复杂模型的对照标尺。

---

## 8. 当日交付清单

- [ ] `bigram/names.txt`（已下载）
- [ ] `bigram/count_based.py`（运行无报错，产出生成结果）
- [ ] `bigram/logs/bigram_samples.txt`（20 个生成的名字）
- [ ] `bigram/logs/W3_day5_log.md`（含：实际 NLL 数值截图、3 道思考题的回答、`argmax` 替换实验的现象）
- [ ] 算法：洛谷 P8306 或 P2580 AC 1 道
- [ ] 数学：协方差矩阵定义与半正定性笔记
- [ ] AI 工具：第 4 次核验练习（最近半年技术新闻类）

完成标准：能复现接近 Karpathy 视频中的 NLL 值（2.40 ~ 2.50），生成的名字大致有元音辅音交替规律。
