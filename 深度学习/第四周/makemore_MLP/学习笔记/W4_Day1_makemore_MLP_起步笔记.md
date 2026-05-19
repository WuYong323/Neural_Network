# Week 4 · Day 1（2026-05-18）：makemore MLP 起步——从 bigram 到 Embedding

> 覆盖任务：
> 1. 建好 `week4_makemore_mlp/` 目录
> 2. 看 EP3 第 0:00-0:40
> 3. 实现 `build_dataset(words, block_size=3)`，验证形状
> 4. 实现 embedding 层，`emb = C[X]` 形状正确
> 5. 完成 `tech_notes/embedding_as_lookup.md`（含 one-hot @ W vs C[X] 速度对比）
>
> 阅读对象：你自己——已完成第3周 micrograd + bigram、刚开始接触 PyTorch fancy indexing 的状态。

---

## 0. 本节学习目标（看完应能回答）

1. 为什么要从 bigram 升级到 MLP？bigram 的硬伤是什么？
2. `block_size` 在 makemore 里是什么？它对应 GPT 里的哪个概念？
3. `build_dataset` 输出 `(X, Y)` 张量的形状和语义？
4. `C[X]` 这一行到底在做什么？为什么它在**数学上**等价于 `one_hot(X) @ C`，但在**工程上**是完全不同的算子？
5. embedding lookup 为什么是 memory-bound 算子？这对 LLM 推理意味着什么？

---

## 1. 项目目录设计

```
week4_makemore_mlp/
├── README.md
├── data/
│   └── names.txt              # Karpathy 的 32k 英文人名数据
├── src/
│   ├── __init__.py
│   ├── vocab.py               # 字符表 stoi / itos
│   ├── dataset.py             # build_dataset
│   └── embedding_demo.py      # embedding 层与 lookup 实验
├── tests/
│   └── test_dataset.py        # 形状与边界验证
├── logs/
└── tech_notes/
    └── embedding_as_lookup.md
```

**为什么这样组织？**

| 目录 | 角色 | 工业类比 |
|---|---|---|
| `src/` | 可复用模块 | 后续 `train.py`、`model.py` 都从这里 import |
| `tests/` | 形状/边界回归 | 第2周梯度检验经验沿用——任何关键转换必须有自动验证 |
| `tech_notes/` | 笔记沉淀 | 笔记和代码同仓库，未来翻回来不会脱钩 |
| `logs/` | 实验输出 | 训练曲线、profiler trace 都丢这里 |

**一次性建好（在仓库根目录执行）：**

```bash
mkdir week4_makemore_mlp && cd week4_makemore_mlp
mkdir -p data src tests logs tech_notes
touch README.md src/__init__.py src/vocab.py src/dataset.py src/embedding_demo.py tests/test_dataset.py tech_notes/embedding_as_lookup.md
# 下载数据
curl -L -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

---

## 2. EP3 第 0:00-0:40 内容速览

Karpathy 这 40 分钟讲了三件事：

### 2.1 为什么 bigram 不够用

bigram 只看**前 1 个字符**。当上下文加到前 3 个字符时，bigram 那张转移矩阵就要变成 `27 × 27 × 27 × 27 = 53 万` 个参数；前 10 个字符是 `27^11 ≈ 5.6 × 10^15`——**指数爆炸**。

> 这就是计数法语言模型的尽头。所有"延长上下文 = 参数指数爆炸"的问题，都要靠**参数共享**来解。

### 2.2 Bengio 2003 的核心思想（这周复现的论文）

| 计数 bigram | Bengio MLP |
|---|---|
| 直接存 P(next \| prev) 的查找表 | 每个字符映射到一个**低维稠密向量** `C[i] ∈ ℝ^d` |
| 参数随上下文指数增长 | 参数随上下文线性增长 |
| 没见过的组合就是 0 概率 | 即使没见过 `abc`，如果模型见过 `abd, abe`，`a, b` 的向量学好了，也能给 `abc` 合理概率 |

**关键洞察：把"离散的字符"映射成"连续的向量"，让相似字符（如 `a` 和 `e` 都是元音）在向量空间里靠近——这就是 Embedding。**

### 2.3 网络结构总览（这周要复现的目标）

```
输入：前 3 个字符的索引       (N, 3)
  ↓ embedding lookup C[X]
embedded：                   (N, 3, 2)    ← 每个字符变成 2 维向量
  ↓ flatten
flat：                       (N, 6)
  ↓ Linear(6 → 100) + tanh
hidden：                     (N, 100)
  ↓ Linear(100 → 27)
logits：                     (N, 27)
  ↓ F.cross_entropy(logits, Y)
loss：                       标量
```

今天只做**前两层（输入 → embedding）**，明天再接 MLP。

---

## 3. 数据集构造：`build_dataset`

### 3.1 设计思路

输入：一堆人名（字符串列表），如 `["emma", "olivia", "ava", ...]`。
目标：把它们切成 "前 `block_size` 个字符 → 第 `block_size+1` 个字符" 的训练样本。

**重要约定：用 `.` 作为起止符**。这样：
- 名字开头：`...emma` → 用 `...` 预测 `e`
- 名字结尾：`emma.` → 用 `mma` 预测 `.`（模型学会"该停了"）

样本展开示例（`block_size=3`，名字 `emma`）：

```
context        target
. . .    →     e
. . e    →     m
. e m    →     m
e m m    →     a
m m a    →     .
```

一个名字 = `len(name) + 1` 个样本。

### 3.2 字符表（`src/vocab.py`）

```python
"""字符表：26 个小写字母 + 起止符 '.'，共 27 个 token。"""
from typing import Iterable


def build_vocab(words: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    """从语料构造 char -> id 和 id -> char 两张表。

    '.' 永远占用 id=0，作为起止符 / padding。

    Args:
        words: 训练语料（小写字符串）

    Returns:
        (stoi, itos)
    """
    chars = sorted(set("".join(words)))
    stoi = {".": 0}
    stoi.update({ch: i + 1 for i, ch in enumerate(chars)})
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


VOCAB_SIZE = 27  # 26 字母 + '.'
```

**为什么把 `.` 固定在 id=0？**
工业代码里 padding token 通常占 0（HuggingFace、fairseq、Karpathy 都这样）——这样 `torch.zeros(...)` 直接就是 "全 pad" 的张量，省一次显式 fill。

### 3.3 `build_dataset` 实现（`src/dataset.py`）

```python
"""把人名列表切成 (X, Y) 训练对。"""
from __future__ import annotations

import torch
from torch import Tensor


def build_dataset(
    words: list[str],
    stoi: dict[str, int],
    block_size: int = 3,
) -> tuple[Tensor, Tensor]:
    """构造上下文-目标对。

    Args:
        words: 人名列表，小写、不含非字母字符
        stoi: 字符到 id 的映射，'.' 必须存在
        block_size: 上下文长度。bigram=1；EP3 默认=3

    Returns:
        X: shape (N, block_size), dtype int64, 每行是 block_size 个字符 id
        Y: shape (N,), dtype int64, 每个元素是下一个字符的 id

    例：
        words=["emma"], block_size=3, stoi['.']=0, stoi['e']=5, stoi['m']=13, stoi['a']=1
        X = [[0,0,0],[0,0,5],[0,5,13],[5,13,13],[13,13,1]]
        Y = [5, 13, 13, 1, 0]
        N = len("emma") + 1 = 5
    """
    assert "." in stoi, "vocab must contain '.' as start/end token"

    X_rows: list[list[int]] = []
    Y_rows: list[int] = []
    pad_id = stoi["."]

    for word in words:
        context = [pad_id] * block_size  # 滑动窗口初始化为全 '.'
        for ch in word + ".":            # 名字末尾追加 '.'，让模型学会停止
            target = stoi[ch]
            X_rows.append(context.copy())
            Y_rows.append(target)
            context = context[1:] + [target]  # 窗口右移一格

    X = torch.tensor(X_rows, dtype=torch.long)
    Y = torch.tensor(Y_rows, dtype=torch.long)
    return X, Y
```

### 3.4 形状验证（`tests/test_dataset.py`）

```python
"""形状/语义回归测试。命令：pytest tests/ -v"""
import torch

from src.dataset import build_dataset
from src.vocab import build_vocab


def test_single_name_shape():
    words = ["emma"]
    stoi, _ = build_vocab(words)
    X, Y = build_dataset(words, stoi, block_size=3)

    # emma → 5 个样本（包含末尾的 '.')
    assert X.shape == (5, 3)
    assert Y.shape == (5,)
    assert X.dtype == torch.long
    assert Y.dtype == torch.long


def test_first_row_is_all_padding():
    """每个名字的第一个样本，context 必须全部是 '.'（id=0）。"""
    stoi, _ = build_vocab(["emma"])
    X, _ = build_dataset(["emma"], stoi, block_size=3)
    assert (X[0] == 0).all()


def test_last_target_is_dot():
    """每个名字的最后一个样本，target 必须是 '.'（表示"该停了"）。"""
    stoi, _ = build_vocab(["emma"])
    _, Y = build_dataset(["emma"], stoi, block_size=3)
    assert Y[-1].item() == stoi["."]


def test_count_consistency():
    """N = sum(len(w) + 1)."""
    words = ["emma", "olivia", "ava"]
    stoi, _ = build_vocab(words)
    X, Y = build_dataset(words, stoi, block_size=3)
    expected = sum(len(w) + 1 for w in words)
    assert X.shape[0] == expected == Y.shape[0]


def test_block_size_variants():
    """改 block_size 不应该影响样本数量。"""
    words = ["emma"]
    stoi, _ = build_vocab(words)
    for bs in [1, 3, 5, 8]:
        X, Y = build_dataset(words, stoi, block_size=bs)
        assert X.shape == (5, bs)
        assert Y.shape == (5,)
```

**手算一遍**（这是验证你真的理解的唯一方式）：

`words = ["emma"]`，`block_size = 3`，建出来的 `stoi = {'.':0, 'a':1, 'e':2, 'm':3}`（顺序按字典序）。

| 步骤 | context | target | X 这一行 | Y |
|---|---|---|---|---|
| 0 | `...` | `e` | `[0,0,0]` | `2` |
| 1 | `..e` | `m` | `[0,0,2]` | `3` |
| 2 | `.em` | `m` | `[0,2,3]` | `3` |
| 3 | `emm` | `a` | `[2,3,3]` | `1` |
| 4 | `mma` | `.` | `[3,3,1]` | `0` |

`X.shape == (5, 3)`，`Y.shape == (5,)`——和测试断言一致。

---

## 4. Embedding 层：`emb = C[X]` 的本质

### 4.1 一句话定义

**Embedding 层 = 一张可训练的查找表 `C ∈ ℝ^(V, d)`，第 `i` 行是字符 `i` 的稠密向量表示。**

```python
import torch

VOCAB_SIZE = 27
EMBED_DIM = 2          # EP3 一开始用 2 维（方便可视化），之后会调大到 10
C = torch.randn(VOCAB_SIZE, EMBED_DIM, requires_grad=True)
```

### 4.2 PyTorch fancy indexing：`C[X]` 做了什么

```python
X = torch.tensor([[0, 0, 0],
                  [0, 0, 2],
                  [0, 2, 3]])         # shape (3, 3), dtype long
emb = C[X]                            # shape (3, 3, 2)
```

**广播规则**：用一个整数张量作为索引时，结果形状 = 索引张量形状 + 被索引张量"被消掉那一维之后"的形状。

```
C.shape       = (27, 2)
X.shape       = (3, 3)
C[X].shape    = (3, 3,    2)
                └─X的形状┘ └C 第一维被 X 替换之后剩下的┘
```

**含义：**`emb[i, j] = C[X[i, j]]`——把 `X` 里每个整数都换成 `C` 里对应那行。

### 4.3 完整 demo（`src/embedding_demo.py`）

```python
"""Embedding 层最小可运行示例。

运行：python -m src.embedding_demo
"""
import torch

from src.dataset import build_dataset
from src.vocab import VOCAB_SIZE, build_vocab


def main() -> None:
    torch.manual_seed(2147483647)  # Karpathy 视频里用的种子

    words = ["emma", "olivia", "ava", "isabella", "sophia"]
    stoi, itos = build_vocab(words)
    X, Y = build_dataset(words, stoi, block_size=3)

    print(f"X.shape = {tuple(X.shape)}")          # e.g. (32, 3)
    print(f"Y.shape = {tuple(Y.shape)}")          # e.g. (32,)

    # Embedding 表：27 个字符，每个映射到 2 维向量
    C = torch.randn(VOCAB_SIZE, 2, requires_grad=True)
    emb = C[X]                                    # (N, 3, 2)

    print(f"C.shape   = {tuple(C.shape)}")
    print(f"emb.shape = {tuple(emb.shape)}")
    print(f"emb[0]    =\n{emb[0]}")               # 第一个样本：3 个字符 × 2 维

    # 验证：emb[i, j] 必须等于 C[X[i, j]]
    i, j = 0, 1
    assert torch.allclose(emb[i, j], C[X[i, j]])
    print("✓ fancy indexing 语义验证通过")


if __name__ == "__main__":
    main()
```

---

## 5. `embedding_as_lookup.md` 核心内容

> 这一节就是你要写到 `tech_notes/embedding_as_lookup.md` 的全部内容。

### 5.1 数学等价：`one_hot(x) @ C == C[x]`

设字符 id `x = 5`，词表大小 `V = 27`，embedding 维度 `d`。

```
one_hot(x) ∈ ℝ^V        是一个长度 27 的向量，只有第 5 位是 1，其余全 0
one_hot(x) @ C ∈ ℝ^d    把 C 的每一行按 0 或 1 加权求和——只剩第 5 行被保留
                        结果就是 C[5]
```

**严格等价。证明用一行：**
```
(one_hot(x) @ C)[k] = Σ_i one_hot(x)[i] · C[i, k] = 1 · C[x, k] = C[x, k]
```

### 5.2 工程不等价：为什么 `C[X]` 快 10-100 倍

| 维度 | `one_hot(X).float() @ C` | `C[X]` |
|---|---|---|
| 浮点乘加次数 | `N · V · d` | 0 |
| 内存访问 | `N·V`（读 one-hot）+ `V·d`（读 C）| `N`（读索引）+ `N·d`（读 C 对应行） |
| 中间内存 | 必须实例化 `N × V` 的 one-hot 矩阵 | 没有中间张量 |
| 反向传播 | 对整个 `C` 算梯度，绝大部分是 0 | 用 `index_add_` 只更新被访问的行（PyTorch 内部优化） |
| GPU 实现 | 走 GEMM kernel（cuBLAS） | 走 gather kernel（memory copy） |

**核心：one-hot 那条路所做的 99% 的乘法都是 `0 × c = 0`，纯粹浪费算力和显存带宽。**

### 5.3 速度对比实验代码

```python
"""one-hot @ W vs C[X] 速度对比。

把它追加到 src/embedding_demo.py 末尾，或单独存为 benchmark_lookup.py。
"""
import time
import torch
import torch.nn.functional as F


def benchmark(device: str = "cpu", n: int = 10_000, vocab: int = 27, dim: int = 64) -> None:
    """对比两种 embedding 实现的耗时。

    Args:
        device: "cpu" 或 "cuda"
        n: batch 大小
        vocab: 词表大小
        dim: embedding 维度
    """
    device_t = torch.device(device)
    X = torch.randint(0, vocab, (n,), device=device_t)
    C = torch.randn(vocab, dim, device=device_t)

    # 预热（GPU kernel 首次启动有编译开销，会污染计时）
    for _ in range(3):
        _ = F.one_hot(X, vocab).float() @ C
        _ = C[X]
    if device == "cuda":
        torch.cuda.synchronize()

    iters = 100

    # 方法 A：one-hot @ C
    t0 = time.perf_counter()
    for _ in range(iters):
        out_a = F.one_hot(X, vocab).float() @ C
    if device == "cuda":
        torch.cuda.synchronize()
    t_a = (time.perf_counter() - t0) / iters * 1e6  # 微秒

    # 方法 B：fancy indexing
    t0 = time.perf_counter()
    for _ in range(iters):
        out_b = C[X]
    if device == "cuda":
        torch.cuda.synchronize()
    t_b = (time.perf_counter() - t0) / iters * 1e6

    # 验证两者数值一致
    assert torch.allclose(out_a, out_b, atol=1e-5)

    print(f"[{device}] N={n}, V={vocab}, d={dim}")
    print(f"  one_hot @ C : {t_a:8.1f} µs / call")
    print(f"  C[X]        : {t_b:8.1f} µs / call")
    print(f"  speedup     : {t_a / t_b:6.2f}×")


if __name__ == "__main__":
    benchmark("cpu", n=10_000, vocab=27, dim=64)
    if torch.cuda.is_available():
        benchmark("cuda", n=10_000, vocab=27, dim=64)
```

**参考量级（CPU，词表 27，dim=64，N=10000）：**
- `one_hot @ C` ≈ 1500-3000 µs
- `C[X]` ≈ 30-80 µs
- 加速比约 30-100×（具体看硬件）

**词表越大、batch 越大，差距越夸张**——LLM 词表通常 5万+，`one-hot @ W` 那种实现根本跑不起来。

### 5.4 工业延伸：embedding 在 LLM 推理里的角色

把 §5.3 的速度对比代入真实 LLM 数字：

| 项 | GPT-2 small | LLaMA-3 8B |
|---|---|---|
| 词表 V | 50,257 | 128,256 |
| hidden_dim d | 768 | 4,096 |
| embedding 表大小 | 50257 × 768 × 4B ≈ **154 MB** | 128256 × 4096 × 2B ≈ **1 GB** |
| 一次 forward 取多少行 | seq_len × batch_size 行 | 同左 |

**关键观察：**

1. **Embedding 表是 GB 级显存常驻**。LLaMA-3 8B 推理时，参数总共 16GB（FP16），embedding 占 1GB——比任何一层 Linear 都大。
2. **Lookup 算子完全是 memory-bound**——没有计算，只有"读 N 行各 d 个 float"。GPU 算力再强也救不了，瓶颈是显存带宽。
3. **这就是为什么 LLM 推理优化里 embedding 几乎不能进一步优化**——它已经是最朴素的内存访问了，没有冗余可压。能做的优化是把表本身压小：**embedding 量化**（INT8 甚至 INT4）、**embedding 表共享**（input embedding 和 output projection 共用一张表，叫 weight tying）。

> **回扣第3周笔记**：`autograd_explained.md` 里 §Q9 的"稀疏更新优惠券"——bigram 写 `xenc @ W` 时你已经感受到 one-hot 浪费了算力。今天这一节，把那张优惠券从"训练侧"延伸到了"推理侧"：训练时 `C[X]` 让反向传播只更新被访问的行，推理时让前向不做无谓的乘法。**同一个洞察，训练和推理都受益。**

### 5.5 PyTorch 标准模块：`nn.Embedding`

工业代码里不会手动写 `C = randn(...); emb = C[X]`，而是用：

```python
embedding = torch.nn.Embedding(num_embeddings=27, embedding_dim=2)
emb = embedding(X)   # 等价于 embedding.weight[X]
```

**`nn.Embedding` 多做了什么？**
- 自动注册参数（`nn.Module` 接管，可以 `model.parameters()` 收集）
- 提供 `padding_idx` 参数（指定 padding token 的行不参与梯度更新）
- 提供 `max_norm` 参数（每次 lookup 后把这些行重归一化，防止爆炸）
- 提供 `sparse=True` 选项（用 SparseAdam 优化器时反向更稀疏）

**第4周这一天先用 `randn + C[X]` 是为了让你看清"它就是一张可训练的表"**。后面切到 `nn.Embedding` 是工程封装，本质不变。

---

## 6. 今日打卡 + 验收

Day 1 任务清单（对应计划 line 1148-1157）：

- [ ] 算法：洛谷 P1880 石子合并 AC（**今天不在本笔记覆盖范围内**）
- [x] DL：建好 `week4_makemore_mlp/` 目录 → §1 命令一次性建好
- [x] DL：看 EP3 第 0:00-0:40 → §2 关键内容回顾
- [x] DL：实现 `build_dataset(words, block_size=3)`，验证形状 → §3 + `pytest tests/`
- [x] DL：实现 embedding 层，`emb = C[X]` 形状正确 → §4 demo 跑通
- [x] DL：完成 `tech_notes/embedding_as_lookup.md` → §5 全部内容直接复制即可

**验收标准（来自计划 line 950）：**

> 能用一句话解释"为什么 embedding 在数学上和 one-hot @ W 等价，但在工程上是完全不同的算子"。

**参考答案：**

> **数学上**，因为 one-hot 向量只有一位是 1，矩阵乘法 `one_hot(x) @ C` 的结果恰好是 `C` 的第 `x` 行——结果和直接索引完全相同。**工程上**，矩阵乘法走的是 GEMM kernel，要做 `N·V·d` 次乘加并实例化 `N×V` 的中间矩阵；fancy indexing 走 gather kernel，只做 `N` 次内存读，零乘法、零中间矩阵——所以在大词表场景下能快两个数量级，并且是反向传播里"只更新被访问行"这个稀疏优化能生效的前提。

---

## 7. 自测题（合上文档默答）

1. `build_dataset(["abc"], stoi={'.':0,'a':1,'b':2,'c':3}, block_size=2)` 输出的 `X` 和 `Y` 各是什么？（手算）
2. 如果 `C.shape = (27, 10)`、`X.shape = (32, 3)`，那么 `C[X].shape` 是多少？
3. 为什么我们要在名字末尾加 `.`？去掉它会发生什么？
4. `F.one_hot(X, 27).float() @ C` 和 `C[X]` 的反向传播有什么差别？哪个更省显存？
5. LLaMA-3 8B 的 embedding 表占用多少显存？它是 memory-bound 还是 compute-bound？

> 答案在 §3.4、§4.2、§3.1（小节末尾"约定"段）、§5.2、§5.4。

---

## 8. 与已有笔记的串联

| 今天的内容 | 关联点 |
|---|---|
| `build_dataset` 的滑动窗口 | 第3周 bigram 是 block_size=1 的特例 |
| `C[X]` 是 lookup | 第3周 `autograd_explained.md` §Q9 的"稀疏优惠券"工程化 |
| Embedding 是参数共享 | 解决 §2.1 计数法上下文指数爆炸 |
| Memory-bound 算子 | 下周 profiler、第8周 LLM 推理优化的核心概念之一 |

明天（Day 2）的内容：把 `emb` flatten 后接 MLP，跑通完整前向 + 第一次 lr-range test。今天 §5.4 提到的 "embedding 占显存大头" 那个观察，下周看 ResNet 的时候会变成"卷积参数为什么反而比 FC 小"的对照案例。

---

*笔记生成日期：2026-05-18（W4 Day 1）*
