# W5 Day3 学习笔记：im2col 与 FLOPs–MAC–latency 三角关系

> **本周定位**：把卷积从"会调用 `nn.Conv2d`"升级到"能从原理 → 显存 → 推理路径完整讲清楚"。
> **今天两个核心**：① 卷积的本质其实是一次矩阵乘法（im2col）；② 为什么"计算量少"不等于"跑得快"（FLOPs–MAC–latency 三角）。
> **串联**：呼应 [[conv_anatomy]]（Day1 Conv2d 四件套）、[[lenet_vs_modern]]（Day2 经典网络）；为 [[residual_grad_flow]]（Day4 ResNet 梯度流）、[[profiler_chrome_trace]]（Day6 profiler 实战）打地基。

---

## 0. 带着这些问题看（今天的学习目标）

读完这份笔记，你应该能**不看资料**回答下面 6 个问题。能答出来，今天就过关：

1. VGG 为什么整张网络只用 3×3 卷积？两个 3×3 和一个 5×5 是什么关系？
2. Inception 里那个 1×1 卷积，看起来"啥也没干"，它到底有什么用？
3. 卷积明明是"滑窗"，为什么说它"本质是矩阵乘法"？GPU 为什么偏爱这种形式？
4. 为什么 1×1 卷积是"内存瓶颈"，而 3×3 卷积是"算力瓶颈"？
5. MobileNet 把计算量降了 8~9 倍，为什么实测延迟只降了 2~3 倍？
6. 拿到一个算子，怎么一眼判断它是 compute-bound（算力受限）还是 memory-bound（带宽受限）？

---

## 1. 经典 CNN 架构速览（Course 4 Week 2 下半）

> 这部分是"概念预热"。ResNet 明天（Day4）专门一天讲梯度流，今天只建立直觉，重点是 **1×1 卷积**——它直接关系到下午的 FLOPs/MAC 分析。

### 1.1 VGG —— "把大卷积拆成小卷积"

**是什么**：VGG（Visual Geometry Group，牛津大学视觉组，2014 年提出）的核心思想极其朴素——**整张网络只用 3×3 卷积，靠不断堆叠把网络加深**。

**为什么这么做**：关键洞察是"**两个 3×3 卷积叠起来，感受野等于一个 5×5，但更划算**"。

这里先解释一个名词。**感受野（Receptive Field）**：指输出特征图上的一个点，"能看到"原始输入图像多大的一块区域。

> 类比：你站在一栋楼前往后退，退得越远，视野里能装下的东西越多。卷积层堆得越深，每个输出点能"回看"的原图范围就越大——这块范围就是感受野。

现在算笔账（设输入输出通道都是 C）：

- **一个 5×5 卷积**：感受野 5×5，参数量 = 5×5×C×C = **25C²**
- **两个 3×3 卷积**：感受野同样是 5×5，参数量 = 2×(3×3×C×C) = **18C²**

两个 3×3 不仅省了 28% 参数，中间还多夹了一层 ReLU（多一次非线性），表达能力反而更强。这就是 VGG 的全部哲学：**小卷积 + 深堆叠 > 大卷积 + 浅堆叠**。

**工业锚点**：3×3 之所以成为现代 CNN 的"标准砖块"，不只是因为参数省，更因为 cuDNN 对 3×3 卷积有专门的高度优化 kernel（**Winograd 算法**，一种用更少乘法换更多加法的快速卷积算法）。3×3 在硬件上几乎是"免费的尺寸红利"——这是后续几乎所有 backbone（主干网络）都偏爱 3×3 的根本原因。

### 1.2 1×1 卷积 —— 看起来啥也没干，其实是"通道混合器"

> 这是今天最反直觉、也最重要的一个点，请务必吃透——它是下午 FLOPs/MAC 分析的主角。

**是什么**：1×1 卷积，卷积核尺寸是 1×1。乍一看它每次只盯着一个像素，好像没法提取任何空间信息。

但关键在于：**1×1 卷积不在空间上做融合，而是在"通道（channel）"维度上做全连接。**

> 类比：把每个像素位置想象成一个人，每个人手里攥着 256 张卡片（= 256 个通道）。1×1 卷积做的事是——发给每个人一套**相同的"换牌规则"**，把手里 256 张卡线性组合成 64 张新卡（256→64 降维）。它完全不管邻居手里有什么牌（不看空间），只在每个人自己的牌堆里做线性混合。

它有三个实打实的作用：

1. **降维 / 升维**：把 256 通道压成 64 通道，大幅砍掉后续卷积的计算量。Inception 和 ResNet 的 Bottleneck 都靠它"先压缩 → 再卷积 → 再还原"。
2. **跨通道信息融合**：让不同通道的特征互相混合。
3. **几乎零成本加非线性**：`1×1 conv + ReLU`，等于在不改变图像分辨率的前提下，多加一层非线性。

**工业锚点（重要伏笔）**：1×1 卷积在数学上**等价于一个纯矩阵乘法**。后面讲 im2col 你会看到，当 kernel=1 时，im2col 退化成一次 reshape，整个卷积直接变成 `(N·H·W, C_in) @ (C_in, C_out)` 的矩阵乘。这让它成为典型的 **memory-bound 算子**——正是今天第三部分的核心案例。

### 1.3 Inception（GoogLeNet）—— "既要又要"的多分支

**是什么**：Inception（名字来自电影《盗梦空间》的梗 "we need to go deeper"）是 Google 2014 年的架构。核心思想：**在同一层里同时用 1×1、3×3、5×5 卷积和池化，把各分支结果拼接起来**，让网络自己决定哪种感受野有用。

**遇到的问题**：5×5 卷积太贵。**解决办法**：在每个昂贵分支前先用 1×1 卷积把通道数砍下来（如 256→64），算完再还原。这叫 **bottleneck（瓶颈）结构**——形如沙漏，中间细、两头粗。

这正是 1×1 卷积价值的最佳示范：用便宜的 1×1 先把通道数压下去，让后面昂贵的 3×3/5×5 在更少的通道上运算，整体计算量大幅下降。

### 1.4 ResNet 预热（明天细讲）

一句话：ResNet（2015）用 `y = F(x) + x` 的"残差连接"，让上百层的深网络也能正常训练。明天 Day4 会专门一天讲它的梯度流，今天只需记住一点——**它同样大量使用 1×1 卷积来做 Bottleneck（先降维、卷积、再升维）**。

### 1.5 架构演进的内在逻辑（把这条线串起来）

- **AlexNet（2012）**：证明深度学习能 work。
- **VGG（2014）**：用小卷积堆出深度。
- **Inception（2014）**：用多分支 + 1×1 bottleneck 提效率。
- **ResNet（2015）**：用残差连接解决"网络一深就训不动"。

**贯穿所有架构的，其实是同一个工程问题：怎么用更少的计算和显存，换到更强的表达能力？** 这个问题就把我们直接引向今天的两个硬核主题——卷积到底是怎么算出来的（im2col），以及怎么科学衡量"算得贵不贵、跑得快不快"（FLOPs–MAC–latency）。

---

## 2. im2col —— 卷积的本质是一次矩阵乘法

### 2.1 问题背景：朴素卷积为什么慢

先看"教科书版"的卷积怎么写。一个标准卷积要做的事是：把卷积核在图像上滑动，每个位置做一次"对应元素相乘再求和"。直接翻译成代码就是这样：

```python
# 环境：Python 3.10+，numpy>=1.24（仅用于教学，演示朴素卷积有多慢）
# 这段代码能跑，但故意写成最朴素的形式——6 层嵌套循环
import numpy as np

def conv2d_naive(X, W, stride=1, pad=0):
    """
    X: 输入 (N, C_in, H, W)        N=batch, C_in=输入通道
    W: 卷积核 (C_out, C_in, kh, kw)
    返回: (N, C_out, H_out, W_out)
    """
    N, C_in, H, Wd = X.shape
    C_out, _, kh, kw = W.shape
    # 先做 padding：在 H、W 两个维度四周补 0
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    H_out = (H + 2 * pad - kh) // stride + 1
    W_out = (Wd + 2 * pad - kw) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out))

    # 6 层循环：batch × 输出通道 × 输出高 × 输出宽 × (求和时还要遍历 C_in×kh×kw)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i * stride, j * stride
                    region = Xp[n, :, h0:h0 + kh, w0:w0 + kw]  # 取出当前滑窗
                    out[n, co, i, j] = np.sum(region * W[co])    # 逐元素乘再求和
    return out
```

**它慢在哪？** 这里要解释一个底层名词。

**局部性（Locality）/ 缓存友好（cache-friendly）**：CPU/GPU 读内存不是一个字节一个字节读的，而是一次拉一整块连续内存到高速缓存里。如果你的访问模式是"东摸一下西摸一下"（跳着访问），缓存就频频失效，速度暴跌。

朴素卷积的内层 `Xp[n, :, h0:h0+kh, w0:w0+kw]` 每次只取一小块，且相邻滑窗之间大量重叠区域被**重复读取**，访问模式碎、循环深、无法批量化。Python 解释器还要为每次循环付出额外开销——这是双重灾难。

> 类比：朴素卷积像让一个人拿着小尺子，在墙上一格一格地量、量完一格挪一下。而我们真正想要的，是把整面墙的尺寸"一次性拍照"丢给一台高速运算机器去并行处理。

### 2.2 核心思想：把"滑窗"摊平成"矩阵行"

**im2col**（image to column，"图像转列"）的想法极其聪明：

> **既然卷积的每个输出点 = 一个滑窗区域和卷积核做点积，那我就把所有滑窗区域，每个都摊平成一行，堆成一个大矩阵；再把所有卷积核也摊平成列。这样整个卷积，就变成了一次大矩阵乘法。**

类比：原来要算 100 道"两个向量的点积"题。与其一道一道算，不如**把 100 个左向量摞成一个矩阵 A，把右向量摞成矩阵 B，一次 `A @ B` 全算完**。矩阵乘法正是 GPU 最擅长、优化最极致的运算（这套高度优化的库叫 **BLAS**，Basic Linear Algebra Subprograms，基础线性代数子程序；NVIDIA 的版本叫 cuBLAS）。

具体怎么变形（设输入 1 张图、`C_in` 通道、卷积核 `kh×kw`、输出尺寸 `H_out×W_out`）：

| 步骤 | 形状变化 | 含义 |
|---|---|---|
| ① im2col 展开 | `(C_in, H, W)` → `(C_in·kh·kw, H_out·W_out)` | 每一列 = 一个滑窗位置摊平后的所有像素 |
| ② 卷积核展开 | `(C_out, C_in, kh, kw)` → `(C_out, C_in·kh·kw)` | 每一行 = 一个卷积核摊平 |
| ③ 矩阵乘 | `(C_out, C_in·kh·kw) @ (C_in·kh·kw, H_out·W_out)` = `(C_out, H_out·W_out)` | 一次 GEMM 搞定所有输出 |
| ④ reshape | `(C_out, H_out·W_out)` → `(C_out, H_out, W_out)` | 折回成图像形状 |

**GEMM**（General Matrix Multiply，通用矩阵乘法）：就是 `C = αAB + βC` 这种标准矩阵乘法的统称，是 BLAS 库的核心函数。"把问题转化成 GEMM"几乎是所有深度学习算子加速的母题。

<!--CONTINUE2-->

### 2.3 动手实现（与 `F.conv2d` 误差 < 1e-5）

下面是今天的核心交付物。我刻意写成**工业风格而非玩具风格**：支持 batch、stride、padding，并且用 numpy 的高级索引（fancy indexing）一次性构造，而不是再套循环——这才是真正能体现 im2col 价值的写法。

```python
# 文件：week5_cnn/src/im2col_numpy.py
# 环境：Python 3.10+，numpy>=1.24，torch>=2.0（torch 仅用于最后做数值对拍验证）
import numpy as np

def get_im2col_indices(C_in, H_out, W_out, kh, kw, stride):
    """
    预先算好 im2col 的三组索引 (k, i, j)，用于一次性 fancy indexing。
    为什么这么写：避免 Python 层的滑窗循环，把"取所有滑窗"变成一次向量化索引——
    这是 im2col 真正快起来的关键，否则又退化回朴素卷积的循环地狱。
    """
    # k: 每个展开元素来自哪个输入通道，形状 (C_in*kh*kw, 1)
    k = np.repeat(np.arange(C_in), kh * kw).reshape(-1, 1)

    # i: 每个展开元素在原图中的行坐标
    i0 = np.repeat(np.arange(kh), kw)                 # 卷积核内部的行偏移
    i0 = np.tile(i0, C_in)                            # 每个通道重复一遍
    i1 = stride * np.repeat(np.arange(H_out), W_out)  # 滑窗起点的行坐标
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)         # 广播相加 = 绝对行坐标

    # j: 每个展开元素在原图中的列坐标（同理）
    j0 = np.tile(np.arange(kw), kh * C_in)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    return k, i, j


def im2col(X, kh, kw, stride=1, pad=0):
    """X: (N, C_in, H, W) → cols: (C_in*kh*kw, H_out*W_out*N)"""
    N, C_in, H, W = X.shape
    H_out = (H + 2 * pad - kh) // stride + 1
    W_out = (W + 2 * pad - kw) // stride + 1
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

    k, i, j = get_im2col_indices(C_in, H_out, W_out, kh, kw, stride)
    # 一次高级索引就把所有滑窗取出来：cols 形状 (C_in*kh*kw, H_out*W_out, N)
    cols = Xp[:, k, i, j]                       # 广播到 batch 维
    cols = cols.transpose(1, 2, 0).reshape(C_in * kh * kw, -1)
    return cols, H_out, W_out


def conv2d_im2col(X, W, b=None, stride=1, pad=0):
    """用 im2col + GEMM 实现卷积。W:(C_out,C_in,kh,kw)"""
    N = X.shape[0]
    C_out, C_in, kh, kw = W.shape
    cols, H_out, W_out = im2col(X, kh, kw, stride, pad)

    W_row = W.reshape(C_out, -1)                # 卷积核摊平成 (C_out, C_in*kh*kw)
    out = W_row @ cols                          # ← 整个卷积浓缩成这一行 GEMM
    if b is not None:
        out += b.reshape(-1, 1)

    # 折回图像形状：注意 batch 在最后，要先 reshape 再调维
    out = out.reshape(C_out, H_out, W_out, N).transpose(3, 0, 1, 2)
    return out
```

**数值对拍（验证误差 < 1e-5）**——这一步绝不能省，是判断你写对没写对的唯一标准：

```python
import torch
import torch.nn.functional as F

np.random.seed(0)
X = np.random.randn(2, 3, 8, 8).astype(np.float64)   # batch=2, 3通道, 8x8
W = np.random.randn(4, 3, 3, 3).astype(np.float64)   # 4个 3x3 卷积核
b = np.random.randn(4).astype(np.float64)

out_mine = conv2d_im2col(X, W, b, stride=1, pad=1)
out_torch = F.conv2d(torch.tensor(X), torch.tensor(W),
                     torch.tensor(b), stride=1, padding=1).numpy()

err = np.abs(out_mine - out_torch).max()
print(f"shape: {out_mine.shape}, max abs error: {err:.2e}")
# 预期输出：shape: (2, 4, 8, 8), max abs error: 1.xxe-15
# 用 float64 时误差在 1e-15 量级（远小于 1e-5）；若用 float32 约 1e-6，也达标
```

> **完成标准（计划里的硬指标）**：max abs error < 1e-5。如果你跑出来误差很大，看下面的排错清单。

### 2.4 常见错误与调试技巧（实战踩坑）

| 症状 | 大概率原因 | 怎么查 |
|---|---|---|
| 误差爆炸（>1）或形状对不上 | im2col 展开顺序和卷积核 reshape 顺序不一致 | 卷积核必须按 `(C_in, kh, kw)` 摊平，和 im2col 里 `k,i,j` 的生成顺序严格对应 |
| 输出尺寸算错 | `H_out` 公式写错 | 死记 `H_out = (H + 2·pad − kh) // stride + 1` |
| batch 维度错乱 | transpose/reshape 顺序反了 | 用 `print(cols.shape)` 逐步打印，对照 2.2 的形状表 |
| 结果整体偏移一个常数 | 忘了加 bias，或 bias 广播方向错 | bias 要 reshape 成 `(C_out, 1)` 再加 |
| float32 下误差 ~1e-3 | 不是 bug，是精度问题 | 教学验证用 `float64`；工业训练用 float32/float16 本就有这个量级误差 |

**调试黄金法则**：先用**最小输入**（如 1×1×3×3 输入 + 1×1×2×2 卷积，stride=1, pad=0）手算出标准答案，再让代码去对。小到能手算，错误无处可藏。

### 2.5 工业锚点：从 im2col 到 implicit GEMM

im2col 有个明显缺点：它要**显式地**把展开后的大矩阵存到内存里。由于滑窗重叠，展开矩阵比原图大好几倍——这叫 **内存膨胀（memory blow-up）**。

> 例子：3×3 卷积、stride=1，展开矩阵的元素数约是原图的 9 倍。一张占 100MB 的特征图，im2col 后可能要 900MB 临时显存。

所以真正的工业级库（cuDNN）用的是 **implicit GEMM（隐式矩阵乘法）**：

**是什么**：它在做矩阵乘法的过程中，**临时、按需地**从原图里取数拼装，**绝不把整个展开矩阵真的写到显存**。等于"边算边摊平，算完即弃"，既享受了 GEMM 的速度，又避开了内存膨胀。

> 类比：im2col 是"先把所有食材都洗好切好摆满整张桌子，再开始炒"（占满桌面）；implicit GEMM 是"炒到哪一步、临时洗切对应的那点食材"（桌面始终干净）。

**这就是你今天该带走的最重要的一句话**：
> **GPU 上的卷积，本质就是矩阵乘法（GEMM）。** cuDNN、TensorRT、乃至所有推理引擎的卷积加速，骨子里都是"如何把这个 GEMM 做得更快、更省内存"。你将来读 cuDNN 文档看到 `IMPLICIT_GEMM`、`WINOGRAD`、`FFT` 这几种卷积算法，它们都是这个母题的不同变体。

<!--CONTINUE3-->

---

## 3. FLOPs–MAC–latency 三角关系

> **今天最该改变你认知的部分**。很多人选模型只看"FLOPs 多少"，以为 FLOPs 小就一定跑得快——**这是错的**。理解这一节，你看模型性能的眼光会和大多数只会调包的人不一样。

### 3.1 三个名词，先逐个说清

**FLOPs（Floating Point Operations，浮点运算次数）**：模型跑一次需要做多少次浮点加法和乘法。注意区分：
- **FLOPs（小写 s = operations）**：运算"次数"，衡量计算量。
- **FLOPS（大写 S = per Second）**：每秒运算"速度"，衡量硬件算力（如 A100 约 312 TFLOPS）。
- 两者关系：`理论时间 ≈ FLOPs / FLOPS`。但"理论"二字是今天的核心——现实往往差很远。

> 类比：FLOPs 是"这趟活儿要搬多少块砖"，FLOPS 是"工人每小时能搬多少块砖"。

**MAC（Memory Access Cost，内存访问代价）**：算这个算子，总共要从内存搬运多少字节的数据 = 读输入 + 读权重 + 写输出。

> 注意：MAC 这个缩写有歧义。在另一些语境里 MAC = Multiply-Accumulate（乘加运算），那是算 FLOPs 的单位。**今天我们说的 MAC 专指 Memory Access Cost（内存访问代价）**，是"搬了多少数据"，不是"算了多少次"。看论文时务必根据上下文区分。

> 类比：MAC 是"为了搬这些砖，工人要在仓库和工地之间往返跑多少趟、搬多重的东西"。

**latency（延迟）**：算子真正跑完花了多少时间（毫秒）。这是我们**最终真正在乎的东西**——用户不关心你算了多少次，只关心多久出结果。

### 3.2 核心概念：算术强度（Arithmetic Intensity）

这是把三者串起来的钥匙。

**算术强度（Arithmetic Intensity）= FLOPs / MAC**，单位是 ops/byte（每搬运一字节数据，能做多少次运算）。

> 类比：你跑一趟仓库（搬数据），到了工地能干多少活（做运算）？
> - **强度高**：跑一趟仓库能干一整天活 → 时间都花在"干活"上 → **算力受限**。
> - **强度低**：跑一趟仓库只够拧一颗螺丝，然后又得跑下一趟 → 时间全耗在"路上" → **带宽受限**。

由此引出两个决定性概念：

**compute-bound（计算受限 / 算力瓶颈）**：算术强度高，瓶颈在 GPU 的算力（FLOPS）。这种算子"喂得饱"GPU，是我们喜欢的状态。

**memory-bound（内存受限 / 带宽瓶颈）**：算术强度低，瓶颈在内存带宽（每秒能搬多少字节）。GPU 算力再强也没用，因为数据喂不上来，核心一直在"等数据"。

> 一句话记忆：**FLOPs 决定了"理论上要算多久"，MAC 决定了"实际上数据搬运拖了多少后腿"，而 latency = 这两者里更慢的那个说了算。**

### 3.3 Roofline 模型：一张图看懂瓶颈在哪

**Roofline（屋顶线）模型**：NVIDIA/学界用来判断算子瓶颈的经典图。横轴是算术强度，纵轴是实际达到的算力（FLOPS）。图形像一个"屋顶"：

```
实际算力 (FLOPS)
  ▲
  │            ┌──────────────  ← 屋顶平台：被硬件峰值算力限制（compute-bound）
  │           /
  │          /  ← 斜坡：被内存带宽限制（memory-bound），强度越高爬得越高
  │         /
  │        /
  └───────┴────────────────────►  算术强度 (FLOPs/MAC)
        拐点
```

- 算子落在**左边斜坡**（强度低）→ memory-bound，受带宽限制，提升空间在"少搬数据"。
- 算子落在**右边平台**（强度高）→ compute-bound，受算力限制，提升空间在"换更强的卡 / 用 Tensor Core"。
- **拐点**的位置 = 硬件峰值算力 / 峰值带宽，是这块 GPU 的固有属性。

工业里拿到一个算子，第一件事就是估它的算术强度，往 Roofline 上一放，立刻知道该往哪个方向优化——这比盲目改代码高效得多。

### 3.4 经典案例：为什么 1×1 是 memory-bound，3×3 是 compute-bound

现在用上午的 1×1 卷积来算实账。设输入特征图 `H×W`，输入通道 `C_in`，输出通道 `C_out`，数据按 float32（4 字节）算。

**3×3 卷积**（每个输出点要做 `C_in×3×3` 次乘加）：
- FLOPs ≈ `2 × H × W × C_out × C_in × 9`（×2 是因为一次乘加 = 1 乘 + 1 加）
- MAC ≈ 读输入 + 读权重 + 写输出 ≈ `(H·W·C_in + C_out·C_in·9 + H·W·C_out) × 4` 字节
- 算术强度高（分子有 ×9 的卷积窗，权重还能在多个输出点间复用）→ **compute-bound**

**1×1 卷积**（每个输出点只做 `C_in` 次乘加）：
- FLOPs ≈ `2 × H × W × C_out × C_in × 1`（少了 ×9）
- MAC 几乎没变（还是要把整张输入特征图读进来、整张输出写出去）
- **FLOPs 砍到 1/9，MAC 几乎不变 → 算术强度暴跌 → memory-bound**

**结论**：1×1 卷积虽然计算量小，但它"搬数据多、算得少"，GPU 大量时间花在等数据搬运上，算力利用率很低。**计算量小 ≠ 跑得快**——这就是今天要彻底扭转的直觉。

### 3.5 反直觉的工业大案例：MobileNet 的 depthwise 卷积

这是把上面所有概念串起来的"压轴题"，请认真读。

**depthwise separable convolution（深度可分离卷积）**：MobileNet（Google 为手机端设计的轻量网络）的核心。它把一个标准卷积拆成两步：
1. **depthwise（逐通道卷积）**：每个输入通道**单独**用一个卷积核卷自己，通道之间不交流。
2. **pointwise（逐点卷积）**：就是 1×1 卷积，负责把通道间的信息融合回来。

这一拆，**理论 FLOPs 降到原来的约 1/8 ~ 1/9**。论文数字漂亮极了。

**但实测延迟只降了 2~3 倍。为什么？**

因为 depthwise 卷积是**极端 memory-bound 的算子**：
- 它每个通道只卷自己，**权重几乎无法复用**（标准卷积里一个权重要服务很多输出，depthwise 里基本一对一）。
- 算术强度极低——搬一堆数据进来，只做很少的运算就得把结果写回去。
- 于是 GPU/手机芯片的算力大量闲置，时间全耗在内存搬运上。

**省下来的是 FLOPs，但 latency 的瓶颈根本不在 FLOPs 上，而在 MAC 上。** 你把 FLOPs 砍掉 8 倍，可瓶颈那一侧（带宽）只松了一点，所以墙钟时间只快了 2~3 倍。

> **这个案例值得背下来**——它是面试里"你怎么理解模型效率"这类问题的标准回答素材，也是你简历里"懂推理优化"的硬证据。它彻底说明：**评估模型快不快，只看 FLOPs 是外行；要看 FLOPs、MAC、硬件三者一起。**

### 3.6 动手测一测（把理论落到真实数字）

光算公式不够，工业里讲究"测了才算数"。下面这段代码实测 1×1 vs 3×3 在 GPU 上的真实延迟，验证 3.4 的结论：

```python
# 环境：Python 3.10+，torch>=2.0，需要 CUDA GPU（CPU 也能跑，但看不出带宽瓶颈）
import torch, time

def bench(conv, x, iters=100):
    """正确测 GPU 延迟的写法。为什么这么写见下方注释。"""
    conv, x = conv.cuda(), x.cuda()
    # ① warmup：头几次有 cudnn 算法选择/缓存分配开销，必须丢弃，否则数据虚高
    for _ in range(10):
        conv(x)
    torch.cuda.synchronize()          # ② GPU 是异步的，计时前必须同步，否则测到的是"发指令"时间而非"算完"时间
    t0 = time.perf_counter()
    for _ in range(iters):
        conv(x)
    torch.cuda.synchronize()          # ③ 计时结束前再同步一次，确保所有 kernel 真的跑完
    return (time.perf_counter() - t0) / iters * 1000  # 毫秒/次

x = torch.randn(32, 256, 56, 56)     # batch=32, 256通道, 56x56
conv1x1 = torch.nn.Conv2d(256, 256, kernel_size=1)
conv3x3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)

if torch.cuda.is_available():
    t1 = bench(conv1x1, x)
    t3 = bench(conv3x3, x)
    print(f"1x1: {t1:.3f} ms   3x3: {t3:.3f} ms   3x3/1x1 = {t3/t1:.1f}x")
    # 关键观察：3x3 的 FLOPs 是 1x1 的 9 倍，但延迟比通常远小于 9 倍
    #          （常见 2~4x）——因为 1x1 是 memory-bound，没把 GPU 喂饱，
    #          它的延迟里很大一块是"白等带宽"，并没有随 FLOPs 等比例下降。
else:
    print("无 GPU；带宽瓶颈现象在 CPU 上不明显，建议在 GPU 上跑")
```

**这段代码的隐藏知识点（GPU 计时的正确姿势，新手最常踩的坑）**：
- 不 warmup → 把第一次的算法选择开销也算进去，数据虚高。
- 不 `torch.cuda.synchronize()` → GPU 是**异步执行**的，CPU 发完指令就往下跑了，你测到的是"发指令的时间"而不是"算完的时间"，会得到荒谬的"0.01ms 跑完 ResNet"。
- 这两个坑，明天（Day4）和 Day6 的 profiler 实战还会反复用到，现在先建立肌肉记忆。

<!--CONTINUE4-->

---

## 4. 两个主题怎么连起来（今天的内在逻辑）

很多人把"im2col"和"FLOPs/MAC"当成两个不相干的知识点。其实它们是一条线：

1. **im2col 告诉你卷积"怎么算"**——本质是 GEMM。
2. **GEMM 在硬件上跑得快不快，取决于它的算术强度**——这就接到了 FLOPs/MAC。
3. **1×1 卷积是 im2col 退化成的最纯粹的 GEMM**，恰恰又是 memory-bound 的典型——两个主题在 1×1 卷积这里完美交汇。

所以今天真正学到的是一个**完整的分析框架**：
> 拿到任何一个算子 → 想它会变成什么样的 GEMM（im2col 视角）→ 估它的算术强度 → 判断 compute-bound 还是 memory-bound（Roofline 视角）→ 决定优化方向。

这个框架，明天分析 ResNet 的 Bottleneck、Day6 看 profiler 的 chrome trace、乃至第 8 周看 FlashAttention（它解决的正是 Attention 的 memory-bound 问题）——全都要用。**今天是你 AI Infra 路上"性能直觉"的地基。**

---

## 5. 自测题（合上笔记，能答出来才算过关）

> 参考答案在每题下方，先自己答，再对照。

1. **两个 3×3 卷积 vs 一个 5×5 卷积，感受野和参数量各是什么关系？为什么前者更好？**
   <details><summary>答案</summary>感受野都是 5×5；参数量 18C² vs 25C²，前者省 28%，且多一层 ReLU 表达力更强。</details>

2. **1×1 卷积不看空间信息，它到底有什么用？**
   <details><summary>答案</summary>在通道维度做线性混合：降维/升维、跨通道融合、低成本加非线性。数学上等价于纯矩阵乘。</details>

3. **为什么说"GPU 上的卷积本质是矩阵乘法"？im2col 干了什么？**
   <details><summary>答案</summary>im2col 把每个滑窗摊平成矩阵的一列、卷积核摊平成行，整个卷积变成一次 GEMM，从而能用 GPU 高度优化的矩阵乘库（cuBLAS）。</details>

4. **im2col 的缺点是什么？cuDNN 怎么解决？**
   <details><summary>答案</summary>缺点是显式存展开矩阵造成内存膨胀（~9 倍）；cuDNN 用 implicit GEMM，边算边按需取数，不真正存展开矩阵。</details>

5. **算术强度是什么？怎么用它判断 compute-bound / memory-bound？**
   <details><summary>答案</summary>算术强度 = FLOPs / MAC（每搬一字节做多少运算）。高 → compute-bound（算力瓶颈）；低 → memory-bound（带宽瓶颈）。用 Roofline 图：落斜坡是带宽限，落平台是算力限。</details>

6. **MobileNet 把 FLOPs 降 8 倍，为什么延迟只降 2~3 倍？**
   <details><summary>答案</summary>depthwise 卷积权重几乎无法复用，算术强度极低，是 memory-bound 算子；延迟瓶颈在内存带宽（MAC）而非 FLOPs，所以砍 FLOPs 对延迟帮助有限。</details>

7. **测 GPU 算子延迟，必须做哪两件事？不做会怎样？**
   <details><summary>答案</summary>① warmup 丢弃头几次（避开算法选择开销）；② 计时前后都 `torch.cuda.synchronize()`（GPU 异步，否则测到的是发指令时间而非执行时间）。</details>

---

## 6. 与已有笔记的串联表

| 今天的内容 | 关联笔记 | 关系 |
|---|---|---|
| Conv2d 四件套（参数/输出/FLOPs/激活） | [[conv_anatomy]]（Day1） | 今天的 FLOPs 计算直接复用 Day1 的公式 |
| 1×1 卷积 / bottleneck | [[lenet_vs_modern]]（Day2） | 现代 CNN 相比 LeNet 多出的"通道工程" |
| GEMM / 算子融合思想 | [[batchnorm_inference]]（W4D5） | fused BN-Conv 也是"把多个算子合成一次 GEMM"的思路 |
| memory-bound vs compute-bound | [[residual_grad_flow]]（Day4，明天） | ResNet Bottleneck 用 1×1 降维，正是 memory-bound 算子的取舍 |
| 算术强度 / Roofline | [[profiler_chrome_trace]]（Day6） | Day6 用 profiler 实测，验证今天理论预测的瓶颈 |
| memory-bound 的极致案例 | 第8周 FlashAttention | Attention 长序列时 memory-bound，FlashAttention 正是为此而生 |

---

## 7. 完成标准 Checklist（今日过关条件）

- [ ] `src/im2col_numpy.py` 写完，与 `F.conv2d` 数值对拍 **max abs error < 1e-5**
- [ ] 能用一句话口述"为什么 GPU 上卷积的本质是矩阵乘"
- [ ] 能口述"为什么 1×1 卷积是 memory-bound"（FLOPs 砍 9 倍但 MAC 不变 → 算术强度暴跌）
- [ ] 能复述 MobileNet 的反直觉案例（FLOPs 降 8 倍、延迟只降 2~3 倍的原因）
- [ ] `tech_notes/flops_vs_latency.md` 写完（FLOPs/MAC/算术强度/Roofline + MobileNet 案例）
- [ ] 第3章的 GPU benchmark 代码跑通，观察到 3x3/1x1 延迟比远小于 9x
- [ ] 自测题 7 道能不看答案口述

> **过关后给自己 5 分钟**：合上电脑，对着墙复述一遍"拿到一个算子，我怎么判断它的瓶颈、怎么优化"。说得顺，今天才算真的拿下了——这是 AI Infra 工程师的核心肌肉记忆。

