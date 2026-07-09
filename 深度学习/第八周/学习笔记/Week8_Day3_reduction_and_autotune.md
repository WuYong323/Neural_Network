# Day 3 学习笔记：从「逐元素」跨到「跨元素」——Reduction 与调优（RMSNorm / Softmax）

> **本节定位**：Day 1 / Day 2 的 pointwise（逐元素）kernel，每个元素各算各的、互不干扰。
> 今天的 **reduction（归约）** 要把"一整行的很多个数汇总成一个数"，program 内部第一次需要**协作通信**。
> 这是 LayerNorm、Softmax、Attention 的共同地基，也是真实 LLM kernel 的第一道真门槛。
>
> 读完你应该能**口述**两句话：
> 1. reduction 为什么天生比 pointwise 难写成 kernel？
> 2. `@triton.autotune` 到底在替你调什么？调歪了会怎样？

---

## 目录

0. 先回忆：昨天的 pointwise 到底"简单"在哪
1. 什么是 Reduction（归约）——为什么 `sum` 比 `+` 难
2. 底层锚点：一个 program 内部怎么把 BLOCK 个数加起来（树形归约 + CUDA 源码）
3. Softmax：为什么"必须减最大值再 exp"（数值稳定的血泪账）
4. RMSNorm vs LayerNorm：现代 LLM 到底砍掉了什么
5. 动手 A：手写 RMSNorm 的 Triton kernel（教学版 + 生产版）
6. Autotune：`@triton.autotune` 在调什么，BLOCK 大小的两难
7. 三方对标：torch vs torch.compile vs 你的 kernel（memory-bound 的意义）
8. 误差分析：为什么 reduction 之后要从 `allclose` 退一步到 cosine
9. Track B：用 `nsys stats` 数 decode 一步有几个 kernel
10. 完成标准自检 + 口述稿

---

## 0. 先回忆：昨天的 pointwise 到底"简单"在哪

昨天写 `z = x + y` 的 kernel，本质是这样一件事：

> "第 5 号数据"的结果，**只依赖**"第 5 号 x"和"第 5 号 y"。谁也不用管别人。

我们把这种"输出的每个格子只看输入对应位置"的操作，叫 **pointwise（逐元素 / 点对点）操作**。
它天生适合 GPU，因为 GPU 有上万个线程，你把 1 万个加法丢给 1 万个线程，**它们之间零沟通**，各干各的，一步到位。

用一句大白话概括 pointwise 的"简单"：

> **输入形状 = 输出形状，线程之间不需要说话。**

```python
# 昨天的世界：形状不变，线程零通信
x = torch.randn(4, 1024)
y = torch.randn(4, 1024)
z = x + y            # 输出还是 (4, 1024)，第 i 个数只碰第 i 个数
```

今天这件"零通信"的好事，要被打破了。

---

## 1. 什么是 Reduction（归约）——为什么 `sum` 比 `+` 难

### 1.1 是什么

**Reduction（归约）**：把一组数，按某种"能合并的运算"（加、乘、取最大、取最小……）**塌缩成更少的数**（通常是一个数）。

- `[3, 1, 4, 1, 5].sum()` → `14`（5 个数塌成 1 个）
- `[3, 1, 4, 1, 5].max()` → `5`
- `x.sum(dim=-1)`：`(4, 1024)` → `(4,)`，把最后一维 1024 个数每行加成 1 个

关键词是**"塌缩一个维度"**。pointwise 是"形状不变"，reduction 是"形状变小"。这一字之差，是今天所有难度的来源。

> 🧠 **类比**：pointwise 像"每个工人给自己手上的零件刷漆"，互不打扰。
> reduction 像"全车间 1024 个工人手里各有一个数字，现在要算出总和"——**必须有人把大家的数收上来加一起**，这就产生了"沟通成本"。谁先交、交给谁、怎么不算重、不漏算，全是问题。

### 1.2 为什么 `x.sum(dim=-1)` 比 `x + y` 难写成 kernel

三个层层递进的原因，建议在笔记里背下来：

**原因①：输出要"塌缩"，多个输入争夺同一个输出格子。**
`x+y` 里，1024 个线程写 1024 个不同的输出地址，天下太平。
`sum` 里，1024 个线程算出的部分结果，最后都要汇进**同一个** `out[row]`。多个线程写同一个地址 = **写冲突（race condition，竞态）**。你不能让 1024 个线程同时 `out[row] += x[i]`，那结果是错的、随机的。

**原因②：program 内部的数据必须"互相通信"。**
要算总和，第 0 号线程的数得和第 512 号线程的数相遇、相加。它们分属不同线程，数据在不同寄存器里。**让两个线程交换数据**，就需要一块公共内存 + 一次"到齐了没"的同步。这在 CUDA 里就是 `shared memory`（共享内存）+ `__syncthreads()`（线程同步栅栏）。pointwise 完全不需要这些。

**原因③：顺序自由，但结果会因顺序略有不同。**
数学上 `(a+b)+c = a+(b+c)`，但**浮点数不满足结合律**（见第 8 节）。所以并行加法的结果，和 numpy 一个一个顺着加的结果，会差一点点。这是 reduction 特有的"正确但不相等"，pointwise 里几乎遇不到。

一句话总结这一节：

> **pointwise 难在"没有难点"；reduction 难在"线程必须协作，还要处理写冲突和浮点顺序"。**

---

## 2. 底层锚点：一个 program 内部怎么把 BLOCK 个数加起来

在 Triton 里，你只写一行 `total = tl.sum(x)`。但你**必须知道这一行底下发生了什么**——因为暑假后段下沉 CUDA 时，你要亲手写这段。这是 CUDA 面试和真实 kernel 优化的经典题：**parallel reduction（并行归约）**。

### 2.1 最蠢的办法：一个线程从头加到尾（串行）

```
线程0：sum = x[0]+x[1]+x[2]+...+x[1023]   ← 一个人干 1024 步
其他 1023 个线程：干看着
```
这不叫并行，这叫"1024 个人上班，1 个人干活"。GPU 的算力全浪费了。时间复杂度 O(N)。

### 2.2 聪明的办法：树形归约（tree reduction）

**核心思想**：让线程两两配对相加，每一轮把"还活着的数"砍一半。1024 → 512 → 256 → … → 1，只需 **log₂(1024) = 10 步**。这就是"树形"——像淘汰赛对阵图，10 轮决出总冠军。

```
第0轮: [a0 a1 a2 a3 a4 a5 a6 a7]   8个数
        a0+=a4  a1+=a5  a2+=a6  a3+=a7   (线程i 加上 线程 i+4)
第1轮: [A0 A1 A2 A3]                4个数
        A0+=A2  A1+=A3
第2轮: [B0 B1]                      2个数
        B0+=B1
第3轮: [总和]                       1个数   ← 3 步搞定 8 个数 = log2(8)
```

时间复杂度从 O(N) 降到 **O(log N)**。N=1024 时，1024 步 vs 10 步，这就是并行的威力。

### 2.3 底层代码：这就是 `tl.sum` 帮你藏起来的 CUDA

下面是**教科书级别**的 CUDA 树形归约。你现在不用会写，但要**看懂三个东西**：共享内存、同步栅栏、砍半循环。看懂它，你就永远理解了 `tl.sum` 这一行的分量。

```cuda
// 编译/运行环境：CUDA 11+，nvcc 编译；这是一个 block 内把 blockDim.x 个数求和的经典写法
// 假设 blockDim.x = 256（一个 block 256 个线程），处理 256 个数

__global__ void reduce_sum(const float* input, float* output, int n) {
    // ① shared memory（共享内存）：一个 block 内所有线程都能读写的"公共黑板"
    //    为什么需要它？因为线程A的数在自己的寄存器里，线程B看不见。
    //    要通信，就得先把数抄到这块公共黑板上，大家才能互相拿。
    __shared__ float sdata[256];

    int tid = threadIdx.x;                       // 我是 block 内第几号线程 (0~255)
    int i   = blockIdx.x * blockDim.x + tid;     // 我负责全局第几个数

    // 每个线程把自己那个数搬到黑板上；越界的填 0（0 是加法的单位元，不影响结果）
    sdata[tid] = (i < n) ? input[i] : 0.0f;

    // ② __syncthreads()：同步栅栏。意思是"全 block 线程在这里等，都到齐了再往下走"。
    //    为什么必须有？下一步要读别人写的黑板，得先保证别人真的写完了。
    //    少了这句 = 有人黑板还没写好你就去读 = 读到垃圾 = 结果随机错误（最难查的 bug 之一）
    __syncthreads();

    // ③ 砍半循环：这就是"树形"。s 从 128 → 64 → ... → 1
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {  // s >>= 1 就是 s /= 2
        if (tid < s) {                              // 只有前一半线程干活
            sdata[tid] += sdata[tid + s];           // 我的数 += 我右边 s 格的数
        }
        __syncthreads();                            // 每轮结束都要同步！等大家都加完这一轮
    }

    // 循环结束后，全 block 的总和躺在 sdata[0] 里
    if (tid == 0) output[blockIdx.x] = sdata[0];    // 0号线程负责把结果写出去
}
```

**三个"为什么"（这才是重点）**：

1. **为什么要 `__shared__`（共享内存）？** 线程之间的寄存器互相看不见，共享内存是它们唯一的"公共黑板"。它比全局显存（global memory）快约 100 倍，专为这种 block 内通信而生。
2. **为什么每轮都要 `__syncthreads()`？** 因为第 s=64 轮，线程 0 要读 `sdata[64]`，而 `sdata[64]` 是上一轮（s=128）线程 64 写的。你必须保证"上一轮所有人都写完了"，才能开始这一轮。这就是"到齐了没"的栅栏。**漏一个 syncthreads，结果随机错，且不一定每次都错——最坑的并行 bug。**
3. **为什么 `tl.sum` 值钱？** 上面这 20 行、还有边界处理、bank conflict（共享内存访问冲突）优化、warp shuffle 进一步加速……Triton 把这一切压成 `tl.sum(x)` 一行。**它不是"简单"，是"帮你把难的藏好了"。** 你现在享受它，秋天自己拆开它。

### 2.4 回到 Triton：你的心智模型

```python
# Triton 里你写的：
row = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # 一整行搬进来
sq_sum = tl.sum(row * row, axis=0)   # ← 这一行 = 上面那整段 CUDA 树形归约！
```

> **记住这句话，写进笔记**：`tl.sum` 底下就是"每个线程算部分和 → 树形两两相加 → log N 步得到全行和"，Triton 用一行封装了 CUDA 的 shared memory + `__syncthreads()`。

---

## 3. Softmax：为什么"必须减最大值再 exp"

### 3.1 是什么

**Softmax（软最大 / 归一化指数）**：把一排任意大小的实数，变成一排**和为 1 的正数**（即概率分布）。Transformer 里 attention 分数就靠它变成"注意力权重"。

数学定义（朴素版）：
```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```
注意这里有个 **reduction**：分母 `Σ exp(x_j)` 要把一整行加起来。所以 softmax 本质是"pointwise 的 exp + 一个 reduction 的求和"。

### 3.2 为什么朴素版会炸——数值稳定（numerical stability）

问题出在 `exp`。指数增长极快：

- `exp(11) ≈ 60000`
- `exp(89) ≈ 4.5 × 10^38` ← 已经超过 **FP32 的上限**（约 3.4×10³⁸）
- FP16（半精度）更惨，上限只有 **65504**，`exp(12)` 就已经溢出了

> 📌 **接你 W7 Day6 §4 的 FP16 溢出**：现代 LLM 大量用 FP16/BF16 存 attention 分数。如果某个分数是 12，`exp(12)=162754` 直接变成 **`inf`（无穷大）**。然后 `inf / inf = NaN`（Not a Number，非数）。**一个 NaN 会像瘟疫一样传遍整个网络，loss 直接变 nan，训练当场死亡。** 这是真实炼丹事故里最常见的死法之一。

来看这段能亲手复现"翻车"的代码：

```python
import torch

# 环境：任意 PyTorch 版本，CPU 即可复现
x = torch.tensor([1000.0, 1001.0, 1002.0])   # 模拟一个数值偏大的 attention 打分行

# ❌ 朴素做法：直接 exp
naive = torch.exp(x) / torch.exp(x).sum()
print(naive)   # tensor([nan, nan, nan]) —— exp(1000)=inf，inf/inf=nan，全军覆没
```

### 3.3 怎么做——减去最大值这一手"魔术"

**做法**：先求出这一行的最大值 `m = max(x)`，每个数都减掉 `m`，再做 softmax：
```
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)
```

**为什么这样做结果不变？**（这是关键，不是玄学）
分子分母同时乘以 `exp(-m)`：
```
exp(x_i) / Σ exp(x_j)  =  [exp(x_i)·exp(-m)] / [Σ exp(x_j)·exp(-m)]  =  exp(x_i - m) / Σ exp(x_j - m)
```
数学上**完全等价**。但工程上天差地别：减完 `m` 后，最大的那个指数变成 `exp(0)=1`，其余都是 `exp(负数) ∈ (0,1]`，**永远不会溢出**。代价只是多了一个 `max` 的 reduction。

```python
# ✅ 稳定做法：减去最大值（PyTorch 官方 softmax 内部就是这么干的）
m = x.max()                                # 第一个 reduction：求行最大
stable = torch.exp(x - m) / torch.exp(x - m).sum()   # 第二个 reduction：求和
print(stable)   # tensor([0.0900, 0.2447, 0.6652])  完美，和为 1
```

> 💡 **工业接轨**：`torch.softmax`、FlashAttention、所有正经 LLM 推理引擎（vLLM、TensorRT-LLM）里的 softmax，**无一例外都减最大值**。FlashAttention 甚至更进一步，用"online softmax"边扫边更新 max，避免存整行——那是你 attention 那天的伏笔。今天先记牢：**softmax 里出现 `max` 不是为了取最大，是为了保命（防溢出）。**

---

## 4. RMSNorm vs LayerNorm：现代 LLM 到底砍掉了什么

### 4.1 先说 LayerNorm（层归一化）是什么

**LayerNorm（层归一化）**：对一行（一个 token 的特征向量）做标准化——**减去均值、除以标准差**，让这行数变成"均值 0、方差 1"，再用可学习的 `γ`（缩放）和 `β`（偏移）调整。

```
LayerNorm(x) = γ · (x - mean(x)) / sqrt(var(x) + eps) + β
其中 mean(x) = Σx / N          ← reduction 1：求均值
     var(x)  = Σ(x-mean)² / N  ← reduction 2：求方差（还依赖 reduction 1 的结果）
```

**为什么要它？** 深层网络里，每层输出的数值分布会乱飘（有的层输出很大、有的很小），这叫 internal covariate shift（内部协变量偏移）。LayerNorm 把每行"拉回统一量纲"，让训练更稳、更快收敛。`eps`（一个很小的数如 1e-6）是防止除以 0。

> 🧠 **类比**：一个班考试，语文满分 150、数学满分 100，直接比总分不公平。LayerNorm 就是把每科成绩换算成"你在本科目的相对位置（标准分）"，再统一比较。

### 4.2 RMSNorm 砍掉了什么

**RMSNorm（Root Mean Square Normalization，均方根归一化）**：由 LLaMA 等现代 LLM 采用。它做了一个大胆的简化——**不减均值了**，只用"均方根"来缩放：

```
RMSNorm(x) = γ · x / sqrt( mean(x²) + eps )
其中 mean(x²) = Σ(x²) / N     ← 只剩这一个 reduction！
```

对比一下砍掉了什么：

| | LayerNorm | RMSNorm |
|---|---|---|
| 求均值 mean(x) | ✅ 要（reduction） | ❌ **砍掉** |
| 减均值 (x - mean) | ✅ 要（centering 中心化） | ❌ **砍掉** |
| 求 mean(x²) / var | ✅ 要（reduction） | ✅ 要（一个 reduction） |
| reduction 次数 | **2 次**（且有依赖，得先算完均值才能算方差） | **1 次** |
| 可学习参数 | γ 和 β | 通常只有 γ |

### 4.3 为什么现代 LLM 敢砍——省了还不掉点

两个层面的原因：

**性能层面（你最该在意的）**：RMSNorm 只有 **1 个 reduction**，LayerNorm 有 **2 个且串行依赖**（必须先算出 mean，才能算 `(x-mean)²`）。在 LLM 里，Norm 操作**每层都调、调好多次**，是典型的 **memory-bound（访存受限）** 小算子。少一次 reduction、少一遍数据扫描，累积下来省得很可观。这正是你做推理优化时"能抠出来的油水"。

**效果层面（为什么不掉点）**：一篇论文（Zhang & Sennrich, 2019）通过实验发现，LayerNorm 起主要作用的是**"缩放不变性"（re-scaling invariance）**，而不是**"平移不变性"（re-centering，减均值那部分）**。既然减均值的贡献不大，那就砍掉它省算力。LLaMA、Gemma、Qwen 等一票主流模型验证了：**用 RMSNorm，速度更快，精度不掉。** 于是它成了现代 LLM 的默认选择。

> 💡 **一句话记住**：RMSNorm = LayerNorm 砍掉"减均值"这一步 = **2 个 reduction 变 1 个** = 更快且不掉点。这就是"为什么现代 LLM 用 RMSNorm"的完整答案。

---

## 5. 动手 A：手写 RMSNorm 的 Triton kernel

目标：**一个 program 处理一整行**。`tl.load` 整行 → 算 `x / sqrt(mean(x²)+eps) * weight` → `tl.store`。
`mean(x²)` 就是你人生第一个亲手写的 reduction（用 `tl.sum`）。

> **运行环境**：Linux + NVIDIA GPU（你的 H100 完美适配）；`pip install triton torch`；Triton ≥ 2.1，PyTorch ≥ 2.1，CUDA 12.x。Windows 上 Triton 支持不完整，建议在你的 H100 环境里跑。

### 5.1 教学版（先看懂，一行行带注释）

```python
import torch
import triton
import triton.language as tl

@triton.jit  # @triton.jit：把这个 Python 函数编译成 GPU kernel
def rmsnorm_kernel(
    x_ptr,        # 输入张量的起始地址（指针）
    w_ptr,        # 权重 γ 的地址
    out_ptr,      # 输出张量的地址
    stride_row,   # 每往下走一行，地址要跳多少（= 列数 N）
    N,            # 每行有多少个元素（特征维度）
    eps,          # 防止除零的小常数
    BLOCK_SIZE: tl.constexpr,  # tl.constexpr：编译期常量，Triton 会按它决定并行宽度
):
    # ① 我是第几号 program？一个 program 负责一行，所以 row = 我的编号
    row = tl.program_id(0)

    # ② 算出这一行在显存里的起始位置
    row_start = x_ptr + row * stride_row

    # ③ 造出这一行内 0,1,2,...,BLOCK_SIZE-1 的列偏移
    cols = tl.arange(0, BLOCK_SIZE)
    # mask：如果 N 不是 BLOCK_SIZE 的整数倍，超出 N 的列不能碰（否则读越界）
    mask = cols < N

    # ④ 一次性把整行搬进片上（other=0.0：越界位置填 0，不影响平方和）
    x = tl.load(row_start + cols, mask=mask, other=0.0)

    # ⑤ ★核心 reduction★：mean(x²) = Σ(x²)/N
    #    tl.sum 底下就是第 2 节那段树形归约！这一行是今天的灵魂
    mean_sq = tl.sum(x * x, axis=0) / N

    # ⑥ 计算缩放系数 1/sqrt(mean_sq + eps)。用 rsqrt（倒数平方根）更快，硬件有专门指令
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # ⑦ 载入权重 γ，逐元素相乘，得到结果
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = x * rstd * w

    # ⑧ 把结果写回显存
    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)
```

**为什么这么写（不只是"这行干啥"）**：
- **一行一 program**：RMSNorm 的 reduction 是"行内"的，行与行独立。让一个 program 吃一整行，reduction 就锁在 program 内部，不用跨 program 通信（跨 program 通信在 GPU 上极贵）。这是 norm/softmax 类 kernel 的标准切法。
- **`other=0.0`**：0 是加法单位元，越界填 0 不污染 `Σx²`。如果换成 max reduction，就得填 `-inf`（max 的单位元）——**填充值必须是该运算的单位元**，这是 reduction kernel 的常见坑。
- **`rsqrt`**：`1/sqrt` 合成一个硬件指令，比先 sqrt 再除快。memory-bound 场景省的是次要的，但这是好习惯。

### 5.2 生产版（含 autotune + Python 封装 + 正确性检查）

把它整理成一个能直接 `python 03_rmsnorm.py` 跑的完整文件。这就是你要产出的 `week8_triton/03_rmsnorm.py`。

```python
# ============================================================
# 03_rmsnorm.py  —  Triton RMSNorm kernel（含 autotune + 三方对标）
# 环境：Linux + NVIDIA GPU（H100 佳），triton>=2.1, torch>=2.1, CUDA 12.x
# 运行：python 03_rmsnorm.py
# ============================================================
import torch
import triton
import triton.language as tl

# ---------- ① autotune：给几组候选，让 Triton 自动选最快 ----------
@triton.autotune(
    configs=[
        # 每个 Config 是一套"配方"：BLOCK_SIZE + num_warps
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16),
    ],
    key=['N'],  # 当 N（行宽）变化时，重新跑一遍 autotune 选最优；N 不变就复用缓存
)
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride_row, N, eps,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    # 平方和用 fp32 累加：即使输入是 fp16，也先转 fp32 再算，防精度崩（工业惯例）
    x_f32 = x.to(tl.float32)
    mean_sq = tl.sum(x_f32 * x_f32, axis=0) / N     # ★ reduction
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = (x_f32 * rstd) * w.to(tl.float32)         # 计算全程 fp32
    tl.store(out_ptr + row * stride_row + cols, out.to(x.dtype), mask=mask)  # 存回原精度
```

```python
# ---------- ② Python 封装：外部像调普通函数一样调它 ----------
def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    assert x.is_cuda and weight.is_cuda, "需要在 GPU 上"
    x = x.contiguous()                      # 保证内存连续，指针步进才正确
    M, N = x.shape                          # M 行 N 列
    out = torch.empty_like(x)
    # grid：启动多少个 program。我们一行一 program，所以启动 M 个
    grid = (M,)
    # 注意：BLOCK_SIZE 没传！autotune 会自己从 configs 里挑，帮你填进去
    rmsnorm_kernel[grid](x, weight, out, x.stride(0), N, eps)
    return out

# ---------- ③ 参考实现（PyTorch eager，用来验证正确性） ----------
def rmsnorm_torch(x, weight, eps=1e-6):
    # 全程 fp32 算，作为"标准答案"
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)   # mean(x²)
    return (xf * torch.rsqrt(ms + eps)).to(x.dtype) * weight

# ---------- ④ 正确性检查 ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 4096                       # 模拟 LLaMA 隐藏维度量级
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    w = torch.randn(N, device='cuda', dtype=torch.float16)

    y_triton = rmsnorm_triton(x, w)
    y_torch  = rmsnorm_torch(x, w)

    # 先看 allclose（后面第 8 节会讲为什么它可能不够，要退到 cosine）
    ok = torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2)
    cos = torch.nn.functional.cosine_similarity(
        y_triton.flatten().float(), y_torch.flatten().float(), dim=0)
    print(f"allclose(rtol=1e-2): {ok}")
    print(f"cosine similarity  : {cos.item():.6f}")   # 期望 > 0.9999
```

**几个生产级细节，工业里踩过血坑才总结出来的**：
- **fp32 累加**：输入常是 fp16，但 reduction 若也用 fp16 累加，几千个数加下来误差会滚雪球。**惯例：load 后立刻 `.to(tl.float32)`，全程 fp32 算，最后存回 fp16。** PyTorch、FlashAttention 都这么干。
- **`.contiguous()`**：不连续的张量（比如刚 transpose 过）指针步进会算错，轻则结果错重则越界。封装层强制连续是防御性编程。
- **`key=['N']`**：autotune 结果按 N 缓存。N=4096 调一次，之后同样 N 直接复用，不会每次都重调。

---

## 6. Autotune：`@triton.autotune` 在调什么

### 6.1 是什么

**Autotune（自动调优）**：你给 Triton 一堆"候选配置"，它在**真实 GPU 上把每个配置跑一遍、计时**，然后**记住最快的那个**，以后就一直用它。本质是"帮你做暴力网格搜索找最优超参"。

它主要调两个旋钮：

- **`BLOCK_SIZE`（块大小）**：一个 program 一次处理多少个元素 / 用多宽的向量。
- **`num_warps`（线程束数量）**：一个 program 用几个 warp。**Warp（线程束）** 是 GPU 调度的最小单位，**固定 32 个线程**打包一起走。`num_warps=4` = 128 个线程，`num_warps=8` = 256 个线程。

> 🧠 **类比**：你要搬一仓库货。`BLOCK_SIZE` 是"每辆卡车装多大"，`num_warps` 是"派几组工人（每组 32 人）"。车太小要跑很多趟（launch 开销大），车太大工人搬不动、仓库门（寄存器/共享内存）塞不下。autotune 就是帮你把"车型 × 工人数"的所有组合都试一遍，选最快的方案。

### 6.2 BLOCK 太小 / 太大分别会怎样（面试高频）

**BLOCK_SIZE 太小**：
- 每个 program 干的活太少，但**要启动的 program 变多** → **launch overhead（启动开销）** 变大。GPU 每启动一个 program 都有固定成本，program 太多，光启动就耗时。
- 好处是每个 program 占用资源少，**occupancy（占用率）** 高——但活太碎，跑不满带宽。

**BLOCK_SIZE 太大**：
- 一个 program 要吃巨多元素，需要巨多**寄存器（register）** 和**共享内存（shared memory）**。GPU 每个 SM（流多处理器）上这两样是**硬性有限**的。
- 需求超标时，一个 SM 上能同时跑的 program 数骤降 → **occupancy 崩塌**。极端情况寄存器不够，编译器把数据"溢出"到慢速显存（**register spilling，寄存器溢出**），性能雪崩。

> 📌 **接你 W7 Day2 的 ncu 占用率概念**：`occupancy`（占用率）= SM 上实际活跃的 warp 数 / 理论最大 warp 数。它衡量"GPU 的并行度喂饱了没"。BLOCK 太小或太大都会伤 occupancy，但方式相反。**autotune 干的就是：在这条"太小←→太大"的曲线上，帮你找那个甜点（sweet spot）。** 你在 ncu 里看到的占用率，正是 autotune 在背后优化的目标之一。

### 6.3 体会 `torch.compile` 的 `max-autotune` 在替你做什么

> 📌 **接你 W7 Day5**：`torch.compile(mode="max-autotune")` 编译慢、首次跑很久，你现在懂它在干嘛了——**它在背后对生成的每个 kernel 做和你 `@triton.autotune` 一模一样的事**：枚举 BLOCK/warp 配置、逐个 benchmark、选最快、缓存。区别只是它自动化、规模更大、还会顺带选 kernel 融合策略。**你手写 autotune 这一次，就是把 `max-autotune` 的黑箱亲手拆开看了一遍。** 以后再用 `torch.compile`，你知道它慢在哪、慢得值不值。

---

## 7. 三方对标：torch vs torch.compile vs 你的 kernel

### 7.1 先理解 memory-bound（访存受限）——这决定了对标的意义

一个算子的性能，要么卡在"算得慢"（**compute-bound，计算受限**），要么卡在"数据搬得慢"（**memory-bound，访存受限**）。判断标准是 **arithmetic intensity（算术强度）** = 计算量 / 访存量。

RMSNorm 算什么？每个元素就一个平方、一个乘法。但要**把整行从显存读进来、再把整行写回去**。**计算极少，访存极多** → 铁板钉钉的 **memory-bound**。

> 🧠 **类比**：memory-bound 像"你算术很快，但题目印在很远的仓库里，你大部分时间在跑腿取题、交卷，真正做题没花几秒"。瓶颈是跑腿（显存带宽），不是脑子（算力）。

**这对优化意味着什么？** memory-bound 算子的唯一目标：**减少显存读写次数**。
- 朴素 PyTorch 做 RMSNorm 可能拆成好几个算子（算平方、求和、开方、乘……），**每个算子都要把数据从显存读一遍、写一遍** → 数据在显存和芯片间来回好几趟。
- 你的 Triton kernel **融合（fuse）** 成一个 kernel：**读一次、算完、写一次**。数据只过一趟显存。

所以理论预期：**你的融合 kernel 在 RMSNorm 上能打平甚至略胜 torch eager**，因为你把多趟访存压成了一趟。这就是"为什么 memory-bound 算子适合手写融合 kernel"。

### 7.2 对标代码（测三方，量带宽利用率）

```python
# 接在 03_rmsnorm.py 后面。用 triton 自带的 do_bench 计时（自动预热、多次取中位数）
def benchmark(M=4096, N=4096, dtype=torch.float16):
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    w = torch.randn(N, device='cuda', dtype=dtype)

    # torch.compile 版：让 PyTorch 自己去融合 + autotune
    torch_compiled = torch.compile(rmsnorm_torch, mode="max-autotune")

    fns = {
        "torch eager    ": lambda: rmsnorm_torch(x, w),
        "torch.compile  ": lambda: torch_compiled(x, w),
        "triton (yours) ": lambda: rmsnorm_triton(x, w),
    }
    # RMSNorm 的访存量：读 x + 写 out ≈ 2 * M * N * 每元素字节数
    bytes_moved = 2 * M * N * x.element_size()
    for name, fn in fns.items():
        ms = triton.testing.do_bench(fn)               # 返回毫秒
        gbps = bytes_moved / (ms * 1e-3) / 1e9         # 有效带宽 GB/s
        print(f"{name}: {ms:.4f} ms | {gbps:6.1f} GB/s")

# benchmark()
```

### 7.3 怎么读结果（这才是产出要的"三方数据 + 分析"）

跑完你会得到类似（**具体数字以你 H100 实测为准**，H100 HBM3 峰值带宽约 3.35 TB/s）：

```
torch eager    : 0.35 ms |  ~190 GB/s     ← 拆成多算子，多趟访存，最慢
torch.compile  : 0.20 ms |  ~330 GB/s     ← 自动融合了，接近手写
triton (yours) : 0.19 ms |  ~350 GB/s     ← 手写融合，打平/略胜
```

分析要点（写进笔记）：
1. **看 GB/s，不光看 ms**。memory-bound 算子的"满分"是**打满显存带宽**。你的 GB/s 越接近 H100 峰值，说明越优。达到峰值的 60-80% 就算很好了。
2. **eager 最慢的原因**：多算子 = 多趟显存往返。用你 W7 Day2 的 nsys/ncu 一看，eager 会显示**好几个 kernel**，你的只有**一个**。
3. **你打平 torch.compile 是正常的**，因为 `compile` 底下也是生成 Triton kernel + autotune，你俩本质做了同一件事。**打不过也别慌**——PyTorch 的模板经过海量调优；能打平就证明你理解到位了。
4. **别用太小的 M、N** 测：太小时启动开销主导，测不出带宽差异，结论会误导。

---

## 8. 误差分析：为什么 reduction 之后要从 `allclose` 退一步到 cosine

这是今天最该写进笔记的"深度思考"，也是产出明确要求的分析。

### 8.1 根因：浮点加法不满足结合律

计算机的浮点数（float）只有有限位数。每次加法都可能**四舍五入丢掉尾数**。于是：

```python
# 环境：纯 Python 即可复现
a = 1e20
b = -1e20
c = 1.0
print((a + b) + c)   # 1.0    先把两个大数抵消，再加 1，正确
print(a + (b + c))   # 0.0    先算 b+c，1.0 相对 1e20 太小被"吃掉"，再加 a 抵消成 0
```

**同样三个数，加的顺序不同，结果不同。** 这就是"浮点加法不满足结合律 `(a+b)+c ≠ a+(b+c)`"。

### 8.2 联系到今天：reduction 改变了累加顺序

- PyTorch eager 的 `.sum()`：可能是某种顺序（或它自己的并行归约顺序）。
- 你的 Triton `tl.sum`：是**树形归约**顺序（第 2 节那种两两配对）。

**两者累加顺序不同 → 每行的 `Σx²` 会差一点点 → 缩放系数差一点点 → 输出差一点点。** 这个差异**不是 bug，是浮点的宿命**。fp16 下尤其明显，因为尾数只有 10 位，特别容易积累舍入误差。

> 关键认知：**pointwise 几乎不会遇到这问题**（`x+y` 没有累加顺序可言）。**reduction 一定会遇到**。这正是"reduction 比 pointwise 难"的又一层含义——连"对不对"的判断标准都得跟着变。

### 8.3 为什么 allclose 会"误判"，cosine 才是对的尺子

**`torch.allclose(a, b, rtol, atol)`（逐元素接近）**：检查**每一个**元素是否满足 `|a-b| ≤ atol + rtol*|b|`。它是"**绝对严格的点对点体检**"——任何一个元素超差就判 False。

问题：reduction 的舍入误差是**逐元素随机抖动**的。在 fp16 + 大 N 下，总有那么几个元素抖得超过默认 `rtol=1e-5`，`allclose` 直接报 False。但**整体方向其实完全正确**，你不能因为"1 万个数里 3 个差了 0.001"就说 kernel 写错了。

**`cosine similarity`（余弦相似度）**：把两个输出当成两个高维向量，看它们**方向**是否一致（夹角余弦，1 = 完全同向）。它衡量的是"**整体形状/方向对不对**"，对个别元素的微小抖动**不敏感**。

> 🧠 **类比**：allclose 像"逐字校对两篇文章，一个标点不同就算不合格"。cosine 像"看两篇文章讲的是不是同一件事、同一个立场"。对于浮点 reduction 这种"内容一致、末位标点随机抖动"的情况，**cosine 才是合理的尺子**。

### 8.4 实操标准（写进笔记的结论）

> 📌 **接你 W7 Day6 §6.2 的 cosine 尺子**：
> - **pointwise kernel**：可以用 `allclose`，甚至 fp32 下能 bit-level 精确。
> - **reduction kernel（RMSNorm/softmax/attention）**：先放宽 `allclose` 的 `rtol/atol`（如 fp16 用 `rtol=1e-2`）；**若仍偶发 False，退一步用 `cosine_similarity > 0.9999` 作为验收标准**。
> - **为什么退这一步**：因为 reduction 改变了浮点累加顺序，逐元素严格相等在数学上就不该期待；我们真正要验证的是"结果方向/语义一致"，这正是 cosine 度量的东西。**这不是降低标准，是换用正确的标准。**

一句话："**从 pointwise 到 reduction，验收尺子从 `allclose`（逐字校对）升级到 `cosine`（看方向），因为浮点加法不结合，累加顺序一变结果末位必抖。**"

---

## 9. Track B：用 `nsys stats` 数 decode 一步有几个 kernel

接你 Day2 抓的 AMK trace（`.nsys-rep` / `.qdrep`）和 W7 Day2 方法论。目标：**量化 decode 一步里有多少 kernel、gap（空隙）占多少**，记进 report 草稿。

### 9.1 为什么要数 kernel 数和 gap

LLM 推理的 decode 阶段（一次生成一个 token）是**逐 token 串行**的，每步都要把整个模型跑一遍。如果这一步被拆成几十上百个小 kernel，**每个 kernel 之间都有 launch 间隙（gap）**——GPU 在 gap 里是空闲的。**gap 占比高 = GPU 没喂饱 = launch overhead 主导**。这正是 AutoMegaKernel（AMK，把很多小 kernel 合成一个"巨核"）要解决的问题：**减少 kernel 数量 = 减少 gap = 提升 GPU 利用率**。你数出来的 kernel 数和 gap 占比，就是"AMK 值不值得做"的量化证据。

### 9.2 命令（在你的 H100 环境跑）

```bash
# ① 统计每种 kernel 的调用次数、总耗时、平均耗时（按 GPU kernel 汇总）
nsys stats --report cuda_gpu_kern_sum your_trace.nsys-rep

# ② 看 CUDA API 层面的调用（launch 次数等）
nsys stats --report cuda_api_sum your_trace.nsys-rep

# ③ 导出为 csv 方便后续算 gap（可选）
nsys stats --report cuda_gpu_trace --format csv \
           --output decode_trace your_trace.nsys-rep
```

### 9.3 怎么算 gap、记什么进 report

- **kernel 数**：从 `cuda_gpu_kern_sum` 的 `Instances` 列相加，得到"decode 一步"总 kernel 数（注意锁定到单步的时间窗口）。
- **gap 占比**：`gap = 单步总墙钟时间 − 所有 kernel 实际执行时间之和`；`gap 占比 = gap / 单步总时间`。gap 占比越高，说明越"卡在启动/空隙"而非"卡在算"。
- **记进 report 草稿的三个数**：① decode 单步 kernel 总数；② 单步总耗时；③ gap 占比。再加一句结论，例如"decode 单步 X 个 kernel、gap 占 Y%，说明 launch overhead 显著，AMK 融合有明确收益空间"。

> 💡 这三个数就是你科研 report 里"问题动机"最硬的一段——**用数据说话，而不是'我觉得慢'**。这也呼应你今天学的：小的 memory-bound kernel 太多 → 融合成一个大 kernel，是贯穿 RMSNorm 手写和 AMK 项目的**同一条主线**。

---

## 10. 完成标准自检 + 口述稿

### 10.1 产出清单（对照今天要求）

- [ ] `week8_triton/03_rmsnorm.py`：含 `@triton.autotune` 的 RMSNorm kernel（第 5.2 节）
- [ ] `tech_notes/reduction_and_autotune.md`：本文件
- [ ] 三方对标数据：torch eager / torch.compile / 你的 triton，含 ms 和 GB/s（第 7 节）
- [ ] "allclose → cosine 为何退一步"的分析（第 8 节）
- [ ] Track B：decode 单步 kernel 数 + gap 占比，记进 report 草稿（第 9 节）
- [ ] 正确性：cosine similarity > 0.9999，且速度不慢于 torch eager

### 10.2 口述稿——能脱稿讲出这两段，今天就通关了

**① "reduction 为什么比 pointwise 难？"**
> pointwise 是逐元素、形状不变、线程之间零通信，各算各的。reduction 要把一整行塌缩成一个数，多个线程算的部分和最后要汇进同一个地址——产生写冲突，必须让线程通过共享内存互相通信、还要用同步栅栏保证"到齐了再往下走"。底层就是树形归约：两两相加，log N 步。而且浮点加法不满足结合律，累加顺序一变结果末位就抖，所以验收标准也得从 allclose 退到 cosine。这些 pointwise 全都没有。

**② "autotune 在调什么？"**
> 主要调 BLOCK_SIZE 和 num_warps。它把我给的几组配置在真实 GPU 上各跑一遍计时，选最快的缓存起来。BLOCK 太小，program 太多、launch 开销大，虽然 occupancy 高但活太碎跑不满带宽；BLOCK 太大，寄存器和共享内存不够，occupancy 崩塌甚至寄存器溢出到显存。autotune 就是在这条曲线上帮我找甜点。这跟 torch.compile 的 max-autotune 做的是同一件事，只是它自动化、规模更大。

### 10.3 一张图记住今天

```
        pointwise (Day1-2)              reduction (Day3)
        ─────────────────              ─────────────────
形状      不变 (N→N)                     塌缩 (N→1)
线程      零通信                          必须通信（shared mem + sync）
底层      各写各的地址                     树形归约 log N 步
误差      allclose 甚至 bit 级            必须放宽 → cosine（浮点不结合）
代表算子   加法/激活                       RMSNorm / softmax / attention
优化       —                             memory-bound → 融合成 1 个 kernel
                                         → autotune 找 BLOCK/warp 甜点
```

> **今天的一句话灵魂**：从"每个人管自己"到"全车间协作汇总"，你迈过了 GPU kernel 从玩具走向真实 LLM 算子的第一道坎。RMSNorm 手写里那个 `tl.sum`、AMK 项目里那些该被融合的小 kernel、以后 attention 的 online softmax——**背后是同一件事：让 GPU 少跑腿、多协作、别空转。**

---
*Day 3 笔记完 · 下一步：把 03_rmsnorm.py 在 H100 上跑通，填入你的真实三方数据，再把 Track B 的 kernel 数/gap 占比补进 report 草稿。*
