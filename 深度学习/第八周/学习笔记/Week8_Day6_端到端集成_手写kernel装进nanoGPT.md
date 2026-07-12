# Day 6 · 端到端集成：把手写 kernel 装进 nanoGPT，和 baseline 正面决战

> 主线一句话：前五天你造的是**零件**（RMSNorm、fused kernel），今天造的是**整机**——把零件焊进 nanoGPT，跑真实生成，和 `torch.compile` 做**全套对标**，最后写一段**诚实到扎心的结论**。
>
> 这段结论，比"我快了 N×"值钱一百倍。因为它就是你判断"小米巨核该攻哪个算子、值不值得手写"的**决策依据**，也是你课题 ≥20% 提升的**分母认知**（你得先知道 torch.compile 这条基线到底多强，才知道你还剩多少空间）。

---

## 0. 今天在学什么（先建立大局观）

一句话概括今天的三个灵魂拷问：

1. **单个 kernel 快 ≠ 端到端快，为什么？** —— 答案是 **Amdahl 定律**（阿姆达尔定律，见 §1）。
2. **把一个自定义 kernel 焊进真实模型，要处理哪些工程脏活？** —— 形状/dtype、autograd、KV Cache 交互、回退路径（见 §2）。
3. **你的手写 kernel vs `torch.compile` 整模型，谁赢？** —— 大概率整模型 torch.compile 赢，但**赢在哪、你输在哪、为什么**，才是今天的核心产出（见 §3、§8）。

今天的动手主线（Track A，约 5 小时）：

```
写一个能 drop-in 的 Triton RMSNorm      →  §4
    ↓ 焊进 nanoGPT，torch.no_grad 跑通生成   →  §5
    ↓ 四组端到端对标（eager / compile / +手写 / compile+手写）  →  §6
    ↓ 三把尺子量误差（max err / cosine / top-1）  →  §7
    ↓ 写诚实结论（今天最值钱的东西）  →  §8
```

交付物：`week8_triton/06_nanogpt_integrated.py` + `tech_notes/end_to_end_showdown.md`（四组对标表 + 三尺子误差 + 诚实结论）。

---

## 1. Amdahl 定律：为什么"局部快 2×"换不来"整体快 2×"

### 1.1 是什么

**Amdahl's Law（阿姆达尔定律，中文常译"阿姆达尔定律"）**：给你一个系统，你只加速了其中**一部分**，那么**整个系统能快多少，被"你没优化的那部分"死死卡住**。

先给公式（记不住没关系，看完例子自然懂）：

设你优化的那部分，占端到端总时间的比例是 `p`（proportion，比例），你把这部分加速了 `s` 倍（speedup），那么整体加速比是：

```
整体加速比 = 1 / ( (1 - p) + p/s )
```

- `(1 - p)`：你**没碰**的部分，它的耗时一点没变，是"改不动的地基"。
- `p/s`：你**优化过**的部分，耗时从 `p` 缩到 `p/s`。

### 1.2 用生活例子把它焊进脑子

**类比：你上班通勤 60 分钟。** 其中：

- 步行到地铁站 3 分钟（占 5%）
- 坐地铁 57 分钟（占 95%）

现在有人送你一双"风火轮鞋"，走路快 **10 倍**：步行从 3 分钟变成 0.3 分钟。

- 省了 2.7 分钟。
- 总通勤：60 → 57.3 分钟。
- **整体只快了 4.5%。** 你把步行这段干到飞起（10×），端到端纹丝不动。

为什么？因为**地铁那 95% 你没动**。这就是 Amdahl 定律的全部灵魂：**决定天花板的不是"你优化的部分能多快"，而是"你没优化的部分占多大"。**

反过来，`s → ∞`（你把步行优化到 0 秒）时，公式退化成：

```
整体加速比上限 = 1 / (1 - p) = 1 / 0.95 ≈ 1.053×
```

也就是说，哪怕步行**瞬移**，端到端最多快 5.3%。**这是物理上限，努力再多也撞墙。**

### 1.3 直接套到你今天的场景

nanoGPT 一次前向，时间大头是谁？**Attention 的矩阵乘 + MLP 的两个大 Linear（矩阵乘）**，通常合计占 **80%~90%**。而 **RMSNorm 这种逐元素归一化，往往只占 3%~8%**。

假设 RMSNorm 占 `p = 5%`，你的 Triton kernel 把它干快了 `s = 2×`：

```
整体加速比 = 1 / ( (1 - 0.05) + 0.05/2 )
           = 1 / ( 0.95 + 0.025 )
           = 1 / 0.975
           ≈ 1.026×      # 端到端只快 2.6%
```

**你 RMSNorm 快了 2 倍，端到端快了 2.6%。** 这个数字第一次算出来时会让人有点泄气，但它恰恰是今天要你亲手体会的真相。

> ⚠️ 这不是打击你，是给你装一个"雷达"。以后你在小米课题里看到"我这个 kernel 快了 3×"就热血上头之前，先问一句：**"它占端到端几个百分点？"** 这一问，就是资深和萌新的分水岭。

### 1.4 一段代码：把 Amdahl 定律变成可交互的直觉

```python
# amdahl_playground.py
# 运行环境：任意 Python 3.8+，无需 GPU、无需第三方库
# 目的：把"局部加速 → 整体加速"的关系变成你能拨动的旋钮

def amdahl_speedup(p: float, s: float) -> float:
    """
    p: 被优化部分占端到端总时间的比例 (0~1)
    s: 这部分的局部加速比 (>1 表示更快)
    返回: 端到端整体加速比
    为什么这么写：分母 = 没动的部分(1-p) + 优化后的部分(p/s)。
    整个系统的新耗时是这个分母（以原总时间=1为单位），所以加速比=1/分母。
    """
    assert 0 <= p <= 1, "p 是比例，必须在 0~1"
    assert s > 0, "加速比必须为正"
    return 1.0 / ((1 - p) + p / s)

if __name__ == "__main__":
    # 场景：RMSNorm 分别占 5% / 15% / 40%，各自加速 2× 和 无穷大(→0)
    for p in (0.05, 0.15, 0.40):
        s2 = amdahl_speedup(p, 2.0)      # 现实：加速 2 倍
        s_inf = amdahl_speedup(p, 1e9)   # 极限：这部分耗时→0
        print(f"占比 {p:>4.0%} | 加速2×→端到端 {s2:.3f}× "
              f"| 极限→端到端 {s_inf:.3f}× (物理天花板)")
```

预期输出（心里先有个数）：

```
占比   5% | 加速2×→端到端 1.026× | 极限→端到端 1.053× (物理天花板)
占比  15% | 加速2×→端到端 1.081× | 极限→端到端 1.176× (物理天花板)
占比  40% | 加速2×→端到端 1.250× | 极限→端到端 1.667× (物理天花板)
```

**读表结论**：要想端到端明显变快，你手写的 kernel 必须打在**占比大的算子**上（比如 Attention、MLP 的 matmul），或者用**巨核（mega-kernel）把一堆小算子的占比"攒"起来一起吃掉**——这正是小米课题"巨核大模型推理优化"的底层逻辑。单点优化一个 5% 的 RMSNorm，天生就有天花板。

### 1.5 工业界怎么用这个雷达

- **性能工程第一步永远是 profile，不是写 kernel。** 你得先知道 `p`（占比），才知道值不值得优化。先 profile 再动手，是 NVIDIA / Meta 性能团队的铁律。
- 你 Day6 之后的 H100 profiling（AMK 课题）本质就是在**测每个算子的 `p`**——nsys 给你的时间轴，就是一张张"哪段占比大"的地图。
- **"该攻哪个算子"这个决策，Amdahl 定律给了你量化答案**：优先攻占比大的、或能被融合成巨核的一批小算子。

---

## 2. 集成一个自定义 kernel，要处理哪些工程脏活

写 kernel 是"1"，集成是后面的"0"——但没有这些"0"，你的 kernel 在真实模型里根本跑不起来，或者跑起来是错的。这一节讲**四个必须处理的工程问题**，每个都配"为什么"和"翻车现场"。

### 2.1 形状（shape）与 dtype 兼容性

**是什么**：你的 kernel 在测试时可能只喂过 `[4096, 768]` 这种规整形状，但真实模型里输入千奇百怪。

**训练态 vs 推理态的形状差异（这是最容易翻车的地方）**：

- **训练/prefill（预填充）**：输入 `[batch, seq_len, dim]`，比如 `[8, 1024, 768]`，token 一大片。
- **推理 decode（逐 token 解码）**：配合 KV Cache 时，每步只喂 **1 个新 token**，输入是 `[batch, 1, dim]`，比如 `[1, 1, 768]`。

> **KV Cache（Key-Value 缓存，键值缓存）**：自回归生成时，之前 token 算过的 Key/Value 存下来复用，所以每一步新的前向**只处理最新那 1 个 token**。这就是为什么 decode 阶段的 `seq_len` 恒等于 1。

**翻车现场**：很多人手写 kernel 时假设"行数是 2 的幂"或"至少几百行"，结果 decode 阶段喂进来一个 `[1, 768]`（只有 1 行），kernel 的 grid（网格，Triton 里指并行的 program 数量）配置直接退化，甚至因为块大小假设崩掉。

**怎么做**：kernel 要对"任意行数"鲁棒。RMSNorm 是逐行独立归一化，天然适配——每行开一个 program 就行，1 行也没问题。写完一定要**同时测 prefill 形状和 decode 形状**（§4 代码里会体现）。

关于 **dtype（data type，数据类型）**：nanoGPT 推理常用 `bfloat16`（脑浮点 16 位）或 `float16`。而 RMSNorm 里"求平方和 → 求均值 → 开方"这几步，如果全程用 fp16 累加，**数值会不稳**（平方和容易溢出或精度丢失）。

**行业铁律**：**归一化类算子，累加（reduction）用 fp32，输出再转回原 dtype。** PyTorch 官方 LayerNorm、所有正经 Triton RMSNorm 实现都这么干。§4 代码里你会看到 `.to(tl.float32)` 这一步，就是为这个。

### 2.2 autograd（自动微分）要不要管

**是什么**：**autograd（automatic differentiation，自动求导）** 是 PyTorch 自动算梯度、支撑反向传播的机制。你自定义的 kernel，PyTorch **不知道怎么给它求导**——因为它看不见你 kernel 内部的数学。

**关键判断：今天你只做推理（inference），不训练。** 推理时用 `torch.no_grad()` 包起来，**根本不需要反向传播**，所以：

- ✅ **今天可以完全不写 backward。** 只要 forward 正确即可。
- ❌ 如果哪天要拿这个 kernel 去**训练**，就必须用 `torch.autograd.Function` 手写 `forward` + `backward` 两个静态方法，否则一 `.backward()` 就报错"没有 grad_fn"。

**翻车现场（新手最常见）**：忘了包 `torch.no_grad()`，PyTorch 默默给整个前向建了计算图（浪费显存、拖慢速度），或者你的自定义 op 混在里面导致图构建失败。

**怎么做（今天的做法）**：

```python
@torch.no_grad()          # 装饰器：函数内全程不建计算图，等价于整段包在 no_grad 里
def generate(model, idx, max_new_tokens):
    ...                    # 推理逻辑，省显存、免去 autograd 开销
```

> 记住这条决策树：**只推理 → no_grad + 只写 forward；要训练 → autograd.Function 写全 forward/backward。** 今天走左边这条，省一半事。

### 2.3 和 KV Cache 的交互

**是什么**：前面说了 decode 阶段每步只喂 1 个 token。你的 kernel 焊进去后，会在**两种截然不同的负载**下被调用：

| 阶段 | 输入形状 | 特征 | 谁是瓶颈 |
|------|----------|------|----------|
| prefill（首次，处理 prompt） | `[B, seq_len, D]`，seq_len 大 | 计算密集（compute-bound） | 算得多 |
| decode（后续，逐 token） | `[B, 1, D]`，行数极少 | 访存/启动开销密集（launch-bound） | **kernel 启动开销** |

**这里藏着今天最重要的洞察之一**：decode 阶段每个算子处理的数据极少（就 1 个 token），但**每次 kernel 启动（launch）的固定开销（几微秒）省不掉**。一层 Transformer 里有 RMSNorm、QKV、Attention、Proj、MLP…十几个 kernel，每步 decode 就要**串行启动十几次**，累加起来，**启动开销本身就成了大头**。

**这正是"巨核（mega-kernel）"的用武之地**：把一整层甚至整个模型的十几个 kernel **融合成一个大 kernel**，只启动一次。decode 阶段 launch 开销占比越大，巨核收益越猛。**你小米课题"巨核大模型推理优化"的价值，很大一块就在 decode 这个战场上。** 记住这个连接，后面 §8 还会回扣。

**怎么做**：今天你不必真去改 KV Cache，但**对标时一定要分别测 prefill 和 decode**（用"首 token 延迟"抓 prefill，用"稳定 tok/s"抓 decode），因为你的 kernel 在这两种负载下的表现可能天差地别。

### 2.4 回退路径（fallback）

**是什么**：**fallback（回退/兜底路径）** 指的是"当你的自定义 kernel 不能用时，自动退回到安全的原生实现"。

**为什么必须有**：Triton kernel 有很多"用不了"的场景——

- 环境里没装 Triton，或跑在 CPU 上（Triton 只跑 GPU）。
- 输入形状触发了你的 kernel 没覆盖的边界（比如 `dim` 不是你假设的对齐值）。
- dtype 不支持。

没有 fallback，你的模型换个环境就直接崩，工程上完全不可接受。

**怎么做（工业级写法）**：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, use_triton=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        # 只有"想用 且 环境支持 且 有 CUDA"时才走 Triton，否则自动回退
        self.use_triton = use_triton and _triton_available() and torch.cuda.is_available()

    def forward(self, x):
        if self.use_triton and x.is_cuda:
            try:
                return triton_rmsnorm(x, self.weight, self.eps)   # 快路径
            except Exception as e:
                # 生产环境这里应打日志，而不是静默吞掉——否则你以为在用 Triton，其实一直在回退
                import warnings; warnings.warn(f"Triton RMSNorm 回退到 PyTorch: {e}")
        return _torch_rmsnorm(x, self.weight, self.eps)           # 慢但永远对的兜底
```

> **易错点**：回退时**静默吞异常**是灾难——你辛苦写的 kernel 可能因为一个小 bug 一直在走 fallback，你却以为在享受加速。**一定打警告/日志。** 这是我在真实项目里踩过的坑，写进来给你避雷。

---

## 3. 手写 kernel vs `torch.compile`：你到底在跟谁较量

在跑对标之前，先搞懂对手是谁，不然你只会看到"我输了"，看不懂"为什么输"。

### 3.1 `torch.compile` 是什么，凭什么强

**`torch.compile`** 是 PyTorch 2.0 起的一键编译器。你写普通的 PyTorch 代码，它在背后帮你做**全图优化**。它强在三点：

1. **算子融合（operator fusion，把多个小算子合并成一个 kernel）**：比如 `x * scale + bias` 这三步，eager（即时执行）模式下是 3 个 kernel、数据在显存里来回搬 3 趟；compile 后融合成 1 个 kernel，数据只读写一次。**这直接砍掉了 kernel 启动开销和显存搬运。**

   > **eager mode（即时执行模式）**：PyTorch 默认模式，一行代码执行一个操作，立即出结果。直观好调试，但每个算子都是独立 kernel，融合不了。

2. **自动生成 Triton kernel**：`torch.compile` 的后端 **Inductor（感应器，PyTorch 的编译后端）** 会**自动帮你写 Triton kernel**——是的，你手写的那个 RMSNorm，它自己也能生成一个，而且往往和相邻算子融合在一起。

3. **`max-autotune`（最大化自动调优）模式**：`torch.compile(model, mode="max-autotune")` 会为每个 matmul 等重算子，**实测多种配置选最快的那个**。这一步能榨出接近手工调优的 matmul 性能。

### 3.2 为什么"整模型 torch.compile"大概率赢你的"单点手写"

关键在**视野范围**：

- **你的手写 kernel**：只看得见**它自己那一个算子**（RMSNorm）。它前面的输出、后面的输入，你管不着，它们还是各自独立的 kernel，该有的启动/搬运开销一个不少。
- **`torch.compile` 整模型**：看得见**整张计算图**。它能把 RMSNorm 和**它前后的算子融合在一起**——比如把 RMSNorm 和后面 QKV 投影的一部分揉进同一个 kernel。**这种"跨算子融合"是你单点手写做不到的。**

用 Amdahl 的语言说：**torch.compile 优化的 `p` 接近 100%（全图），你手写的 `p` 只有 5%（单算子）。** 天生不是一个量级的战斗。

**所以最可能的对标结果是：**

| 对标组 | 预期表现 | 原因 |
|--------|----------|------|
| ① eager（纯即时） | 最慢，基准线 | 每个算子独立 kernel，无融合 |
| ② torch.compile（max-autotune） | **通常最快** | 全图融合 + 自动调优 |
| ③ eager + 你的手写 kernel | 比①快一点点（约 §1 算的 2.6%） | 只优化了 5% 的算子 |
| ④ compile + 你的手写 kernel | **可能反而比②慢**（见 §3.3 陷阱） | 手写 kernel 打断了 compile 的融合 |

### 3.3 一个反直觉的陷阱：④ 组可能"1+1 < 1"

你可能以为"compile 已经很快了，我再塞进去一个更快的手写 kernel，岂不是更快"。**大概率错。**

**原因**：`torch.compile` 追踪计算图时，遇到你的自定义 kernel（一个它看不懂内部的黑盒），会触发 **graph break（图断裂，编译图被迫从中间断开）**。断裂点两侧无法融合，你等于**亲手在 compile 精心融合的流水线上砸了一刀**。结果：你替换的那个点或许快了，但你**破坏的融合损失更大**，净效果变慢。

> **graph break（图断裂）**：`torch.compile` 把模型追踪成一张连续的图来整体优化。当它遇到无法追踪的东西（自定义 op、某些 Python 控制流、`.item()` 等），就把图从这里"剪断"，断点前后各自编译、中间回退到 eager。**断点越多，融合机会越少，性能越差。**

**怎么让 ④ 不翻车（进阶，选做）**：用 `torch.library.custom_op` 把你的 kernel **注册成 torch 认识的正规算子**，并提供 `register_fake`（假实现，告诉 compile 输出的形状/dtype 以便追踪）。这样 compile 至少不会因为它 graph break。**但即便如此，compile 仍然无法"穿透"你的 kernel 去和它内部融合**——所以 ④ 追平 ② 就算胜利，超过 ② 很难。

**这个"④ 打不过 ②"的现象，本身就是今天最有价值的诚实结论之一。** 它告诉你：**在 torch.compile 已经吃掉的战场上，单点手写 kernel 很难再捞到好处。你的机会在 compile 吃不到的地方——比如把整层融成一个巨核（超出 compile 的融合粒度）。** 又一次回扣小米巨核课题。

---

## 4. 动手第一步：写一个能 drop-in 的 Triton RMSNorm

### 4.1 先复习 RMSNorm 的数学（为什么它长这样）

**RMSNorm（Root Mean Square Normalization，均方根归一化）** 是 LayerNorm 的简化版，被 LLaMA、Qwen 等主流大模型采用。公式：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

对每一行（每个 token 的特征向量）独立做：

1. `mean(x²)`：这一行所有元素平方，求平均 → 得到"能量"的度量。
2. `sqrt(... + eps)`：开方得到 RMS（均方根），`eps` 防止除零。
3. `x / rms`：把这一行缩放到"单位能量"附近。
4. `* weight`：每个特征通道乘一个可学习缩放。

**为什么比 LayerNorm 简单**：LayerNorm 还要**减均值**（`x - mean(x)`）再除标准差；RMSNorm **省掉了减均值**，只做缩放。实践发现减均值那步对大模型效果影响很小，省掉能快一点——这本身就是一次"值不值得"的工程取舍，和你今天要做的判断同源。

**为什么它是绝佳的练手对象**：逐行独立、访存密集（memory-bound，瓶颈在读写显存而非算力）、每行计算简单——适合一行开一个 Triton program，且能清楚看到"融合读写"的收益。

### 4.2 完整 kernel 代码（带逐行"为什么"）

```python
# rmsnorm_triton.py
# 运行环境依赖：
#   - NVIDIA GPU（Triton 只支持 GPU 后端）
#   - PyTorch >= 2.1, triton >= 2.1（pip install torch triton）
#   - 建议 CUDA 12.x
import torch
import triton
import triton.language as tl

def _triton_available():
    try:
        import triton  # noqa
        return True
    except ImportError:
        return False

@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,          # 输入张量首地址（已 reshape 成 [n_rows, n_cols]）
    w_ptr,          # weight 首地址，长度 n_cols
    y_ptr,          # 输出张量首地址
    x_row_stride,   # 相邻两行在内存里差多少个元素（通常 = n_cols）
    n_cols,         # 每行元素个数（= 模型 hidden dim）
    eps,            # 防除零小量
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：一次处理的列数（取 >= n_cols 的 2 的幂）
):
    # 每个 program 负责一行。program_id(0) 就是行号。
    # 为什么按行并行：RMSNorm 逐行独立，行与行之间没有数据依赖，天然可并行。
    row_idx = tl.program_id(0)
    x_ptr_row = x_ptr + row_idx * x_row_stride  # 定位到本行起始地址

    # 列方向的偏移 [0,1,...,BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 掩码：BLOCK_SIZE 可能比 n_cols 大，超出部分不能读，用 mask 挡住
    mask = col_offsets < n_cols

    # 读入本行。other=0.0：被 mask 挡住的位置填 0，不影响后面平方和。
    x = tl.load(x_ptr_row + col_offsets, mask=mask, other=0.0)

    # 关键：转 fp32 再算平方和。为什么？见 §2.1——归一化的 reduction 必须高精度，
    # 否则 bf16/fp16 累加会掉精度甚至溢出。这是工业级实现的铁律。
    x_fp32 = x.to(tl.float32)
    mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols  # mean(x²)
    rrms = 1.0 / tl.sqrt(mean_sq + eps)                 # 1/rms，用乘代替除更快

    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    # 先在 fp32 下缩放，乘 weight，最后 tl.store 时自动转回 y 的 dtype
    y = x_fp32 * rrms * w.to(tl.float32)

    y_ptr_row = y_ptr + row_idx * x_row_stride
    tl.store(y_ptr_row + col_offsets, y, mask=mask)     # 写回，mask 保证不越界

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    对最后一维做 RMSNorm。x 可以是任意前置维度，如 [B,S,D] 或 [B,D]。
    """
    orig_shape = x.shape
    dim = orig_shape[-1]
    # 把所有前置维度压平成"行"。为什么：kernel 只认 [n_rows, n_cols]，
    # 这样无论 [B,S,D] 还是 [B,1,D]（decode）都能统一处理——解决 §2.1 的形状问题。
    x2d = x.reshape(-1, dim)
    x2d = x2d.contiguous()          # 确保内存连续，否则 stride 假设不成立会读错
    n_rows, n_cols = x2d.shape

    y = torch.empty_like(x2d)
    # BLOCK_SIZE 取 >= n_cols 的最小 2 的幂：让一行一次性读完，reduction 最简单
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)                # 启动 n_rows 个 program，一行一个

    _rmsnorm_fwd_kernel[grid](
        x2d, weight, y,
        x2d.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 2048 else 8,  # 列多时多给 warp，提升并行度
    )
    return y.reshape(orig_shape)    # 还原成原始形状，对调用方透明

def _torch_rmsnorm(x, weight, eps=1e-6):
    """纯 PyTorch 参考实现，用作 fallback 和 §7 的正确性基准。"""
    dtype = x.dtype
    x = x.to(torch.float32)                                    # 同样 fp32 累加
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(dtype)                              # 最后转回原 dtype
```

### 4.3 焊进模型前，先单元测试（别跳过）

```python
# test_rmsnorm.py —— 焊进大模型前，先在小数据上验正确性 + 测两种形状
import torch
from rmsnorm_triton import triton_rmsnorm, _torch_rmsnorm

def test_correctness_and_shapes():
    torch.manual_seed(0)
    dim = 768
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    # 关键：同时测 prefill 形状 和 decode 形状（§2.1 / §2.3 的坑）
    for shape in [(8, 1024, dim),   # prefill：一大片 token
                  (1, 1, dim),      # decode：只有 1 个 token —— 最容易崩的边界
                  (32, dim)]:       # 2D 也要能吃
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        y_triton = triton_rmsnorm(x, weight)
        y_ref = _torch_rmsnorm(x, weight)
        # bf16 下不能要求逐位相等，用相对+绝对容差
        assert torch.allclose(y_triton, y_ref, atol=1e-2, rtol=1e-2), \
            f"形状 {shape} 不匹配！max_err={ (y_triton-y_ref).abs().max().item() }"
        print(f"✅ {shape} 通过")

if __name__ == "__main__":
    test_correctness_and_shapes()
```

> **调试技巧**：如果 `(1, 1, dim)` 这个 decode 形状崩了，八成是你 kernel 里对"行数/BLOCK_SIZE"有隐藏假设。先用最小形状 `(1, dim)` 打断点，`print` 出 `n_rows, n_cols, BLOCK_SIZE` 三个值，通常一眼看出问题。**永远先在小形状上验对，再上大模型——大模型里报错定位成本高 10 倍。**

---

## 5. 动手第二步：焊进 nanoGPT，`torch.no_grad` 跑通生成

### 5.1 集成策略：不改模型源码，"热替换"归一化层

最干净的集成方式，不是去改 nanoGPT 的 `model.py`，而是**加载好模型后，遍历所有子模块，把归一化层原地换成你的 Triton 版**。这样做的好处：模型源码保持原样、随时能一键切回、对标时公平（结构完全一致，只换实现）。

> **前提说明**：原版 nanoGPT 用的是 `LayerNorm`。为了让对标**公平**（三把误差尺子才有意义），我们让 **baseline 和实验组都用 RMSNorm**——一个纯 PyTorch RMSNorm 当基准（组①②），一个 Triton RMSNorm 当实验（组③④）。如果你的 nanoGPT 本来就是 RMSNorm 版（LLaMA 式），那更省事，直接替换即可。**关键原则：对标双方必须是同一个数学，只有 kernel 实现不同——否则你分不清速度差异是"kernel 快"还是"算法不同"。**

```python
# integrate.py —— 把模型里所有 RMSNorm 换成指定实现
import torch, torch.nn as nn
from rmsnorm_triton import triton_rmsnorm, _torch_rmsnorm

class RMSNorm(nn.Module):
    """统一的 RMSNorm 层，用 backend 开关切换实现——对标的核心开关。"""
    def __init__(self, dim, eps=1e-6, backend="torch"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.backend = backend          # "torch" 或 "triton"

    def forward(self, x):
        if self.backend == "triton" and x.is_cuda:
            try:
                return triton_rmsnorm(x, self.weight, self.eps)
            except Exception as e:
                import warnings; warnings.warn(f"Triton 回退: {e}")  # §2.4 不静默
        return _torch_rmsnorm(x, self.weight, self.eps)

def swap_norm_backend(model: nn.Module, backend: str):
    """
    原地把 model 里所有 RMSNorm 的 backend 切换掉。
    为什么用遍历替换：不碰模型定义、随时可逆、保证对标双方结构 100% 一致。
    """
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.backend = backend
    return model
```

### 5.2 跑通端到端生成（`torch.no_grad` + KV Cache 视角）

```python
# generate.py —— 最小可用的自回归生成，推理态只读、不建计算图
import torch
import torch.nn.functional as F

@torch.no_grad()   # §2.2：只推理，全程不建计算图，省显存也免去 autograd 麻烦
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    idx: [B, T] 初始 token（prompt）
    这是不带 KV Cache 的朴素版：每步把全序列重算一遍。
    真实高性能推理会用 KV Cache（§2.3）只算最新 token——这里为聚焦"kernel 集成"先简化。
    """
    model.eval()
    for _ in range(max_new_tokens):
        # 若序列超过模型上下文长度，截断（block_size 是模型能吃的最大长度）
        idx_cond = idx if idx.size(1) <= model.config.block_size \
                   else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)          # 前向：这里内部会调用到你换掉的 RMSNorm
        logits = logits[:, -1, :] / temperature   # 只取最后一个位置的 logits 做采样
        if top_k is not None:                # top-k 采样：只在概率最高的 k 个里选
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)   # 采样下一个 token
        idx = torch.cat((idx, idx_next), dim=1)              # 拼接，继续下一步
    return idx
```

**跑通验证（先确认"能生成、且没崩"，再谈快慢）**：

```python
# 假设你已按 nanoGPT 方式加载了 model 和 tokenizer/编码器
model = swap_norm_backend(model, "triton")   # 切到手写 kernel
model = model.cuda().to(torch.bfloat16)

start_ids = encode("Hello, I am")             # 你的编码函数
x = torch.tensor(start_ids, dtype=torch.long, device="cuda")[None, ...]
out = generate(model, x, max_new_tokens=50, top_k=50)
print(decode(out[0].tolist()))                # 先肉眼看：生成的是不是通顺的文本
```

> **里程碑判据**：这一步只要满足两条就算过关——(1) 不报错、能生成；(2) 生成的文本**读起来通顺、不是乱码**。如果换 Triton 后开始出乱码，说明你的 kernel 有数值 bug，回 §4.3 单元测试查（多半是 fp32 累加那步没写对）。**速度对标是下一节的事，先保证"对"。**

---

## 6. 动手第三步：四组端到端对标（今天的硬核产出）

### 6.1 先讲清楚"怎么测才不骗自己"（GPU 计时三铁律）

GPU 计时最容易测出**假数据**，新手 90% 的对标表都是错的。三条铁律：

**铁律一：必须 warmup（预热）。** 第一次跑 kernel 会触发 JIT 编译、CUDA 上下文初始化、显存分配；`torch.compile` **第一次调用要花几秒到几十秒编译**。这些一次性开销**绝不能算进计时**。做法：正式计时前先空跑几次。

**铁律二：必须 `torch.cuda.synchronize()`（同步）。** GPU 是**异步**的——你 Python 里的调用只是"下单"，GPU 在后台慢慢做。不同步就 `time.time()`，你测的是"下单时间"，不是"做完时间"，数字small到离谱且没意义。做法：计时的起止点都要 `synchronize`。

> **异步（asynchronous）**：CPU 把任务丢给 GPU 队列后立刻返回、继续跑下一行 Python，不等 GPU 做完。所以要测真实耗时，必须显式等 GPU 把队列清空（synchronize）。

**铁律三：多次取中位数，不取平均。** 单次有抖动（其他进程、频率波动）。跑多次取**中位数（median）**比平均数抗离群值。

### 6.2 两个指标：TTFT 和 tok/s（分别对应 prefill 和 decode）

- **TTFT（Time To First Token，首 token 延迟）**：从喂进 prompt 到吐出**第一个** token 的时间。它主要由 **prefill 阶段**（处理整个 prompt）决定，反映"计算密集"负载下的性能。用户体感上的"AI 反应快不快"就看它。
- **tok/s（tokens per second，每秒生成 token 数）**：进入稳定生成后，平均每秒吐多少 token。它主要由 **decode 阶段**（逐 token）决定，反映"访存/启动密集"负载下的性能。它是吞吐量的核心指标。

**为什么两个都要测**（回扣 §2.3）：你的 kernel 在 prefill（大 batch 计算密集）和 decode（1 token 启动密集）下表现可能完全相反。只测一个会得出片面结论，误导你对小米课题"该攻哪"的判断。

### 6.3 对标代码（可直接改用）

```python
# benchmark.py —— 四组端到端对标：eager / compile / +手写 / compile+手写
import torch, time, statistics
from generate import generate
from integrate import swap_norm_backend

def measure_ttft(model, x, top_k=50):
    """首 token 延迟：只生成 1 个 token 的端到端时间。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = generate(model, x, max_new_tokens=1, top_k=top_k)
    torch.cuda.synchronize()             # 铁律二：等 GPU 真正做完
    return time.perf_counter() - t0

def measure_tokps(model, x, n_tokens=100, top_k=50):
    """稳定吞吐：生成 n_tokens 个 token，算 tok/s。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = generate(model, x, max_new_tokens=n_tokens, top_k=top_k)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return n_tokens / dt

def bench_one(name, model, x, warmup=3, iters=10):
    """对一个配置做完整测量：预热 + 多次取中位数。"""
    # 铁律一：预热。torch.compile 的首次编译开销必须在这里被吃掉。
    for _ in range(warmup):
        _ = generate(model, x, max_new_tokens=8, top_k=50)
    torch.cuda.synchronize()

    ttfts, tokps = [], []
    for _ in range(iters):               # 铁律三：多次
        ttfts.append(measure_ttft(model, x))
        tokps.append(measure_tokps(model, x))
    # 取中位数抗抖动
    return {
        "name": name,
        "ttft_ms": statistics.median(ttfts) * 1000,
        "tok_s": statistics.median(tokps),
    }

def run_showdown(build_model_fn, x):
    """
    build_model_fn(): 每次返回一个全新加载的模型（避免四组互相污染状态）。
    """
    results = []

    # ① eager + torch RMSNorm（纯基准线）
    m = swap_norm_backend(build_model_fn(), "torch").cuda().to(torch.bfloat16).eval()
    results.append(bench_one("① eager (baseline)", m, x))
    del m; torch.cuda.empty_cache()

    # ② torch.compile(max-autotune) + torch RMSNorm（最强对手）
    m = swap_norm_backend(build_model_fn(), "torch").cuda().to(torch.bfloat16).eval()
    m = torch.compile(m, mode="max-autotune")   # 全图融合 + 自动调优
    results.append(bench_one("② compile(max-autotune)", m, x))
    del m; torch.cuda.empty_cache()

    # ③ eager + 你的手写 Triton kernel
    m = swap_norm_backend(build_model_fn(), "triton").cuda().to(torch.bfloat16).eval()
    results.append(bench_one("③ eager + 手写kernel", m, x))
    del m; torch.cuda.empty_cache()

    # ④ compile + 你的手写 kernel（警惕 §3.3 的 graph break 陷阱）
    m = swap_norm_backend(build_model_fn(), "triton").cuda().to(torch.bfloat16).eval()
    m = torch.compile(m, mode="max-autotune")
    results.append(bench_one("④ compile + 手写kernel", m, x))
    del m; torch.cuda.empty_cache()

    return results

def print_table(results):
    base = results[0]["tok_s"]                       # 以 eager 为 1.00×
    print(f"{'配置':<26}{'TTFT(ms)':>12}{'tok/s':>12}{'相对eager':>12}")
    print("-" * 62)
    for r in results:
        print(f"{r['name']:<26}{r['ttft_ms']:>12.2f}"
              f"{r['tok_s']:>12.1f}{r['tok_s']/base:>11.2f}×")
```

### 6.4 你会得到一张这样的表（示意，你的真实数字为准）

```
配置                          TTFT(ms)       tok/s     相对eager
--------------------------------------------------------------
① eager (baseline)              45.20        88.5        1.00×
② compile(max-autotune)         38.10       141.2        1.60×
③ eager + 手写kernel            44.30        90.8        1.03×
④ compile + 手写kernel          39.50       138.7        1.57×
```

**这张表要会读**（数字是示意，你机器上跑出来的才算数）：

- **② 比 ① 快 60%**：torch.compile 全图融合的威力，这就是你要正视的"分母"——**基线很强**。
- **③ 只比 ① 快 3%**：和 §1 Amdahl 算的 2.6% 高度吻合。你的 RMSNorm kernel 确实更快，但它只占 5%，端到端就这点。
- **④ 反而比 ② 慢一点点（138.7 < 141.2）**：§3.3 预言的 graph break 陷阱兑现——手写 kernel 打断了 compile 的融合。

> **如果你的 ③ 反而比 ① 慢**：别慌，这也是真实结论。常见原因：你的 RMSNorm kernel 在这个 dim 下没 compile 自动生成的 kernel 快（Inductor 也会生成 Triton），或者你没预热导致 JIT 编译算进去了。**记录下来、分析原因，比强行"P 出一个我赢了"诚实得多，也有用得多。**

---

## 7. 动手第四步：三把尺子量误差——证明"换了 kernel，结果还对"

### 7.1 为什么"快"之外还必须证"对"

换 kernel 提速了，但如果生成结果错了，等于白干甚至有害。而且 bf16/fp16 下，**不同实现给出的数值永远不会逐位相等**（浮点累加顺序不同就有微小差异）。所以问题不是"有没有误差"，而是**"误差是否可控、可解释、不影响最终决策"**。三把尺子从三个层次回答这个问题。

### 7.2 尺子一：逐元素 max error（最严，看最坏情况）

**是什么**：两个实现输出张量，对应位置相减取绝对值，找**最大**的那个差。

**为什么**：它抓的是"最坏的一个点偏了多少"。max err 小 → 说明没有哪个元素爆炸性出错（比如某个 NaN、某个溢出）。

```python
def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()
```

**怎么判读**：bf16 下 max err 在 `1e-2 ~ 1e-3` 量级是正常的（bf16 只有约 3 位有效数字）；如果到了 `1e-1` 甚至更大，说明有真 bug（大概率是 fp32 累加那步没做，§2.1）。

### 7.3 尺子二：logits cosine similarity（余弦相似度，看方向一致性）

**是什么**：**logits（模型最后一层的原始输出分数，未经 softmax）** 是一个长度=词表大小的向量，决定下一个 token 选谁。**cosine similarity（余弦相似度）** 衡量两个向量**方向**有多一致，范围 [-1, 1]，越接近 1 越一致。

**为什么用它而不是只看 max err**：采样只关心 logits 向量的**相对形状/方向**（谁比谁大），不关心绝对数值整体缩放。cosine 恰好只测方向、无视整体缩放，正好对上"采样在乎什么"。

```python
import torch.nn.functional as F
def logits_cosine(logits_a, logits_b):
    # 展平成向量算 cosine；越接近 1.0 越好
    a = logits_a.float().flatten()
    b = logits_b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()
```

**怎么判读**：`> 0.9999` 是很好；`> 0.999` 可接受；掉到 `0.99` 以下要警惕，说明 kernel 误差已经开始扭曲 logits 的形状，可能改变采样结果。

### 7.4 尺子三：top-1 一致率（最贴近业务，看最终决策）

**是什么**：对每个位置，两个实现各自选出概率最高的那个 token（**top-1，最高分的候选**），统计"两边选的是同一个 token"的比例。

**为什么它最重要**：前两把尺子测的是"中间数值像不像"，这把测的是**"最终决策一不一样"**。哪怕 logits 有点偏，只要 top-1 选的 token 一样，生成结果就一样。**这是最贴近"用户真正看到什么"的尺子。**

```python
def top1_agreement(logits_a, logits_b):
    # 在最后一维（词表维）取 argmax，比较两边是否选了同一个 token
    top_a = logits_a.argmax(dim=-1)
    top_b = logits_b.argmax(dim=-1)
    return (top_a == top_b).float().mean().item()
```

**怎么判读**：理想是 `1.0`（100% 一致）；`> 0.99` 通常可接受（偶尔在两个 logits 极接近的位置分歧，不影响大局）；明显低于 0.99 要查 kernel。

### 7.5 把三把尺子串起来用（关键：用 teacher forcing，别用自由生成）

```python
# error_rulers.py —— 固定输入下，对比 torch 基准 vs triton 的三把尺子
import torch
from integrate import swap_norm_backend

@torch.no_grad()
def compare_three_rulers(build_model_fn, x_fixed):
    """
    x_fixed: 固定的一段输入 token [B, T]。
    关键：两个模型喂【完全相同的输入】做【单次前向】，对比 logits。
    为什么不用自由生成对比：自由生成里一步 token 选择不同，后面会雪崩式发散
    （蝴蝶效应），你就分不清是"kernel 误差"还是"采样分岔"。teacher forcing
    （用同一份固定输入喂两边）才能干净地隔离出 kernel 本身的误差。
    """
    # 基准：纯 torch RMSNorm
    m_ref = swap_norm_backend(build_model_fn(), "torch").cuda().to(torch.bfloat16).eval()
    logits_ref, _ = m_ref(x_fixed)
    del m_ref; torch.cuda.empty_cache()

    # 实验：triton RMSNorm
    m_tri = swap_norm_backend(build_model_fn(), "triton").cuda().to(torch.bfloat16).eval()
    logits_tri, _ = m_tri(x_fixed)

    print(f"尺子1 逐元素 max err : {max_abs_err(logits_ref, logits_tri):.3e}")
    print(f"尺子2 logits cosine  : {logits_cosine(logits_ref, logits_tri):.6f}")
    print(f"尺子3 top-1 一致率   : {top1_agreement(logits_ref, logits_tri):.4f}")
```

预期输出（示意）：

```
尺子1 逐元素 max err : 4.882e-03      # bf16 正常量级
尺子2 logits cosine  : 0.999987       # 方向几乎完全一致
尺子3 top-1 一致率   : 1.0000          # 最终决策 100% 相同 —— 换 kernel 生成不变
```

**这三行就是你"误差可控可解释"的铁证**：max err 在 bf16 正常量级、cosine 逼近 1、top-1 完全一致 → **换 kernel 后生成结果实质不变，提速是"免费"的（没牺牲正确性）。** 这句话写进 `end_to_end_showdown.md`。

> **工业惯例**：这套"三把尺子"就是推理框架（vLLM、TensorRT-LLM）做 kernel 替换时的标准验收流程——先对 logits 数值，再看 top-1/生成一致率。你现在练的，就是工业界真实的 kernel 验收手法。

---

## 8. 动手第五步：写"诚实结论"——今天最值钱的东西

这一节不是让你写作文，是让你把今天的数据**炼成一个能指导小米课题决策的判断**。`end_to_end_showdown.md` 里必须回答下面四个问题，每个都要用**你自己跑出来的数字**支撑。

### 8.1 结论模板（照着填，填的是你的真实数据）

**① 我的手写 kernel 在哪组赢了、哪组输了？**

> 例：我的 Triton RMSNorm 在【组③ eager+手写】里比【组① eager】快 3%，赢了；但【组② torch.compile】比我的【组③】快 55%，我输了；【组④ compile+手写】比【组② 纯 compile】还慢 1.8%，也输了。

**② 为什么赢、为什么输？（用 Amdahl + 融合视野解释）**

> 例：③ 赢在 RMSNorm 本身确实更快，但赢得少——因为 RMSNorm 只占端到端约 5%（Amdahl 天花板 §1），单点优化上限就 5% 出头。② 赢在 torch.compile 做了**全图融合 + max-autotune**，优化的是接近 100% 的图，而我只优化了 5% 的一个点，量级不对等（§3.2）。④ 输在我的自定义 kernel 触发了 **graph break**，打断了 compile 的跨算子融合，破坏的收益 > 我省下的（§3.3）。

**③ 换了 kernel，生成结果对不对？（三把尺子）**

> 例：max err 4.9e-3（bf16 正常）、logits cosine 0.99998、top-1 一致率 100%。结论：换 kernel 后生成结果实质不变，提速没有牺牲正确性。

**④ 这告诉我小米巨核课题该怎么打？（最关键的一句）**

> 例：单点手写小算子（RMSNorm 这类占比 <10% 的）在 torch.compile 面前几乎没有端到端价值——compile 已经把这块吃干净了。**我的真正机会在 compile 吃不到的地方**：(a) 占比大的算子（Attention/MLP 的 matmul）；(b) 用**巨核把一整层十几个算子融成一个 kernel**，一次启动——这超出了 compile 的融合粒度，尤其在 **decode 阶段（launch 开销占比大，§2.3）** 收益最猛。这就是"巨核大模型推理优化"值得做的底层依据。

### 8.2 为什么这段"诚实"比"我快了 N×"值一百倍

- **它是决策依据，不是炫耀**：老板/导师看"我快了 3×"没用（快的是啥？占多少？），看"compile 是 1.6× 基线、我的机会在巨核吃掉 decode 的 launch 开销"——**这才是能拍板往哪投人力的信息**。
- **它建立了正确的分母认知**：你课题要 ≥20% 提升，分母是**谁**？是 eager 还是 torch.compile？今天这张表告诉你：**必须以 torch.compile 为分母**（因为那是真实生产基线），不能拿 eager 当分母偷偷放水。**用 eager 当分母刷出来的 20%，在评审面前一戳就破。**
- **它是资深工程师的思维方式**：诚实面对"我在哪打不过基线"，才能把有限精力投到真正有增量的地方。**这是 Amdahl 定律教给性能工程师最深的一课：不是"我能让什么变快"，而是"让什么变快才对端到端有意义"。**

### 8.3 `end_to_end_showdown.md` 建议结构

```markdown
# 端到端对标：手写 RMSNorm vs torch.compile

## 1. 实验设置
- 模型 / 参数量 / dim / 序列长度 / dtype(bf16)
- 硬件（GPU 型号）、torch / triton 版本
- 对标口径：warmup 3 次、iters 10 次取中位数、synchronize（§6.1 三铁律）

## 2. 四方对标表（TTFT + tok/s + 相对 eager）
（贴 §6.3 print_table 的输出）

## 3. 三把尺子误差
（贴 §7.5 输出：max err / cosine / top-1）
一句话：换 kernel 后生成实质不变，提速无损正确性。

## 4. 诚实结论（§8.1 四问）
- 赢在哪、输在哪
- 为什么（Amdahl + 融合视野 + graph break）
- 对小米巨核课题的决策指引：主攻大占比算子 / 巨核融合 / decode 战场
```

---

## 9. 完成标准自检（对照打勾）

- [ ] `week8_triton/06_nanogpt_integrated.py`：Triton RMSNorm 焊进 nanoGPT，`torch.no_grad` 下端到端生成，文本通顺不乱码。
- [ ] 一张**四方对标表**：① eager ② compile(max-autotune) ③ eager+手写 ④ compile+手写，各有 TTFT + tok/s。
- [ ] 计时遵守三铁律：warmup、synchronize、多次取中位数。
- [ ] **三把尺子**齐全：max err、logits cosine、top-1 一致率，且用 teacher forcing（固定输入单次前向）隔离误差。
- [ ] `tech_notes/end_to_end_showdown.md`：四方表 + 三尺子 + **一段"我的 kernel 相对 torch.compile 的真实位置"的诚实分析**。
- [ ] 诚实分析里明确写出：**以 torch.compile 为分母**、单点小算子的 Amdahl 天花板、巨核课题的主攻方向。

---

## 10. 一页速查（明天回顾用）

| 概念 | 一句话记忆 |
|------|-----------|
| Amdahl 定律 | 天花板 = 1/(1-p)，你没优化的部分卡死上限 |
| p（占比） | 先 profile 拿到 p，再决定值不值得写 kernel |
| eager 模式 | 一算子一 kernel，无融合，最慢基准 |
| torch.compile | 全图融合 + max-autotune，通常最强对手，你的真实分母 |
| graph break | 自定义 kernel 打断 compile 融合，导致 ④ 可能<② |
| fp32 累加 | 归一化 reduction 必须 fp32，否则 bf16 掉精度出乱码 |
| fallback | 必须有兜底 + 不静默吞异常，否则你以为在加速其实在回退 |
| KV Cache / decode | 每步 1 token，launch 开销占大头 → 巨核主战场 |
| GPU 计时三铁律 | warmup + synchronize + 多次取中位数 |
| 三把尺子 | max err（看最坏）+ cosine（看方向）+ top-1（看决策） |
| teacher forcing | 固定输入单次前向，隔离 kernel 误差，避免生成发散 |
| 今日核心心法 | 不是"我能让什么变快"，是"让什么变快才对端到端有意义" |

---

> **收尾寄语**：今天你做的不是"又写了个更快的 kernel"，而是**第一次站在系统层面，用数据诚实地给自己的优化定位**。这份"知道自己打不过 torch.compile、也知道该往哪打才打得过"的清醒，就是把你和"只会写 kernel 的人"区分开的东西——它直接决定你在小米巨核课题里，能不能选对那个值得 all-in 的战场。

