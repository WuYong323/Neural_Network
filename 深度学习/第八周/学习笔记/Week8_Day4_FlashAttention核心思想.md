# Day 4 · FlashAttention 的思想:为什么"不落地 T×T 矩阵"是关键

> **本周理解高峰。** W6 你手写的 attention 会显式生成 T×T 的注意力矩阵(`wei = q @ k.T`);W7 你知道 decode 是 memory-bound(受访存带宽限制)。今天把这两件事撞在一起。
>
> **一句话记住今天:FlashAttention 的核心不是新数学,而是"把 softmax 融进 attention,让那个 T×T 矩阵永远不落 HBM"。** 这正是你 W7 Day2"第二笔账(访存账)"在 attention 上的终极形态,也是你未来在张老师组做巨核推理优化时要亲手复现的核心技巧。

---

## 目录

- [0. 今日心智地图:两条线索在这里合流](#0-今日心智地图两条线索在这里合流)
- [1. 问题一:标准 attention 为什么是 memory-bound](#1-问题一标准-attention-为什么是-memory-bound)
- [2. 问题二:online softmax —— 今天唯一的"魔法"](#2-问题二online-softmax--今天唯一的魔法)
- [3. 问题三:FlashAttention 省的是哪笔账](#3-问题三flashattention-省的是哪笔账)
- [4. 对照 W6:我那版哪一步把 T×T 落回了 HBM](#4-对照-w6我那版哪一步把-tt-落回了-hbm)
- [5. 动手 Track A:简化版 fused attention(Triton)](#5-动手-track-a简化版-fused-attentiontriton)
- [6. 对标官方:"官方比我快在哪"差距地图](#6-对标官方官方比我快在哪差距地图)
- [7. Track B:AMK 巨核里的 attention 与 H100 反主场](#7-track-bamk-巨核里的-attention-与-h100-反主场)
- [8. 完成标准:脱稿讲稿(面试级)](#8-完成标准脱稿讲稿面试级)

---

## 0. 今日心智地图:两条线索在这里合流

在动任何代码之前,先把今天要"撞"的两件事摆清楚,否则你会陷在 Triton 语法里,忘了自己在证明什么。

**先理清几个基础名词(第一次出现,深度解释):**

- **HBM(High Bandwidth Memory,高带宽显存)**:就是你 `nvidia-smi` 里看到的那 80GB。它容量大、离计算核心远。类比:**仓库**。东西多,但每次取货要开车过去,慢。H100 的 HBM 带宽约 **3.35 TB/s**——听着快,但对 attention 来说远远不够。
- **SRAM(Static RAM,片上高速缓存 / 在 GPU 里通常指 Shared Memory + 寄存器)**:紧贴计算单元,容量极小(A100 上每个 SM 仅 **192KB**),但带宽极高(约 **19 TB/s**,是 HBM 的 10 倍以上)。类比:**你工位上的小抽屉**。放不了几样东西,但伸手就拿到。
- **SM(Streaming Multiprocessor,流式多处理器)**:GPU 的"一个车间"。H100 有 132 个 SM,每个 SM 自带一小块 SRAM。**注意:一个 SM 的 SRAM 另一个 SM 看不到**——这个事实是理解 Track B"跨 SM 同步为什么贵"的关键,先记住。
- **memory-bound(访存受限)**:一个操作跑得多快,不取决于 GPU 算得多快,而取决于数据从 HBM 搬进搬出的速度。类比:**厨师刀工再快,也快不过你从冰箱(HBM)往砧板(SRAM)搬菜的速度**。反义词是 **compute-bound(计算受限)**。

**两条线索:**

```
线索 A(来自 W6):我手写的 attention
    wei = q @ k.T        # 显式造出一个 T×T 的大矩阵
    wei = softmax(wei)   # 对这个大矩阵做 softmax
    out = wei @ v        # 大矩阵再乘 V

线索 B(来自 W7):decode 阶段是 memory-bound
    瓶颈不在"算",在"搬"。第二笔账(访存)才是大头。

          ↓  今天让它们相撞  ↓

结论:那个 T×T 大矩阵,是标准 attention 最沉重的"第二笔账"。
      FlashAttention 的全部功力,就是让它永远不落 HBM。
```

今天的目标不是写出比官方快的 kernel(你打不过,这正常),而是**彻底想通这个矩阵为什么该消失、以及靠什么数学技巧(online softmax)能让它消失**。

---

## 1. 问题一:标准 attention 为什么是 memory-bound

### 1.1 是什么:先把标准 attention 的三步摊开

注意力的数学就三步(单头,序列长 `T`,每个 token 向量维度 `d`):

```
输入:Q, K, V   形状都是 (T, d)      # T=序列长度, d=head_dim
1) S = Q @ Kᵀ          形状 (T, T)   # 每个 query 对每个 key 打分
2) P = softmax(S)      形状 (T, T)   # 逐行归一化成概率
3) O = P @ V           形状 (T, d)   # 用概率加权求和 value
```

关键就在 **`S` 和 `P` 都是 `(T, T)`**。这个矩阵随序列长度**平方级膨胀**。

### 1.2 为什么:手算一遍 T×T 矩阵在 HBM 里往返几趟

W6 §attention_complexity 你已经算过:**T=1024、单头、fp32,一个 T×T 矩阵 = 1024×1024×4 字节 = 4MB**。

而 SM 的 SRAM 只有 192KB。**4MB 根本塞不进 192KB 的抽屉**,所以标准实现只能把它扔回仓库(HBM)。我们逐步跟踪它的往返(GPU 一步就是一次 kernel launch,每次 kernel 都要"从 HBM 读输入 → 算 → 把输出写回 HBM"):

```
标准 attention 的 HBM 往返(❗标记访存,这就是"第二笔账"):

Step 1  matmul kernel:  读 Q,K(小)  →  算 S  →  ❗写 S 到 HBM      (+4MB 写)
Step 2  softmax kernel: ❗从 HBM 读回 S (+4MB 读) → 算 → ❗写 P 到 HBM (+4MB 写)
Step 3  matmul kernel:  ❗从 HBM 读回 P (+4MB 读) → 读 V → 算 O → 写 O

           ┌─────────────────────────────────────────────┐
           │  那个 4MB 的 T×T 矩阵,一来一回穿过 HBM ≈ 4 次 │
           │  ≈ 16MB 的纯访存,而它只是个"中间产物"        │
           └─────────────────────────────────────────────┘

对比:真正的输入输出 Q/K/V/O 各只有 T×d×4 = 1024×64×4 = 256KB,小得可怜。
```

**核心洞察:HBM 流量被那个"用完就扔的中间矩阵"主宰了,而不是被真正的输入输出主宰。** 这就是浪费的本质。

### 1.3 用"算术强度"把 memory-bound 钉死

**算术强度(Arithmetic Intensity)**:每从 HBM 搬 1 字节数据,能顺带做多少次浮点运算(FLOPs/Byte)。这是判断一个 op 是"算得慢"还是"搬得慢"的黄金标尺。

类比:算术强度就像**"你开一趟车去仓库,能顺便干多少活"**。开车(搬数据)成本固定,如果拉一趟货只干一点点活,那你大部分时间在路上——这就是 memory-bound。

- GPU 的**平衡点**(以 H100 为例):算力 ≈ 1000 TFLOPS(fp16),带宽 ≈ 3.35 TB/s。平衡点 ≈ 1000e12 / 3.35e12 ≈ **~300 FLOPs/Byte**。也就是说:**每搬 1 字节,得干够 ~300 次浮点运算,才"配得上"这次搬运**,否则算力就在空转等数据。
- 标准 attention 的 softmax + 中间矩阵读写:对 `T×T` 个元素,每个元素只做几次 exp/加法,却要完整读一遍、写一遍。算术强度低到个位数 —— **远低于 300,妥妥的 memory-bound**。

> **一句话总结问题一:** 标准 attention 慢,不是因为矩阵乘法算得慢(那部分其实很快),而是因为那个 `T×T` 中间矩阵在 HBM 里被反复读写,算术强度极低,GPU 大部分时间在等数据搬运。**这是"第二笔账"的典型受害者。**

---

## 2. 问题二:online softmax —— 今天唯一的"魔法"

> **底层理解锚点。想通这一节,你就懂了这个统治性算子。** FlashAttention 全部的巧妙,浓缩在这一个数学技巧里。

### 2.1 是什么:先看普通 softmax 为什么"必须先看到一整行"

softmax 的定义,对一行分数 `x = [x₁, x₂, ..., x_T]`:

```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

朴素实现有个致命问题:**数值溢出**。`exp(89)` 在 fp32 里就 ≈ 4.5e38,超过 fp32 上限(~3.4e38)直接变 `inf`。attention 分数经常大到几十上百,一 `exp` 就爆。

所以工业界**永远用"减最大值"的稳定版 softmax**(safe softmax):

```
m = max(x)                              # 第 1 遍:扫全行,找最大值
softmax(x)ᵢ = exp(xᵢ - m) / Σⱼ exp(xⱼ - m)   # 减去 m,最大的 exp 变成 exp(0)=1,绝不溢出
```

减 `m` 后,分子分母同时缩放,结果**数学上完全等价**,但每一项 `exp(xᵢ - m)` 都 ≤ 1,安全。

**问题来了:** 这个稳定版需要**先扫一遍全行拿到 `m`,再扫一遍算分母 `Σexp`,第三遍才能算最终值**。也就是所谓的 **3-pass softmax(三遍扫描)**。要做到"三遍扫描",你就**必须把一整行 `T` 个分数都存下来**——这正是 T×T 矩阵不得不落地 HBM 的数学根源!

```
朴素稳定 softmax = 3 遍扫描 = 必须先攒齐一整行 = T×T 矩阵必须存下来
                                                        ↑
                                           这就是我们要干掉的东西
```

### 2.2 为什么:online softmax 让你"边算边更新,不必先看到整行"

**online softmax(在线 / 流式 softmax)** 的野心:**只扫一遍,分块处理,每来一小块就增量修正之前的累积结果。** 这样你永远不需要把整行存下来——处理完一块就可以扔掉,只保留两个标量状态。

它维护两个"运行状态(running state)":

- `m` = **running max(运行最大值)**:到目前为止见过的最大分数。
- `l` = **running sum(运行分母)**:到目前为止累积的 `Σ exp(xᵢ - m)`。

**核心难点(也是唯一的魔法):** 分母 `l` 是"相对于当前 `m`"算的。如果新来的一块里出现了一个**比 `m` 还大的值**,那么 `m` 要更新成 `m_new`,可**之前累积的 `l` 是用旧的 `m_old` 算的,基准错了!** 怎么办?

**答案:用一个修正因子 `α = exp(m_old − m_new)` 把旧账"整体缩放"到新基准上。** 推导只需一行代数:

```
旧的  l_old = Σ exp(xᵢ - m_old)          （基于旧基准 m_old）
想要  改成 Σ exp(xᵢ - m_new)             （换到新基准 m_new）

因为  exp(xᵢ - m_new) = exp(xᵢ - m_old) · exp(m_old - m_new)
                       = exp(xᵢ - m_old) · α

所以  只需  l_old · α  就完成了基准切换,不必重新扫描旧数据!
            ↑
      这就是全部的魔法:旧账乘一个修正因子 α,就"追溯性地"对齐了新基准。
```

于是每来一块 `x_block`,更新规则是:

```
m_new = max(m_old, max(x_block))            # 更新运行最大值
α     = exp(m_old - m_new)                  # 修正因子(旧账要缩水多少)
l_new = l_old · α + Σ exp(x_block - m_new)  # 旧账缩水 + 新块贡献
m     = m_new
```

**因为 `m_new ≥ m_old`,所以 `α = exp(m_old − m_new) ≤ 1`——修正永远是"把旧账缩小",符合直觉:发现了更大的值,之前那些相对就没那么重要了。**

### 2.3 增量更新图(手绘这张图,今天就通关了)

用一个具体例子走一遍。假设一行分数被切成两块处理:

```
════════════════════════════════════════════════════════════════
 处理 Block 1:  scores = [2, 5, 1]
────────────────────────────────────────────────────────────────
   m = max(2,5,1) = 5
   l = e^(2-5) + e^(5-5) + e^(1-5)
     = 0.050   + 1.000   + 0.018   = 1.068
   状态: m=5, l=1.068
════════════════════════════════════════════════════════════════
 处理 Block 2:  scores = [7, 3]    ← ❗出现了比 5 更大的 7!基准要变
────────────────────────────────────────────────────────────────
   m_block = max(7,3) = 7
   m_new   = max(m_old=5, 7) = 7
   α       = e^(m_old - m_new) = e^(5-7) = e^(-2) = 0.135   ← 修正因子

   ┌───────────────── 关键一步:修正旧账 ─────────────────────┐
   │  旧的 l 是按基准 5 算的,现在基准变 7,先缩水:               │
   │      l_old · α = 1.068 × 0.135 = 0.144                │
   │  再加上新块(按新基准 7)的贡献:                            │
   │      e^(7-7) + e^(3-7) = 1.000 + 0.018 = 1.018        │
   │  →  l_new = 0.144 + 1.018 = 1.162                     │
   └───────────────────────────────────────────────────────┘
   状态: m=7, l=1.162
════════════════════════════════════════════════════════════════
 验证:如果一次性算 softmax([2,5,1,7,3]) 的分母(基准=7):
   e^(2-7)+e^(5-7)+e^(1-7)+e^(7-7)+e^(3-7)
   = 0.0067 + 0.135 + 0.0025 + 1.000 + 0.018 = 1.162  ✅ 完全一致!
════════════════════════════════════════════════════════════════
```

**这张图就是 FlashAttention 的心脏。** 每来一块新数据,发现更大的 max,就用 `α` 把之前所有累积量(分母 `l` 和后面会讲的输出 `acc`)"整体缩水"到新基准,然后叠加新块贡献。**全程只保留几个标量,从不存整行。**

### 2.4 怎么做:纯 Python 底层实现(任何机器都能跑,先建立肌肉记忆)

```python
# online_softmax_demo.py
# 依赖:仅需 Python 3 标准库(math)。目的是脱离 GPU、脱离框架,
#      用最朴素的代码把"增量修正"这件事跑给自己看。
import math

def naive_softmax(x):
    """稳定版 softmax:3 遍扫描,必须先攒齐一整行 x。"""
    m = max(x)                                  # 第 1 遍:找 max(为什么?防 exp 溢出)
    exps = [math.exp(xi - m) for xi in x]       # 第 2 遍:算分子
    l = sum(exps)                               # 分母
    return [e / l for e in exps]                # 第 3 遍:归一化

def online_softmax(x, block_size=1):
    """在线 softmax:1 遍扫描,分块增量。永远不存整行,只留 m 和 l 两个标量。"""
    m = float('-inf')   # running max:见过的最大分数
    l = 0.0             # running sum:相对当前 m 的 Σexp
    for i in range(0, len(x), block_size):
        block = x[i:i + block_size]
        m_block = max(block)
        m_new = max(m, m_block)
        alpha = math.exp(m - m_new)             # ← 魔法:旧账修正因子(m=-inf 时 alpha=0,天然正确)
        # 旧账缩水 α 倍,再叠加新块相对新基准的贡献
        l = l * alpha + sum(math.exp(xi - m_new) for xi in block)
        m = m_new
        print(f"  处理到第 {i+len(block):2d} 个元素后: running_max={m:.3f}, running_sum={l:.4f}")
    # 最终每一项的概率(第二遍才需要,attention 里这一步会和 @V 融合,见 2.5)
    return [math.exp(xi - m) / l for xi in x]

if __name__ == "__main__":
    x = [2.0, 5.0, 1.0, 7.0, 3.0]
    print("=== naive(3遍,需整行) ===")
    print([round(p, 4) for p in naive_softmax(x)])
    print("=== online(1遍,分块增量,block_size=2) ===")
    p = online_softmax(x, block_size=2)
    print([round(pi, 4) for pi in p])
    # 两者输出应完全一致,证明 online softmax 数学等价、且不必先看到整行
```

运行你会看到 `running_max`、`running_sum` 随每块更新,最终两种方法结果**逐位相同**。这就是"数学上完全等价,但内存行为天差地别"。

### 2.5 把 online softmax 和 `@V` 融合(这才是 FlashAttention 完整形态)

attention 的输出是 `O = softmax(S) @ V`。softmax 的分子还没除以分母时,就已经可以乘 `V` 累加了——**输出累加器 `acc` 和分母 `l` 用同一个 `α` 一起修正**,最后再统一除以 `l`。这样连"第二遍算概率"都省了,`S` 彻底不必存下。

```python
# flash_attention_one_query.py
# 依赖:numpy。演示"单个 query"如何在只保留 m/l/acc 三个状态的情况下,
#      流式地把 softmax 和 @V 融成一遍,全程不生成整行注意力权重。
import numpy as np

def flash_attention_single_query(q, K, V):
    """
    q: (d,)      单个 query 向量
    K: (T, d)    所有 key
    V: (T, d)    所有 value
    返回: (d,)   该 query 的注意力输出,等价于 softmax(qKᵀ/√d) @ V
    """
    d = q.shape[0]
    scale = 1.0 / np.sqrt(d)
    m = -np.inf                 # running max(标量)
    l = 0.0                     # running sum(标量)
    acc = np.zeros(d)           # running 输出累加器(d 维向量,不是 T×T!)
    for j in range(K.shape[0]):
        s = np.dot(q, K[j]) * scale     # 当前这一个分数,一个标量
        m_new = max(m, s)
        alpha = np.exp(m - m_new)       # 修正因子
        p = np.exp(s - m_new)           # 当前项的未归一化权重
        l = l * alpha + p               # 分母:旧账缩水 + 新项
        acc = acc * alpha + p * V[j]    # ← 输出累加器同步修正!和 l 用同一个 alpha
        m = m_new
    return acc / l                       # 最后统一除以分母

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d = 64, 32
    q = rng.standard_normal(d)
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d))

    out_flash = flash_attention_single_query(q, K, V)

    # 参考:标准三步法(显式造出长度 T 的权重向量)
    scale = 1.0 / np.sqrt(d)
    s = (q @ K.T) * scale
    p = np.exp(s - s.max()); p /= p.sum()
    out_ref = p @ V

    cos = np.dot(out_flash, out_ref) / (np.linalg.norm(out_flash) * np.linalg.norm(out_ref))
    print(f"cosine 相似度 = {cos:.8f}  (应 ≈ 1.0)")
    # 注意 acc 始终是 (d,) 维,循环里从没出现过 (T,) 的完整权重向量
```

**盯住这两行,它们就是 FlashAttention:**

```python
l   = l   * alpha + p          # 分母增量修正
acc = acc * alpha + p * V[j]   # 输出增量修正(和分母共用 alpha)
```

`acc` 永远是 `(d,)` 维,`l`/`m` 是标量。**扩展到一个 query 块 `(BLOCK_M, d)`,`acc` 就是 `(BLOCK_M, d)`——依然只和 `d` 相关,和 `T` 无关。那个 `T×T` 矩阵,就这样从内存里被"抹掉"了。**

---

## 3. 问题三:FlashAttention 省的是哪笔账

### 3.1 是什么:一句话对齐 W7 的"两笔账"

W7 你建立了"两笔账"的框架:

- **第一笔账 = 计算账(FLOPs)**:要做多少次浮点乘加。
- **第二笔账 = 访存账(HBM 流量)**:要在仓库(HBM)和工位(SRAM)之间搬多少字节。

**FlashAttention 省的是第二笔账,几乎不动第一笔账。** 这一点极其重要,面试常考,别搞反:

```
                    标准 attention        FlashAttention
计算(FLOPs)         ~4·T²·d              ~4·T²·d       ← 几乎没变!甚至略增(重算)
访存(HBM 流量)      O(T²)  ❗大头         O(T·d)  ✅暴降  ← 全部功劳在这
```

> **反直觉但关键:FlashAttention 并没有减少计算量,矩阵乘法一次没少做。它只是让那个 T×T 中间矩阵"生在 SRAM、用在 SRAM、死在 SRAM",从不写回 HBM。** 因为 attention 本来是 memory-bound(第 1 节证明过),砍访存就等于砍总时间。这是"针对瓶颈下刀"的教科书案例。

### 3.2 为什么:分块(tiling)让 T×T 永远住在片上

把 `Q`、`K`、`V` 切成小块,让每一小块的中间结果都能塞进 192KB 的 SRAM 抽屉:

```
外层循环:遍历 Q 的每个块 Q_i  (BLOCK_M 行)
  在 SRAM 里初始化该块的 m_i, l_i, acc_i
  内层循环:遍历 K/V 的每个块 K_j, V_j  (BLOCK_N 行)
    S_ij = Q_i @ K_jᵀ            # 小块打分,住在 SRAM,大小 BLOCK_M×BLOCK_N
    用 online softmax 增量更新 m_i, l_i, acc_i   ← 第 2 节的魔法
    ——— S_ij 用完即弃,从不写 HBM ———
  最后 acc_i / l_i,把结果 O_i 写回 HBM

关键:内存里同时存在的只有几个小块,峰值远小于 192KB。T 再大也不怕。
```

### 3.3 顺带一提:反向传播的"以算换存"

前向不存 `S`/`P`,反向传播需要它们怎么办?**FlashAttention 选择在反向时用 `Q/K/V` 重新算一遍 `S`(recomputation,重计算),而不是把它们存下来。** 这又是一次"第一笔账换第二笔账":多花一点算力(便宜),省下巨额显存和访存(贵)。今天以前向理解为主,这里知道有这回事即可。

---

## 4. 对照 W6:我那版哪一步把 T×T 落回了 HBM

把你 W6 手写的 attention 拿出来逐行标注。这是今天"对照阅读"的核心动作——**在自己的代码上,亲手指认出那个该死的矩阵在哪里落地**。

```python
# 这是 W6 风格的手写 attention(PyTorch)。每个中间张量都是一次 HBM 落地。
import torch
import torch.nn.functional as F

def attention_w6(q, k, v, causal=True):
    # q,k,v: (T, d)
    T, d = q.shape
    scale = d ** -0.5

    wei = q @ k.transpose(-2, -1) * scale     # ❗ 落地①:wei 是 (T,T),PyTorch 立刻在 HBM 分配 4MB
                                              #    这一步结束,一个完整的 T×T 矩阵已经躺在 HBM 里
    if causal:
        mask = torch.tril(torch.ones(T, T, device=q.device))
        wei = wei.masked_fill(mask == 0, float('-inf'))  # ❗ 落地②:又读又写这个 T×T 矩阵

    wei = F.softmax(wei, dim=-1)              # ❗ 落地③:softmax 从 HBM 读回 (T,T),写回 (T,T)
                                              #    (PyTorch 内部其实是 3-pass,反复读写)
    out = wei @ v                             # ❗ 落地④:@v 之前再从 HBM 读回整个 (T,T) 的 wei
    return out                                # out 是 (T,d),小
```

**逐行对照 FlashAttention 怎么避开:**

| W6 这一行 | 发生了什么落地 | FlashAttention 怎么避开 |
|---|---|---|
| `wei = q @ k.T` | 造出完整 `(T,T)` 写进 HBM | 只算 `Q_i @ K_jᵀ` 的小块 `S_ij`,留在 SRAM |
| `masked_fill(...)` | 再读写整个 `(T,T)` | mask 在小块 `S_ij` 上就地施加 |
| `F.softmax(wei)` | 独立 kernel,把 `(T,T)` 读回再写回 | 融进循环,online softmax 增量更新,不独立成 kernel |
| `wei @ v` | 又把整个 `(T,T)` 读回 | `acc += P_ij @ V_j`,`P_ij` 只是 SRAM 里的小块 |

> **写进你笔记的一句话:** 我 W6 那版的病根是"**每一步都是一个独立 PyTorch 算子,算子之间必须靠 HBM 传递那个 4MB 的中间矩阵**"。FlashAttention 把这四步**融合(fuse)成一个 kernel**,中间矩阵从生到死都在 SRAM,四趟 HBM 往返归零。

---

## 5. 动手 Track A:简化版 fused attention(Triton)

> **今天以理解 + 读为主、写为辅。目标不是打过官方,而是亲手体会"softmax 和两个 matmul 融在一个 kernel 里"是什么感觉。**

### 5.0 名词补课:Triton 是什么

**Triton** 是 OpenAI 开源的、用 Python 语法写 GPU kernel 的语言/编译器。类比:CUDA 是让你**手动挡开车**(管到每个线程),Triton 是**自动挡**(你只管"块"这个粒度,线程调度、SRAM 分配交给编译器)。它是当前写融合算子(fused kernel)的主流工具,PyTorch 2.x 的 `torch.compile` 后端 Inductor 生成的就是 Triton 代码。**你未来在张老师组写巨核,Triton 是绕不开的。**

> **⚠️ 运行环境提醒(你在 Windows,这条很重要):** Triton 官方主要支持 Linux + NVIDIA GPU。**别在你的 Windows 笔记本上折腾**,直接在你那台 **H100 的 Linux 机器**上跑(或临时用 Colab 的 T4/A100)。依赖:`pip install torch triton`(近版 PyTorch 的 Linux wheel 已自带 triton)。第 2 节的纯 Python/numpy 版本才是你在任何机器上都能跑的。

### 5.1 阶梯一:最简"整块读进 SRAM"版(先不做 online softmax)

这是任务建议的最简版:**假设 T 很小,一次把整行 K/V 读进 SRAM,做一次普通 softmax**。它已经实现了"融合"(三步在一个 kernel 内),但因为要求整行塞进 SRAM,**只能处理小 T**——这恰好暴露了为什么需要 online softmax。

```python
# 04_fused_attention_simplified.py  ——  阶梯一:naive fused(无 online softmax)
# 依赖: torch, triton;环境: Linux + NVIDIA GPU(在你的 H100 机器上跑)
import torch
import triton
import triton.language as tl

@triton.jit
def fused_attn_naive_kernel(
    Q, K, V, O,                       # 指针,张量形状均为 (T, D)
    T,                                # 序列长度(运行时值)
    scale,                            # 1/sqrt(D)
    D: tl.constexpr,                  # head_dim,编译期常量
    BLOCK_M: tl.constexpr,            # 每个 program 负责的 query 行数
    BLOCK_N: tl.constexpr,            # 一次性覆盖所有 key(要求 BLOCK_N >= T)
    IS_CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)                              # 我是第几个 query 块
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)      # 我负责的 query 行号
    offs_n = tl.arange(0, BLOCK_N)                      # 所有 key 行号(整块)
    offs_d = tl.arange(0, D)

    # 为什么用 mask:T 不一定整除 BLOCK,越界的行读进来是垃圾,必须屏蔽
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)    # (BLOCK_M, D) 进 SRAM
    k = tl.load(K + offs_n[:, None] * D + offs_d[None, :],
                mask=offs_n[:, None] < T, other=0.0)    # (BLOCK_N, D) 整个 K 进 SRAM
    v = tl.load(V + offs_n[:, None] * D + offs_d[None, :],
                mask=offs_n[:, None] < T, other=0.0)    # (BLOCK_N, D) 整个 V 进 SRAM

    s = tl.dot(q, tl.trans(k)) * scale                 # (BLOCK_M, BLOCK_N) 分数,住在 SRAM
    s = tl.where(offs_n[None, :] < T, s, float('-inf'))  # 屏蔽越界 key 列
    if IS_CAUSAL:                                       # 因果 mask:只能看自己和左边
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))

    # —— 一次性 softmax(需要整行 s 都在 SRAM,所以只适用小 T)——
    m = tl.max(s, axis=1)                               # 每行 max(防溢出)
    p = tl.exp(s - m[:, None])
    l = tl.sum(p, axis=1)
    p = p / l[:, None]                                  # (BLOCK_M, BLOCK_N) 归一化权重

    o = tl.dot(p.to(v.dtype), v)                        # (BLOCK_M, D) 输出
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], o,
             mask=offs_m[:, None] < T)


def fused_attn_naive(q, k, v, causal=True):
    T, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = triton.next_power_of_2(T)                 # 整块覆盖:小 T 才行
    grid = (triton.cdiv(T, BLOCK_M),)
    fused_attn_naive_kernel[grid](
        q, k, v, o, T, D ** -0.5,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IS_CAUSAL=causal,
    )
    return o
```

**这一版的教学价值:** 它证明了"融合可行"——三步进一个 kernel,`s` 从没落 HBM。**但它的死穴是 `BLOCK_N >= T`:T 一大,`s` 就塞不进 SRAM,整个方案崩溃。** 这就逼出了阶梯二。

### 5.2 阶梯二:真正的 FlashAttention(online softmax + 分块)

和阶梯一的**唯一区别**,就是把"一次性 softmax"换成第 2 节的"分块 online softmax 循环"。**盯住带 `← 魔法` 注释的几行,它们和阶梯一的差异,就是 FlashAttention 的全部。**

```python
# 04_fused_attention_simplified.py  ——  阶梯二:真·FlashAttention(online softmax 分块)
@triton.jit
def flash_attn_kernel(
    Q, K, V, O,
    T, scale,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)   # 只加载我这块 Q,常驻 SRAM

    # online softmax 的三个运行状态(全程只有这几个,和 T 无关)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)   # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                 # running sum
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)               # running 输出

    for start_n in range(0, T, BLOCK_N):                # ← 分块:K/V 一块一块流过来
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)
        v = tl.load(V + offs_n[:, None] * D + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale              # (BLOCK_M, BLOCK_N) 小块分数,SRAM
        s = tl.where(offs_n[None, :] < T, s, float('-inf'))
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))

        # ===== online softmax 增量更新:今天的魔法,和第 2 节完全对应 =====
        m_ij  = tl.max(s, axis=1)                       # 本块每行 max
        m_new = tl.maximum(m_i, m_ij)                   # ← 魔法:更新 running max
        alpha = tl.exp(m_i - m_new)                     # ← 魔法:旧账修正因子 α
        p     = tl.exp(s - m_new[:, None])              # 本块相对新基准的权重
        l_i   = l_i * alpha + tl.sum(p, axis=1)         # ← 魔法:分母 = 旧账缩水 + 新块
        acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # ← 魔法:输出同步修正
        m_i   = m_new
        # ===============================================================

    acc = acc / l_i[:, None]                            # 循环结束,统一除分母
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], acc,
             mask=offs_m[:, None] < T)


def flash_attn(q, k, v, causal=True, BLOCK_M=64, BLOCK_N=64):
    T, D = q.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(T, BLOCK_M),)
    flash_attn_kernel[grid](
        q, k, v, o, T, D ** -0.5,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IS_CAUSAL=causal,
    )
    return o
```

> **对照阶梯一 ↔ 阶梯二的 diff,就是本周理解高峰的落点:** 阶梯一"一次性 softmax、要整行"→ 只能小 T;阶梯二"分块 online softmax、只留 m/l/acc"→ 任意 T,且 `s` 永不落 HBM。**这个 diff,你要能在白板上默写。**

### 5.3 对标官方 + cosine / top-1 误差(接 W7 Day6 §6.3)

```python
# 05_benchmark_and_error.py
# 依赖: torch, triton;把你的 kernel 和 PyTorch 内置 FlashAttention 对拍
import torch
import torch.nn.functional as F
# from 04_fused_attention_simplified import flash_attn, fused_attn_naive   # 按你的文件名导入

def cosine_sim(a, b):
    """attention 输出用 cosine 最合适:它衡量方向一致性,对整体缩放不敏感。"""
    a, b = a.flatten().float(), b.flatten().float()
    return F.cosine_similarity(a, b, dim=0).item()

def top1_match(a, b):
    """辅助 sanity check:逐 token 看输出向量的 argmax 特征位是否一致。"""
    return (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()

def benchmark(fn, q, k, v, causal, iters=100):
    # 预热(触发 Triton JIT 编译,别把编译时间算进去——常见新手坑)
    for _ in range(10):
        fn(q, k, v, causal)
    torch.cuda.synchronize()                 # ← 关键:GPU 异步,不同步测的是"下发时间"不是"执行时间"
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn(q, k, v, causal)
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # 毫秒/次

if __name__ == "__main__":
    torch.manual_seed(0)
    T, D = 1024, 64
    dtype = torch.float16                    # 用 fp16:贴近工业实践,也能上 tensor core
    q = torch.randn(T, D, device="cuda", dtype=dtype)
    k = torch.randn(T, D, device="cuda", dtype=dtype)
    v = torch.randn(T, D, device="cuda", dtype=dtype)

    out_mine = flash_attn(q, k, v, causal=True)

    # 官方内置 FlashAttention:F.scaled_dot_product_attention
    # 需要 (batch, heads, T, D) 形状,补上 batch/head 维
    out_ref = F.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], is_causal=True
    )[0, 0]

    print(f"cosine 相似度 = {cosine_sim(out_mine, out_ref):.6f}   (目标 > 0.999)")
    print(f"top-1 一致率  = {top1_match(out_mine, out_ref):.4f}")
    print(f"我的  kernel : {benchmark(flash_attn, q, k, v, True):.4f} ms")
    print(f"官方  SDPA   : {benchmark(lambda *a: F.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None],is_causal=True), q,k,v,True):.4f} ms")
```

**预期结果:cosine ≈ 0.999+(数学对了),但耗时大概率被官方甩开几倍。这非常正常,也非常重要——差距本身就是你今天最有价值的产出。**

> **误差方法论小结(呼应 W7 Day6):** attention 输出为什么用 **cosine** 而不是逐元素绝对误差?因为 fp16 累加顺序不同会带来微小数值差异,绝对误差会虚高;而我们真正关心的是"输出向量方向对不对"。**cosine 对缩放/微小抖动鲁棒,是 attention/embedding 类输出的标准对拍指标。** top-1 一致率作为辅助 sanity check。

---

## 6. 对标官方:"官方比我快在哪"差距地图

**你打不过官方,把原因写清楚,就是你和工业级 kernel 之间的"差距地图"。这是今天最能进简历的东西。** 面试官问"你和 FlashAttention 官方实现差在哪",你要能答出这张表:

| 维度 | 你的简化版 | 官方 FlashAttention(FA-2 / FA-3) | 差距根源 |
|---|---|---|---|
| **online softmax** | 已实现(阶梯二) | 已实现,且和反向重计算深度配合 | 前向你追平了,思想一致 |
| **Tensor Core 利用** | `tl.dot` 用上了,但块大小/精度没调优 | 精心排布,fp16/bf16 输入 + fp32 累加,吃满 Tensor Core | 你没做 warp 级数据布局 |
| **warp 级优化(warp specialization)** | 无,全交给 Triton 默认调度 | 不同 warp 分工(有的搬数据、有的算),流水线重叠 | warp 是 32 线程的调度单位;分工能让"搬"和"算"重叠 |
| **异步流水线(async pipeline)** | 无 | 用 `cp.async` / Hopper 的 **TMA**(张量内存加速器)异步预取下一块 K/V,搬数据时不停算 | 隐藏了访存延迟 |
| **块大小/split 调优(autotune)** | 手填 64 | 针对不同 GPU / 序列长自动搜最优配置 | 缺 autotune |
| **Hopper 专用(FA-3)** | 无 | 用 **wgmma**(warp-group 级矩阵乘)、FP8、生产者-消费者流水线 | 吃到 H100 专有指令 |

**几个新名词快速补:**

- **warp(线程束)**:GPU 调度的最小单位,32 个线程锁步执行。类比一个"32 人的施工小队,一起抬同一根梁"。
- **warp specialization(warp 专精)**:让不同小队分工——A 队专门从 HBM 搬砖(load),B 队专门砌墙(compute),两队重叠工作,搬砖时墙没停。
- **TMA(Tensor Memory Accelerator,张量内存加速器)**:H100 新增的硬件,专门高效地把一大块张量从 HBM 异步搬进 SRAM,搬的同时计算单元继续干活。

> **写进笔记的定性结论:** 我的版本和官方在**数学思想(online softmax + tiling)上完全一致**,cosine ≈ 1 证明了这点;差距**全在工程实现层**——Tensor Core 排布、warp 分工、异步预取、按硬件 autotune。**思想我今天已经拿下,剩下的是把片上流水线榨干,那是暑假后段和巨核阶段的活。**

---

## 7. Track B:AMK 巨核里的 attention 与 H100 反主场

> 结合你 W7 的 AMK(AutoMegaKernel)report:**attention region 约 364µs,是整个巨核的大头**。今天把它和 FlashAttention 对照,理解"为什么论文说 AMK 在 H100 上打不过"。

**先说 megakernel(巨核)是什么:** 通常一个大模型推理要 launch 成百上千个小 kernel(每层的 matmul、norm、attention…),**每次 launch 都有固定开销,且 kernel 之间靠 HBM 传数据**。巨核的野心是**把整个模型(或一大段)融进一个 kernel**,数据尽量留在片上,消灭 launch 开销和中间 HBM 往返。类比:与其派 1000 个快递员各送一件(每人都要出门),不如一辆车一趟拉完。

**为什么 attention 是巨核里最难啃、也最耗时的一块(呼应 364µs):**

1. **attention 需要跨整个序列做归约(reduction)。** softmax 的分母、`@V` 的加权和,本质是"把 T 个 key 的贡献汇总起来"。在巨核里,不同 key 块可能被分到**不同的 SM** 上算。
2. **一个 SM 的 SRAM,另一个 SM 看不到(回到第 0 节记住的事实)。** 要把分散在多个 SM 上的部分和(partial `m`/`l`/`acc`)合并成最终结果,就需要**跨 SM 同步(cross-SM synchronization)**——数据得绕经 HBM 或 L2、还要用全局屏障等所有 SM 到齐。**这种同步在 GPU 上非常贵**,因为 GPU 的设计哲学是"成千上万线程各干各的、尽量别互相等"。

**为什么偏偏在 H100 上,巨核方案的 attention 打不过专用 FlashAttention:**

- H100(Hopper 架构)为 attention 这类算子准备了**专属武器**:TMA 异步搬运、wgmma warp-group 矩阵乘、以及 **FlashAttention-3 专门设计的生产者-消费者流水线**。这套东西把单个 attention kernel 的片上流水线榨到极致。
- **巨核为了"把一切融进一个 kernel"这个全局目标,反而没法为 attention 单独启用这套 Hopper 专用的极致流水线**;而它试图省下的 launch 开销,在 H100 上又因为跨 SM 同步的代价被吃回去。**于是在 H100 这种"专用算子被硬件加持到牙齿"的平台上,巨核的 attention region 成了反主场——通用融合打不过专用极致。**

> **Track B 一句话总结(填进你的 AMK 对照笔记):** FlashAttention 是"把 attention 这一个算子的片上流水线做到极致";AMK 巨核是"把整个模型融成一个 kernel 求全局最优"。两者目标不同。在 H100 上,attention 因需要**跨 SM 归约同步**、又用不上 Hopper 为专用 FA-3 准备的 TMA/wgmma 流水线,导致巨核里的 attention region(≈364µs)成为短板——**这正是你接手 H100 profiling 时要重点用 nsys 去坐实的"跨 SM 同步开销"信号。**

---

## 8. 完成标准:脱稿讲稿(面试级)

**合上笔记,对着这四句能脱稿讲出来,今天就通关了:**

1. **标准 attention 为什么 memory-bound?**
   > 因为它显式造出 `T×T` 的注意力矩阵(T=1024 单头就 4MB),这个中间矩阵要在 HBM 里往返约 4 趟(写 S → 读 S 写 P → 读 P),而它的算术强度只有个位数,远低于 H100 ~300 FLOPs/Byte 的平衡点。瓶颈是搬,不是算——典型的"第二笔账"受害者。

2. **online softmax 凭什么不用先看到整行?**
   > 它只维护 running max `m` 和 running sum `l` 两个状态,分块处理。新块若出现更大的 max,就用修正因子 `α = exp(m_old − m_new) ≤ 1` 把旧账整体缩水到新基准,再叠加新块贡献。数学上和一次性 softmax 完全等价,但全程只存标量,从不存整行。

3. **FlashAttention 靠什么把 T×T 留在片上?**
   > 靠 online softmax + tiling:把 Q/K/V 切块,`S_ij` 小块只住在 SRAM,用 online softmax 增量更新 `m/l/acc`,小块用完即弃,从不写回 HBM。输出累加器 `acc` 和分母 `l` 共用同一个 `α` 同步修正,最后统一归一化。

4. **FlashAttention 省的是哪笔账?**
   > 省的是访存账(第二笔账),不是计算账。FLOPs 几乎没变(反向甚至因重计算略增),但 HBM 流量从 `O(T²)` 降到 `O(T·d)`。因为 attention 本来 memory-bound,砍访存就等于砍时间——针对瓶颈下刀。

---

### 今日产出清单

- [x] `week8_triton/04_fused_attention_simplified.py`(阶梯一 naive fused + 阶梯二 online softmax flash;见 §5.1 / §5.2)
- [x] `tech_notes/flashattention_idea.md`(本文件:含 online softmax 增量更新图 §2.3 + "官方比我快在哪"差距地图 §6)
- [x] 纯 Python / numpy 版 online softmax 与单 query flash(§2.4 / §2.5,任何机器可跑,建立肌肉记忆)
- [x] cosine + top-1 对拍脚本(§5.3)
- [x] Track B:AMK attention region 与 H100 反主场的定性结论(§7,待你用 nsys profiling 坐实)

> **一句话收尾:** 今天你没有发明新数学,你只是想通了一件事——**当一个中间结果又大、又只用一次、又拖慢整体时,最好的优化不是把它算得更快,而是让它根本不存在。** 这个思想,从 attention 一路通向你未来要做的巨核。
