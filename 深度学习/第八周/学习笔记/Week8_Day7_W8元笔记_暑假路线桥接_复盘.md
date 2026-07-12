# Day 7 · W8 元笔记 + 暑假路线桥接 + 复盘

> 本周收口。W8 是你从"会测"（W7）跨到"会改"的第一周——你亲手写了 5 类 kernel，每个都对标了 torch.compile、都用三把尺子验了误差。今天不写新 kernel，做三件事：
>
> 1. 把这周的手艺**串成一套可复用的方法论**（不是"我写过 5 个 kernel"，而是"给我任意 memory-bound 算子，我有一套固定打法"）。
> 2. 画一张**诚实的能力地图**——我现在能写什么、打不打得过 torch.compile、和官方 FlashAttention 差在哪。清楚边界，才知道下一步攻哪。
> 3. 定下**暑假后段的下沉路线**：CUDA → FlashAttention → vLLM。

---

## 0. 元笔记是什么，为什么这周必须写

**元笔记（meta-note）**：不是记录"我学了什么"，而是记录"我学的东西之间是什么关系、我现在站在哪、下一步往哪走"。普通笔记是**点**，元笔记是把点连成**线和地图**。

**为什么必须写（一个类比）**：你这周学的 5 类 kernel，像是打完了 5 个副本、捡了一堆装备。如果不整理，下次要用时还得翻箱倒柜。元笔记就是**把散装备整理进背包分类栏**——elementwise 放这格、reduction 放那格，并在包上贴张纸条写"我这套装备能打过 torch.compile 吗、打不过谁"。

**为什么现在写最值**：记忆还热。一周后你会忘掉"Day3 reduction 那个 mask 的坑""Day6 graph break 那个反直觉结论"。**元笔记是把短期记忆固化成可检索的长期资产**——这正是你 W4/W5/W6/W7 一直在做的事，W8 延续它。

今天的产出（4 个文件 + GitHub）：

- `tech_notes/week8_industrial_view.md` —— 本周元笔记（方法论总表 + 纪律兑现 + 能力地图 + 串联表）
- 更新 `inference_optimization_landscape.md` —— 新增 Triton/手写 kernel 能力
- `W8_review.md` —— 最有效动作 / 最大阻塞 / 下周砍什么
- `week8_triton/` 清理 + README + GitHub ≥3 次有意义 commit

---

## 1. 写 kernel 方法论总表：从 elementwise 到端到端集成

### 1.1 先看这条能力进阶链（为什么是这个顺序）

这周 5 步不是随便排的，它是一条**难度和视野双双递增**的链：

```
Day1 elementwise    → 学"怎么把数据切块喂给 GPU"（并行的地基）
Day2 fused pointwise → 学"怎么少搬几趟内存"（融合的核心收益）
Day3 reduction      → 学"怎么跨元素求和/求均值"（RMSNorm 的骨架）
Day4 fused attention → 学"怎么把 attention 一堆算子融成一个"（FlashAttention 思想）
Day6 端到端集成      → 学"怎么把 kernel 焊进真模型并诚实对标"（系统视野）
```

**内在逻辑**：前三步是"单算子内部"的功夫，越来越难；Day4 是"跨算子融合"的质变；Day6 是"跳出算子、看整个系统"的又一次质变。**每一步都在为理解 FlashAttention 和巨核铺路。**

### 1.2 方法论总表（今天的核心产出之一）

先解释几个反复出现的术语，后面表格才看得懂：

- **elementwise（逐元素运算）**：每个输出元素只依赖对应位置的一个输入元素，互不干扰。比如 `y = x + 1`，第 3 个输出只看第 3 个输入。**类比**：流水线上每个工人只管拧自己面前那颗螺丝，谁也不用等谁——天然完美并行。
- **memory-bound（访存受限）**：算子的瓶颈不在"算得慢"，而在"数据在显存和计算核心之间搬得慢"。**类比**：厨师切菜（计算）飞快，但食材要一趟趟从很远的仓库（显存）搬来，整体速度卡在搬运工身上。**归一化、逐元素、加法这类都是 memory-bound**——这正是 Triton 融合的主战场。
- **fusion（算子融合）**：把本来要分几个 kernel 做、数据来回搬几趟的操作，合并成一个 kernel、数据只搬一趟。**这是 memory-bound 算子提速的头号手段。**
- **reduction（归约）**：把一组数"塌缩"成一个数，比如求和、求最大、求均值。**类比**：全班分数求平均——要遍历所有人。难点在于它需要"跨元素通信"，不像 elementwise 那样各管各的。

| 步骤 | kernel 类型 | 练了什么能力 | 对应课题主线 | 典型对标结论 |
|------|------------|-------------|-------------|-------------|
| Day1 | elementwise（vector-add） | 切块 / grid / mask / 内存合并访问 | 并行编程地基 | 追平 torch，理解"launch 开销" |
| Day2 | fused pointwise | 融合省内存往返（读写一次 vs 多次） | memory-bound 提速核心 | 比 eager 快，逼近/追平 compile |
| Day3 | reduction（RMSNorm） | 跨元素归约 + fp32 累加精度 | 归一化算子、巨核零件 | 单算子快，端到端受 Amdahl 限 |
| Day4 | fused attention | online softmax + 分块不落大矩阵 | FlashAttention 思想内核 | 手写版打不过官方，但懂了原理 |
| Day6 | 端到端集成 | 热替换 / 三铁律对标 / 三尺子验误差 | 系统视野 + 分母认知 | compile 全图融合通常最强 |

> **怎么用这张表**：这不是"我做过什么"的清单，是**"给我一个新算子，我从表里定位它属于哪类、套哪套打法"的查询表**。看到一个归一化算子 → 归到 reduction 那行 → 想起 fp32 累加、逐行并行的套路。**这就是方法论的价值：把经验变成可复用的模式。**

### 1.3 提炼：一套"给我任意 memory-bound 算子"的固定打法

这是完成标准里要求你"5 分钟能讲清"的东西，把它背下来：

```
第 0 步 profile：先确认它真是 memory-bound、且占端到端比例够大（Amdahl，值不值得写）
第 1 步 写朴素 Triton kernel：一行/一块开一个 program，load → 算 → store
第 2 步 融合：把相邻的逐元素操作塞进同一个 kernel，数据只 load/store 一次
第 3 步 精度：reduction 部分用 fp32 累加，输出转回原 dtype
第 4 步 对标：warmup + synchronize + 多次取中位数，对比 eager / torch.compile
第 5 步 验误差：max err + logits cosine + top-1 一致率（三尺子）
第 6 步 诚实结论：赢在哪、输在哪、以 torch.compile 为分母、下一步攻哪
```

**这七步就是你 W8 的全部手艺凝结成的 SOP（standard operating procedure，标准作业流程）。** 面试被问"你会写 kernel 吗"，你答的不该是"会写 vector-add"，而是把这七步讲一遍——**这是"会一套方法"和"会一个例子"的差距。**

---

## 2. 贯穿纪律的兑现：加速比 + cosine 汇总表

### 2.1 是什么、为什么单独拉一节

**贯穿纪律（discipline）**：这周你给自己立的规矩——**每写一个 kernel，都必须 (a) 对标 torch.compile，(b) 用三把尺子验误差**。不允许"我感觉它快"就收工。

**为什么这条纪律是本周的灵魂**：新手写 kernel 最容易犯两个错——① 只和 eager 比（那太弱，赢了不算本事）；② 只看快不看对（提速了但结果错，等于负分）。这条纪律逼你**每次都用生产级标准（torch.compile 分母 + 三尺子）验收自己**。今天把散落在 Day1–6 的验收数据**汇成一张总表**，就是这条纪律的"兑现凭证"。

### 2.2 加速比 + 误差汇总总表（填你自己的真实数字）

```
（示意结构，数字用你各天跑出来的真实值替换）
```

| kernel | vs eager | vs torch.compile | max err (bf16) | logits/输出 cosine | top-1 一致率 | 诚实定位 |
|--------|---------|------------------|----------------|--------------------|-------------|---------|
| Day1 vector-add | ~1.0× | ~1.0× | — | 1.000000 | — | 追平，练地基 |
| Day2 fused pointwise | 1.8× | 1.05× | 3e-3 | 0.99999 | — | 逼近 compile |
| Day3 RMSNorm | 2.0×（单算子） | 0.9~1.1× | 5e-3 | 0.99998 | — | 单算子有效 |
| Day4 fused attention | 3×（vs 朴素） | 打不过官方 | 8e-3 | 0.9998 | — | 懂原理，非生产级 |
| Day6 端到端集成 | 1.03×（端到端） | 0.98×（compile+手写） | 5e-3 | 0.99998 | 1.0000 | compile 全图更强 |

> ⚠️ **上表数字全是示意**。今天的动作是：**翻回你 Day1–6 的记录，把真实数字填进去**。如果某天你没记全（比如漏了 cosine），今天补跑一次——这本身就是元笔记的价值：暴露"哪些验收没做扎实"。

### 2.3 从总表能读出的三个规律（这才是汇总的意义）

汇总不是为了好看，是为了**看出单个实验看不出的模式**：

1. **越 memory-bound、越"逐元素"的算子，融合收益越大**（Day2 > Day3）。因为它们的瓶颈就是内存往返，融合直击要害。
2. **单算子加速比 ≠ 端到端加速比**（Day3 单算子 2× 但 Day6 端到端仅 1.03×）。Amdahl 定律的实测印证——这是 W8 最重要的一课。
3. **误差始终可控**（cosine 全程 > 0.999、top-1 = 1.0）。说明 bf16 下 fp32 累加这条纪律守住了数值正确性。**"又快又对"才是完整的胜利。**

> **工业惯例**：真实推理框架（vLLM、TensorRT-LLM）的 kernel 库里，每个 kernel 都配一张类似的"性能 + 精度"双维度验收表。你现在做的，就是工业级 kernel 交付的标准动作。**这张表以后可以直接放进你的项目 README / 简历作品集。**

---

## 3. 诚实的能力地图：我在哪、边界在哪、下一步攻哪

这一节是今天最有价值的部分。**"清楚知道边界，才知道下一步攻哪"**——含糊地觉得"我会写 kernel 了"是危险的，会让你在错的地方浪费暑假。

### 3.1 三档能力自评（诚实打分）

| 能力项 | 我现在的水平 | 证据 | 边界（还不能做什么） |
|--------|-------------|------|---------------------|
| 写 memory-bound 融合 kernel | ✅ 能独立写 | Day2/Day3 kernel | 复杂 tiling、寄存器级优化还不熟 |
| 对标 & 验误差 | ✅ 有纪律、成体系 | 三铁律 + 三尺子 | — |
| 打过 torch.compile | ⚠️ 单点偶尔追平 | Day6 对标表 | 全图/跨算子融合打不过 |
| 写生产级 attention kernel | ❌ 只懂思想 | Day4 玩具版 | 打不过官方 FlashAttention（下详） |
| CUDA 手写（比 Triton 更底层） | ❌ 没碰过 | — | 暑假下沉目标 |

### 3.2 关键边界：我的 attention 和官方 FlashAttention 差在哪

这是你最需要想清楚的一条边界，因为它直接决定你要不要在暑假投入 CUDA。

**FlashAttention（快速注意力，一种 IO 感知的精确 attention 算法）** 的核心思想你 Day4 已经摸到了：**不把巨大的 attention 分数矩阵完整写进显存**，而是分块（tiling）计算，用 **online softmax（在线 softmax，边遍历边更新的 softmax）** 把结果一块块累加出来。

先讲透 online softmax——这是 FlashAttention 的算法心脏，也是你 Day4 练的东西：

**问题背景**：普通 softmax 要先看到**整行所有分数**，才能算出分母（所有 `exp` 的和）。但 FlashAttention 想分块处理，每次只看一小块，怎么在"没看全"的情况下算 softmax？

**核心技巧（用代码讲最清楚）**：

```python
# online_softmax_demo.py
# 环境：纯 numpy，pip install numpy。目的：理解 FlashAttention 怎么"边走边算 softmax"
import numpy as np

def naive_softmax_weighted_sum(scores, values):
    """普通做法：必须一次看到整行 scores。"""
    m = scores.max()                       # 减最大值防 exp 溢出（数值稳定）
    p = np.exp(scores - m)
    return (p @ values) / p.sum()          # 分母是全部 exp 之和

def online_softmax_weighted_sum(scores, values, block=4):
    """
    online 版：分块遍历，每块更新累加，永远不落完整的 p 向量。
    这就是 FlashAttention 不落大矩阵、省显存的核心。
    """
    m = -np.inf          # running max：目前见过的最大分数
    l = 0.0              # running sum：目前的 exp 分母累加
    acc = np.zeros_like(values[0], dtype=np.float64)  # running 加权和
    for i in range(0, len(scores), block):
        s_blk = scores[i:i+block]
        v_blk = values[i:i+block]
        m_new = max(m, s_blk.max())        # 更新全局最大值
        # 关键：旧的累加量要按"新旧最大值之差"重新缩放，才能和新块对齐
        # 为什么：之前的 exp 是以旧 m 为基准算的，m 变了必须校正，否则加错
        correction = np.exp(m - m_new)
        l = l * correction + np.exp(s_blk - m_new).sum()
        acc = acc * correction + np.exp(s_blk - m_new) @ v_blk
        m = m_new
    return acc / l

if __name__ == "__main__":
    np.random.seed(0)
    scores = np.random.randn(16).astype(np.float64)
    values = np.random.randn(16, 8).astype(np.float64)
    a = naive_softmax_weighted_sum(scores, values)
    b = online_softmax_weighted_sum(scores, values)
    print("最大误差:", np.abs(a - b).max())   # 应≈1e-16，两者数学等价
```

**这段代码就是你和 FlashAttention 共享的算法内核。** 你 Day4 写的就是它的 Triton 版。**那你到底差官方什么？** 差在"算法之外的所有硬件级工程"：

| 维度 | 你的 Day4 版 | 官方 FlashAttention |
|------|-------------|---------------------|
| 算法（online softmax + tiling） | ✅ 一样 | ✅ 一样 |
| 分块大小针对 SRAM 精调 | ❌ 拍脑袋 | ✅ 按 GPU 共享内存容量精调 |
| warp 级调度 / 避免 bank conflict | ❌ 没做 | ✅ 手工优化 |
| 反向传播（训练用） | ❌ 没写 | ✅ 有 |
| 支持各种 mask/变长/GQA | ❌ 没有 | ✅ 全支持 |
| 针对不同 GPU 架构特化 | ❌ 一份代码 | ✅ Ampere/Hopper 各有实现 |

> **一句话诚实定位**：**你懂了 FlashAttention 的"为什么快"（算法层），但不会它的"怎么榨干硬件"（工程层）。** 前者靠 Triton 就能摸到，后者需要 CUDA 级的控制力——**这正是你暑假下沉 CUDA 的理由：不是为了重造 FlashAttention，是为了看懂它、以及为巨核课题做真正的硬件级优化。**

### 3.3 这张能力地图直接推导出你的下一步

- 打不过 torch.compile 的全图融合 → 你的增量在**巨核**（超出 compile 融合粒度）。
- 懂 FlashAttention 算法但不会硬件级优化 → **下沉 CUDA** 补上这块。
- 巨核要在真实硬件上验证 → 你手里的 **H100 profiling（AMK 课题）** 就是练兵场。

**三条线在这里交汇：Triton 手艺（W8）+ CUDA 下沉（暑假）+ H100 真实 profiling（AMK）→ 小米巨核课题。** 能力地图不是自我批评，是**给暑假精力做 Amdahl 式的分配**——把时间投在边界最该突破的地方。

---

## 4. 串联表：把 W8 缝回你的整条知识线

### 4.1 为什么要缝

**为什么单独做这件事**：知识如果只是"一周一周孤立地学"，会变成一堆断线的珠子。串联表是**把 W8 这颗新珠子，用线穿回 W6/W7 已有的珠子**——让新旧知识互相印证、互相召回。这是你 W4 起就坚持的格式，今天延续。

**一个类比**：你脑子里的知识网络像一张地铁图。今天学的 W8 是新开的一条线，串联表就是**标出它和哪些老线换乘**——这样以后从任何一站，都能通过换乘找到相关知识，而不是每条线各走各的。

### 4.2 串联表（延续格式）

| W8 内容 | 关联笔记 | 怎么关联（一句话说清换乘逻辑） |
|---------|---------|------------------------------|
| fused pointwise（Day2） | W7 Day4 两笔账 / Day5 Inductor 生成代码 | W7 你**读懂**了 Inductor 融合出的 Triton 代码，W8 你**亲手写**了同类融合——从"看得懂"到"写得出" |
| reduction / autotune（Day3） | W7 Day5 max-autotune | W7 你知道 compile 会 autotune 选最优配置，W8 你**自己也给 kernel 加了 autotune**，理解它在调什么 |
| FlashAttention 思想（Day4） | W6 attention_complexity / W7 Day4 第二笔账 | W6 你算过 attention 的 O(N²) 复杂度**为什么是瓶颈**，W8 你**用 online softmax 亲手化解**了它 |
| 三尺子误差（Day6/7） | W7 Day6 数值一致性 | W7 你建立了"数值一致性"的验收意识，W8 你**把它固化成三把尺子的标准流程** |
| AMK report v1 | W7 AMK report v0 | v0 是"我会读 profile 找瓶颈"，v1 升级到"我会写 kernel 改瓶颈"——report 记录你能力的迭代 |

> **怎么用这张表**：以后复习 attention 时，从"W6 为什么是瓶颈 → W8 怎么亲手解决"一条线看下来，比孤立看任何一天都深刻。**串联表的价值，是让每次复习都是"带着上下文的复习"。**

### 4.3 一条清晰的能力演进主线（把整学期缝成一句话）

```
W6：我知道 attention/KV Cache 为什么是瓶颈（懂问题）
 ↓
W7：我会用 profile + torch.compile 判断"该往哪优化"（会判断）
 ↓
W8：我会亲手写 kernel 优化、并证明它更快更对（会动手 + 会证明）
 ↓
暑假：下沉 CUDA / FlashAttention / vLLM，补上硬件级工程能力
 ↓
课题：在 H100 上做巨核大模型推理优化（真实产出）
```

**这条线就是你简历/保研/面试时讲自己的故事骨架。** 每个 W 都是一次能力量级的跃迁，不是零散的知识点。

---

## 5. 更新 `inference_optimization_landscape.md`：推理优化全景图

### 5.1 landscape 是什么、为什么要持续更新

**landscape（全景图/技术版图）**：一张"推理优化这个领域里，都有哪些手段、我掌握了哪些、还有哪些没碰"的地图。它和能力地图（§3）的区别：能力地图是"我"，landscape 是"整个领域"——你在领域地图上标出自己走到了哪。

**为什么持续更新**：这个领域手段很多（量化、蒸馏、融合、并行、投机…），一次学不完。landscape 让你**始终看得见全局、不迷失在单点**——避免"我 kernel 写得很嗨，却忘了 kernel 只是推理优化的一小块"。

### 5.2 本次更新：新增"手写 kernel"这一块

在你已有的 landscape 上，把 W8 的收获标进去（✅ 已掌握 / 🔶 懂概念 / ⬜ 待学）：

```markdown
## 推理优化全景（W8 更新版）

### A. 算子/kernel 层
- ✅ Triton 手写融合 kernel（elementwise / pointwise / reduction）  ← W8 新增
- ✅ 对标 torch.compile + 三尺子验误差方法论                        ← W8 新增
- 🔶 FlashAttention（懂 online softmax 思想，不会硬件级实现）        ← W8 摸到
- ⬜ CUDA 手写 kernel（比 Triton 更底层）                          ← 暑假目标
- ⬜ 巨核 mega-kernel（整层融合，超出 compile 粒度）               ← 课题方向

### B. 图/编译层
- ✅ torch.compile / Inductor 全图融合 + max-autotune（W7 会用会读）
- ⬜ 手写整图调度

### C. 模型/算法层（推理专属加速）
- ⬜ speculative decoding（投机解码）        ← 待后续，见下方解释
- ⬜ 量化 quantization（INT8/FP8/INT4）

### D. 系统/分布式层
- ⬜ tensor parallelism（张量并行）           ← 待后续，见下方解释
- ⬜ PagedAttention / continuous batching（vLLM 核心）
```

### 5.3 把两个"待后续"的名词先讲明白（免得它们一直是黑盒）

虽然还没学，但先建立直觉，将来上手快。

**speculative decoding（投机解码/推测解码）**

- **是什么**：用一个**小而快的"草稿模型"**先一口气猜出接下来好几个 token，再让**大模型一次性并行验证**这些猜测，猜对的直接用、猜错的从错的地方重来。
- **为什么快（核心）**：大模型 decode 是**逐 token 串行**的（一次只出 1 个，慢），而"验证 K 个已有 token"可以**并行一次做完**。相当于把"串行生成"部分换成"并行验证"。
- **类比**：老师出题（大模型）很慢，但改判断题（验证）飞快。让一个学生（草稿模型）先把后面几道题的答案猜出来，老师一眼扫过去，对的跳过、错的才亲自重做。平均下来比老师一题题亲自做快得多。
- **和你的关系**：它是 decode 阶段（你 Day6 认定的巨核主战场）的另一条正交加速路线，vLLM 里就有。**暑假学 vLLM 时会正式碰到。**

**tensor parallelism（张量并行，TP）**

- **是什么**：一个大模型的单层权重矩阵大到一张 GPU 放不下 / 算不快，就把这个**大矩阵切成几块，分给多张 GPU 同时算**，最后拼起来。
- **为什么需要**：大模型（几百亿参数）的一个权重矩阵可能几十 GB，单卡装不下；就算装得下，单卡算一个巨型 matmul 也慢。切开并行是唯一出路。
- **类比**：一面超大的墙要刷漆，一个人刷太慢/够不着，就把墙**竖着切成 4 段，4 个人同时刷**，刷完这一段的结果拼成整面墙。切法有讲究（怎么切才让"拼接"的通信开销最小），这就是 TP 的技术核心。
- **和你的关系**：它是"多卡"层面的优化，和你现在"单卡单 kernel"是不同层级。**但巨核 + TP 在超大模型推理里会叠加使用**，属于你课题往后走会遇到的系统层知识。

> **为什么现在只标不学**：landscape 的用法是**"先占位、按优先级逐个攻"**。你暑假的优先级是 CUDA→FlashAttention→vLLM（补 kernel 和系统地基），speculative/TP 是再往后的系统课题。**标出来，是为了让你始终知道"全局还有什么"，而不是学到哪算哪。**

---

## 6. 写 `W8_review.md`：复盘 + 暑假下沉路线桥接

### 6.1 复盘的三问（模板，填你自己的真实感受）

复盘不是写流水账。就回答三个能指导行动的问题：

**① 本周最有效的动作是什么？（为了以后多做）**

> 例：坚持"每个 kernel 都以 torch.compile 为分母对标"。这个纪律逼我认清了单点优化的天花板（Amdahl），让我不再幻想"随便写个 kernel 就能赢"，直接锁定了巨核这个真正有增量的方向。

**② 本周最大的阻塞是什么？（为了下周消除）**

> 例：Day4 fused attention 的 online softmax，分块累加时的"重缩放校正"卡了很久——因为我对 GPU 共享内存/寄存器的理解还停在 Triton 抽象层，看不到底层。**这个阻塞直接指向暑假要下沉 CUDA。**

**③ 下周砍什么？（为了聚焦）**

> 例：砍掉"手写更多种类 kernel"的贪心——已经证明单点小算子打不过 compile，再堆数量没意义。把时间集中到 CUDA 基础 + 读 FlashAttention 源码上。

> **为什么"砍什么"最重要**：新手复盘总在加（还要学这、还要学那），资深的复盘在**减**。**精力是最稀缺的资源，Amdahl 定律不光管 kernel，也管你的时间**——砍掉低占比的努力，才有精力投在高占比的方向。

### 6.2 暑假后段下沉路线（CUDA → FlashAttention → vLLM）

这三步的顺序有严格逻辑，不能乱：

```
CUDA 基础            →  FlashAttention 源码      →  vLLM
（补硬件级控制力）        （用 CUDA 视角读懂它）        （看 kernel 怎么进生产系统）
```

**第一步：CUDA（补的是"硬件级控制力"）**

- **为什么先学它**：§3.2 已经证明——你懂 FlashAttention 的算法，但缺"榨干硬件"的能力。这个能力只有 CUDA 级别才有（线程块、共享内存、寄存器、warp 调度都要手动管）。
- **学到什么程度**：能手写一个带共享内存的 tiled matmul（分块矩阵乘），理解 shared memory / bank conflict / occupancy 这几个核心概念。**不用成为 CUDA 大师，够读懂 FlashAttention 即可。**

**第二步：FlashAttention 源码（用 CUDA 视角"验证"你的理解）**

- **为什么第二**：有了 CUDA 视角，你才看得懂官方为什么那样切块、那样调度。**这一步是把 §3.2 那张"我 vs 官方"的差距表，一行行填平的过程。**
- **学到什么程度**：能讲清它的分块策略为什么贴着 SRAM 容量设计、反向传播怎么重算不存中间量。

**第三步：vLLM（看"kernel 怎么装进生产系统"）**

- **为什么最后**：前两步是"单个 kernel"，vLLM 是"kernel 怎么在真实推理服务里协同"——PagedAttention、continuous batching、KV Cache 管理。**这一步把你从"会写 kernel"接到"懂推理系统"，直接对口小米课题的系统视角。**
- **学到什么程度**：能画出 vLLM 一次请求的生命周期，指出你写的 kernel 会插在哪个环节。

> **路线和课题的闭环**：CUDA 给你硬件手感 → FlashAttention 给你顶级 kernel 的范本 → vLLM 给你系统落地视角 → 三者合力，你才能在 H100 上把巨核这件事**从想法做到能对标出 ≥20% 的真实产出**。**这条暑假路线不是随便排的，是你能力地图（§3）的每个 ❌ 精准对应的补课计划。**

---

## 7. 工程收尾：清理 `week8_triton/` + README + GitHub 提交

### 7.1 为什么代码整理和写 kernel 一样重要

**是什么**：把一周攒下的实验脚本，整理成一个**别人（包括三个月后的你自己）能看懂、能跑起来**的项目。

**为什么不能省**：面试官/导师看你的 GitHub，不是看你 commit 了多少行，是看**你的工程素养**——README 清不清楚、结构乱不乱、commit 有没有意义。**一个整理干净、有对标数据的 kernel 仓库，比十个 half-baked 的玩具脚本值钱。** 这是你科研/求职的门面。

### 7.2 建议的目录结构

```
week8_triton/
├── README.md                      # 门面：这周做了什么 + 怎么跑 + 对标结论
├── requirements.txt               # torch>=2.1, triton>=2.1, numpy（钉版本，别用 latest）
├── day1_vector_add.py
├── day2_fused_pointwise.py
├── day3_rmsnorm_autotune.py
├── day4_fused_attention.py
├── 06_nanogpt_integrated.py       # Day6 端到端集成
├── benchmark_utils.py             # 复用的三铁律计时 + 三尺子误差工具
└── results/
    ├── showdown_table.md          # 四方对标表
    └── error_rulers.md            # 三尺子误差
```

> **易错点**：`requirements.txt` 一定**钉具体版本**（`triton==2.1.0` 而非 `triton`）。Triton 版本间 API 变动大，不钉版本，别人（和未来的你）大概率跑不起来。这是开源项目最常见的"在我机器上能跑"翻车。

### 7.3 README 该写什么（工业惯例）

```markdown
# Week 8 · Triton 手写 Kernel + 端到端对标

一周从零手写 Triton kernel，并以 torch.compile 为基准做端到端对标。

## 核心结论（先给最值钱的）
- 单点 memory-bound kernel 可比 eager 快 ~2×，但端到端受 Amdahl 限（RMSNorm 仅 ~2.6%）
- torch.compile 全图融合通常最强；单点手写难超越 → 增量在巨核/大占比算子
- 全程三尺子验误差：cosine > 0.999，top-1 一致率 100%，提速无损正确性

## 环境
pip install -r requirements.txt   # 需 NVIDIA GPU + CUDA 12.x

## 怎么跑
python 06_nanogpt_integrated.py    # 复现四方对标表

## 对标数据
见 results/showdown_table.md
```

> **README 铁律**：**第一屏必须给"核心结论"**，不是安装步骤。看的人 30 秒内要知道"这仓库证明了什么"。把你最诚实、最有洞察的结论放最前面——那是你和"只会跑通代码的人"的区别。

### 7.4 GitHub ≥3 次有意义的 commit

**"有意义"的定义**：一次 commit = 一个完整、可独立理解的改动，commit message 说清"做了什么 + 为什么"，不是 `update`、`fix`、`111` 这种废话。

按知识点切成三次（对应你的能力进阶）：

```bash
# commit 1：地基 + 融合
git add day1_vector_add.py day2_fused_pointwise.py benchmark_utils.py
git commit -m "feat: elementwise + fused pointwise kernel，对标 eager/compile

- vector-add 建立 grid/mask/合并访问基础
- fused pointwise 验证融合省内存往返，vs eager 1.8x"

# commit 2：归约 + autotune
git add day3_rmsnorm_autotune.py day4_fused_attention.py
git commit -m "feat: RMSNorm(reduction+fp32累加) + fused attention(online softmax)

- RMSNorm 加 autotune，fp32 累加保证 bf16 精度
- attention 用 online softmax 分块，不落 O(N^2) 大矩阵"

# commit 3：端到端集成 + 对标结论
git add 06_nanogpt_integrated.py results/
git commit -m "feat: 手写 kernel 集成进 nanoGPT + 四方端到端对标

- 三铁律计时 + 三尺子验误差
- 诚实结论：端到端受 Amdahl 限，compile 全图融合更强"
```

> **为什么这样切 commit**：三次 commit 恰好复现你的**能力叙事**（地基→进阶→系统集成）。以后你 review 自己的 git log，就是一部能力成长史。**commit message 里写"为什么"和对标数字，是工业界 code review 文化的基本功**——让改动可追溯、可理解。

> ⚠️ **提交前检查**：`.gitignore` 里加上 `*.pth`、`__pycache__/`、大模型权重、`results/*.png` 里的大图——**别把几百 MB 的模型权重推上 GitHub**（超限且没意义）。只提交代码和小的 markdown 结果表。

---

## 8. 完成标准：5 分钟讲清这套手艺（对着练）

完成标准是"能 5 分钟讲清：给我一个 memory-bound 算子，我怎么用 Triton 写融合 kernel、怎么对标证明它更快、怎么验证误差可控"。**把下面这段背到能脱口而出——这是你从 W7"会判断"升级到 W8"会动手并证明"的验收。**

### 5 分钟讲稿（照着背，填自己的数字）

> "给我一个 memory-bound 算子，比如 RMSNorm，我的完整打法是这样：
>
> **第一步，先判断值不值得写。** 我会先 profile 确认它确实是访存受限、且占端到端比例够大——因为 Amdahl 定律告诉我，优化一个只占 5% 的算子，端到端最多快 5%，得先算这笔账。
>
> **第二步，写融合 kernel。** RMSNorm 是逐行归约，我一行开一个 Triton program，`load` 进来后，**平方和用 fp32 累加**（bf16 直接累加会掉精度），算出 rms 缩放，再乘 weight，一次 `store` 回去。核心是把本来多个算子多趟内存往返，融合成读写各一次。
>
> **第三步，对标证明更快。** 我遵守 GPU 计时三铁律：warmup 吃掉编译开销、synchronize 等 GPU 真做完、多次取中位数。对标对象是 eager 和 **torch.compile**——因为 compile 是真实生产基线，拿它当分母才诚实。
>
> **第四步，验证误差可控。** 用三把尺子：逐元素 max err 看最坏点、logits cosine 看方向、top-1 一致率看最终决策。cosine 大于 0.999、top-1 100% 一致，就证明提速没有牺牲正确性。
>
> **最后，诚实结论。** 我的单点 kernel 比 eager 快，但打不过 torch.compile 的全图融合——所以我的增量方向不是堆更多单点 kernel，而是巨核：把整层十几个算子融成一个，这超出了 compile 的融合粒度，尤其在 decode 阶段收益最大。"

**如果你能不看稿讲完上面这段，W8 就真正毕业了。** 注意这段讲的不是某个 kernel，是**一套方法 + 一个诚实的自我定位**——这正是资深和萌新的分界。

### 完成标准自检

- [ ] `tech_notes/week8_industrial_view.md`：方法论总表（§1.2）+ 加速比/cosine 汇总表（§2.2）+ 能力地图（§3）+ 串联表（§4.2）
- [ ] `inference_optimization_landscape.md`：新增手写 kernel 能力，speculative/TP 标为待后续
- [ ] `W8_review.md`：最有效动作 / 最大阻塞 / 下周砍什么（§6.1）+ 暑假路线（§6.2）
- [ ] `week8_triton/` 清理 + README（第一屏给核心结论）+ ≥3 次有意义 commit
- [ ] 能脱稿讲完上面的 5 分钟讲稿

---

## 9. 一页速查（明天回顾用）

| 概念 | 一句话记忆 |
|------|-----------|
| 元笔记 | 记的不是"学了什么"，是"知识间的关系 + 我在哪 + 下一步" |
| 写 kernel 七步 SOP | profile→写→融合→fp32累加→对标→三尺子→诚实结论 |
| elementwise | 每个输出只看对应一个输入，天然完美并行 |
| memory-bound | 瓶颈在搬数据不在算，融合是头号解药 |
| reduction | 一组数塌缩成一个数，需跨元素通信，用 fp32 累加 |
| fusion | 多算子合一、内存只搬一趟，memory-bound 提速核心 |
| online softmax | 分块遍历、running max/sum 重缩放，不落大矩阵 |
| 我 vs FlashAttention | 算法层一样，硬件工程层（tiling/warp/架构特化）差一截 |
| speculative decoding | 小模型猜多个 token，大模型并行验证，串行变并行 |
| tensor parallelism | 大矩阵切块分多卡同算再拼接 |
| 复盘核心 | 资深复盘在"砍什么"，精力也遵守 Amdahl |
| 暑假路线 | CUDA(硬件手感)→FlashAttention(顶级范本)→vLLM(系统落地) |
| commit 有意义 | 一次一个完整改动，message 写"做了什么+为什么+数字" |

---

> **收尾寄语**：W8 你完成了一次关键跃迁——从 W7"我知道该往哪优化"到 W8"我能亲手优化、并用生产级标准证明它又快又对"。但今天最珍贵的产出，不是那几个 kernel，而是这份**诚实的能力地图**：你清楚地知道自己打不过 torch.compile 的哪里、和 FlashAttention 差在哪个层级、下一步该沉到 CUDA 补什么。**知道边界在哪，比多写一个 kernel 重要得多**——它让你的暑假每一分精力，都精准投在能突破边界的地方。这，就是你和小米巨核课题之间，最短的那条路。

