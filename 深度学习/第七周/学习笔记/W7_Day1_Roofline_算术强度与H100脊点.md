# W7 Day1 · Roofline:把 "memory-bound" 从"我以为"变成"我算过"

> **本笔记的唯一目标**:让你真正能用一张纸 + 一个脚本,**定量回答**"nanoGPT 在 H100 上 decode 一个 token,瓶颈到底是算力还是带宽"。
>
> 串联:这是 [W7 学习计划](./W7_学习计划_AI_Infra主线.md) Track A Day1 的产出,对应小米课题**主线 1(性能画像与瓶颈分析)**;为 W6 §5.3 那句"decode memory-bound"补上**定量证据**;为 Day2 的 `nsys`、Day4 的算子融合动机铺垫。

---

## 0. 开篇:你要能回答的 5 个问题

读完这份笔记 + 跑完脚本,你应该能不看资料、口头答出来:

1. 什么叫 **arithmetic intensity(算术强度)**?为什么它是判断一个算子 compute-bound 还是 memory-bound 的**唯一**判据?
2. Roofline 模型的"屋顶"和"斜坡"分别代表硬件的什么物理极限?为什么可达性能是 `min(屋顶, 斜坡)`?
3. 什么叫 **ridge point(脊点)**?H100 的脊点为什么大到约 295?
4. nanoGPT decode 一步里,matmul / attention / layernorm / KV-cache 的 `cat`,各自落在 Roofline 的哪个区域?为什么**全都**落在斜坡上?
5. 为什么这张散点图,就是你给小米课题写的 profiling report 的**第一张图**?

如果第 4 题你能脱口而出"它们的算术强度全都约等于 1,而脊点是 295,所以全部 memory-bound",那这一天就值了。

---

## 1. 问题背景:为什么要发明 Roofline?

### 1.1 一个真实的困惑

假设你在 H100 上跑 nanoGPT 生成文本,发现一秒只能吐出几十个 token,GPU 利用率(`nvidia-smi` 里那个百分比)看着挺高,但你换更强的卡、或者把矩阵乘法写得更快,**速度几乎不变**。你会很困惑:算力明明翻倍了,为什么没用?

答案是:**这个负载根本不缺算力,它缺的是"把数据从显存搬进计算单元"的带宽**。你优化错了方向——好比一家餐厅出餐慢,你以为是厨师不够快(算力),拼命再雇厨师,结果真正的瓶颈是**只有一条窄过道,菜端不出来**(带宽)。再多厨师也堵在门口。

Roofline 模型(屋顶线模型)就是用来在**动手优化之前**,先一眼看出"这个负载到底卡在厨师还是过道"的工具。

### 1.2 名词:compute-bound vs memory-bound

这两个词贯穿整个推理优化领域,先把它讲死。

- **compute-bound(受算力限制 / 计算密集)**:计算单元在满负荷干活,数据供应跟得上。瓶颈是"算得不够快"。**例子**:训练时的大矩阵乘法(batch 大、序列长),搬一次权重要做海量乘加,计算单元忙得冒烟。
- **memory-bound(受访存限制 / 访存密集)**:计算单元经常**空等**数据从显存(HBM)运过来。瓶颈是"数据来得不够快"。**例子**:就是我们今天的主角——LLM 的 **decode 阶段**,每生成 1 个 token 要把整个模型的权重从显存读一遍,却只做了"一个向量乘矩阵"这么点计算。算力大量闲置。

> **类比**:compute-bound 像"题目难、计算量大,你在草稿纸上奋笔疾书";memory-bound 像"题目简单,但你每做一步都要跑到隔壁楼翻一次资料",大部分时间花在跑腿上,脑子(算力)是闲的。

**关键认知**:一个算子是 compute-bound 还是 memory-bound,**不是它本身的固有属性**,而是**它的算术强度**和**这块硬件的脊点**比较的结果。同一个算子换张卡,结论可能反转。这就是为什么必须"算",不能"背"。

---

## 2. 核心原理(一):算术强度 = 计算量 ÷ 搬运量

### 2.1 严格定义

> **算术强度(arithmetic intensity,常记作 I 或 AI)= 总计算量(FLOPs) ÷ 总数据搬运量(bytes)**
>
> 单位:**FLOP/byte**(每搬运 1 字节数据,配套做了多少次浮点运算)。

把两个名词拆开讲:

- **FLOPs(floating-point operations,浮点运算次数)**:注意是**小写 s 表示复数"次数"**(总量),不要和 **FLOP/s**(每秒浮点运算,是**速度**)搞混——这是新手最常踩的坑。一次乘法算 1 FLOP,一次加法算 1 FLOP。矩阵乘法 `C[M,N] = A[M,K] @ B[K,N]` 的计算量是 `2*M*K*N`(每个输出元素要 K 次乘 + K 次加,约 2K 次,共 M*N 个输出)。
- **bytes(字节)**:这个算子为了完成计算,**必须经过显存↔计算单元这条慢通道**搬运的数据总量。包括:读输入、读权重、写输出。单位是字节,**和数据精度强相关**(fp32 每个数 4 字节,fp16/bf16 每个数 2 字节,int8 每个数 1 字节)。

### 2.2 为什么是"除法"?这个比值的物理意义

算术强度回答的是一个非常具体的问题:

> **"我每费力气从显存搬来 1 个字节,能榨出多少次计算?"**

- 比值**高** → 搬一次数据能算很久 → 计算单元有活干 → 倾向 compute-bound。
- 比值**低** → 搬一堆数据只算一点点 → 计算单元干完就空等下一批数据 → 倾向 memory-bound。

> **类比(快递与拆箱)**:把"搬运 bytes"想成快递员送箱子上楼(慢、是瓶颈),把"FLOPs"想成你拆箱子干的活。
> - 算术强度高 = 一个箱子里有一台需要组装两小时的家具 → 快递员送一趟,你忙半天 → 快递员(带宽)闲着,你(算力)是瓶颈。
> - 算术强度低 = 一个箱子里只有一颗螺丝 → 快递员拼命跑,你一秒拧完就喊"下一个" → 你(算力)闲着,快递员(带宽)是瓶颈。

### 2.3 决定性结论:矩阵乘法在 decode 阶段的算术强度 ≈ 1

这是今天最重要的一个推导,**自己动手算一遍**,别只看结论。

decode 阶段每次只处理 **1 个新 token**,所以那些大矩阵乘法退化成**矩阵 × 向量**(数学上叫 GEMV,General Matrix-Vector multiply,通用矩阵-向量乘),即 `M=1`:

```
权重矩阵 B 形状 [K, N],输入向量 A 形状 [1, K],输出 [1, N]

FLOPs = 2 * M * K * N = 2 * 1 * K * N = 2*K*N
bytes ≈ 读权重 B = K * N * 2(fp16,每个数 2 字节)   ← 这是绝对大头
        (读输入 A = K*2、写输出 C = N*2 都小到可忽略)

算术强度 = 2*K*N / (K*N*2) = 1 FLOP/byte
```

**结论**:decode 里几乎所有大矩阵乘法(QKV 投影、输出投影、MLP 两层),算术强度都**恒等于约 1**,和矩阵多大无关!因为权重每个数只被这 1 个 token 用了一次("用完即弃"),没有任何复用。

对比 **prefill / 训练**阶段:同时处理 `M` 个 token(比如一个 1024 长的 prompt 或一个 batch),同一份权重被 M 个 token 共享:

```
算术强度 = 2*M*K*N / (K*N*2) = M FLOP/byte
```

M 越大,权重复用越充分,算术强度越高 → 越容易 compute-bound。**这就是 prefill 算力密集、decode 访存密集的数学根源**——同一个矩阵乘法,只因为"一次喂几个 token"不同,bound 类型就反转了。

> **记死这句话**:`decode 的算术强度 ≈ 1,prefill 的算术强度 ≈ batch×序列长度`。decode 之所以慢且省不了,本质是"权重零复用"。

---

## 3. 核心原理(二):Roofline 的两条线与脊点

### 3.1 两条线分别是什么

Roofline 图:**横轴是算术强度(对数轴,FLOP/byte),纵轴是可达性能(对数轴,FLOP/s)**。图上有两条线,围成一个"屋顶+斜坡"的轮廓:

1. **算力屋顶(水平线,roof)**:高度 = 这块卡的**峰值算力 `peak FLOP/s`**。这是物理上限,再怎么样也不可能算得比它快。是一条**水平**线,因为不管算术强度多高,算力就这么多。

2. **带宽斜坡(斜线,ramp)**:高度 = `peak带宽(bytes/s) × 算术强度(FLOP/byte)`。
   - 量纲对一下:`(byte/s) × (FLOP/byte) = FLOP/s`,正好是性能,完美。
   - 为什么是斜的?因为算术强度越高,**同样的带宽能"喂"出越多计算**(每个字节榨出的 FLOP 更多),所以在带宽受限区,性能随算术强度线性上升。在对数-对数坐标下,这条线斜率正好是 1(一条 45° 线)。

### 3.2 可达性能 = min(屋顶, 斜坡)

> 任何算子的**理论可达性能 = min(算力屋顶, 带宽斜坡在该算术强度处的高度)**。

直觉:你的实际速度被**两个上限里更低的那个**卡住(木桶效应)。

- 算术强度很低时,斜坡很矮,远低于屋顶 → 被**带宽**卡住 → **memory-bound**,你处在那条 45° 斜坡上。
- 算术强度很高时,斜坡升到比屋顶还高,但你不可能突破算力上限 → 被**算力**卡住 → **compute-bound**,你处在那条水平屋顶上。

### 3.3 脊点:两线交点 = 屋顶 ÷ 带宽

两条线相交的那个算术强度,就是 **ridge point(脊点 / 拐点)**。求交点:令斜坡 = 屋顶:

```
peak带宽 × ridge = peak算力
=>  ridge point = peak算力(FLOP/s) ÷ peak带宽(byte/s)     单位:FLOP/byte
```

**脊点的物理意义**(一定要会用大白话说):

> **脊点 = "在这块卡上,一个算子每从显存搬 1 字节,至少要配套做多少次浮点运算,才能不被带宽拖后腿"的门槛值。**
> - 算子算术强度 **< 脊点** → 计算量配不上搬运量 → 落在斜坡 → **memory-bound**。
> - 算子算术强度 **> 脊点** → 计算量充足,搬运跟得上 → 落在屋顶 → **compute-bound**。

---

## 4. 动手算 H100 的脊点(科研严谨性从这里开始)

### 4.1 代入官方 spec

```
H100 SXM(FP16/BF16 Tensor Core,非稀疏):
  峰值算力 ≈ 990 TFLOP/s = 990e12 FLOP/s
  HBM3 带宽 ≈ 3.35 TB/s   = 3.35e12 byte/s

ridge point ≈ 990e12 / 3.35e12 ≈ 295 FLOP/byte
```

### 4.2 读懂这个 295

> **在 H100 上,一个算子每从显存搬 1 个字节,必须配套做够约 295 次浮点运算,才能把这张卡的算力喂饱;否则就是带宽在拖后腿。**

现在把第 2.3 节的结论接上:**decode 的矩阵乘法算术强度 ≈ 1**。

```
1  (decode 实际算术强度)   vs   295  (H100 脊点)
```

**1 远远小于 295**(差了约两个数量级)。所以 decode 阶段的算子**几乎必然 memory-bound**,而且不是"勉强 memory-bound",是"被甩出脊点 295 倍远"的重度 memory-bound。

这就是 **W6 §5.3 那句"decode memory-bound"的定量证据**:不再是"我以为"或"书上说",而是"我拿 H100 真实 spec 算过,实际算术强度 1 比脊点 295 小了 295 倍"。

### 4.3 ⚠️ 科研严谨性:数字必须对应你的真实硬件

你现在手里这张卡是 [project-amk-h100] 里记的 **H100-80GB-HBM3(学校集群)**,但写报告前**必须亲手核对**,因为不同型号脊点能差不少:

| 型号 | 算力(FP16 TC) | 带宽 | 脊点 |
|---|---|---|---|
| H100 **SXM** 80GB HBM3 | ≈ 990 TFLOP/s | ≈ 3.35 TB/s | ≈ **295** |
| H100 **PCIe** 80GB HBM2e | ≈ 756 TFLOP/s | ≈ 2.0 TB/s | ≈ **378** |
| H100 **NVL/94GB** HBM3 | 更高 | ≈ 3.9 TB/s | 又不同 |

核对命令(在 hn33 登录节点的 H100 上跑,记得先 `module load CUDA/12.4`):

```bash
# 看型号、是 SXM 还是 PCIe、显存大小
nvidia-smi -q | grep -iE "Product Name|Product Brand|Total" | head

# 只看一行型号
nvidia-smi --query-gpu=name,memory.total --format=csv

# 带宽要查 NVIDIA 官方 datasheet 对应型号(spec 表里的 "Memory Bandwidth"),
# nvidia-smi 不直接给峰值带宽,只给当前显存占用。
```

> **为什么这是科研严谨性而不是吹毛求疵**:你给师兄、给小米课题的报告里如果写"脊点 295",审的人第一句就会问"你那张是 SXM 还是 PCIe?"。报告里**每个数字都要能溯源到一行命令或一份 datasheet**,否则整张 Roofline 图的可信度归零。这正是把"玩具级 profiling"升级到"方法论级"的分水岭。

---

## 5. 动手(主菜):对 nanoGPT decode 逐算子算 FLOPs / bytes 并画 Roofline

### 5.1 先把每类算子的算术强度推清楚(看懂表,再看代码)

以 GPT-2 small 配置(`n_layer=12, n_embd=768, n_head=12`),decode 第 T+1 个 token(已有 KV cache 长度 T),fp16(2 字节/数)为例:

| 算子类别 | 在干什么 | FLOPs(主项) | bytes(主项) | 算术强度 | bound |
|---|---|---|---|---|---|
| **QKV / 输出投影 / MLP 矩阵乘** | 向量 × 权重(GEMV) | `2·K·N` | `K·N·2`(读权重) | **≈ 1** | memory |
| **Attention 打分 `q·Kᵀ`** | 1 个 q 点乘 T 个 k | `2·T·d` | `T·d·2`(读 K cache) | **≈ 1** | memory |
| **Attention 加权 `scores·V`** | T 个权重加权 T 个 v | `2·T·d` | `T·d·2`(读 V cache) | **≈ 1** | memory |
| **LayerNorm** | 逐元素求均值方差再归一 | `≈ 几·n_embd` | `n_embd·2·2`(读+写) | **< 1** | memory |
| **KV-cache 的 `torch.cat`** | 把新 K/V 拼到旧 cache 后面 | **0**(纯搬数据) | 搬整段 cache | **= 0** | 纯访存 |

**一眼看穿的规律**:这张表里**没有任何一个算子的算术强度接近 295**,全都在 0~1 量级。所以 decode 这一步,**整条流水线从头到尾都贴着带宽斜坡**——这就是第 4 题的答案。

> **特别说明 `torch.cat` 这一行(你点名想懂的)**:它的 FLOPs 是 **0**,因为拼接不做任何算术,纯粹是内存搬运。它的算术强度是 **0/bytes = 0**,落在 Roofline 图的**最左边、贴着横轴**——是"纯访存、零计算"的极端 memory-bound。第 7 节会把它的底层 C++ 实现扒开,告诉你"为什么这次搬运还附带浪费空间"。

### 5.2 完整可运行脚本

```python
# file: week7_roofline/src/roofline_nanogpt.py
# 运行环境:Python 3.10+,只需 numpy + matplotlib(本脚本是"纸面估算",不真跑 GPU,
#          所以在你本地 Windows 上就能跑出图;真实 profiling 是 Day2 的 nsys 的活)
#   pip install numpy matplotlib
# 运行:  python roofline_nanogpt.py
# 产出:  roofline_nanogpt.png  +  终端打印每个算子的 FLOPs/bytes/算术强度/bound 判定

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 硬件 spec —— 改这里以匹配你的真实 H100(见笔记 §4.3,务必核对!)
# ============================================================
PEAK_FLOPS = 990e12     # H100 SXM FP16 Tensor Core 峰值算力 (FLOP/s)
PEAK_BW    = 3.35e12    # HBM3 峰值带宽 (byte/s)
RIDGE      = PEAK_FLOPS / PEAK_BW   # 脊点 = 屋顶 / 带宽
print(f"[硬件] H100: 算力 {PEAK_FLOPS/1e12:.0f} TFLOP/s, 带宽 {PEAK_BW/1e12:.2f} TB/s, "
      f"脊点 = {RIDGE:.1f} FLOP/byte\n")

# ============================================================
# 2. nanoGPT (GPT-2 small) 配置 + decode 上下文长度
# ============================================================
n_layer = 12
n_embd  = 768
n_head  = 12
head_dim = n_embd // n_head     # 每个注意力头的维度 = 64
T        = 1024                 # decode 时已有的 KV cache 长度(序列已生成到第 1024 位)
DTYPE_BYTES = 2                 # fp16 / bf16,每个数 2 字节
                                # 为什么用 2:H100 的 990 TFLOP/s 是 fp16 TC 的数字,
                                # 精度必须和算力口径一致,否则脊点和算子不在同一坐标系。

def gemv(K, N):
    """矩阵-向量乘(decode 时 M=1)的 FLOPs 与 bytes。
    返回 (flops, bytes)。bytes 只数读权重这个大头——读输入向量(K)、
    写输出(N)相比 K*N 小两三个数量级,工业估算里通常忽略。"""
    flops = 2 * K * N                       # 每个输出 N 要做 K 次乘+K 次加 ≈ 2K
    bytes_ = K * N * DTYPE_BYTES            # 权重整块从 HBM 读进来,且只用一次(零复用)
    return flops, bytes_

# ============================================================
# 3. 逐类算子估算(单层的量,decode 单 token)
# ============================================================
ops = {}

# --- (a) QKV 投影:把 [1,768] 投到 [1,3*768] ---
ops["QKV_proj"]   = gemv(n_embd, 3 * n_embd)
# --- (b) 注意力输出投影:[1,768] -> [1,768] ---
ops["attn_out"]   = gemv(n_embd, n_embd)
# --- (c) MLP 第一层(升维 4x)+ 第二层(降维) ---
f1 = gemv(n_embd, 4 * n_embd)
f2 = gemv(4 * n_embd, n_embd)
ops["MLP"]        = (f1[0] + f2[0], f1[1] + f2[1])

# --- (d) Attention 打分 q·Kᵀ:1 个 query 去点乘 T 个 key,跨所有 head ---
#     FLOPs = 2 * T * n_embd(= 2*T*head_dim*n_head)
#     bytes = 读整块 K cache = T * n_embd * 2
attn_score_flops = 2 * T * n_embd
attn_score_bytes = T * n_embd * DTYPE_BYTES
# --- (e) Attention 加权 scores·V:结构同上,读整块 V cache ---
attn_av_flops    = 2 * T * n_embd
attn_av_bytes    = T * n_embd * DTYPE_BYTES
ops["Attention"] = (attn_score_flops + attn_av_flops,
                    attn_score_bytes + attn_av_bytes)

# --- (f) LayerNorm:对 768 维做归一,FLOPs 约 5*n_embd(求和/方差/缩放),
#     bytes = 读一遍 + 写一遍 ---
ln_flops = 5 * n_embd
ln_bytes = n_embd * DTYPE_BYTES * 2
ops["LayerNorm"] = (2 * ln_flops, 2 * ln_bytes)   # 每层有 2 个 LayerNorm

# --- (g) KV-cache 的 cat:把新 1 个 token 的 K、V 拼到 [T,768] 后面 ---
#     纯搬运,FLOPs=0。bytes 取决于实现:最坏情况(下面 §7 解释)会重读+重写整段 cache。
#     这里按"读旧 cache + 写新 cache"的悲观估计,体现它有多 memory-bound。
cat_flops = 0
cat_bytes = (T * n_embd * DTYPE_BYTES) * 2 * 2    # K 和 V 各一份,各读+写
ops["KVcache_cat"] = (cat_flops, cat_bytes)

# ============================================================
# 4. 打印分类表 + 计算每个算子的算术强度与 bound 判定
# ============================================================
print(f"{'算子':<14}{'FLOPs':>12}{'bytes':>12}{'算术强度':>12}{'  判定':<10}")
print("-" * 62)
points = []   # (算术强度, 可达性能, 名字)
for name, (flops, bytes_) in ops.items():
    ai = flops / bytes_ if bytes_ > 0 else 0.0
    # 可达性能 = min(屋顶, 斜坡)。落在哪条线上,就说明被谁卡住。
    attainable = min(PEAK_FLOPS, PEAK_BW * ai)
    bound = "compute" if ai > RIDGE else "memory"
    print(f"{name:<14}{flops:>12.2e}{bytes_:>12.2e}{ai:>12.3f}   {bound}")
    if ai > 0:                          # cat 的 ai=0,对数轴画不了,单独标注
        points.append((ai, attainable, name))
print("-" * 62)
print(f"脊点 = {RIDGE:.1f} FLOP/byte → 算术强度小于它的全是 memory-bound\n")

# ============================================================
# 5. 画 Roofline 图
# ============================================================
ai_axis = np.logspace(-1, 4, 500)                  # 横轴:算术强度 0.1 ~ 10000
roof    = np.full_like(ai_axis, PEAK_FLOPS)        # 算力屋顶(水平线)
ramp    = PEAK_BW * ai_axis                        # 带宽斜坡(斜线)
attain  = np.minimum(roof, ramp)                   # 可达性能 = min(两者)

plt.figure(figsize=(9, 6))
plt.loglog(ai_axis, attain, 'k-', lw=2.5, label="Roofline = min(屋顶, 斜坡)")
plt.loglog(ai_axis, roof, 'r--', lw=1, alpha=0.6, label=f"算力屋顶 {PEAK_FLOPS/1e12:.0f} TFLOP/s")
plt.loglog(ai_axis, ramp, 'b--', lw=1, alpha=0.6, label=f"带宽斜坡 {PEAK_BW/1e12:.2f} TB/s")
plt.axvline(RIDGE, color='gray', ls=':', label=f"脊点 ≈ {RIDGE:.0f} FLOP/byte")

# 把每类算子标成散点
for ai, perf, name in points:
    plt.scatter(ai, perf, s=90, zorder=5)
    plt.annotate(name, (ai, perf), textcoords="offset points",
                 xytext=(6, 6), fontsize=9)

plt.xlabel("算术强度 Arithmetic Intensity (FLOP/byte)")
plt.ylabel("可达性能 Attainable Performance (FLOP/s)")
plt.title("nanoGPT decode @ H100 —— 所有算子都贴在带宽斜坡上")
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, which="both", ls=":", alpha=0.4)
plt.tight_layout()
plt.savefig("roofline_nanogpt.png", dpi=130)
print("[OK] 已保存 roofline_nanogpt.png")
```

### 5.3 预期输出(终端)

你会看到所有算子的算术强度都挤在 1 附近,判定栏整列写着 `memory`:

```
[硬件] H100: 算力 990 TFLOP/s, 带宽 3.35 TB/s, 脊点 = 295.5 FLOP/byte

算子                 FLOPs       bytes        算术强度     判定
--------------------------------------------------------------
QKV_proj         3.54e+06    3.54e+06       1.000   memory
attn_out         1.18e+06    1.18e+06       1.000   memory
MLP              9.44e+06    9.44e+06       1.000   memory
Attention        3.15e+06    3.15e+06       1.000   memory
LayerNorm        7.68e+03    6.14e+03       1.250   memory
KVcache_cat      0.00e+00    6.29e+06       0.000   memory
--------------------------------------------------------------
脊点 = 295.5 FLOP/byte → 算术强度小于它的全是 memory-bound
```

**怎么读这张图(报告里要写的话)**:横轴拉到几千的位置才是脊点 295,而我们所有算子的点全挤在最左边 `AI≈1` 那条竖线附近,死死贴在那条蓝色 45° 斜坡上,**离红色算力屋顶还差约 295 倍**。这一张图就直接证明:**优化 decode 不该去堆算力,而该去削减"搬运量"**——这正好引出后面几天的所有手段(KV cache 量化、算子融合减少 HBM 往返、megakernel)。

---

## 6. 工业锚点:这张图为什么是 profiling report 的第一张

### 6.1 对应小米课题主线 1

小米课题主线 1 是"量化计算/访存/通信/调度开销占比"。资深 Infra 工程师拿到一个**没见过的新负载**,第一个动作不是上 `nsys`、不是读代码,而是**先估一个 Roofline 位置**——因为它用 5 分钟的纸面估算,就划定了"往哪个方向优化才有意义"的大方向:

- 点落在斜坡(memory-bound)→ 一切努力都该围绕**减少 HBM 访问**:量化(把权重从 fp16 压到 int8/int4,字节数直接砍半甚至砍到 1/4)、算子融合(中间结果留在片上 SRAM,不写回 HBM)、KV cache 压缩。
- 点落在屋顶(compute-bound)→ 才轮到**堆算力 / 用更高效的 Tensor Core kernel / 上更强的卡**。

如果方向搞反(对 memory-bound 负载拼命优化算力),就是开头那个"再雇厨师却堵在过道"的故事——白干。

### 6.2 直接喂给 AMK 任务

你 Day2 起会把**完全相同的流程**搬到 [project-amk-h100] 的 AMK 上。AMK 的卖点正是 **megakernel(巨核算子)**——把整个 decode 的几十个小 kernel 融成一个大 kernel,目的就是**砍掉算子之间反复读写 HBM 的搬运量**。换句话说:**megakernel 是"对一个重度 memory-bound 负载"的对症下药**。你今天这张 Roofline 图,就是"为什么需要 megakernel"这个故事的**第一页论据**——没有它,你说"AMK 减少了访存"就是空口无凭。

---

## 7. 深挖内核:`torch.cat` 拼 KV cache 时,为什么"还会浪费空间"?

你点名想懂这个。这一节把"类比 → 底层 C++ → 为什么不能按需精确申请"一次讲透。它同时解释了 §5.1 表里 `cat` 那行为什么 bytes 那么大、为什么是重度 memory-bound。

### 7.1 现象:`torch.cat` 干了两件费钱的事

```python
# KV cache 的朴素写法(很多教程里就是这么写的,decode 每步都调一次)
k_cache = torch.cat([k_cache, k_new], dim=2)   # 把新 token 的 k 拼到尾部
v_cache = torch.cat([v_cache, v_new], dim=2)
```

`torch.cat` 在底层做的事:

1. **申请一整块全新的连续显存**,大小 = 旧 cache + 新 token。
2. 把**旧 cache 整个拷贝**进去,再把新 token 拷在后面。
3. 旧的那块显存被释放(其实是还给缓存分配器,见下)。

所以它每一步都 **重读 + 重写整段 cache**(这就是 §5.1 表里 cat 的 bytes ∝ `T·n_embd` 的来历,FLOPs 却是 0 → 极致 memory-bound),而且 T 越大越贵。这也是为什么**生产级推理引擎从不用 `cat` 续 KV cache**,而是**预分配一块最大长度的 buffer,decode 时只往固定槽位写**(vLLM 的 PagedAttention 更进一步做了分页)。

### 7.2 为什么"申请的显存会比需要的多"?——CUDA 缓存分配器

这是你最想懂的点:**为什么不按 `cat` 后的精确大小申请,偏要多给一截?**

答案是:**你看到的"申请"根本不是向操作系统/驱动申请,而是向 PyTorch 自己的 `CUDACachingAllocator(CUDA 缓存分配器)`要**。原因要从"为什么不能每次都精确申请"说起。

**根本原因:`cudaMalloc`(向驱动真申请显存)极慢,而且会强制设备同步。**

如果每次 `cat` 都老老实实 `cudaMalloc(精确字节数)` + 用完 `cudaFree`,那 decode 每生成一个 token 都要陷入驱动、阻塞整个 GPU 流水线——慢到不可接受。所以 PyTorch 自己维护了一个显存"二房东":

- 第一次找它要显存,它向驱动一次性批发一大块(几 MB 起),切一小块给你。
- 你"释放"时,这块**不还给驱动,而是留在它的空闲链表里**,下次秒级复用。

而这个二房东为了让"释放的块"能被后续请求高效复用,**会把每次请求的大小向上取整到某个粒度(rounding,对齐)**。这就是"浪费"的来源。看真实源码逻辑(简化自 PyTorch `c10/cuda/CUDACachingAllocator.cpp`):

```cpp
// 简化自 PyTorch c10/cuda/CUDACachingAllocator.cpp 的 round_size()
// 作用:把你请求的字节数 size 向上取整 —— 这就是"多申请"的根源
static size_t round_size(size_t size) {
  if (size < kMinBlockSize) {                  // kMinBlockSize = 512 字节
    return kMinBlockSize;                      // 再小的请求也至少给 512B
  }
  // 对齐到 512 的倍数:不足 512 的零头,统统向上凑满一个 512
  return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

// 申请逻辑(高度简化)
Block* malloc(size_t orig_size) {
  size_t size = round_size(orig_size);         // ← 先取整,所以实际占用 ≥ 你要的
  Block* block = find_free_block(size);         // 先在空闲链表里找够大的块复用
  if (!block) {
    // 链表里没有合适的 → 才向驱动批发(注意是按更大的 granularity 拿)
    void* ptr;
    size_t alloc_size = get_allocation_size(size);  // 通常远大于 size(2MB/20MB 档)
    cudaMalloc(&ptr, alloc_size);                   // 真·系统调用,慢,所以尽量少做
    block = new Block(ptr, alloc_size);
  }
  // 如果拿到的块比需要的大很多,会 split 成"用的部分"+"剩余放回链表"
  return block;
}
```

把这段读懂,你就能精确回答你自己提的那串问题:

- **"为什么请求会有多余浪费的空间?"** → 因为分配器把 size **向上取整到 512 字节(或更大档位)的倍数**。你要 700 字节,它给你 1024 字节,中间 324 字节就是"为了对齐和未来复用而牺牲的浪费"。
- **"为什么不能按需要的大小精确请求?"** → 能精确算出需要多少,但**精确请求会导致显存碎片化**:每个块大小都奇形怪状,释放后留下的空洞拼不出下一个请求需要的连续块,最终"明明总空闲够、却没有一块连续的够用",触发昂贵的 `cudaMalloc` 甚至 OOM。**统一对齐到固定粒度**,是用"少量空间浪费"换"块可互相复用、几乎不碰驱动"的工程取舍。
- **"为什么 `torch.cat` 会请求一整块连续显存?"** → 因为张量在底层是**一块连续内存 + 形状/步长(stride)元数据**,绝大多数 CUDA kernel 假设输入是连续的才能高效访问;`cat` 的结果必须是一个能被后续 kernel 正常读的张量,所以它必须落在**一整块连续显存**里,不能东一块西一块拼。

> **一句话工业结论**:`CUDACachingAllocator` 用"向上取整 + 不还给驱动 + 块复用"三件套,把"显存分配"从每次几十微秒的系统调用,变成几乎零成本的链表操作;代价是**一点可预测的空间浪费**和**碎片**。这就是你观察到"`cat` 浪费空间"的根本原因——它不是 bug,是**速度换空间的刻意设计**。理解这一点,你才会明白生产引擎为何要彻底绕开 `cat`、改用预分配 + 定槽写入。

---

## 8. 常见陷阱(报告里别犯)

1. **FLOP/s 和 FLOPs 混用**:一个是速度(性能、纵轴),一个是总量(算术强度的分子)。写错单位,整张图的量纲就崩了。
2. **精度口径不一致**:H100 的 990 TFLOP/s 是 **fp16 Tensor Core** 的数;如果你的算子按 fp32 数 bytes、却拿 fp16 算力当屋顶,脊点和点不在同一坐标系。**算 bytes 用的精度,必须和算力屋顶的精度口径一致。**
3. **拿稀疏算力当屋顶**:NVIDIA 宣传页常给"带稀疏"的翻倍数字(如 1979 TFLOP/s)。真实稠密推理用不到,**报告里要用稠密(non-sparse)数字**,并注明。
4. **把"纸面 Roofline"当成"实测性能"**:本笔记的脚本是**理论上限估算**,真实达到的性能往往低于斜坡(还有 kernel launch 开销、未打满带宽等)。**实测是 Day2 `nsys` 的事**;Roofline 给的是"天花板和努力方向",不是"你现在的成绩"。
5. **型号没核对**:见 §4.3。SXM/PCIe/94GB 脊点不同,直接照抄 295 可能被审。

---

## 9. 自测题(合上笔记,口头答)

1. 用一句话定义算术强度,并说出它的单位。(答:§2.1)
2. 为什么 decode 的矩阵乘法算术强度恒约等于 1,而 prefill 约等于 batch×序列长?(答:§2.3,关键词"权重零复用 vs 复用")
3. 脊点的公式是什么?H100 SXM 的脊点为什么约 295?它的大白话含义?(答:§3.3 + §4)
4. nanoGPT decode 的 5 类算子各落在 Roofline 哪里?为什么全是 memory-bound?(答:§5.1 表)
5. `torch.cat` 续 KV cache 为什么慢、为什么浪费空间、为什么生产引擎不用它?(答:§7)
6. 你的负载是 memory-bound,资深工程师会优先用哪三类优化?(答:§6.1,量化/融合/cache 压缩)

---

## 10. 与已有笔记的串联

| 关联笔记 | 关系 |
|---|---|
| W6 §5.3「decode memory-bound」 | 本笔记给它补上**定量证据**(算术强度 1 vs 脊点 295) |
| W6 KV Cache 实现(Day6) | §7 解释了朴素 `cat` 写法的底层代价,引出预分配 buffer 的必要性 |
| W7 Day2(`nsys`) | 本笔记是**理论上限**,Day2 用 nsys 测**真实达到值**,两者对比才完整 |
| W7 Day4(算子融合 / megakernel 动机) | §6.2 说明"重度 memory-bound"正是 megakernel 的对症场景 |
| [project-amk-h100] AMK profiling | 同一套 Roofline 流程将原样迁移到 AMK,本图是"为什么要 megakernel"的第一页论据 |
| `tech_notes/optimizer_memory.md`(W4D6) | 同属"显存/访存代价"主题,延续 AI Infra 工业视角 |

---

## 11. 完成标准 checklist

- [ ] 跑通 `roofline_nanogpt.py`,得到 `roofline_nanogpt.png` 和算子分类表
- [ ] **用 `nvidia-smi -q` 核对你那张 H100 是 SXM/PCIe、显存大小**,把脚本里的 `PEAK_FLOPS/PEAK_BW` 改成对应 datasheet 数字,重算脊点
- [ ] 能脱口而出第 0 节的 5 个问题
- [ ] 能口头讲清 §7 的 `torch.cat` 底层(对齐取整 + 缓存分配器 + 为什么不精确申请)
- [ ] 把这张图 + 一段"所有算子贴在斜坡上 → decode 重度 memory-bound → 该减访存而非堆算力"的解读,存进 `tech_notes/roofline_nanogpt.md`,作为小米课题 profiling report 的第一张图
```
