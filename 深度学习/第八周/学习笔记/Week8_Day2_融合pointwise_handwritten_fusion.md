# Week 8 · Day 2 —— fused pointwise:亲手把"第二笔账"省出来

> **昨天(Day 1)你学会了"一个 program 处理一整块数据"。今天要用这个能力做第一件有商业价值的事:把好几个逐元素算子塞进一个 kernel,亲手省掉"中间张量往返 HBM"这笔浪费。**
> W7 Day4 你**量出**了这笔浪费;W7 Day5 你**读懂**了 Inductor 自动融合的代码;今天你**自己写**这个融合 kernel,并用自己跑的数字说清它到底省了多少、为什么大张量才省得明显。

---

## 0. 三个学习目标问题的"电梯答案"

先把结论钉在最前面,读完全文回来应能自己复述。

1. **`relu(x*a+b)` 不融合是 3 个 kernel、融合是 1 个,省的两笔账在代码里对应哪几行?**
   - **启动账(launch)**:看你的**封装函数里 kernel 被调用了几次**——融合版只有 **1 行** `fused_kernel[grid](...)`,不融合版是 3 次 torch 算子 = 3 次 launch。省的是 2 次 kernel 启动的固定开销。
   - **访存账(memory)**:看 `tl.load` / `tl.store` 各几次。融合版是 **3 次 load(x/a/b)+ 1 次 store(out)= 4 次碰 HBM**;中间量 `x*a+b`、`maximum(·,0)` 全待在**寄存器**里(就是 `y = x*a+b` 和 `y = tl.maximum(y, 0.0)` 这两行,它们之间没有任何 load/store)。不融合版要多出 `t1`、`t2` 的**写回 + 读回**,共 8 次碰 HBM。省的就是这 4 次多余的 HBM 往返。

2. **为什么 pointwise 最好融?什么算子不能简单融进来?**
   pointwise(逐元素)的每个输出只依赖**同一个位置**的输入(`out[i] = f(x[i],a[i],b[i])`),各元素之间零依赖,所以一个 program 把自己那块的数在寄存器里一路算到底就行,天生可拼接。**不能简单融的是"需要跨元素看数据"的算子**:reduction(求和/求最大等,一个输出要看一整行)、matmul/conv(每个输出是一堆元素的点积)、以及内部含 reduction 的 softmax/layernorm。它们需要跨线程通信,构成**融合边界**。

3. **张量多大融合才明显?为什么小张量融了也白融?**
   要大到进入 **memory-bound(访存受限)** 区间(经验值 ~10⁷ 元素以上)才见干净的收益,本例上限是理论 **2×**。小张量(~10⁴)是 **launch-bound / latency-bound(启动/延迟受限)**:整个操作只有几微秒、GPU 根本没吃饱,访存这根"主杠杆"压根没使上力,省下的那点访存时间淹没在启动延迟里——省了也感觉不到,而且这点绝对时间对真实模型无关紧要。

下面把每个答案背后的机制讲透,并给你能亲手复现的代码与曲线。

---

## 1. 问题背景:那笔"往返 HBM"的浪费到底长什么样

### 1.1 先把内存层级摆清楚(这是今天所有直觉的地基)

GPU 上的数据不是"一个地方存着",而是一座**金字塔**,越往上越快越小、越往下越慢越大:

> - **register(寄存器)**:金字塔尖。每个线程私有,速度≈计算单元本身(访问延迟约 1 个时钟周期),但极小(每线程几十~上百个)。计算真正发生的地方——ALU(算术逻辑单元)直接从寄存器取数。
> - **shared memory / L1(共享内存)**:每个 SM 一块,H100 上约 228 KB,快(几十周期),同一 program 内的线程可共享。
> - **L2 cache**:全 GPU 共享,H100 约 50 MB。
> - **HBM(High Bandwidth Memory,高带宽内存)**:金字塔底座,就是你说的"显存"。H100 是 HBM3,容量 80 GB、带宽约 **3.35 TB/s**——听着快,但**每次访问要等几百个时钟周期**,而且带宽是全体 SM 抢着用的稀缺资源。

**核心矛盾**:计算单元(在塔尖)算得飞快,但数据大部分躺在塔底的 HBM 里。**把数据从 HBM 搬上来、再把结果搬回去,才是绝大多数深度学习算子真正的瓶颈**——这类"算得快、卡在搬运"的算子,就叫 **memory-bound(访存受限)**。

> **类比**:寄存器是你手里的笔,HBM 是楼下仓库。做算术题(计算)一秒钟就写完,但每要一个数字都得跑下楼去仓库取、算完再跑下楼放回去。真正耗时间的是**上下楼(访存)**,不是**动笔(计算)**。

### 1.2 不融合:中间结果被反复"搬下楼又搬上楼"

在 PyTorch 里朴素地写 `relu(x*a+b)`,eager 模式会**拆成 3 个独立算子、3 次 kernel 启动**:

```python
t1 = x * a        # kernel 1
t2 = t1 + b       # kernel 2
out = torch.relu(t2)   # kernel 3
```

站在内存的角度看,每个 kernel 都得"从仓库取料 → 算 → 把半成品放回仓库":

```
kernel1  (x*a):    读 x, 读 a           → 算 → 写 t1        访存 = 3N
kernel2  (t1+b):   读 t1, 读 b          → 算 → 写 t2        访存 = 3N
kernel3  (relu):   读 t2               → 算 → 写 out       访存 = 2N
                                                  ───────────────
                                            总访存 = 8N 个元素
```

(N = 元素个数;每个元素 fp32 = 4 字节。)

**看出浪费了吗?** `t1` 和 `t2` 是**中间张量(intermediate tensor)**——它们唯一的作用就是把数据从上一个 kernel 传给下一个 kernel。但因为分成了 3 个 kernel,每个 kernel 结束时 GPU 只能把 `t1`、`t2` **写回 HBM**(塔底),下一个 kernel 再**从 HBM 读回来**。这一"写下去 + 读上来"就是 W7 Day4 你量出的那笔浪费:

- `t1`:kernel1 写 HBM(N)+ kernel2 读 HBM(N)= 2N 纯浪费
- `t2`:kernel2 写 HBM(N)+ kernel3 读 HBM(N)= 2N 纯浪费
- 合计浪费 **4N** 次 HBM 访问,只为传递本可以直接留在手边的中间值。

### 1.3 融合:中间结果一直待在"笔尖"(寄存器),从不下楼

把三步塞进**一个** kernel,中间值 `x*a`、`x*a+b` 全程留在寄存器里,只在最开始读一次输入、最后写一次输出:

```
fused (relu(x*a+b)):  读 x, 读 a, 读 b   → 寄存器里连算 x*a+b、再 relu → 写 out
                                                  ───────────────
                                            总访存 = 4N 个元素
```

**两笔账一起省下来了**:

| | 不融合 | 融合 | 省了 |
|---|---|---|---|
| **启动账**:kernel 启动次数 | 3 | 1 | 少 2 次固定启动开销 |
| **访存账**:HBM 访问量 | 8N | 4N | 少 4N(即腰斩,理论加速上限 8N/4N = **2×**)|

> **这就是今天的全部价值**:融合不是"算得更快"(加法乘法本来就快),而是**让数据少下几趟楼**。对 memory-bound 算子,少搬一半数据 ≈ 快一倍。

---

## 2. 底层追问:两笔账在机器指令层面是什么样

"中间值留在寄存器"听起来抽象,我们下到接近汇编的层面看一眼,你会对"省了什么"有肌肉记忆。下面是**概念化**的 PTX/SASS(GPU 汇编),重点看 HBM 访问指令(`ld.global` / `st.global`)出现了几次。

```
========== 融合后(1 个 kernel,中间量在寄存器)==========
ld.global.f32   %fx, [x + off];       # 读 x   ← 碰 HBM
ld.global.f32   %fa, [a + off];       # 读 a   ← 碰 HBM
ld.global.f32   %fb, [b + off];       # 读 b   ← 碰 HBM
fma.rn.f32      %fy, %fx, %fa, %fb;    # x*a+b:一条【乘加融合】指令,结果进寄存器 %fy
max.f32         %fy, %fy, 0f00000000;  # relu:max(y,0),还在寄存器 %fy
st.global.f32   [out + off], %fy;      # 写 out ← 碰 HBM
# 全程:3 次 ld + 1 次 st 碰 HBM。中间量 %fy 从没离开寄存器。

========== 不融合(3 个 kernel,中间量往返 HBM)==========
# --- kernel1 ---
ld.global x; ld.global a; mul %ft1,%fx,%fa; st.global [t1];   # t1 写回 HBM  ← 浪费
# --- kernel2 ---
ld.global [t1]; ld.global b; add %ft2,%ft1,%fb; st.global [t2];  # t1 读回 + t2 写回 ← 浪费
# --- kernel3 ---
ld.global [t2]; max %ft3,%ft2,0f00000000; st.global [out];    # t2 读回 ← 浪费
# 全程:碰 HBM 8 次。
```

两个值得记住的细节:

1. **`fma`(fused multiply-add,乘加融合指令)**:`x*a+b` 编译成**一条**硬件指令,而不是"先乘、再加"两条。这是**指令级别的融合**——GPU 硬件本身就把乘加合一了。所以"融合"这件事其实有两层:算子级(把 3 个 kernel 拼一起)和指令级(乘加合一)。今天讲的是算子级,但 `fma` 顺带告诉你:硬件天然鼓励"算完立刻用,别落地"。

2. **数值会有极小差异,所以用 `allclose` 而不是 `==`**:`fma` 计算 `x*a+b` 只做**一次浮点舍入**;而不融合的 `mul` 再 `add` 做了**两次舍入**。两者可能差 ~1 个 ULP(最低有效位)。这不是 bug,是浮点运算的正常现象(融合版往往还更准)。所以验证正确性必须用 `torch.allclose(out_fused, out_ref, rtol, atol)` 允许微小误差,而不是要求逐位相等。

---

## 3. 动手:`week8_triton/02_fused_pointwise.py`(完整可运行)

下面是完整脚本:融合 kernel + 三方对标 + "加速比 vs 张量大小"扫描并出图 + 误差验证。存成 `week8_triton/02_fused_pointwise.py`,在 H100(或任意 CUDA GPU)上 `python 02_fused_pointwise.py` 即可。

```python
"""
Week 8 Day 2 — 手写 fused pointwise: relu(x*a+b)
可运行环境 / 依赖:
    - NVIDIA GPU + CUDA 12.x
    - Python 3.10+,PyTorch 2.x(自带 triton),matplotlib
    - 若在无显示器的集群跑图,已用 Agg 后端,直接存 PNG
运行:
    python 02_fused_pointwise.py
产物:
    - 控制台:三方 benchmark 表 + 正确性检查
    - 文件:fusion_speedup_vs_size.png(加速比 vs 张量大小 曲线)
"""

import matplotlib
matplotlib.use("Agg")          # 集群无显示器:用非交互后端,只存文件不弹窗
import matplotlib.pyplot as plt

import torch
import triton
import triton.language as tl


# ======================================================================
# 1) 融合 kernel:一次读、寄存器里连算、一次写
# ======================================================================
@triton.jit
def fused_mul_add_relu_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr,   # 4 个张量的显存首地址
    n_elements,                     # 元素总数(运行时值)
    BLOCK_SIZE: tl.constexpr,       # 每个 program 处理多少元素(编译期常量)
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # —— 访存账:这里只有 3 次 load 碰 HBM ——
    x = tl.load(x_ptr + offs, mask=mask)
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)

    # —— 关键:中间结果全在寄存器,绝不写回 HBM ——
    # 这两行之间没有任何 tl.load/tl.store,x*a+b 的结果一直在寄存器里
    y = x * a + b                 # 编译后大概率是一条 fma(乘加融合)
    y = tl.maximum(y, 0.0)        # relu,依然在寄存器

    # —— 访存账:这里只有 1 次 store 碰 HBM ——
    tl.store(out_ptr + offs, y, mask=mask)


def fused_triton(x, a, b):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    # —— 启动账:整个操作只有这 1 次 launch ——
    fused_mul_add_relu_kernel[grid](x, a, b, out, n, BLOCK_SIZE=1024)
    return out


# ======================================================================
# 2) 三个对标对象
# ======================================================================
def naive_torch(x, a, b):
    # 不融合:3 个独立算子 = 3 次 launch + 中间量 t1/t2 往返 HBM
    t1 = x * a
    t2 = t1 + b
    return torch.relu(t2)


# torch.compile:让 Inductor 自动融合(底层会生成一个 Triton kernel)
# mode="max-autotune":多花编译时间,换更激进的调优(会自动选 BLOCK/num_warps)
compiled_fused = torch.compile(naive_torch, mode="max-autotune")


# ======================================================================
# 3) 计时:torch.cuda.Event(接 W7 Day2 / Day1 的计时法)
# ======================================================================
def bench(fn, *args, warmup=30, iters=100):
    """返回单次调用平均耗时(毫秒)。GPU 时间,不含 Python 端。"""
    for _ in range(warmup):        # warmup:触发 JIT 编译 + GPU 升频 + 缓存预热
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()       # 必须等 GPU 真跑完再读时间(launch 是异步的)
    return start.elapsed_time(end) / iters


def eff_bandwidth_gbps(ms, n, dtype_bytes=4):
    """融合版有效带宽:读 x/a/b + 写 out = 4N 次访问。"""
    return (4 * n * dtype_bytes) / (ms * 1e-3) / 1e9


# ======================================================================
# 4) 正确性 + 单点三方对标(取一个足够大的 memory-bound 尺寸)
# ======================================================================
def correctness_and_headline():
    torch.manual_seed(0)
    n = 1 << 24                    # 16M 元素,已在 memory-bound 区间
    x = torch.randn(n, device="cuda")
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")

    ref = naive_torch(x, a, b)
    out = fused_triton(x, a, b)
    # 用 allclose 而非 ==:fma 单次舍入 vs 分步乘加两次舍入,允许 ~1 ULP 差异
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5), "融合结果不对!"
    max_err = (out - ref).abs().max().item()
    print(f"[正确性] allclose 通过,最大逐元素误差 = {max_err:.2e}\n")

    t_naive = bench(naive_torch, x, a, b)
    t_compile = bench(compiled_fused, x, a, b)
    t_triton = bench(fused_triton, x, a, b)

    print(f"张量规模 n = {n:,}(fp32,memory-bound 区间)")
    print(f"{'方案':<22}{'耗时(ms)':>12}{'带宽(GB/s)':>14}{'相对朴素加速':>14}")
    print("-" * 62)
    for name, t in [("① 朴素 3 算子", t_naive),
                    ("② torch.compile", t_compile),
                    ("③ 手写 Triton 融合", t_triton)]:
        print(f"{name:<22}{t:>12.4f}{eff_bandwidth_gbps(t, n):>14.1f}"
              f"{t_naive / t:>14.2f}x")
    print("\n[预期] ②③ 都把访存从 8N 砍到 4N,加速比都逼近 2×,且彼此接近")
    print("       —— 因为都撞到同一堵 HBM 带宽墙,手写不会显著超过编译器。\n")


# ======================================================================
# 5) 关键实验:加速比 vs 张量大小(小 → 大),画曲线
# ======================================================================
def sweep_and_plot():
    torch.manual_seed(0)
    # 从 16K 扫到 64M;显存够的话可加 1<<27(128M)
    exps = [14, 16, 18, 20, 22, 23, 24, 25, 26]
    sizes, sp_triton, sp_compile = [], [], []

    for e in exps:
        n = 1 << e
        x = torch.randn(n, device="cuda")
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")

        t_naive = bench(naive_torch, x, a, b)
        t_triton = bench(fused_triton, x, a, b)
        t_compile = bench(compiled_fused, x, a, b)

        sizes.append(n)
        sp_triton.append(t_naive / t_triton)
        sp_compile.append(t_naive / t_compile)
        print(f"n=2^{e:<2}={n:>12,}  手写={t_naive/t_triton:5.2f}x  "
              f"compile={t_naive/t_compile:5.2f}x")

        del x, a, b
        torch.cuda.empty_cache()   # 大张量扫描时及时回收,避免碎片/OOM

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, sp_triton, "o-", label="手写 Triton 融合")
    plt.plot(sizes, sp_compile, "s--", label="torch.compile")
    plt.axhline(2.0, color="gray", ls=":", label="理论上限 2× (8N→4N)")
    plt.axhline(1.0, color="red", ls=":", label="1× (无收益)")
    plt.xscale("log")
    plt.xlabel("张量元素数 N (log)")
    plt.ylabel("相对朴素 3 算子 的加速比")
    plt.title("融合加速比 vs 张量大小:小张量无感,大张量逼近 2×")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fusion_speedup_vs_size.png", dpi=150)
    print("\n[已保存] fusion_speedup_vs_size.png")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    correctness_and_headline()
    sweep_and_plot()
```

**跑之前先在脑子里预判(然后用真实结果验证你的模型)**:

- 单点(16M)三方表:②`torch.compile` 和 ③手写 Triton 的加速比都应**逼近 2×**、且彼此接近。因为两者都把访存砍到 4N,都撞到同一堵 HBM 带宽墙——**手写不会显著快过编译器**,这很重要,别指望手写就一定赢(见 §6 工业实践)。
- 曲线:小 N(2^14)加速比接近 1×甚至上下抖动;随 N 增大爬升,到 memory-bound 区间(≳10⁷)稳定在 2× 附近。下一节解释为什么是这个形状。

---

## 4. 学习目标 Q2:为什么 pointwise 好融,什么不能融

### 4.1 pointwise 好融的本质:输出只依赖"同位置"的输入

> **pointwise / elementwise(逐元素算子)**:输出的第 i 个元素**只由输入的第 i 个元素决定**,`out[i] = f(x[i], a[i], b[i], ...)`,元素与元素之间**互不影响**。加、减、乘、除、relu、gelu、sigmoid、乘加(FMA)全属此类。

正因为"零跨元素依赖",一个 program 只要:**读进自己那块的 x/a/b → 在寄存器里把这块的 f 从头算到尾 → 写出自己那块的 out**。它**不需要看别的 program 的数据,也不需要和别人通信**。所以把任意多个 pointwise 首尾相接,本质上就是"在寄存器里多写几行算术",拼接零成本——这就是它最好融的原因。

> **类比**:pointwise 像流水线上"每个工位只加工自己手上这一件产品"。你可以把打磨、上漆、抛光三道 pointwise 工序合并到同一个工位一次做完,产品不用在工序间来回搬。

### 4.2 融合边界:什么算子不能简单融进来(接 W7 Day5 §4.3)

一旦某个输出要**跨元素看数据**,寄存器里"各算各的"就不成立了,融合就撞墙:

> **reduction(归约)**:把一整个维度"压扁"成更少的数,如 `sum`、`mean`、`max`、`min`、`argmax`。特点是**一个输出依赖很多(甚至全部)输入**——`s = sum(x)` 这一个数要看遍所有 `x[i]`。

reduction 为什么不能像 pointwise 那样融?因为一个 program 只持有自己那一块数据,而求和/求最大需要**把所有 program 的局部结果汇总**。这要用到跨线程/跨 block 的通信手段:shared memory 内规约、`tl.atomic_add` 原子累加、或分多趟(multi-pass)。这些都不是"在寄存器里接着算"能解决的。

其他天然的融合边界:

- **matmul / convolution(矩阵乘 / 卷积)**:每个输出是一整行 × 一整列的点积,是重度跨元素运算,有专门的分块 + 共享内存算法,不能当 pointwise 拼。
- **softmax / layernorm**:内部藏着 reduction(softmax 要沿行求 max 和求 sum;layernorm 要求均值和方差)。所以它们**不能整段无脑融**,得把 reduction 那步单独处理。

### 4.3 但"边界"不等于"完全不能融"——工业界的融合姿势

真实框架(Inductor、FlashAttention)玩得更精细:pointwise 可以**贴着 reduction/matmul 的前后**融进去:

> - **prologue fusion(前融)**:把 reduction/matmul **之前**的 pointwise 算子,融进"读输入"那一步。例:`sum(relu(x))`——读 x 时顺手做 relu,再进规约,relu 不单独落地。
> - **epilogue fusion(后融)**:把 reduction/matmul **之后**的 pointwise 算子,融进"写输出"那一步。例:matmul 算完一块结果,**趁它还在寄存器/共享内存里**,顺手加 bias、过激活,再写回。这是推理框架里最赚的融合之一——matmul 的输出很大,少一次它的往返 HBM 收益巨大。

**规则总结**:pointwise ↔ pointwise 随便融;pointwise 能作为前/后缀融进一个 reduction/matmul;但**两个互相依赖的 reduction 之间**(前一个的完整结果是后一个的输入)必须**物化(materialize,把中间结果真的写回 HBM)**,这就是一道硬融合边界。W8 后面写 fused-softmax 时你会正面遇到它。

---

## 5. 学习目标 Q3:张量多大才见收益,为什么小张量白融

这是今天要用**自己的曲线**讲清的重点。用一个简单模型就能预测你会看到的形状。

### 5.1 两笔账的"大小依赖性"正好相反

把耗时拆成两部分(L = 单次 kernel 启动/延迟的固定开销,约几微秒;BW = 有效带宽):

```
T_不融合 ≈ 3·L  +  (8N × 4字节) / BW
T_融合   ≈ 1·L  +  (4N × 4字节) / BW
          └启动账┘   └────访存账────┘
```

- **启动账**(`k·L`):**固定绝对开销**,不随 N 变。N 很小时,它占总时间的**大头**;N 很大时,几乎可忽略。
- **访存账**(`∝ N`):**随 N 线性增长**。N 很大时它主导,融合把它腰斩 → 干净的 2×。

### 5.2 三个区间,三种表现

```
小 N (~1e4):  T ≈ 由 L 主导 → launch/latency-bound
             访存那根主杠杆几乎没使力(数据太少,BW 用不满)。
             整个操作就几微秒,GPU 没吃饱。融合省下的访存微不足道,
             省下的 2 次启动也就几微秒——绝对值太小,对真实模型无意义。
             → 加速比接近 1×、且抖动大。这就是"融了也白融"。

中 N (~1e6):  访存账开始追上启动账,加速比爬升中(1× → 2× 过渡带)。

大 N (≳1e7): T ≈ 由访存账主导 → memory-bound
             此时 HBM 带宽是唯一瓶颈,8N→4N 直接兑现 ≈ 2× 加速,
             稳定、可复现、且正好对应真实模型里的瓶颈。
             → 加速比稳定在 2× 附近(本例的理论上限)。
```

> **为什么"小张量白融"要理解到位**:不是"融合在小张量上没用",而是**小张量整体就不值得优化**——它本来就只花几微秒,GPU 大半空着,你省 50% 也是省几微秒的绝对时间,在真实模型里被别的开销淹没。融合真正的战场是**大中间张量的 memory-bound 算子**,那里省一半访存 = 快一倍,而且这些算子在大模型里成百上千个,累加起来是实打实的吞吐提升。

### 5.3 怎么用你的曲线复述 W7 Day4(对齐完成标准)

跑完 `sweep_and_plot()`,指着 `fusion_speedup_vs_size.png` 说这三句话,今天就达标:

1. "**左端(小 N)加速比 ≈ 1× 且抖**——这里是 launch-bound,访存账没使上力,印证 W7 Day4 说的'小张量省访存无感'。"
2. "**曲线随 N 上升**——因为访存账 ∝ N,越大越主导。"
3. "**右端(大 N)压在 2× 那条虚线附近**——这就是访存账兑现:8N→4N,访存腰斩、时间腰斩,正是 W7 Day4 量出的那笔浪费被我亲手省掉了。"

> 注意:你的确切数字会因 GPU 型号、驱动、Triton 版本而不同(尤其小 N 区受启动开销和噪声影响大)。**重点不是数字漂亮,是能用两笔账的模型解释你看到的形状**。

---

## 6. 底层理解锚点:我手写的 vs Inductor 自动生成的,差在哪

W7 Day5 §4.2 你读过 Inductor 自动融合生成的 Triton kernel。把它和你今天手写的摆一起——**结构几乎一模一样**。下面是 Inductor 对 `relu(x*a+b)` 生成的典型样子(命名、风格是它的招牌特征):

```python
# Inductor 自动生成(TORCH_COMPILE_DEBUG=1 可导出),已简化
@triton.jit
def triton_poi_fused_add_mul_relu_0(   # 名字直接把融进来的算子(add/mul/relu)列出来;poi=pointwise
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]   # 和你的 offs 一个意思
    xmask = xindex < xnumel                       # 和你的 mask 一个意思
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)         # 读 x
    tmp1 = tl.load(in_ptr1 + (x0), xmask)         # 读 a
    tmp3 = tl.load(in_ptr2 + (x0), xmask)         # 读 b
    tmp2 = tmp0 * tmp1                            # x*a(寄存器)
    tmp4 = tmp2 + tmp3                            # +b (寄存器)
    tmp5 = tl.full([1], 0, tl.int32)             # 常量 0(它用 tl.full 造)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)    # relu(用 helper,保证 NaN 语义和 torch 一致)
    tl.store(out_ptr0 + (x0), tmp6, xmask)       # 写 out
```

**核心骨架和你的完全相同**:`program_id → xindex/offs → xmask/mask → 3 次 load → 寄存器里连算 → 1 次 store`。这说明你已经掌握了编译器在干的事。

**差异主要在这几处(每一处都指向 Day3 的调优)**:

| 维度 | 你的手写版 | Inductor 版 | 影响 |
|---|---|---|---|
| BLOCK 选择 | 写死 `BLOCK_SIZE=1024` | `XBLOCK` 由 autotune/启发式选 | 影响 occupancy 和访存效率 → **Day3 主题** |
| num_warps / num_stages | 用默认 | 按启发式或 autotune 选 | 影响并行度与流水 → **Day3** |
| relu 实现 | `tl.maximum(y, 0.0)` | `triton_helpers.maximum` | 后者严格对齐 torch 的 NaN 语义 |
| 常量 0 | 字面量 `0.0` | `tl.full([1], 0, ...)` | 生成代码风格,性能等价 |
| 索引 | 一维 `offs` | 通用 `xindex`(可拆多维、处理 stride/广播) | 编译器要应付任意形状,你只处理连续一维 |
| other 参数 | 省略(靠 store 的 mask 兜住) | 也常省略 | 等价 |

**一句话**:结构你已经会了,差距在"**参数怎么选最优**"(BLOCK、num_warps)和"**通用性/边界严谨度**"。前者正是 **Day3(autotune 调优)** 要补的课——今天你能把这张对比表写进笔记,Day3 就有了明确靶子。

---

## 7. 工业实践:什么时候该手写,什么时候别手写

1. **单纯的 pointwise 链,别手写**。`torch.compile` 的 Inductor 已经能自动把连续 pointwise 融成一个 Triton kernel(就是 §6 那段),而且在 memory-bound 区间和你手写的一样撞 HBM 带宽墙——**手写通常追平、极少超过**。你今天手写是为了**理解**,不是为了在生产里替换掉编译器。

2. **真正值得手写的是编译器融不动或融不好的地方**:
   - **matmul 的 epilogue fusion**:GEMM 后紧跟 bias+激活+残差,想在结果还在片上时一次做完。这是 LLM 推理里最赚的融合之一(FFN 层的 `matmul → +bias → gelu`)。
   - **attention**:FlashAttention 就是把 `QKᵀ → softmax → ·V` 融进一个 kernel,靠不把巨大的注意力矩阵物化到 HBM 来省访存和显存——这是"融合思想"在 W6 你搭过的东西上的巅峰应用。
   - **自定义/新算子**:框架还没支持、或形状特殊的算子(各种量化 kernel、MoE 的分发合并)。

3. **判断该不该融的思维模板(把它变成肌肉记忆)**:
   - 这些算子是不是 memory-bound?(pointwise、逐元素基本都是)→ 是,则融合有价值。
   - 中间张量大不大?越大,少一次往返 HBM 越赚。
   - 张量规模够不够大到 memory-bound 区间?→ 小的别费劲。
   - 中间有没有 reduction/matmul 边界?→ 有,则考虑前融/后融,而非无脑合并。

4. **务实心法**:先 `TORCH_COMPILE_DEBUG=1` 看编译器已经融成了什么、有没有该融没融的,再决定手写补哪一块。别一上来就手搓——先让编译器干活,你只在它掉链子的地方出手。

---

## 8. 常见陷阱与调试技巧

1. **计时不 warmup / 不 synchronize** → 量到编译时间或异步空档,数字假到离谱。尤其 `torch.compile(mode="max-autotune")` **首次调用会编译很久**,必须充分 warmup。

2. **用 `==` 判正确性** → 因 `fma` 单次舍入 vs 分步两次舍入,几乎必然不逐位相等。用 `torch.allclose(..., rtol=1e-4, atol=1e-5)`。

3. **在小张量上得意于"手写快 3 倍"** → 那多半是 launch/噪声,不是真实收益。**只认 memory-bound 区间(大 N)的加速比**,那才是能落到真实模型上的数字。

4. **大张量扫描 OOM** → 朴素版会额外分配 `t1`、`t2`,峰值显存高。循环里 `del` + `torch.cuda.empty_cache()`,或调低最大 exp。

5. **输入形状/连续性** → 本 kernel 假设三个输入同形状、连续(`is_contiguous()`)。遇到广播(broadcast)或转置切片,要么先 `.contiguous()`/`expand` 对齐,要么在 kernel 里正确处理 stride——否则结果错乱。

6. **`torch.compile` 结果和手写对不上** → 先确认输入 dtype 一致(fp32/fp16 舍入不同),再确认 relu 的 NaN 语义(`tl.maximum` vs `triton_helpers.maximum`)。调试时把张量调小、`tl.device_print("y", y)` 打印中间量。

---

## 9. Track B(~1h):抓一份 AMK decode 的 nsys trace(先存着,Day5 细读)

```bash
module load CUDA/12.4
# 抓 AMK small 的 decode 过程,存成 .nsys-rep,先不分析
nsys profile -o amk_small_decode --force-overwrite true \
    amk run small --gpu h100 --phase decode
# 产物:amk_small_decode.nsys-rep  → 归档,Day5 用 nsys-ui / nsys stats 细读
```

> **nsys(Nsight Systems)**:NVIDIA 的**系统级**时间线剖析器,看的是"CPU 发命令、kernel 排队、拷贝、GPU 空隙"这种宏观时间线(和 ncu 的"单 kernel 内部微观指标"互补)。今天只负责**采集并存档**,别陷进去分析——那是 Day5 的活。

---

## 10. 完成标准自测

- [ ] **手写融合 kernel 数值正确**:`torch.allclose` 通过,并能解释为什么用 allclose 而非 `==`(fma 单次舍入)。
- [ ] **能指着代码说清两笔账**:1 次 `fused_kernel[grid](...)` = 启动账;3 次 `tl.load` + 1 次 `tl.store` + 中间量留寄存器 = 访存账(8N→4N)。
- [ ] **三方数据齐全**:朴素 / `torch.compile` / 手写,且理解大 N 下②③都逼近 2× 且彼此接近(同撞带宽墙)。
- [ ] **画出"加速比 vs 张量大小"曲线**,并能用两笔账模型讲清左端≈1×(launch-bound)、右端≈2×(memory-bound),**用自己的数字复述 W7 Day4 的两笔账**。
- [ ] **写下"手写 vs Inductor 差在哪"**:结构相同,差在 BLOCK/num_warps 选择与通用性 → 指向 Day3 调优。
- [ ] Track B:`amk_small_decode.nsys-rep` 已归档。

---

### 附:今日一句话总结

**融合不让加法变快,它让数据少下楼。把 `relu(x*a+b)` 的 3 个 kernel 拼成 1 个,省下"启动"和"中间张量往返 HBM"两笔账;访存从 8N 砍到 4N,理论上限 2× ——但这 2× 只在张量大到 memory-bound 时才兑现,小张量本就没吃饱 GPU,融了也白融。**
