# Day 5 · AMK 真实 Profiling 收口:把「预测延迟」换成「H100 实测」

> **一句话主线**:W7 你交的 AMK report v0,延迟数字全是 **cost model 纸上算出来的**(标着 `PREDICTED`),GPU 真实路径没接通(`gpu_mismatch`)。今天你要用一台真实的 H100,把这些预测数字**换成 nsys 抓出来的实测数字**,并且指着时间线说清楚:论文点名的那个「跨 SM 同步」瓶颈,到底长什么样、占了多少时间。
>
> **为什么这是你的独特价值**:论文自己承认,他们在 Modal 云上跑,`ncu`(NVIDIA 的 kernel 级性能分析器)不可用,所以只能靠 cost model 预测,并把「真实硬件 profiling」明确写进了 future work。你手上有真实 H100,你能做他们做不到的事——这不是重复师姐 Gemma+vLLM 的活儿,是从 0 补上一块论文缺失的证据。

---

## 0. 读这份笔记的正确姿势

今天你其实要回答一个特别朴素、但特别深的问题:

> **「为什么在纸上算出来的耗时,和真机上跑出来的耗时不一样?差在哪?」**

这个问题背后藏着 AI Infra 最核心的一条认知分水岭:
- **新手**相信理论峰值(「H100 有 990 TFLOPS,所以我的 matmul 应该 X 微秒跑完」);
- **老手**知道理论和现实之间隔着一整层「纸上算不出来的开销」——kernel 启动、线程同步、访存打架、占用率不足……而 profiling 的全部意义,就是**把这层看不见的开销可视化、量化**。

所以这份笔记的组织逻辑是:

```
先啃三个硬骨头(Megakernel / Cost Model / 跨SM同步)
        ↓
用这三块地基回答两个学习目标问题(为什么预测≠实测 / 同步瓶颈长啥样)
        ↓
动手:nsys 怎么抓、怎么读、怎么算 gap 占比
        ↓
收口:report v1 怎么写(预测→实测 + 同步定位 + 诚实声明)
        ↓
Track A:把小算子映射回 AMK 的 task 图
```

---

## 1. 三个硬骨头(今天所有东西的地基)

### 1.1 Megakernel(巨核):把整个模型塞进「一个」kernel

**是什么。**
先说传统做法。你在 PyTorch 里写一层 Transformer,前向大概是这样一串操作:

```
RMSNorm → QKV 投影(matmul) → attention → 输出投影(matmul) → 残差加 → RMSNorm → MLP 上投影 → 激活 → MLP 下投影 → 残差加
```

**传统方式下,上面每一个箭头基本就是一次 CUDA kernel 启动**(kernel = 一段在 GPU 上并行执行的函数)。一层就十几个 kernel,一个模型几十层,一次前向就是**成百上千次 kernel 启动**。

> **CUDA kernel(核函数)是什么** —— 它是你写给 GPU 执行的一个函数,启动时你告诉 GPU「开多少个线程、怎么分组」,然后成千上万个线程并行跑同一段代码。可以把它想成:你(CPU)是包工头,kernel launch 就是你朝工地喊一嗓子「这批活儿开工!」,然后一大群工人(GPU 线程)一起干。喊一嗓子本身是有成本的(见下)。

**Megakernel** 走另一条极端路线:**把整层、甚至整个模型的前向,融合进一个巨大的 kernel 里**,只喊一次「开工」,里面把所有活儿从头干到尾。**AMK = Auto-Mega-Kernel,就是「自动」生成这种巨核的编译器/框架**——你给它模型结构,它自动把计算拆成一堆细粒度的 **task(任务)**(论文里 small 配置是 5826 个 task),再自动排进一个 kernel 里执行。

**为什么要这么干?** 两个真实痛点:

1. **kernel 启动开销(launch overhead)**:每次「喊一嗓子」CPU 都要走一遍驱动、把 kernel 参数打包、丢进 GPU 的执行队列,这一趟大约 **3–10 微秒(µs)**。听起来很小,但一次前向上千次启动,光启动就吃掉几毫秒——对于追求极致低延迟的推理,这是不能忍的。
2. **访存往返(HBM round-trip)**:两个独立 kernel 之间,前一个的输出必须写回显存(HBM),后一个再从显存读回来。数据在**芯片和显存之间反复搬运**,而显存带宽是最稀缺的资源。Megakernel 把中间结果留在片上(寄存器/共享内存),省掉这些往返。

**类比。** 传统 kernel-per-op 像**一条流水线上每个工位都各自独立开关机、上下料**:每换一道工序,半成品都要搬去仓库(显存)存一下,下一道工序再搬回来。Megakernel 像**把所有工序压进一个巨型一体化机床**:原料进去,成品出来,中间半成品一直在机器内部流转,不落地。

**但——**(这是今天的核心伏笔)一体化机床有个代价:里面的各个工位必须**严格对齐节拍**。谁快了得等,谁慢了全线卡住。这个「对齐节拍」就是后面 1.3 要讲的**跨 SM 同步**,也正是 AMK 在 H100 上打不过 cuBLAS 的根因。

**底层代码:launch 开销到底花在哪。**
下面这段 CUDA C++ 让你直观看到「喊一嗓子」的成本。它启动一个几乎什么都不干的 kernel,连启动 1 万次,看看纯启动要多久。

```cpp
// launch_overhead.cu
// 编译: nvcc -O3 launch_overhead.cu -o launch_overhead
// 运行环境: 任意 NVIDIA GPU + CUDA Toolkit(11.x/12.x 均可)
// 目的: 测量「一次 kernel 启动」的纯开销(不含真正的计算)

#include <cstdio>
#include <cuda_runtime.h>

// 一个几乎空的 kernel:只写一个字节,计算量可忽略
// 这样测出来的时间 ≈ 纯启动/调度开销
__global__ void noop(int* flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *flag = 1;
}

int main() {
    int* d_flag;
    cudaMalloc(&d_flag, sizeof(int));

    const int N = 10000;

    // warmup:第一次启动会触发 JIT/上下文初始化,必须丢弃,否则严重高估
    for (int i = 0; i < 100; ++i) noop<<<1, 32>>>(d_flag);
    cudaDeviceSynchronize();  // 等 GPU 真正干完,否则计时的是「入队时间」不是「执行时间」

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < N; ++i) {
        // 这一行 <<<...>>> 背后:CPU 打包参数 → 走驱动 → 塞进 GPU 硬件队列
        // 这一整趟就是「launch overhead」,和 kernel 里算多少无关
        noop<<<1, 32>>>(d_flag);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("平均每次 kernel 启动: %.3f us\n", ms * 1000.0f / N);
    // 典型 H100 结果: 3~6 us/次。乘以一次前向的上千次启动,就是毫秒级纯浪费。

    cudaFree(d_flag);
    return 0;
}
```

**看懂这段代码你就懂了 megakernel 的动机**:它想把上千个 `<<<...>>>` 合并成 1 个。省下的就是 `(次数-1) × 每次几微秒`。

---

### 1.2 Cost Model(成本模型):v0 里那些预测数字是怎么「算」出来的

**是什么。** AMK 要决定「哪些 task 先算、怎么排」,它需要**估计每个 task 大概要多久**。它不可能每排一种方案就真跑一次(太慢),所以用一个**解析成本模型(analytic cost model)**——纯靠公式,根据这个 task 要做多少次浮点运算(FLOPs)、要搬多少字节数据(bytes),除以硬件的理论峰值,算出一个预测耗时。

> **FLOPs(浮点运算次数)** —— 一次乘法、一次加法各算一次。一个 `[M,K]×[K,N]` 的矩阵乘,约 `2×M×N×K` 次 FLOPs(每个输出元素要 K 次乘加,乘加算 2 次)。
>
> **Roofline(屋顶线模型)** —— 判断一个计算是「算力受限」还是「带宽受限」的经典模型。核心思想:一个 task 的耗时,不会低于「算它需要的时间」和「搬它数据需要的时间」这两者中的**较大值**。

**核心公式(这就是 v0 里 `PREDICTED` 的来源):**

```
预测耗时 = max( FLOPs / 峰值算力 , 需搬字节数 / 峰值带宽 )
```

**底层代码:一个简化但接近真实的 cost model。**

```python
# cost_model.py
# 环境: Python 3.8+,无需第三方库
# 目的: 复现 AMK v0 里「PREDICTED」延迟是怎么算出来的,并暴露它系统性漏掉了什么

# ---- H100 SXM 理论峰值(公开规格,做示例用)----
PEAK_BF16_FLOPS = 989e12    # ~989 TFLOPS(bf16,不含稀疏)
PEAK_HBM_BW     = 3.35e12   # ~3.35 TB/s(HBM3)

def predict_matmul_us(M, K, N, dtype_bytes=2):
    """预测一个 [M,K]x[K,N] matmul 的耗时(微秒)。这就是 cost model 干的事。"""
    flops = 2.0 * M * N * K                       # 乘加各一次 → 2MNK
    # 需要搬的数据: 读 A、读 B、写 C(理想情况,假设各读写一遍)
    bytes_moved = (M*K + K*N + M*N) * dtype_bytes

    t_compute = flops / PEAK_BF16_FLOPS           # 算力受限下界
    t_memory  = bytes_moved / PEAK_HBM_BW         # 带宽受限下界

    t_ideal = max(t_compute, t_memory)            # roofline: 取较大者
    return t_ideal * 1e6                          # 秒 → 微秒

if __name__ == "__main__":
    # 用 small 配置里一个典型 attention 投影的规模举例
    pred = predict_matmul_us(M=512, K=2048, N=2048)
    print(f"cost model 预测: {pred:.1f} us")
    # 注意: 这个数字里【完全没有】以下东西 ——
    #   1. kernel 启动开销(见 1.1)
    #   2. 线程/网格同步等待(见 1.3)——最致命
    #   3. 多个 SM 同时抢 HBM 造成的带宽打折(内存竞争)
    #   4. 占用率不足(occupancy)导致算力吃不满
    #   5. tile 边界的尾部效应(最后一波线程只干半截活)
    # 真机实测 > 预测,几乎必然,差的就是上面这堆。
```

**这就是学习目标 Q1 的答案雏形**:cost model 算的是**理想物理下界**——假设机器完美、没有任何协调成本、带宽随便用。而真机上,上面注释里那 5 条「协调成本」全都要算钱。所以 `PREDICTED` 永远偏乐观,实测永远更大。**W7 Day4 说的「纸上算不出来的开销」,就是这 5 条。**

`gpu_mismatch` 标记,通常意思是:**v0 的数字是 cost model 预测的,还没经过真实 GPU 执行校准,预测值和(缺失的)实测值对不上、没接通**。今天你干的就是把这个 `mismatch` 消掉,换成真值。

---

### 1.3 跨 SM 网格级同步(Grid-level cross-SM synchronization):AMK 的阿喀琉斯之踵

这是今天**最硬、也最出彩**的一块。搞懂它,你的 report v1 §7 才写得出真东西。

**先补两个前置概念。**

> **SM(Streaming Multiprocessor,流式多处理器)** —— GPU 内部真正干活的「核心大单元」。H100 有 **132 个 SM**。你可以把每个 SM 想成一个**独立的车间**,里面有一堆工人(CUDA core / tensor core)、自己的小仓库(共享内存 + 寄存器)。一个 kernel 启动后,它的很多线程块(block)会被分派到这 132 个车间上并行开工。
>
> **grid(网格)** —— 一次 kernel 启动的**全部线程块的总集合**。「grid 级同步」就是「让全部 132 个车间在某个点上全部停下、对齐、再一起继续」。

**为什么 megakernel 必须做跨 SM 同步?**
回到 1.1 的一体化机床。假设一个巨核里要算:
```
第 1 步:所有 SM 一起算完 QKV 投影(把结果写进片上)
第 2 步:attention 要用【全部】QKV 结果 —— 所以必须等第 1 步【全部 SM】都算完才能开始
```
问题来了:132 个 SM 各干各的,进度不一样。第 2 步依赖第 1 步的**全部**输出。于是巨核必须在两步之间插一道**全局栅栏(barrier)**:**所有 SM 都到齐了,才放行下一步**。这就是 `grid.sync()`。

**为什么这道栅栏这么贵?** 两个原因,都能在时间线上看到:
1. **等最慢的那个**:栅栏的耗时 = 最慢 SM 的到达时间。只要有一个 SM 因为分到的 tile 略大、或者访存略慢而拖后腿,**其余 131 个 SM 全部空等**。这就是「SM 空等」。
2. **软件同步 vs 硬件同步**:cuBLAS/vLLM 用「多个独立 kernel」,kernel 之间的边界是由 **GPU 硬件队列**天然保证的(前一个 kernel 全部退出,后一个才开始),这个同步是硬件级、几乎免费的。而 megakernel 在**一个 kernel 内部**要自己用软件实现全局同步,还得反复做(每个 tile / 每个依赖边界都可能来一次)。论文说的「**每 tile 一次 grid 级跨 SM 同步**」,就是说这个昂贵的软件栅栏被触发了成千上万次。

**这就是学习目标 Q1/Q2 的核心结论**:
- AMK 在 H100 上只有 cuBLAS/vLLM 的 **0.60–0.72×**(即慢 1.4–1.7 倍),不是因为它算得慢,而是因为**它花了大量时间在栅栏前排队空等**。
- cost model 预测里**根本没有这一项**(它假设 task 一个接一个无缝执行),所以预测严重偏乐观。

**底层代码:grid.sync() 长什么样,为什么它是全局栅栏。**

```cpp
// grid_sync_demo.cu
// 编译: nvcc -O3 -arch=sm_90 grid_sync_demo.cu -o grid_sync_demo   (sm_90 = Hopper/H100)
// 运行环境: H100 + CUDA 12.x
// 目的: 演示 megakernel 内部的「跨 SM 全局同步」到底是什么,以及它为什么让快的 SM 空等

#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// 模拟一个「巨核里的两阶段计算」:阶段1所有 block 各写一格 → 全局同步 → 阶段2读全部
__global__ void two_phase(float* buf, int n) {
    // grid_group 代表「本次启动的全部线程块」——跨 SM 的总集合
    cg::grid_group grid = cg::this_grid();

    int bid = blockIdx.x;

    // ---- 阶段 1:每个 block(可能在不同 SM 上)算自己那份 ----
    if (bid < n && threadIdx.x == 0) {
        // 故意制造「快慢不均」:靠后的 block 多空转,模拟 tile 大小不均/访存慢
        for (volatile int i = 0; i < bid * 1000; ++i) { /* busy wait,拖慢这个 block */ }
        buf[bid] = bid * 1.0f;
    }

    // ---- 全局栅栏:所有 block(所有 SM)必须都到这里,才放行 ----
    // 关键点: 先到的 block 会【卡在这里空等】最慢的 block。
    //         这就是 nsys 时间线上看到的「SM 空等 / gap」的物理来源。
    grid.sync();   // ← 这一行就是「跨 SM 网格级同步」

    // ---- 阶段 2:此时才能安全读【全部】阶段1的结果 ----
    if (bid == 0 && threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < n; ++i) s += buf[i];
        printf("sum = %.1f\n", s);
    }
}

int main() {
    const int n = 132;               // 假装每个 SM 一个 block
    float* buf; cudaMalloc(&buf, n * sizeof(float));

    void* args[] = { &buf, (void*)&n };
    dim3 grid(n), block(64);

    // 注意: grid.sync() 要求用 cooperativeLaunch 启动,普通 <<<>>> 不支持全局同步
    // 这本身也是一个约束:cooperative kernel 的 block 数不能超过 GPU 一次能同时容纳的上限
    cudaLaunchCooperativeKernel((void*)two_phase, grid, block, args);
    cudaDeviceSynchronize();

    cudaFree(buf);
    return 0;
}
```

**读代码的关键**:`grid.sync()` 这一行,先到的 block 会**原地空转等**,直到最后一个 block 也到达。你在 `two_phase` 里看到我故意让靠后的 block 多空转——这模拟真实里「tile 大小不均 / 某个 SM 访存慢」。**megakernel 每个 tile 边界都来这么一下,累计起来就是巨大的空等**,而 cost model 完全没算这笔钱。

---

## 2. 两个学习目标问题:现在可以精准回答了

### Q1:为什么 `PREDICTED` ≠ 实测?`gpu_mismatch` 是什么?

一句话:**cost model 算的是「理想物理下界」,实测是「加上所有协调成本后的真实值」。** 差值来源可以列成一张表,这张表建议直接进 report v1:

| 开销来源 | cost model 有没有算 | 在真机上从哪来 | 大致量级 |
|---|---|---|---|
| 纯计算(FLOPs/算力) | ✅ 算了 | tensor core 干活 | 基准 |
| 纯访存(bytes/带宽) | ✅ 算了 | HBM 读写 | 基准 |
| **kernel 启动开销** | ❌ 没算 | CPU→驱动→GPU 队列 | 每次 3–10µs |
| **跨 SM 同步空等** | ❌ 没算 | `grid.sync()` 等最慢 SM | **最致命,可占大头** |
| **HBM 带宽竞争** | ❌ 没算 | 多 SM 同时抢显存,实际带宽打折 | 峰值的 60–80% |
| **占用率不足** | ❌ 没算 | 寄存器/共享内存不够,SM 装不满线程 | 算力吃不满 |
| **尾部效应(tail)** | ❌ 没算 | 最后一波 tile 干半截,SM 空转 | 视 tile 划分 |

`gpu_mismatch` = v0 的延迟只有「基准」那两行(预测),GPU 实测路径没接通,所以下面那 5 行全是缺失的、对不上的。**今天你把这 5 行用 nsys 量出来,`mismatch` 就消了。**

### Q2:同步瓶颈在 nsys 时间线上长什么样?

因为 megakernel 是**一个超长 kernel**,你在时间线上不会看到「一格一格的小 kernel」。同步瓶颈体现为下面三种可观测信号(report v1 §7 就靠它们):

1. **SM 占用率的锯齿波(sawtooth)** —— 用 `--gpu-metrics-device` 采样 SM active %。你会看到占用率周期性地**冲高→骤降→再冲高**:骤降的那个「谷」,就是大批 SM 到了 `grid.sync()` 空等最慢者的时刻。**谷越深、越宽,同步浪费越大。**

   ```
   SM活跃% ┐   ╱╲      ╱╲      ╱╲          ← 波峰:在算 tile
        90 ┤  ╱  ╲    ╱  ╲    ╱  ╲
        50 ┤ ╱    ╲  ╱    ╲  ╱    ╲        ← 波谷:卡在 grid.sync() 空等
        10 ┤╱      ╲╱      ╲╱      ╲
           └────────────────────────── 时间
             ↑gap  ↑gap  ↑gap   ← 这些谷加起来 = 同步开销
   ```

2. **NVTX 区间之间的 gap** —— 如果 AMK 用 NVTX 标了 task/阶段(所以命令里有 `--trace=...,nvtx`),两个 NVTX 区间之间的空白,就是同步/调度的间隙。

   > **NVTX(NVIDIA Tools Extension)** —— 一套让你在代码里手动打「时间戳标签」的 API,比如 `nvtxRangePush("attention")` / `nvtxRangePop()`。nsys 会把这些标签画到时间线上,让你知道「这段时间机器在算哪个模块」。类比:给你的程序贴上带时间的便利贴。

3. **CUDA API 层的 gap** —— `cuda_gpu_trace` 里,GPU 实际忙的时间 vs 整个 iteration 墙钟时间的差,就是 GPU 空闲(含同步空等 + 启动间隙)。

**gap 占比的可操作定义**(report 里要给数字,就用这个):
```
GPU 空闲占比 = (一次 iteration 墙钟时间 − 该 iteration 内 GPU 真正忙的时间) / 墙钟时间
同步空等占比 ≈ 1 − 平均SM占用率        (用 gpu-metrics 采样,近似)
```

---

## 3. 动手:nsys 怎么抓、怎么读、怎么算(Track B 主线,~4h)

### 3.1 为什么要「多 iteration + 丢 warmup + 抓稳态」

第一次跑任何 GPU 程序,都会有一堆一次性开销污染数据:CUDA 上下文初始化、cuBLAS 句柄创建、JIT 编译、cache 冷启动、显存首次分配。**这些只发生在前几次,不代表真实推理速度。**

所以标准做法:**跑很多次(比如 50 次),前 10 次当 warmup 丢掉,只统计后面稳定的那段**。这就是「抓稳态(steady state)」。类比:测一辆车的百公里油耗,不能算它刚点火那几秒的猛喷油,得等它匀速跑起来再测。

### 3.2 nsys profile 命令(每个 flag 为什么这么写)

```bash
# ── 在 H100 上跑,抓 AMK small 配置的稳态 profile ──
# 前提: 你的 AMK 跑脚本支持多 iteration(比如 --iters 50),且理想情况下代码里
#        对稳态区间打了 NVTX(或用 --capture-range 圈定)。

nsys profile \
  -o amk_h100_small \                 # 输出文件名 → amk_h100_small.nsys-rep(师兄要的就是这个)
  --trace=cuda,nvtx \                 # 抓 CUDA 活动(kernel/内存/API)+ NVTX 标签。够用且开销小
  --gpu-metrics-device=all \          # 【关键】采样 SM 占用率等硬件指标 → 才能看到 1.3 的锯齿波
  --cuda-memory-usage=true \          # 顺带记录显存使用,便于交叉验证 cost model 的 bytes 假设
  --force-overwrite=true \            # 覆盖旧的同名报告,避免手动删
  python run_amk.py --config small --iters 50   # 你的实际跑命令(占位,按项目改)

# 产出: amk_h100_small.nsys-rep
```

**为什么不加 `--trace=cudnn,cublas` 之类?** AMK 是自己生成的巨核,不走 cuBLAS 路径,加了也没东西。**为什么不直接用 ncu?** ncu 是 kernel 深挖(单 kernel 拆到指令/stall 级别),开销极大、会把 kernel 拖慢几十倍,而且——这正是论文在 Modal 上做不了的原因。今天先用 nsys 拿到**时间线 + 占用率**这一层,足够支撑 report v1 的结论。

### 3.3 nsys stats 提取数据(命令 + 怎么读)

`.nsys-rep` 是二进制,用 `nsys stats` 出各种汇总报告。接 W7 Day2 §5.3 的命令体系:

```bash
# 1) kernel 汇总:总 kernel 数、各 kernel 总耗时/平均耗时/占比
#    → 看 megakernel 是不是占了绝大部分时间;总 kernel 数验证「是否真的融合成少数几个」
nsys stats --report cuda_gpu_kern_sum amk_h100_small.nsys-rep

# 2) GPU 侧完整 trace(每个 kernel/memcpy 的起止时间)
#    → 算「GPU 真正忙的时间」,进而算空闲 gap 占比
nsys stats --report cuda_gpu_trace amk_h100_small.nsys-rep

# 3) NVTX 区间在 GPU 上的投影耗时(如果 AMK 打了 NVTX)
#    → 直接拿到 attention / mlp 各阶段的【实测】耗时,去和 v0 的预测对比
nsys stats --report nvtx_gpu_proj_sum amk_h100_small.nsys-rep

# 4) CUDA API 汇总:cudaLaunchKernel 等调用的次数和总耗时
#    → 量化 launch overhead(表格里第 3 行)
nsys stats --report cuda_api_sum amk_h100_small.nsys-rep

# 5) 只统计稳态那段(丢 warmup):加时间窗过滤
#    假设前 10 iter 在前 200ms,就从 200ms 之后开始统计
nsys stats --report cuda_gpu_kern_sum --filter-nvtx="steady" amk_h100_small.nsys-rep
#    (若用 NVTX 圈了 "steady" 区间;否则用 --start / GUI 手动框选稳态窗口)
```

**读 `cuda_gpu_kern_sum` 输出的重点列:**
- `Instances`(次数):稳态每 iteration 应有多少个 kernel。若 megakernel 生效,这个数应该**远小于**传统路径。
- `Total Time` / `Avg`:这就是**实测耗时**。把它和 v0 的 `attention 364µs / mlp 251µs` 预测值并排,差多少就是那 5 项漏算开销的总和。
- `%`:megakernel 占 GPU 总时间的比例,佐证「时间都花在这个巨核里」。

### 3.4 算「同步 gap 占比」的脚本

nsys 也能导出 sqlite/csv,便于精确算 gap。下面给一个自动化思路(把 stats 的 CSV 喂进去):

```python
# compute_gap.py
# 环境: Python 3.8+;先用 `nsys stats --report cuda_gpu_trace --format csv \
#        -o trace amk_h100_small.nsys-rep` 导出 trace_cuda_gpu_trace.csv
# 目的: 算稳态区间内「GPU 空闲(含同步空等)占比」——report v1 §7 要引用的数字

import csv

STEADY_START_NS = 200_000_000   # 丢掉前 200ms warmup(按你实际情况调)
STEADY_END_NS   = None          # None = 到结尾

rows = []
with open("trace_cuda_gpu_trace.csv") as f:
    for r in csv.DictReader(f):
        # 不同 nsys 版本列名略有差异,常见为 "Start (ns)" / "Duration (ns)"
        start = float(r.get("Start (ns)") or r.get("Start"))
        dur   = float(r.get("Duration (ns)") or r.get("Duration"))
        if start < STEADY_START_NS:            # 跳过 warmup
            continue
        if STEADY_END_NS and start > STEADY_END_NS:
            continue
        rows.append((start, start + dur))

rows.sort()
# 合并重叠区间(不同流可能并行),得到「GPU 至少有一件事在忙」的总时长
busy = 0.0
cur_s, cur_e = rows[0]
for s, e in rows[1:]:
    if s <= cur_e:                 # 重叠 → 合并
        cur_e = max(cur_e, e)
    else:
        busy += cur_e - cur_s      # 结算一段连续忙碌
        cur_s, cur_e = s, e
busy += cur_e - cur_s

wall = rows[-1][1] - rows[0][0]    # 稳态墙钟总时长
idle = wall - busy                 # 空闲 = 墙钟 − 忙碌
print(f"稳态墙钟: {wall/1e3:.1f} us")
print(f"GPU 忙碌: {busy/1e3:.1f} us")
print(f"GPU 空闲(含启动间隙+同步空等)占比: {idle/wall*100:.1f} %")
# 这个百分比 + gpu-metrics 的 SM 占用率锯齿,一起支撑「同步瓶颈占 X%」的结论
```

> ⚠️ **诚实提醒**:上面这个 `idle` 占比,严格说包含了「同步空等 + kernel 间隙 + 尾部空转」的**混合**,它是同步瓶颈的**上界近似**,不是纯同步。要更纯,得靠 `--gpu-metrics` 的 SM 占用率锯齿谷做交叉印证。**report 里必须把这句话写清楚**——这就是「诚实声明」的一部分。

---

## 4. 收口:写 `amk_profiling_report_v1.md`

核心动作:**在 v0 骨架上改两处、加一处**,其余保持不动,保证 v0→v1 可追溯。

### 4.1 §3 延迟画像:预测 → 实测(替换)

把 v0 里 `PREDICTED(analytic cost model)` 的表,换成 nsys 实测,并**保留预测列做对比**——对比本身就是你的核心发现:

```markdown
## §3 延迟画像(v1:H100 实测)

| Region     | v0 预测(cost model) | v1 实测(nsys, H100 稳态) | 实测/预测 | 差距来源(见 §3.1) |
|------------|--------------------:|--------------------------:|---------:|--------------------|
| attention  | 364 µs              | <填 nsys 数>              | <×>      | 同步空等为主        |
| mlp        | 251 µs              | <填 nsys 数>              | <×>      | 启动+带宽竞争       |
| 单层总计    | ...                 | ...                       | ...      | ...                |

> 数据来源:amk_h100_small.nsys-rep,稳态区间(丢弃前 10 iter warmup)。
> 采集命令见附录。实测 > 预测,差值即 cost model 未建模的协调开销。
```

### 4.2 §7 跨 SM 同步瓶颈定位(新增)

这是你的**独特产出**,结构建议:

```markdown
## §7 跨 SM 同步瓶颈定位(v1 新增,真实硬件证据)

### 7.1 论文声明
论文指出 AMK 在 H100 上为 cuBLAS/vLLM 的 0.60–0.72×,归因于
「megakernel 每 tile 一次 grid 级跨 SM 同步」。v0 因 Modal 上 ncu 不可用,
无法给出硬件证据,列为 future work。

### 7.2 本报告的实测证据
- GPU 空闲(含同步空等)占稳态墙钟 X%(compute_gap.py 测得)。
- SM 占用率呈锯齿波(图见 amk_h100_small.nsys-rep 时间线),
  周期性谷值对应 grid.sync() 空等,谷宽 ≈ Y µs / 次,一次 iteration 约 Z 次。
- [截图:nsys 时间线 + gpu-metrics SM active% 曲线,标注同步谷]

### 7.3 结论
论文所述同步瓶颈在 H100 上得到实测印证,估计占稳态时间约 X%,
是 AMK 落后 cuBLAS/vLLM 的主因,与其算力/带宽利用无关。
```

### 4.3 诚实声明(W7 一直强调,别丢)

```markdown
## 附:测量边界声明(Honesty Statement)
【测了什么】H100 单卡、small 配置、稳态多 iteration 的 nsys 时间线级 profiling:
  kernel 数/耗时、NVTX 区间实测、GPU 空闲占比、SM 占用率锯齿。
【没测什么】
  - 未做 ncu kernel 级深挖(指令/stall/warp 效率),故「同步占比」为时间线级
    近似上界,非指令级精确值。
  - 空闲占比混合了同步空等 + kernel 间隙 + 尾部效应,未完全解耦。
  - 仅 small 配置、单卡,未覆盖 large / 多卡。
【可信度】实测数字来自真实 H100,可复现(命令见附录);解释性结论(同步为主因)
  与论文声明一致并由 SM 占用率锯齿佐证,但精确归因需后续 ncu 验证。
```

---

## 5. Track A(~2h,不断手感):把小算子映射回 AMK task 图

**目标**:理解你 Day3 写的 **RMSNorm** / Day2 的 **fused pointwise**,在 AMK 的 5826 个 task 里对应哪一类。

**关键认知**:AMK 把模型拆成 task,大致分两类:
- **重算子 task**:matmul / attention,吃 tensor core,是 megakernel 里的「大块头」。
- **轻算子 task**:RMSNorm、激活、残差加、pointwise —— 这些**计算量小、几乎纯访存(带宽受限)**,数量极多(5826 里一大半是这种)。

**为什么它们在 megakernel 里特别关键?** 因为在传统路径里,每个轻算子都是一次独立 kernel(启动开销 > 计算本身,极亏)。**megakernel 的最大收益恰恰来自把这些轻算子融进去**——省掉它们的启动和访存往返。你 Day2/Day3 手写的 fused pointwise / RMSNorm,本质就是「手动做了一次 AMK 自动做的 task 融合」。

```cpp
// rmsnorm_task.cu(片段) —— 对应 AMK 里的一个「轻算子 / 带宽受限 task」
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// 关键: 它 FLOPs 极少,主要成本是把 x 读一遍、y 写一遍 → 带宽受限
//       所以在 cost model 里它的预测 = bytes/带宽,几乎没有算力项。
__global__ void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    // 一个 block 处理一行(一个 token 的 hidden 向量)
    extern __shared__ float smem[];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + row * n;
    float* yr = y + row * n;

    // 1) 求平方和(块内规约)—— 这是 RMSNorm 唯一需要「跨线程协作」的地方
    float local = 0;
    for (int i = tid; i < n; i += blockDim.x) local += xr[i] * xr[i];
    smem[tid] = local; __syncthreads();          // block 内同步(便宜,同一个 SM 内)
    for (int s = blockDim.x/2; s > 0; s >>= 1) { // 树形规约
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / n + eps);        // 1/sqrt(mean+eps)

    // 2) 归一化 + 加权
    for (int i = tid; i < n; i += blockDim.x) yr[i] = xr[i] * rms * w[i];
}
```

**映射练习(写进笔记留痕)**:
- 这个 RMSNorm 用的是 `__syncthreads()`(**block 内、同一 SM 内**同步,便宜)——注意它和 1.3 的 `grid.sync()`(**跨 SM**,昂贵)**不是一回事**。想清楚这个区别,你就真懂了 megakernel 的成本结构:轻算子内部同步便宜,**贵的是 task 之间的跨 SM 栅栏**。
- 在 AMK task 图里,这个 RMSNorm 会被拆成若干个「带宽受限 task」,和它前后的 matmul task 之间,就隔着那道昂贵的 `grid.sync()`。

---

## 6. 完成标准自检 + 常见陷阱

**完成标准(对照今天目标逐条打勾):**
- [ ] 产出 `amk_h100_small.nsys-rep`(H100、small、稳态多 iter)——师兄要的文件 ✅
- [ ] `amk_profiling_report_v1.md`:§3 从预测换成实测(带 v0 对比列)✅
- [ ] §7 新增同步瓶颈定位:**能指着时间线说「论文说的跨 SM 同步,在这里、占了 X%」** ✅
- [ ] 诚实声明:测了什么/没测什么写清楚 ✅
- [ ] Track A:RMSNorm/fused pointwise ↔ AMK 轻算子 task 的映射理解 ✅
- [ ] 发给师兄/张老师组 ✅

**常见陷阱(踩一个就白跑):**
1. **忘了丢 warmup** → 数字被首次初始化污染,实测虚高,结论不可信。**必跑 ≥30 iter,只统计稳态。**
2. **没加 `--gpu-metrics-device`** → 拿不到 SM 占用率,§7 的锯齿波无从谈起,同步瓶颈变成「嘴上说说」。**这个 flag 是 §7 的命根子。**
3. **把 `idle 占比`直接当「同步占比」** → 它是混合上界。必须在诚实声明里说明,并用 SM 锯齿交叉印证。
4. **在 ncu 上死磕** → 今天用不上,而且 ncu 会把 megakernel 拖慢到没法看稳态。nsys 时间线层足够。
5. **把 v0 预测数删掉** → 对比列没了,你最核心的发现(实测/预测差多少)就消失了。**保留,并排放。**
6. **`grid.sync()` 需要 `cudaLaunchCooperativeKernel` 启动**,普通 `<<<>>>` 不支持——读 AMK 源码时若看到 cooperative launch,就是它在做全局同步的实锤,可写进 §7 佐证。

**一句话收尾**:今天你干的事,本质是**把「理论下界」和「工程现实」之间那道缝,用真实 H100 的证据量出来**。这道缝的名字叫「协调成本」,其中最贵的一块叫「跨 SM 同步」。你能量出它、指着时间线讲清它——这就是你第一段科研的第一个硬产出,也是 cost model 永远给不了的东西。
```
