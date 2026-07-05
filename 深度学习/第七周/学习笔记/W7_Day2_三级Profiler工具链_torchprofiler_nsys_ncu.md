# W7 Day2 · 三级 Profiler 工具链:torch.profiler → nsys → ncu

> **本笔记的唯一目标**:让你真正学会"**逐级放大**"地给一段 GPU 代码做体检 —— 先用 `torch.profiler` 知道**哪个算子(op)吃时间**,再用 `nsys` 看**这些算子在时间线上是怎么排队、谁在等谁、空了多少**,最后用 `ncu` 钻进**某一个 kernel 内部看它跑得够不够满**。读完你要能独立对 nanoGPT 的 `generate_kv` 跑出一份 nsys trace,**数清 decode 一步有几个 kernel、gap 占多少、launch 占多少**,并把同一套命令平移到 AMK,产出小米课题的第一份原始 profiling 数据。
>
> 串联:这是 [W7 学习计划](./W7_学习计划_AI_Infra主线.md) **Track A+B 合流**的一天,承接 [W7 Day1 Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md) 的"定性判断"——Roofline 告诉你 decode 是 memory-bound,但**它是一张纸上的理论上限**;今天的三级工具是**真实测量**,要回答"理论说该慢,那实际慢在哪一行、慢成什么样"。对应小米课题主线 1(性能画像与瓶颈分析)。产出对齐 [项目规范](./W7_学习计划_AI_Infra主线.md) 中的 `tech_notes/profiling_toolchain.md`。

---

## 0. 开篇:读完你要能脱口而出的 6 个问题

1. 为什么需要**三个**工具,而不是一个?它们的"放大倍率"分别是什么级别?
2. `torch.profiler` 是怎么做到"知道每个 op 花了多少 GPU 时间"的?它在底层挂了什么钩子?为什么它给的 GPU 时间**不能**直接当墙钟时间(wall-clock)用?
3. 什么叫 **kernel launch overhead(核函数启动开销)**?为什么 decode 阶段会被它"拖死",而 prefill/训练几乎感受不到?
4. nsys 时间线上,GPU 那一行两个 kernel 之间的**空白(gap)**到底意味着什么?怎么区分"GPU 在等 CPU 发指令"还是"GPU 在等数据/依赖"?
5. `ncu` 为什么慢得离谱(一个 kernel 测几十秒)?**kernel replay(核函数重放)**是什么、为什么必须重放?
6. 为什么 nanoGPT decode 一步会有**几十个小 kernel + 一堆 gap**,而 AMK 的目标是把它压成**一个巨核(megakernel)**?这两份 nsys trace 摆在一起,就是你给小米写的报告的"Before / After"。

如果第 3、4 题你能不看资料讲清楚"decode 慢一半不是因为算得慢,而是 CPU 发指令发不过来 + GPU 空等",这一天就值了。

---

## 1. 问题背景:Roofline 之后,为什么还要 profiler?

### 1.1 Roofline 的盲区

Day1 我们用 Roofline 算出:nanoGPT decode 的算术强度 ≈ 1,远低于 H100 脊点 ≈ 295,所以是 **memory-bound(受访存带宽限制)**。

但 Roofline 是一张**理想化的纸**,它有两个致命盲区:

1. **它假设 kernel 一个接一个、零间隙地跑**。现实里 GPU 经常**空等**——上一个 kernel 算完了,下一个还没被"发"过来。这部分浪费 Roofline 完全看不见。
2. **它只看单个算子的理论上限**,看不见"一步 decode 由几十个算子拼起来,它们之间怎么衔接"。

> **类比**:Roofline 像是给一名运动员算"理论上百米能跑多快"(看肌肉、看摄氧量)。但真实比赛慢了,可能不是肌肉不行,而是**起跑反应慢、交接棒掉了、跑道之间还要停下来系鞋带**。Profiler 就是去录下整场比赛的录像,一帧一帧看到底卡在哪。

### 1.2 为什么是"三级",而不是一个万能工具?

因为**不同粒度的问题,需要不同放大倍率的镜头**。一个工具想同时看全局又看细节,要么数据量爆炸,要么严重拖慢被测程序。三级工具是一条**从"望远镜"到"显微镜"逐级放大**的链条:

| 工具 | 中文 | 放大粒度 | 回答的问题 | 类比 |
|---|---|---|---|---|
| `torch.profiler` | PyTorch 内置剖析器 | **算子级(op-level)** | 哪个 op(matmul / attention / layernorm / cat)总共吃了多少时间? | **体检报告**:告诉你"心肺功能这项分数低" |
| `nsys` (Nsight Systems) | 系统级时间线剖析器 | **时间线级(timeline)** | 这些 op 在 CPU/GPU 上怎么排队?kernel 之间空了多少?谁在等谁? | **24 小时动态心电图**:看出"你每隔几秒心跳停一下" |
| `ncu` (Nsight Compute) | 单核函数剖析器 | **单 kernel 级** | 某一个具体 kernel,SM 占用率多少?访存吞吐打到峰值的百分之几? | **心脏造影**:钻进单个器官看血管堵在哪 |

**工作流铁律(记死)**:**永远从粗到细**。先 `torch.profiler` 找到吃时间的大头 → 再 `nsys` 看这个大头在时间线上是"算得慢"还是"等得久" → 只有当确认某个 kernel 本身算得慢时,才动用 `ncu` 去解剖它。

为什么不能反着来?因为 `ncu` 一次只看一个 kernel 且极慢,你要是没有前两步定位,等于拿显微镜在一整个人身上乱扫,几辈子也扫不完。

---

## 2. 第一级:torch.profiler —— 算子级,"哪个 op 吃时间"

### 2.1 是什么

`torch.profiler` 是 PyTorch 自带的剖析器,你在 Python 代码里用一个 `with` 块把要测的代码包起来,它就能告诉你:**每一种算子(op)在 CPU 上花了多少时间、在 GPU(CUDA)上花了多少时间、被调用了多少次、占总时间的百分比**。

### 2.2 为什么它能"看见"GPU 时间?—— 底层挂了 CUPTI 钩子

这是第一个必须深挖的原理。很多人以为 profiler 就是"在代码前后记两个时间戳相减",**对 GPU 完全不是这样**。

原因在于 **CUDA 是异步的(asynchronous)**:

> 你在 Python 里写 `y = x @ w`,CPU 并**不会**等 GPU 算完。它只是把"做一次矩阵乘"这条命令**塞进一个队列(CUDA stream,见 §3.2)**就立刻返回,继续执行下一行 Python。真正的计算稍后才在 GPU 上发生。

所以如果你天真地这样测:

```python
import time
t0 = time.time()
y = x @ w            # CPU 瞬间返回,GPU 还没开始算
t1 = time.time()     # 你测到的只是"把命令塞进队列"的时间,几乎是 0!
print(t1 - t0)       # 完全错误的结果
```

你测到的根本不是计算时间。**正确测 GPU 时间,必须用 GPU 自己的硬件时钟在 kernel 真正开始/结束的瞬间打点。**

`torch.profiler` 在底层是通过 NVIDIA 的 **CUPTI(CUDA Profiling Tools Interface,CUDA 剖析工具接口)** 做到这一点的(PyTorch 里封装 CUPTI 的库叫 **Kineto**)。机制分两路:

1. **Callback API(回调接口)**:在每一次 CUDA 运行时/驱动 API 被调用时(比如 `cudaLaunchKernel`),CUPTI 会回调一下,记录"CPU 在这个时刻发起了一次 kernel 启动"。这是 **CPU 侧**的时间戳。
2. **Activity API(活动记录接口)**:GPU 上每个 kernel **真正执行**的开始/结束时间,由 GPU 硬件计时器记录,异步写进一块缓冲区,事后再"收割"出来。这是 **GPU 侧**的真实执行时间戳。

最后 Kineto 把这两路用一个"关联 ID(correlation id)"对上号——于是你既能看到"CPU 第几行发起了这次 launch",又能看到"它对应的 kernel 在 GPU 上几点几分真正跑了多久"。

> **类比**:你寄快递(发起 op),快递公司给你一个**单号(correlation id)**。你这边记下"我几点几分下的单"(CPU callback),包裹真正送达时快递柜记下"几点几分签收"(GPU activity)。事后凭同一个单号,就能拼出"从下单到送达"的完整时间线。profiler 干的就是这个对单号的活。

### 2.3 怎么用(工业级写法,可直接跑)

```python
# ============================================================
# 环境: PyTorch >= 2.0, CUDA GPU(H100/A100 等)
# 运行: python prof_demo.py
# 目的: 用 torch.profiler 测出 decode 风格的小矩阵乘里,哪个 op 吃时间
# ============================================================
import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function

dev = "cuda"
# 模拟 decode: 一个 token(M=1)过一层 MLP。注意 M=1 正是 decode 的 GEMV 形态
hidden, inter = 4096, 11008
x = torch.randn(1, hidden, device=dev, dtype=torch.float16)
w1 = torch.randn(hidden, inter, device=dev, dtype=torch.float16)
w2 = torch.randn(inter, hidden, device=dev, dtype=torch.float16)

def decode_step():
    # record_function: 给这段代码打一个自定义标签,profiler 里会单独成一行
    # 为什么这么写: 真实模型里 op 很多,手动打标签能把"逻辑块"圈出来,否则一片 aten::mm 看不出归属
    with record_function("mlp_block"):
        h = torch.relu(x @ w1)   # 第一层: [1,4096]x[4096,11008] -> GEMV
        out = h @ w2             # 第二层: [1,11008]x[11008,4096] -> GEMV
    return out

# schedule: 控制"预热/记录"节奏。为什么需要?
#   - wait=1:  跳过第1步(此时可能在初始化)
#   - warmup=2: 再跑2步热身(CUDA 首次启动 kernel 要 JIT 编译/建缓存,首次极慢,不能算进去)
#   - active=3: 真正记录 3 步
# 这是工业惯例: 永远丢掉冷启动那几步,否则数据被首次开销污染
my_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 同时抓 CPU 和 GPU
    schedule=my_schedule,
    record_shapes=True,    # 记录每个 op 的输入张量形状(排查"为什么这个 mm 这么慢"靠它)
    profile_memory=True,   # 记录显存分配/释放(排查 OOM、看 cat 这类算子的显存行为)
    with_stack=False,      # 记录 Python 调用栈(精确但更慢,定位"哪行代码"时才开)
) as prof:
    for _ in range(6):     # 步数要 >= wait+warmup+active = 6
        decode_step()
        prof.step()        # 关键: 告诉 profiler "一步结束了",它据此推进 schedule

# 按 GPU 自身耗时排序打印 top 10。注意是 self_cuda_time_total:
#   "self" = 只算这个 op 自己的时间, 不含它调用的子 op, 避免父子重复计数
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# 导出 Chrome trace, 浏览器打开 chrome://tracing 或 https://ui.perfetto.dev 可视化
prof.export_chrome_trace("trace.json")
```

### 2.4 怎么读输出 + 常见陷阱

输出表里你重点看四列:

- `Self CUDA %`:这个 op 自己占了多少 GPU 时间(找大头就看它)。
- `CUDA total`:含子调用的总 GPU 时间。
- `# of Calls`:调用次数(decode 里某些 op 会被调用几十次,单次小但累计大)。
- `Input Shapes`(开了 `record_shapes`):同名 op 可能因形状不同性能天差地别。

**三个高频陷阱:**

1. **没热身**:首次 kernel 启动包含 JIT 编译、cuBLAS 选 kernel、显存分配器建池,可能比稳态慢 10 倍以上。不丢掉就得出"matmul 慢得离谱"的错误结论。
2. **把 CPU 时间当总时间**:CUDA 异步,CPU 早早返回了。判断真实瓶颈要看 **CUDA 时间**或墙钟时间,不是 CPU self time。
3. **profiler 本身有开销**:CUPTI 抓取会让程序变慢(尤其开了 `with_stack`)。所以 **torch.profiler 给的是"相对占比"很准,但"绝对时间"略有放大**——要拿绝对时间下结论,用下一级的 nsys。

> **它的能力边界**:torch.profiler 告诉你"`aten::mm` 占了 40% GPU 时间",但它**看不到** kernel 之间的空隙、看不到 launch overhead 的全貌、也看不到 GPU 内部 SM 占用。它是望远镜,接下来要换镜头。

---

## 3. 第二级:nsys —— 时间线级,"谁在等谁、空了多少"

这是今天**最核心**的一级,也是你 AMK 课题报告的主力工具。

### 3.1 是什么

**nsys** 全称 **Nsight Systems**,是 NVIDIA 的**系统级时间线剖析器**。它把一段时间内 **CPU 线程在干什么、CUDA API 在何时被调用、每个 kernel 在 GPU 上何时真正执行、内存拷贝何时发生**,全部画在**同一条时间轴**上,你能像看视频剪辑软件的多轨道一样,看到 CPU 行和 GPU 行的对齐关系。

它和 torch.profiler 的本质区别:torch.profiler 给你**聚合后的表格**(每种 op 加总),nsys 给你**没有聚合的、按真实发生顺序排开的时间线**。表格看不出"空隙",时间线一眼就能看出。

### 3.2 必须先理解:CUDA stream 与异步执行模型

要看懂 nsys 时间线,必须先把"CPU 怎么驱动 GPU 干活"这件事彻底搞清。这是整个 GPU 性能优化的地基。

> **CUDA stream(CUDA 流)** = 一条**先进先出(FIFO)的命令队列**。CPU 把"启动 kernel A""做一次 memcpy""启动 kernel B"这些命令**按顺序塞进流**,GPU 则从流里**按顺序取出来执行**。同一条流里的命令严格按序;CPU 塞命令(enqueue)和 GPU 执行命令(execute)是**异步并行**的两件事。

整个过程像这样:

```
CPU 线程(Python/调度):  [发 kA][发 memcpy][发 kB][发 kC] ...  → 塞进 stream 队列
                              │       │        │     │
                              ▼       ▼        ▼     ▼
CUDA stream 队列:        [kA][memcpy][kB][kC] ...   (FIFO)
                              │
                              ▼   GPU 从队头不断取出执行
GPU 执行单元(SM):        ███kA███  ░░░  ██memcpy██  ░░░  ███kB███ ...
                                    ↑gap          ↑gap
```

**关键洞察**:GPU 执行行里那些空白 `░░░`(gap),意味着 **GPU 干完上一个、想取下一个时,队列里还没有命令**——也就是 **CPU 还没把下一条命令塞进来**。GPU 在**空等 CPU**。

### 3.3 核心原理:kernel launch overhead,以及 decode 为什么被它拖死

**kernel launch overhead(核函数启动开销)** 是指:CPU 每发起一次 `cudaLaunchKernel`,从"调用 API"到"GPU 真正开始执行这个 kernel",中间有一段固定的**额外开销**——大约 **几微秒(µs)级别**(典型 3~10 µs,取决于驱动、参数个数等)。

这段开销花在哪?CPU 要:打包 kernel 参数 → 陷入驱动 → 在流队列里写入一条命令 → 通知 GPU 的命令处理器(front-end)。这些都是**固定成本,和 kernel 大不大无关**。

现在把它和 Day1 的结论合起来看,decode 的悲剧就出现了:

- **训练 / prefill**:一个 kernel 处理一大批数据(比如 batch×序列 = 几千个 token 的大矩阵乘),kernel 本身要在 GPU 上跑**几百微秒甚至毫秒**。这时 5 µs 的 launch 开销,占比连 1% 都不到,**忽略不计**。
- **decode**:每个 kernel 只处理 **1 个 token**(M=1 的 GEMV,见 Day1),计算量极小,kernel 本身在 H100 上可能只跑 **几微秒**。这时 5 µs 的 launch 开销,**和 kernel 执行时间一样长甚至更长**!

> **类比**:launch overhead 像"每做一道菜前,服务员都要走到后厨口头报一遍菜名"(固定 5 秒)。如果这道菜要炖 1 小时(prefill),报菜名那 5 秒无所谓。但如果这道菜是"切一片柠檬"只要 3 秒(decode),那报菜名的 5 秒比做菜还久,**厨房一大半时间在等服务员报菜名,而不是在做菜**。

更糟的是,decode 一步不是一个 kernel,而是**几十个小 kernel 串起来**(每层:QKV 投影、attention、output 投影、两层 MLP、两个 layernorm、各种 elementwise……乘以几十层)。每个都要交 5 µs 的"过路费",而且**前一个没发完,GPU 就空等**。于是时间线变成:

```
GPU: ██k1██ ░░░░ ██k2██ ░░░░ ██k3██ ░░░░ ...
          ↑      ↑      ↑
       全是 gap,GPU 实际利用率可能只有 20~40%
```

这就是 **launch-bound(受启动开销限制)** 或叫 **CPU-dispatch-bound**:瓶颈既不是算力也不是带宽,而是 **CPU 发命令的速度跟不上 GPU 吞命令的速度**。**这正是 W6 §8.3 要你验证的预测。**

### 3.4 怎么做:给 nanoGPT 的 generate_kv 跑 nsys(动手核心)

**第一步:用 NVTX 给 decode 步打标记**(让时间线可读)

> **NVTX(NVIDIA Tools Extension)** = 你在代码里手动插入的"彩色标签",nsys 会把它画成时间轴上的一个有色区间。没有它,几十个 kernel 糊成一片,你根本分不清"哪一段是第 3 步 decode"。

```python
# 在 nanoGPT 的 generate_kv 循环里(model.py / sample.py)插桩
import torch.cuda.nvtx as nvtx

for step in range(max_new_tokens):
    nvtx.range_push(f"decode_step_{step}")   # 标记一步 decode 的开始
    logits = model(idx_cond, use_kv_cache=True)   # 你的 KV-cache 前向
    idx_next = sample(logits)
    nvtx.range_push("kv_cat")                 # 单独标出 KV-cache 拼接(见 §3.6)
    kv_cache = torch.cat([kv_cache, new_kv], dim=2)
    nvtx.range_pop()                          # kv_cat 结束
    nvtx.range_pop()                          # decode_step 结束
```

**第二步:跑 nsys profile**

```bash
# ============================================================
# 环境: 装了 Nsight Systems 的 H100 机器; nsys 在 CUDA toolkit 里自带
# 关键: 只测少量 decode 步(比如 max_new_tokens=8), 并先在脚本里跑几步热身
# ============================================================
nsys profile \
  -o nanogpt_decode \                # 输出文件 nanogpt_decode.nsys-rep
  --trace=cuda,nvtx,osrt \           # 抓: CUDA kernel/API, 我们的NVTX标签, OS运行时(看CPU线程在干啥)
  --cuda-graph-trace=node \          # 若用了 CUDA Graph, 展开到节点级(否则只看到一个大块)
  --force-overwrite=true \
  python sample_kv.py --max_new_tokens=8
```

为什么这样配:`--trace=cuda` 给你 GPU kernel 行,`nvtx` 给你刚打的彩色标签,`osrt`(OS runtime)让你看到 **CPU 线程是不是卡在某个系统调用上**——这能区分"CPU 忙不过来"还是"CPU 在等锁/IO"。

**第三步:导出统计,数清三个关键数字**

```bash
# 报告1: 每种 kernel 的耗时汇总(数 kernel 种类、看谁最贵)
nsys stats --report cuda_gpu_kern_sum nanogpt_decode.nsys-rep

# 报告2: 逐个 kernel 的执行轨迹(数一步有几个 kernel、看每个的起止时间→算 gap)
nsys stats --report cuda_gpu_trace nanogpt_decode.nsys-rep

# 报告3: CUDA API 调用汇总(看 cudaLaunchKernel 总共调了多少次、花了多少 CPU 时间 → launch 占比)
nsys stats --report cuda_api_sum nanogpt_decode.nsys-rep
```

**第四步:算出你要交的三个指标**(验证 W6 §8.3)

针对**稳态的某一步** `decode_step_k`(挑中间一步,避开首尾):

1. **kernel 数**:`cuda_gpu_trace` 里落在该 NVTX 区间内的 kernel 行数。记为 `N`。
2. **gap 占比**:
   ```
   GPU 忙时间 = 该步内所有 kernel 执行时长之和
   该步墙钟时间 = decode_step_k 区间的总时长(NVTX 给你)
   gap 占比 = (该步墙钟时间 − GPU 忙时间) / 该步墙钟时间
   ```
   gap 占比越高,说明 GPU 空等越严重 → 越 launch-bound。
3. **launch 占比**:
   ```
   launch 占比 ≈ N × 单次launch开销(~5µs) / 该步墙钟时间
   ```
   或直接用 `cuda_api_sum` 里 `cudaLaunchKernel` 的总耗时 / 总墙钟时间。

> **预期结果(就是要去验证的)**:nanoGPT decode 一步有**几十个 kernel**,gap 占比**显著(可能 30%~60%)**,launch 相关开销占比**不可忽略**。这定量证明了"decode 不只是 memory-bound,还严重 launch-bound",从而**论证了 AMK megakernel(把几十个 kernel 融成一个)的价值**——把 N 从几十降到 1,launch 开销和 gap 几乎归零。

### 3.5 怎么读时间线 + 区分两种 gap(关键技巧)

在 Nsight Systems GUI(或 Perfetto)里打开 `.nsys-rep`,你会看到多条横轨:CPU 线程行、CUDA API 行、GPU kernel 行、memcpy 行。

**两种 gap,优化方向完全不同,必须分清:**

| 现象 | 含义 | 怎么认 | 优化方向 |
|---|---|---|---|
| GPU 行有 gap,**同一时刻 CPU 行正在忙着发 launch** | GPU 在等 CPU 发命令 → **launch-bound** | gap 紧挨着 CPU 的 `cudaLaunchKernel` | CUDA Graph / megakernel / 算子融合 |
| GPU 行有 gap,**CPU 行也是空的(在等)** | 在等数据(memcpy)或同步点 | gap 前有 memcpy,或有 `cudaStreamSynchronize` | 重叠传输与计算、去掉多余同步 |

> **类比**:看一段后厨监控录像。厨师(GPU)停手时——如果服务员(CPU)正满头大汗地报菜名,那是"传菜系统太慢";如果服务员也在干等(比如在等食材从冷库运来),那是"供应链问题"。同样是厨师停手,病根完全不同,药方也不同。

### 3.6 深挖一个底层真相:nsys 里那条 KV-cache `cat`,为什么会出现一次 memcpy + 多余显存?

你之前问过"`torch.cat` 会预先请求一整块连续显存、而且浪费空间,底层到底怎么实现的、为什么不能按需精确申请"。在 nsys 时间线上,你**真的会在 decode 每一步看到这条 `cat` 触发的 `Memcpy DtoD`(设备到设备拷贝)**,正好借它把原理讲死。

**现象**:`kv_cache = torch.cat([kv_cache, new_kv], dim=2)`,在 nsys 里表现为:一次显存分配 + 一次把旧 KV 整块拷到新地址的 `Memcpy DtoD`。序列越长,拷贝越大,这条 memcpy 在时间线上越来越宽——这本身就是个性能问题(后面 PagedAttention 就是来解决它的)。

**为什么 cat 必须新分配一整块、还"浪费"?** 看 PyTorch 底层:

```cpp
// 简化自 PyTorch aten/src/ATen/native/TensorShape.cpp 的 cat 逻辑
// 核心: cat 的语义是"产出一个全新的、内存连续的张量"
Tensor cat(TensorList tensors, int64_t dim) {
    // 1) 先把所有输入在 dim 维上的尺寸加起来, 算出输出总形状
    int64_t cat_dim_size = 0;
    for (const Tensor& t : tensors)
        cat_dim_size += t.size(dim);
    auto out_shape = compute_out_shape(tensors, dim, cat_dim_size);

    // 2) ★关键★ 一次性 malloc 一整块"连续"显存来放结果
    //    张量在 PyTorch 里必须满足 strided 连续布局, 不能"东一块西一块拼起来当一个张量用"
    Tensor result = at::empty(out_shape, tensors[0].options());

    // 3) 把每个输入张量逐个 memcpy 到 result 的对应偏移处
    int64_t offset = 0;
    for (const Tensor& t : tensors) {
        result.narrow(dim, offset, t.size(dim)).copy_(t);  // ← 这就是那条 Memcpy DtoD
        offset += t.size(dim);
    }
    return result;
}
```

关键在第 2 步。**为什么不能"按需精确、零浪费"地拼?**

1. **张量的内存模型要求连续**:PyTorch 的 Tensor 本质是"一个连续内存块 + 一组 stride(步长)"。GPU kernel 靠 stride 用 `地址 = 基址 + i*stride` 这种**等差公式**寻址,才能让成千上万线程并行、合并访存。如果 KV cache 是"旧块在地址 A,新 token 在地址 B"两段分离的内存,kernel 没法用一个公式遍历,连续性被破坏。所以 cat 必须**新开一整块连续内存**,把两段都搬进去。

2. **显存的"多余/浪费"来自分配器,不是 cat 本身**:你在 nsys/torch.profiler 的 memory 里看到的显存常常**比理论需要多**,根源是 PyTorch 的 **CUDACachingAllocator(显存缓存分配器)**。`cudaMalloc` 向驱动直接要显存极慢(毫秒级,要陷入内核、改页表),所以 PyTorch **不每次都问驱动要**,而是:

```cpp
// 简化自 c10/cuda/CUDACachingAllocator.cpp 的思想
// 申请时, 把请求 size 向上取整到一个"块尺寸档位", 而不是精确 size
size_t round_size(size_t size) {
    if (size < kMinBlockSize) return kMinBlockSize;       // 最小 512B
    // 大于一定阈值后, 按 2MB 的粒度向上对齐 → 这就是"多余空间"的来源!
    return ((size + kRoundLarge - 1) / kRoundLarge) * kRoundLarge;  // kRoundLarge=2MB
}
// 释放时不还给驱动, 而是塞回自己的空闲链表(free list)缓存起来, 下次同档位直接复用
```

   **所以"多余空间"是故意的**:用"按 2MB 档位向上取整 + 缓存复用"换取"几乎不调用昂贵的 cudaMalloc"。代价是显存占用比精确值大一截(内部碎片),收益是分配快几个数量级。这是典型的**空间换时间**工程权衡。

   > **类比**:你不会每次想喝水才去井边打一杯(调 cudaMalloc),而是一次打一大桶(向上取整的大块)放家里,喝完桶留着下次接(free list 缓存)。桶里总有没喝完的水(浪费),但省下了无数趟跑井边的时间。

3. **这和 AMK / PagedAttention 的关系**:正因为 cat 每步都要"重分配+整块拷贝",KV cache 才会随序列变长越来越慢、越来越占显存。vLLM 的 **PagedAttention** 用"分页(把 KV 切成固定大小的页,像操作系统虚拟内存一样按页存,不要求物理连续)"来避免这次拷贝;AMK 则在 megakernel 内部统一管理这些访存。**你在 nsys 里亲眼看到这条 memcpy 越来越宽,就是这些优化的动机来源。**

---

## 4. 第三级:ncu —— 单 kernel 级,"这个 kernel 跑得够不够满"

### 4.1 是什么

**ncu** 全称 **Nsight Compute**,是**单个 CUDA kernel 的显微镜**。当 nsys 告诉你"`k_attention` 这个 kernel 执行时间本身就很长(不是 gap 的问题)",你才用 ncu 钻进去,读它的**硬件性能计数器**,回答:它把 SM 用满了吗?访存带宽打到峰值的百分之几?寄存器/shared memory 是不是限制了并行度?

### 4.2 两个必懂的核心指标

**(1) occupancy(占用率)**

> **occupancy(占用率)= 一个 SM 上实际活跃的 warp 数 ÷ 该 SM 理论能容纳的最大 warp 数。**

先解释 **warp(线程束)**:GPU 上 32 个线程被绑成一束,作为最小调度单位一起执行同一条指令(SIMT)。一个 **SM(Streaming Multiprocessor,流多处理器)** 是 GPU 上的一个"计算核心车间",能同时驻留很多 warp,并在某些 warp 等数据(访存延迟)时,**立刻切换到另一个 warp 干活来掩盖延迟**。

occupancy 高的意义:有足够多的 warp 在排队,GPU 总有活干、能把访存延迟藏起来。occupancy 低,说明 warp 不够,一旦几个 warp 都在等访存,SM 就闲置。

> **类比**:occupancy 像一个客服(SM)同时挂着多少个聊天窗口(warp)。挂得多,这个客户在打字(等访存)时,客服就切去回别的客户,手不停。只挂一两个窗口,客户一打字客服就发呆。

occupancy 被什么限制?**每个线程用的寄存器太多、或每个 block 用的 shared memory 太多**,都会让一个 SM 装不下更多 warp。ncu 会直接告诉你是哪一项卡住了 occupancy。

**(2) memory throughput(访存吞吐)**

> 这个 kernel 实际达到的显存带宽,占 H100 峰值带宽(HBM3,约 3.35 TB/s)的百分之几。

对 Day1 判定为 memory-bound 的 decode kernel,**理想情况这个值应该接近 100%**——说明它确实在拼命搬数据,瓶颈就是带宽,优化到头了。如果它**远低于 100%**(比如只有 30%),说明这个 kernel **连带宽都没吃满**,问题另有其因(比如访存没合并、occupancy 太低导致延迟没藏住),还有优化空间。这正是 ncu 的价值:**把"理论该 memory-bound"和"实际有没有真打满带宽"对上账**。

### 4.3 为什么 ncu 慢得离谱?—— kernel replay(核函数重放)

这是 ncu 最反直觉的一点:测**一个** kernel 可能要几十秒,比它实际执行慢上千倍。

原因:GPU 上**硬件性能计数器的物理寄存器数量有限**,一次 kernel 运行只能采集一小部分指标。但 ncu `--set full` 要采集**几百个指标**(occupancy、各级 cache 命中、各种访存吞吐……),一次根本测不完。

ncu 的办法是 **kernel replay(核函数重放)**:它把这个 kernel 的输入状态**保存下来,反复运行同一个 kernel 几十遍(每遍叫一个 pass)**,每一遍采集一批不同的计数器,最后拼出完整画像。

> **类比**:你想给一台运转的机器拍 X 光、CT、核磁、超声……但每种设备一次只能拍一项,而且互相干扰。于是你让机器**把同一个动作一模一样重复几十次**,每次只用一种设备拍一项,最后把所有片子拼成完整诊断。重复几十遍,自然就慢了几十倍。

**这条原理直接决定了工作流铁律**:ncu 绝不能拿来扫整个程序,只能在前两级精确定位到"就是这一个 kernel 算得慢"之后,**点名解剖它**。

### 4.4 怎么用

```bash
# ============================================================
# 环境: Nsight Compute(ncu), 通常需要 sudo 或开启 GPU 计数器权限
# 关键: 一定要用 -k / -s 限定只测目标 kernel, 否则它会去 replay 每一个 kernel, 慢到天荒地老
# ============================================================
ncu \
  --set full \                       # 采集完整指标集(会触发多次 replay, 最慢但最全)
  -k "regex:.*gemv.*|.*attention.*" \  # 只测名字匹配的 kernel(强烈建议先用 nsys 拿到准确 kernel 名)
  -c 5 \                             # 最多测 5 个 launch 就停, 避免重复测稳态的同一个 kernel
  --launch-skip 10 \                 # 跳过前 10 次 launch(冷启动), 测稳态
  -o ncu_decode_report \             # 输出 .ncu-rep, 用 Nsight Compute GUI 打开
  python sample_kv.py --max_new_tokens=20

# 只要两个关键指标、不要 full(快很多, 日常体检够用):
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
  -k "regex:.*gemv.*" -c 3 \
  python sample_kv.py
```

读结果重点:`Achieved Occupancy`(实际占用率)、`Memory Throughput`(% of peak)、以及 ncu 自动给的 **"bottleneck" 提示**(它会直接写"This kernel is memory bound"之类)。

---

## 5. 平移到 AMK:产出小米课题第一份原始数据(~1h)

今天的工具学会后,**最有价值的产出是把同一套 nsys 命令原封不动用到 AMK 上**,因为这是你独特的"H100 真实硬件 profiling"价值点(见项目记忆 [[project_amk_h100]])。

**做法:**

1. 找到 AMK 跑 Llama-3.1-8B decode 的入口脚本,在它的 decode 循环里插同样的 NVTX 标记(`decode_step_k`)。
2. 跑**完全相同**的 nsys 命令:
   ```bash
   nsys profile -o amk_decode --trace=cuda,nvtx,osrt --force-overwrite=true \
     python run_amk_decode.py --tokens=8
   nsys stats --report cuda_gpu_kern_sum amk_decode.nsys-rep
   nsys stats --report cuda_gpu_trace amk_decode.nsys-rep
   ```
3. 记录与 nanoGPT **完全对应**的三个指标,做成对照表:

| 指标 | nanoGPT(baseline) | AMK | 说明 |
|---|---|---|---|
| decode 一步 kernel 数 | (你数出的 N,几十) | (预期个位数,甚至 1) | megakernel 把多 kernel 融成一个 |
| gap 占比 | (30%~60%?) | (预期大幅下降) | launch 间隙被消除 |
| launch 相关开销占比 | (不可忽略) | (预期趋近 0) | 一次 launch 顶过去几十次 |
| 时间线结构 | 几十个小块 + 大量空白 | 少数大块,几乎无空白 | 这就是 Before/After 两张图 |

> **这张对照表 + 两份 nsys trace,就是 AMK report 的原始数据底座。** 不需要现在就解释"为什么 AMK 更快"(那是后面的事),今天只要**如实记录两边的时间线结构**,把原始数据攒下来。这是真实硬件实测,不是理论推演,正是你区别于师姐工作的独特贡献。

---

## 6. 工业实践、常见陷阱与最佳实践

### 6.1 行业里真实的工作流

字节、英伟达、vLLM 团队做推理优化,标准三板斧就是今天这条链:

1. **先 nsys 看全局**(很多团队甚至跳过 torch.profiler,直接 nsys + NVTX),一眼定位"是 launch-bound 还是 compute/memory-bound"。
2. **launch-bound → 上 CUDA Graph 或算子融合**(把多次 launch 合并),AMK 这类 megakernel 是这条路线的极致。
3. **某 kernel 确实慢 → ncu 解剖**,看是 occupancy 不够还是访存没合并,再去改 kernel(或换 FlashAttention 这类已优化实现)。

### 6.2 CUDA Graph:对付 launch overhead 的工业利器(理解动机即可)

> **CUDA Graph** = 把"一连串 kernel launch"这个固定序列**录制一次**,之后用**一次** graph launch 就重放整串,把几十次 CPU→GPU 的 launch 开销压成一次。

它和 AMK 是同一个问题的两种答案:CUDA Graph 是"少发几次命令",AMK megakernel 是"干脆只写一个 kernel,连命令都只发一次"。两者都直击 §3.3 的 launch-bound 痛点。你在 nsys 里看到的"几十个 kernel + 一堆 gap",就是它们存在的理由。

### 6.3 常见陷阱清单

1. **忘了热身**:首次 kernel 包含 JIT/选 kernel/建显存池,慢 10 倍。三级工具都要先跑几步再测。
2. **测了太多 token**:nsys 文件会爆炸(几个 GB),GUI 都打不开。decode 测 **4~8 步**足够看清稳态结构。
3. **拿 ncu 扫全程**:replay 机制会让它慢到不可用,**必须 `-k` 限定 kernel、`-c` 限定次数**。
4. **没用 NVTX 就看 nsys**:几十个 kernel 糊成一片,分不清步与步的边界。**插桩是 nsys 可读性的前提**。
5. **GPU 计数器权限**:ncu 在共享/云 H100 上常因权限读不到计数器,报 `ERR_NVGPUCTRPERM`。需要管理员加 `nvidia-modprobe` 权限或用 root。
6. **多卡/多进程干扰**:profiler 默认抓整机,确认只在目标卡、目标进程上测,否则数据被别的负载污染。

### 6.4 一句话总结三级关系

> **torch.profiler 告诉你"病在哪个器官"(哪个 op),nsys 告诉你"器官之间怎么配合出了问题、谁在空等"(时间线/gap/launch),ncu 告诉你"这个器官内部的血管堵在哪"(单 kernel 占用率/带宽)。从粗到细,逐级放大,绝不越级。**

---

## 7. 今日产出清单(对齐计划)

- [ ] `tech_notes/profiling_toolchain.md`(本笔记即其正文,整理进项目仓)
- [ ] nanoGPT `generate_kv` 的 nsys trace(`nanogpt_decode.nsys-rep`)+ 记下的三指标:**kernel 数 N / gap 占比 / launch 占比**
- [ ] 一句话验证结论:W6 §8.3 关于"decode launch-bound"的预测是否成立(用你的实测数字支撑)
- [ ] **AMK nsys 初步记录**(`amk_decode.nsys-rep`)+ §5 那张 nanoGPT vs AMK 对照表 —— 这是 AMK report 的原始数据
- [ ] (可选)对 nanoGPT 最贵的那个 decode kernel 跑一次 ncu,记下 occupancy 和 memory throughput,看它是否真打满了 H100 带宽

---

## 附:三个工具速查命令卡

```bash
# ---- torch.profiler(算子级,代码内) ----
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=2, active=3),
             record_shapes=True, profile_memory=True) as prof:
    for _ in range(6): step(); prof.step()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# ---- nsys(时间线级,命令行) ----
nsys profile -o out --trace=cuda,nvtx,osrt --force-overwrite=true python run.py
nsys stats --report cuda_gpu_kern_sum out.nsys-rep   # kernel 汇总
nsys stats --report cuda_gpu_trace   out.nsys-rep    # 逐 kernel 轨迹(数 gap)
nsys stats --report cuda_api_sum     out.nsys-rep    # API 汇总(看 launch)

# ---- ncu(单 kernel 级,命令行,务必限定 kernel) ----
ncu --set full -k "regex:目标kernel名" -c 5 --launch-skip 10 -o ncu_out python run.py
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed -k "regex:..." -c 3 python run.py
```

> 下一步(Day3+)预告:有了这份"哪里慢、慢多少"的实测,就能名正言顺地选优化手段——算子融合 / CUDA Graph / FlashAttention / PagedAttention,逐个对症下药,并用同一套 nsys 命令量出"优化前后"的提升幅度(小米课题要求 ≥20%)。
