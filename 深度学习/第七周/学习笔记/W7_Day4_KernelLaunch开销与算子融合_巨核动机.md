# W7 Day4 · Kernel Launch 开销与"为什么要融合":巨核动机实测

> **本笔记的唯一目标**:让你从"听说融合能加速"升级到**能用本机实测的数字、并深挖到 CPU/driver 底层代码**,讲清楚三件事——(1) 启动一个 GPU kernel,CPU 到底忙了哪几步、为什么这笔开销**和算多少无关**、大概值多少钱;(2) 把 N 个小算子融成 1 个大核,到底省了**哪两笔账**(省启动 + 省访存),每笔账在硬件上差多少个数量级;(3) `megakernel`(巨核)和 `CUDA Graph`(CUDA 图)这两条"少发命令"的路线,为什么是你小米课题主线 2/3 的**动机地基**。读完你要能指着自己 RTX 5060 上跑出的 `8.56 µs/launch`、`融合 1.61×`、`CUDA Graph 7.98×` 三组数字,把"巨核要省什么"讲成有数、有图、有底层原理的结论。
>
> **串联**:这是 [W7 学习计划](./W7_学习计划_AI_Infra主线.md) **Day4**,直接对接小米课题**主线 3(巨核算子生成)+ 主线 2(图级优化)**。承接 [W7 Day2 三级 Profiler](./W7_Day2_三级Profiler工具链_torchprofiler_nsys_ncu.md)——Day2 你在 nsys 时间线上**看见**了"decode 一步几十个 kernel + 一堆 gap",但只知道 gap 大概是 launch 引起的;**今天把这些 gap 拆到底层、量成数字**。也承接 [W7 Day1 Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md)——Roofline 说 decode 是 memory-bound,今天补上"除了访存,还有第三种 bound:launch-bound(启动受限)",它是 Roofline 那张纸**根本画不出来**的。向下接 Day5 `torch.compile`(自动做今天讲的融合)。
>
> **产出对齐**:本笔记正文即计划要求的 `tech_notes/kernel_launch_and_fusion.md`(仓库里叫这个名;桌面按 `W7_DayX` 惯例平铺)。配套实测脚本 `bench_launch_fusion.py` + 三组 launch/fusion 数据。

---

## 0. 开篇:读完你要能不看资料答出来的问题

1. 启动一个 CUDA kernel,CPU 侧到底做了哪 4 件事?为什么这个开销是**固定**的、和"这个 kernel 要算 1 个数还是 10 亿个数"无关?
2. 为什么 **decode(逐 token 生成)** 阶段特别怕 launch 开销,而 **训练 / prefill** 几乎感受不到?(提示:M=1)
3. 把 N 个小算子融成 1 个大核,到底省了**哪两样东西**?哪一样是 Day2 没讲过的?
4. "中间结果落 HBM"为什么是浪费?寄存器 / shared memory / HBM 三者延迟差几个数量级?
5. **megakernel(巨核)** 和 **CUDA Graph** 都在对付 launch 开销,它们的**根本区别**是什么?各自的代价是什么?
6. 为什么说今天量化的这两笔账,**就是你小米课题主线 3 全部价值的地基**?

> 第 1、3、5 题是灵魂。如果你能把第 1 题从"CPU 要提交任务"讲到"参数打包进一块 buffer → 写命令进 stream 的环形队列 → 敲 doorbell 寄存器通知 GPU",并说清"这条软件路径的长度和 kernel 算多少字节完全无关",这一天就值了。

---

## 1. 问题背景:Roofline / Profiler 都指向了同一个"看不见的敌人"

先把前两天的结论接上,你才知道今天在补哪块拼图。

- **Day1(Roofline)** 说:nanoGPT decode 算术强度 ≈ 1,是 **memory-bound**。潜台词是"瓶颈在**搬数据**,不在算"。
- **Day2(nsys)** 你亲眼看到:decode 一步不是一个大 kernel,而是**几十个小 kernel**,而且 kernel 与 kernel 之间**有大片空白(gap)**。GPU 在这些 gap 里**啥也没干,在干等**。

这里藏着一个 Roofline **根本表达不了**的东西。Roofline 那张图的横轴是算术强度、纵轴是算力,它默认"kernel 一个接一个、严丝合缝地跑"。可现实里,那些 gap 意味着 GPU 有大量时间**既不在算、也不在搬数据,而是在等 CPU 把下一条命令发过来**。

这就引出了第三种瓶颈,和 compute-bound / memory-bound 并列:

> **launch-bound(启动受限)**:程序的瓶颈既不是算力、也不是访存带宽,而是 **CPU 发射 kernel 的速度跟不上 GPU 消费的速度**。GPU 算得飞快(一个小 kernel 几微秒就干完),但 CPU 准备并提交下一个 kernel 要花好几微秒,于是 GPU 频繁空等——**忙的是 CPU,闲的是那块最贵的 GPU**。

**类比(先建直觉,§2 立刻拆底层代码)**:decode 好比一个**顶级大厨(GPU)** 在后厨等菜。可点菜、写小票、把料备齐递进后厨的**服务员(CPU)** 只有一个,而且每道菜(kernel)不管分量多小,服务员走这一趟流程都要固定花 8 秒。大厨颠个勺 2 秒就做完一道,然后**站着等服务员跑下一趟**。你看到的"出菜慢",不是大厨手慢,是**服务员这条腿跟不上**。这就是 launch-bound。

**为什么偏偏 decode 中招,训练 / prefill 没事?** 关键在一个数:**M(一次处理多少 token)**。

| 阶段 | 一次处理 token 数 M | 每个 kernel 的计算量 | launch 开销占比 | 结论 |
|---|---|---|---|---|
| **训练 / prefill** | 几百~几千(整段序列并行) | **大**(大矩阵乘,kernel 要跑几十~几百 µs) | 8µs / 200µs ≈ **忽略不计** | 计算把启动开销"稀释"了 |
| **decode** | **1**(自回归,一次一个) | **极小**(GEMV,kernel 可能只跑 2~5 µs) | 8µs / (8µs+3µs) ≈ **一半以上** | 启动开销吃掉大头 → launch-bound |

一句话:**decode 把每个 kernel 的"分子(计算量)"压到了极小,而 launch 这个"分母里的固定成本"没变,于是占比爆炸。** 这正是 Day1 Roofline 里"decode 算术强度 ≈ 1"从另一个角度看到的同一件事——每个 kernel 干的活太少了。

---

## 2. 底层:一次 kernel launch,CPU 到底忙了哪 4 步?

这一节是全篇地基,也是你要求的"要内核不要表面"。我们不停在"CPU 提交任务"这种话上,直接下沉到 CUDA driver 这一层看它做了什么。

### 2.1 是什么:launch 是一条"软件路径",不是一次"硬件计算"

**kernel launch(核函数启动)** 指的是:你在 CPU 侧调用 `kernel<<<grid, block>>>(args)`(或 PyTorch 底层的 `cudaLaunchKernel`),把"让 GPU 跑这个函数"这件事**提交出去**的全过程。

最反直觉、也最关键的一点:**launch 的开销全部发生在 CPU 和 driver 这条软件路径上,GPU 此时基本没参与。** 所以它和"这个 kernel 要处理多少数据"**完全解耦**——你启动一个只加 16 个数的 kernel,和启动一个加 16 亿个数的 kernel,CPU 走的这条提交流程**一模一样长**。这就是"固定开销"四个字的由来。

### 2.2 拆成 4 步(对应知识点里的"参数打包→driver 提交 stream 队列→配 grid/block→同步元数据")

我把这条路径拆成 4 步,每步说清"CPU 在干嘛、为什么躲不掉":

**① 参数打包(argument marshalling)**
把 kernel 需要的所有参数——指针(张量的显存地址)、标量(维度大小、缩放系数)、配置——按 GPU 的 ABI(应用二进制接口)布局,拷进一块**参数缓冲区(parameter buffer)**。为什么躲不掉:GPU 不能直接读 CPU 栈上的变量,参数必须被序列化成 GPU 能取的格式。参数越多、越大,这步越久(但通常是几百 ns 级)。

**② 配置 grid / block(launch configuration)**
计算并校验启动配置:开多少个 **block(线程块)**、每个 block 多少 **thread(线程)**、需要多少 **shared memory(共享内存)**。为什么躲不掉:GPU 是"给我一个网格,我按网格铺线程"的执行模型,你不告诉它网格形状,它不知道要唤起多少线程。

**③ 提交进 stream 队列(enqueue to stream)**
driver 把"启动这个 kernel + 它的参数 + 配置"打包成一个**命令(command)**,写进该 **CUDA stream(CUDA 流,一条 FIFO 命令队列,见 Day2 §3.2)** 对应的**环形缓冲区(ring buffer / command buffer)**——这块 buffer 是 CPU 和 GPU **共享**的一段内存。为什么躲不掉:CPU 和 GPU 异步,必须通过这个队列传递命令,CPU 写、GPU 读。

**④ 通知 GPU + 维护同步元数据(doorbell + bookkeeping)**
CPU 更新队列写指针,并通过写一个特殊的 **doorbell(门铃)寄存器**告诉 GPU "队列里有新活了";同时 driver 要更新一堆记账元数据(这个命令的序号、依赖、给事件/同步用的追踪信息)。为什么躲不掉:GPU 不会主动轮询,要 CPU"敲门铃"唤醒它去取命令;而记账是后续 `cudaStreamSynchronize`、event 等待、错误归属能工作的前提。

**这 4 步加起来,就是那个"和算多少无关"的固定开销**,量级 ~**5–20 µs**(不同 CPU/驱动/操作系统差异不小,本机 §4 实测 ≈ 8.56 µs)。Windows 上因为 WDDM 驱动模型有额外的批处理/调度层,通常比 Linux 更贵。

### 2.3 底层代码:launch 到底是"写内存 + 敲门铃",不是"调用一个函数"

光说抽象,不如看简化到骨架的伪 C 代码。真实的 CUDA driver 是闭源的,但公开的 GPU 驱动(如 NVIDIA 开源内核模块、Mesa/nouveau、以及各类 ISA 文档)都印证了这套"环形队列 + doorbell"机制。下面是**提炼出核心结构**的示意实现:

```c
/* ============================================================
 * 简化伪代码: 一次 kernel launch 在 driver 侧的骨架
 * 目的: 让你看清"launch = 往共享队列写命令 + 敲门铃寄存器",
 *       而不是"CPU 亲自跑一遍 kernel"。真实 driver 更复杂,
 *       但这三件事(打包/入队/敲门铃)是所有 GPU 的共性。
 * ============================================================ */

// GPU 和 CPU 共享的一段内存, 组织成环形队列(ring buffer)
typedef struct {
    uint32_t *ring;        // 命令环形缓冲区(位于 pinned/mapped 内存, GPU 可直接读)
    uint32_t  size;        // 环大小
    uint32_t  write_idx;   // CPU 维护的写指针
    volatile uint32_t *doorbell;  // 映射到 GPU 的 doorbell 寄存器(MMIO)
} GpuQueue;

// —— 这就是 cudaLaunchKernel 底层做的事(高度简化) ——
void launch_kernel(GpuQueue *q,
                   void *kernel_func,      // GPU 上 kernel 的入口地址
                   dim3 grid, dim3 block,  // ② 配 grid/block
                   size_t shmem,
                   void **args, int n_args) // kernel 参数列表
{
    // ① 参数打包: 把每个参数按 ABI 布局拷进一块参数区
    //    注意: 这里在 memcpy, 是纯 CPU 的内存搬运, 和"算什么"无关
    uint8_t param_blob[MAX_PARAM_SIZE];
    size_t off = 0;
    for (int i = 0; i < n_args; i++) {
        memcpy(param_blob + off, args[i], arg_size(args[i]));
        off += arg_size(args[i]);
    }

    // ③ 组一条 launch 命令, 写进 stream 的环形队列
    //    命令里带上: kernel 地址、grid/block 配置、shmem、参数区指针
    uint32_t *slot = &q->ring[q->write_idx % q->size];
    encode_launch_command(slot, kernel_func, grid, block, shmem, param_blob, off);
    // ↑ encode 也是 CPU 在填结构体字段, 依然与 kernel 计算量无关

    // 内存屏障: 确保命令内容对 GPU 可见后, 再更新写指针(顺序不能反)
    memory_barrier();
    q->write_idx++;

    // ④ 敲 doorbell: 一次 MMIO 写, 通知 GPU "队列有新命令, 来取"
    //    这一步是 CPU→GPU 唯一真正的"通知", GPU 收到才去 fetch 命令
    *(q->doorbell) = q->write_idx;

    // 函数到此返回! CPU 不等 GPU 执行, 立刻回到 Python 发下一条。
    // —— 这就是 CUDA "异步" 的本质, 也是 Day2 说的 "CPU 早早返回"。
}
```

**读这段代码要抓住的三个点(比记 API 重要):**

1. **全程没有一行在"算 kernel 的内容"**。CPU 干的是 `memcpy`(打包)、填结构体(编码命令)、`++`(更新指针)、一次 MMIO 写(敲门铃)。**所以开销是固定的**——这条路径长度只取决于"参数有几个",与"每个线程要算多少浮点"零关系。这就是 §2.1 那句"launch 和算多少解耦"的代码级证据。
2. **函数最后直接 `return`,不等 GPU**。这就是"异步"和"CPU 早早返回"(Day2 §2.2 你测 `time.time()` 得到接近 0 的原因)的底层来源:CPU 把命令扔进队列、敲完门铃,活就交出去了。
3. **launch 慢,慢在这条软件路径本身**:memcpy、加锁(driver 内部要保护共享队列)、系统调用/驱动切换(尤其 Windows WDDM)。你没法让它变快,只能——**少走几趟**。这就直接推出了下一节的两笔账。

> **一个能立刻验证的推论**:既然 launch 开销与计算量无关,那么"启动 50 个只加 16 个数的 kernel"应该几乎全是启动开销、总时间≈50×固定开销。§4 实验 1 就是这么设计的,实测 50 个小 add 花 427µs、单个花 8µs,反推单次 launch ≈ 8.56µs——和这里的理论完全对上。

---

## 3. 核心:算子融合(operator fusion)省的两笔账

上一节我们锁定了敌人:launch 太贵、还老让 GPU 空等。**算子融合**就是对症的药。

**算子融合(operator fusion,把多个小算子合并成一个大 kernel)是什么**:原本要用 N 个独立 kernel 依次完成的一串运算,改写成**一个 kernel** 内部把这 N 步一口气做完。比如 `relu(x @ w + b)` 本来是 matmul、加 bias、relu 三个 kernel,融合后一个 kernel 从头算到尾。

它省的东西,可以清清楚楚记成**两笔账**——这两笔账,就是你小米课题主线 3(巨核)的全部动机来源,务必记死。

### 3.1 第一笔账:省启动(launch 次数 N → 1)

最直接的一笔。原来 N 个算子 = N 次 launch = N × ~8µs 的固定开销;融合成 1 个 kernel = 1 次 launch。

对 decode 的杀伤力有多大?回到知识点里的数:**decode 一步启动 40 个 kernel,光 launch 就 ≈ 40 × 8µs = 320µs**(落在"~200–800µs"区间)。如果把这 40 个融成 1 个,launch 从 320µs 直接砍到 8µs。**在 launch-bound 的 decode 场景,这一笔账省下的时间可能就是端到端延迟的大头。**

同时它还顺带消灭了 gap:kernel 少了,kernel 之间的"等 CPU 发下一条"的空白自然也少了(Day2 nsys 时间线上那些空隙)。

### 3.2 第二笔账:省访存(中间结果不落 HBM)

这一笔账 Day2 没细讲,是今天的**新增重点**,也是很多人忽略、但在 memory-bound 场景往往**比第一笔更值钱**的一笔。

先建立那个决定性的**速度差数量级**——这是理解一切的锚:

| 存储层级 | 中文 | 一次访问延迟(量级) | 类比 |
|---|---|---|---|
| register / shared memory | 寄存器 / 片上共享内存 | **~几 ns** | 手边摊开的草稿纸,伸手就拿 |
| L2 cache | 二级缓存 | ~几十 ns | 抽屉里的资料 |
| **HBM(High Bandwidth Memory,显存)** | 高带宽显存 | **~几百 ns** | 得起身走到隔壁档案室去取 |

**片上(on-chip)和 HBM 差了大约两个数量级(几 ns vs 几百 ns)。** 记住这个"100 倍"的差距,下面全靠它。

**未融合时发生了什么(问题)**:GPU 的计算单元(ALU)**只能直接读片上的寄存器**,不能直接对 HBM 里的数做运算。所以每个独立 kernel 的宿命是:

```
从 HBM 把输入读进片上寄存器  →  算  →  把结果写回 HBM
```

于是一串 N 个算子,中间结果就要**反复地"写回 HBM、又读回来"**:

```
算子1: 读A(HBM) → 算 → 写tmp1(HBM)
算子2: 读tmp1(HBM) → 算 → 写tmp2(HBM)      ← tmp1 刚写回又立刻读出来, 这趟 HBM 往返纯属浪费
算子3: 读tmp2(HBM) → 算 → 写C(HBM)
```

那些 `tmp` 是**中间结果**,它的唯一用途就是喂给下一个算子。可未融合时,它却要老老实实"下楼走到档案室存一趟、下一个算子再走过去取一趟"。**每一趟 HBM 往返都是几百 ns 的延迟 + 实打实占用宝贵的 HBM 带宽**——而带宽正是 decode 这种 memory-bound 场景最紧缺的资源(Day1 结论)。

**融合后怎么省(解法)**:一个 kernel 内部把 N 步连着做完,中间结果 `tmp` **一直待在片上寄存器/shared memory 里,压根不写回 HBM**:

```
融合kernel: 读A(HBM一次) → 算1 → 算2 → 算3(中间 tmp 全在寄存器, ~几ns) → 写C(HBM一次)
```

HBM 访问从"读A + 写tmp1 + 读tmp1 + 写tmp2 + 读tmp2 + 写C"(6 次)压成"读A + 写C"(2 次)。**省下的中间张量 HBM 往返,在 memory-bound 场景直接兑换成延迟下降。**

> **底层直觉(为什么中间结果能"不落地")**:kernel 里的局部变量默认就分配在**寄存器**里。未融合时,`tmp1` 是一个独立 kernel 的**输出张量**,它必须有个 HBM 地址好让下一个 kernel 找到它;融合后,`tmp1` 降级成了**同一个 kernel 内的一个局部变量**,活在寄存器里,算完即用、用完即弃,**从来不需要一个 HBM 地址**。融合的魔法,本质就是"把'跨 kernel 传递的张量'降级成'kernel 内的局部变量'"。

### 3.3 两笔账的代码对照:一眼看穿"落地"与"不落地"

```python
# 环境: PyTorch>=2.0 + CUDA。演示 out = a*b + c 这一串。
import torch
a = torch.randn(8_000_000, device="cuda")
b = torch.randn(8_000_000, device="cuda")
c = torch.randn(8_000_000, device="cuda")

# —— 未融合: 2 个 kernel, 中间 tmp 落 HBM ——
def unfused(a, b, c):
    tmp = a * b        # kernel1: 读a,读b → 算 → 写tmp到HBM  (tmp 是真实存在的显存张量)
    return tmp + c     # kernel2: 读tmp,读c → 算 → 写out到HBM (tmp 又从HBM读回来, 这趟往返是浪费)

# —— 融合: 1 个 kernel(addcmul 是 PyTorch 原生融合算子), tmp 只在寄存器 ——
def fused(a, b, c):
    # addcmul(c, a, b) = c + a*b, 一个 kernel 内部把"乘"和"加"连着做完
    # a*b 的中间结果留在寄存器, 不写回 HBM; 且只 launch 1 次
    return torch.addcmul(c, a, b)  # 为什么用它: 它就是官方替你手写好的融合 kernel
```

`unfused` 里那个 `tmp` 变量,就是**被迫落地 HBM 的中间结果**;`fused` 里它消失了,融进了寄存器。§4 实验 2 实测这一改:**1218µs → 758µs,1.61×**。注意这里两个版本 launch 次数只差 1(2→1),8µs 的启动差可忽略——**1.61× 的加速几乎全来自"省访存"这第二笔账**。这就是为什么说第二笔账在 memory-bound 场景往往更值钱。

### 3.4 收束:这两笔账 = megakernel(巨核)的全部动机

现在把两笔账拼起来,你就理解了小米课题主线 3 到底在追求什么。

**megakernel(巨核)是什么**:把 Transformer 一层里原本几十个算子(Attention 的 QKV 投影 / score / softmax / 加权和 / 输出投影,FFN 的两个大矩阵乘 + 激活,以及各处的 LayerNorm、残差加……)**尽可能融进极少数、甚至单个** GPU kernel 里。极端形态下,一整个 decode step 就一次 launch。

它为什么值得做,答案就是今天这两笔账,而且**两笔在 decode 上同时受益**:

| | 未融合(几十个小 kernel) | megakernel(融成极少数) | 省的是哪笔账 |
|---|---|---|---|
| launch 次数 | 几十次 × 8µs | 1~几次 | 第一笔:省启动(§3.1) |
| 中间张量 HBM 往返 | 每两个算子之间一趟 | 几乎为零(留片上) | 第二笔:省访存(§3.2) |
| GPU gap(空等) | 大量(Day2 nsys 所见) | 趋近于零 | 第一笔的副产品 |

> **一句话锚定课题**:你课题主线 3 写的"把 Attn/FFN/LayerNorm 融成单一大核""显著降低 kernel launch 次数与中间张量/访存",翻译成人话就是——**第一笔账(N→1 次 launch)+ 第二笔账(中间结果不落 HBM)**。今天你把"巨核要省的两样东西"量化成了本机的真实数字(8.56µs/launch、1.61× 访存收益),W8/暑假你自己写巨核时,才有基准去回答"我到底省了多少、够不够 ≥20% 那条验收线"。

---

## 4. CUDA Graph:另一条"少发命令"的路,和巨核什么关系

融合是"把多个 kernel 合成一个"。但有时候你**没法改 kernel 代码**(比如用的是别人的库、cuBLAS/cuDNN 的 kernel 你动不了),又想省 launch 开销,怎么办?**CUDA Graph** 是另一条路。

### 4.1 是什么:把固定的 launch 序列"录一次,重放多次"

**CUDA Graph(CUDA 图)是什么**:decode 每一步启动的那 40 个 kernel,**顺序、配置、依赖关系几乎每步都一模一样**(只有输入数据在变)。CUDA Graph 让你把这一整串 launch **录制(capture)成一张图**,之后每一步只需**一次** `graph.replay()` 就把整串命令重新提交给 GPU——**绕过了每个 kernel 那 4 步 CPU 提交路径中的绝大部分**。

**类比**:未用 Graph,像每天点同样一桌 40 道菜,你每道都要重新跟服务员口述一遍(40 趟)。CUDA Graph 像**把这一桌菜存成一个"套餐编号"**,以后只喊一句"上 3 号套餐"(1 次),厨房就知道要做哪 40 道、什么顺序——省掉了 39 趟口述。

**为什么能省**:回看 §2.3 那 4 步,CUDA Graph 在录制时就把"参数打包、配 grid/block、编码命令"这些**预计算好并缓存起来**;replay 时 GPU 直接按缓存的命令序列执行,CPU 侧几乎不用再逐个走那条昂贵的软件路径。它省的主要是 **CPU 发射开销(第一笔账里的 launch 部分)**,但**不省访存**——中间张量该落 HBM 还是落(这点它不如融合)。

### 4.2 底层直觉:replay 为什么快

```
普通 eager 执行第 t 步 decode:
  CPU: [打包+入队+敲门铃] × 40  ← 每个 kernel 都走一遍 §2.3 的完整路径, 40×8µs
  GPU:  k1 ░ k2 ░ ... k40        ← 一堆 gap(等 CPU)

CUDA Graph replay 第 t 步:
  CPU: [提交整张图] × 1          ← 只敲一次门铃, 命令序列 GPU 侧已缓存
  GPU:  k1 k2 ... k40            ← 几乎无 gap, 一串连着跑
```

### 4.3 代码:录制一张图并重放(可跑,§4.6 实测 7.98×)

```python
# 环境: PyTorch>=2.0 + CUDA。演示把 N 次小 launch 录成图后一次重放。
import torch
x = torch.randn(16, device="cuda")
N = 50

# ★ 录制前必须热身: capture 期间不允许发生一次性初始化(如 cudaMalloc),
#   否则把"分配显存"这种一次性动作也录进图里, 会出错。所以先在别的 stream 跑几遍暖机。
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    y = x
    for _ in range(3):                 # 热身 3 遍
        y = y + 1.0
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):              # ★ 录制: 这个 with 块里的所有 launch 被录进图 g
    y = x
    for _ in range(N):
        y = y + 1.0                    # 这 50 次 launch 的固定序列被记下来

# 之后每次只需 replay, 不再逐个 launch
for _ in range(100):
    g.replay()                         # 一次提交整张图, 省掉 50 次 CPU 发射
torch.cuda.synchronize()
```

### 4.4 CUDA Graph vs megakernel:同一个病,两种药(开篇第 5 题)

这是必须讲清的对比——它俩都治 launch-bound,但根本不同:

| | CUDA Graph | megakernel(算子融合) |
|---|---|---|
| 核心手段 | 把 N 次 launch **录成图、一次重放** | 把 N 个 kernel **改写成 1 个 kernel** |
| 省第一笔账(启动) | ✅ 省(CPU 发射合并成一次) | ✅ 省(本来就只有 1 个 kernel) |
| 省第二笔账(访存) | ❌ **不省**(中间张量照样落 HBM) | ✅ **省**(中间结果留片上) |
| 要不要改 kernel 代码 | **不要**(黑盒 kernel 也能录) | **要**(得亲手把逻辑塞进一个 kernel) |
| 代价 / 限制 | 图是"静态"的:形状/控制流一变就得重录;capture 有诸多限制 | 开发难度高、寄存器/shared memory 有限,融太多会溢出反而变慢 |
| 谁在用 | `torch.compile(mode="reduce-overhead")` 底层就用它(Day5)、vLLM 等 | AMK / FlashAttention / 各家推理引擎的核心竞争力 |

> **一句话**:CUDA Graph 是"**不改代码、少发命令**"的通用便宜药,只治第一笔账;megakernel 是"**重写 kernel、连中间访存一起干掉**"的猛药,两笔账全治,但难写。**你课题主线 3 选的是猛药(巨核),而 Day5 的 `torch.compile` 是自动版的组合拳(既做融合又用 CUDA Graph)——所以它是你巨核必须打败的 baseline。**

---

## 5. 动手实测:本机 RTX 5060 上把三笔账量出来

知识点要求"nsys 量 launch 占比,或 N 个小算子 vs 融合算子对照实验"。本机有 GPU,我直接跑了**可复现的微基准**,把三个数量都量成真实数字。脚本见 `bench_launch_fusion.py`(桌面),下面是设计思路 + 实测结果。

### 5.1 实测环境与结果(2026-07-06 本机)

```
GPU:   NVIDIA GeForce RTX 5060 Laptop GPU
torch: 2.12.0.dev+cu128
计时:  torch.cuda.Event(GPU 硬件时钟), warmup=30, iters=200 取平均
```

| 实验 | 对照 | 实测 | 结论 |
|---|---|---|---|
| **① 单次 launch 固定开销** | 50 个小 add vs 1 个小 add | 427.57µs vs 8.20µs → **8.56 µs/launch** | 落在理论 5–20µs 区间;这就是每个 kernel 的"过路费" |
| **② 融合省访存** | `a*b` 后 `+c`(2 kernel) vs `addcmul`(1 融合 kernel) | 1218µs vs 758µs → **1.61×** | 省的几乎全是中间张量 HBM 往返(第二笔账) |
| **③ CUDA Graph 省启动** | 50 次 eager launch vs 1 次 graph replay | 430.69µs vs 54.00µs → **7.98×** | 省的是 CPU 发射开销(第一笔账),猛 |

### 5.2 三个结果分别验证了什么

- **实验①(8.56µs)**:直接坐实 §2 的理论——launch 是条固定长度的软件路径。50 个只加 16 个数的 add(计算量趋近 0),总时间 427µs 几乎全是启动开销,反推单次 ≈8.56µs。**这就是 decode "40 个 kernel × 8µs ≈ 320µs 纯启动"的本机版证据。**
- **实验②(1.61×)**:验证第二笔账。两个版本 launch 只差 1 次(可忽略),加速几乎全来自"`tmp` 不落 HBM"。张量越大(越 memory-bound),这一笔越值钱。
- **实验③(7.98×)**:验证 CUDA Graph 只治第一笔账但极猛——50 次发射合并成 1 次,把 launch-bound 场景的 CPU 开销几乎清零。**注意它是 §5 三个实验里加速最夸张的,恰恰因为实验用的是"纯 launch-bound"负载(小到没有计算、没有访存,瓶颈 100% 是发射)。**

### 5.3 进阶(有 H100 时做):nsys 量 nanoGPT decode 的 launch 占比

本机微基准证明了"单笔账值多少钱";要量"真实模型里 launch 占端到端多少比例",按 Day2 学的 nsys 流程做(H100 上):

```bash
# 承接 Day2 §3.4。在 nanoGPT generate_kv 的 decode 循环里已插好 NVTX 标记后:
nsys profile -o nanogpt_decode --trace=cuda,nvtx,osrt --force-overwrite=true \
  python sample_kv.py --max_new_tokens=8
# 数三个数(取稳态中间一步 decode_step_k):
nsys stats --report cuda_gpu_trace nanogpt_decode.nsys-rep   # 数该步 kernel 数 N
nsys stats --report cuda_api_sum   nanogpt_decode.nsys-rep   # 看 cudaLaunchKernel 总耗时
# launch 占比 ≈ cudaLaunchKernel总耗时 / 该步墙钟时间; 或 N×8µs / 墙钟
```

预期:decode 一步几十个 kernel,launch 相关占比不可忽略(Day2 预测的 30%~60% gap)。**这一步做完,你就能在 AMK report 里写"nanoGPT decode 有 N 个 kernel、launch 占比 X%,这就是巨核要消灭的量"。**

### 5.4 关联你手上的真实 H100 数据(AMK Llama-3.1-8B)

你桌面那份 AMK 编译报告(`...Llama-3.1-8B-Instruct.h100.report.md`)已经有真实数字可以对号入座:

- 报告里 **tasks: 5826**——这就是"融合前的算子/任务数量级"。一个 8B 模型 decode 一步展开成几千个 task,若每个都独立 launch,光启动开销就是天文数字。这正是 megakernel 存在的理由:**把 5826 个 task 编排/融合进极少数 kernel。**
- region breakdown:**attention 364µs / mlp 251µs**(共 ~616µs / 719.8µs 每 token)——attention 和 mlp 是大头,也正是融合的主战场(Attn 内部一堆小算子、FFN 两个大矩阵乘 + 激活)。
- 报告标注 latency 是 **cost-model 预测、GPU 路径未完全接通**——所以你 Day2 学的 nsys **真实硬件测量**才有独特价值:把预测数换成实测数,正是你区别于师姐工作的贡献点。

---

## 6. 工业实践、常见陷阱与最佳实践

### 6.1 行业里怎么用这三招

推理引擎(vLLM / TensorRT-LLM / SGLang 等)对付 launch-bound 的组合拳,基本就是今天这三层,按"性价比从高到低"上:

1. **先上 CUDA Graph**(最便宜,不改 kernel):把 decode 的固定 launch 序列录成图。vLLM 的 `enforce_eager=False` 默认就开 CUDA Graph;`torch.compile(mode="reduce-overhead")` 也是它。**投入最小、收益立竿见影**,是第一道防线。
2. **用现成的融合 kernel**:attention 直接换 **FlashAttention**(它本身就是把 score/softmax/加权和融进一个 kernel,两笔账全省);RMSNorm、RoPE、SwiGLU 等用各家写好的融合算子。
3. **手写 megakernel**(最贵、最强):当通用融合还不够、要榨干特定模型/硬件时,才自己写巨核。**这正是你小米课题主线 3 的定位**——在 torch.compile 的自动融合(baseline)之上,靠手写巨核再抠出 ≥20%。

### 6.2 常见陷阱清单

1. **用 `time.time()` 测 GPU**:CUDA 异步,CPU 侧计时测到的是"塞队列"的时间(接近 0),不是真实执行时间。**必须用 `torch.cuda.Event` 或 `torch.cuda.synchronize()` 后再计时**(脚本里 `cuda_time_ms` 就是范例)。
2. **不热身就测**:首次 kernel 含 JIT 编译 / cuBLAS 选 kernel / 显存池初始化,慢 10 倍以上,污染结果。三个实验都 `warmup=30`。
3. **融合不是越多越好**:一个 kernel 的寄存器 / shared memory 有硬上限。融太多中间变量,寄存器不够会**溢出(register spilling)到本地内存(其实在 HBM)**,反而更慢;或 occupancy 暴跌(Day2 §4.2)。巨核难写,难就难在这个平衡。
4. **CUDA Graph 的静态假设被打破**:图录的是固定形状/固定控制流。序列长度变了、batch 变了、有依赖数据的 `if`,图就失效要重录。decode 常用"按 batch/长度分桶,每桶录一张图"来应对。
5. **Windows 上没有 Triton**:本机 `torch.compile` 因缺 Triton 报 `TritonMissing`(实测踩到了),所以实验 2 我改用原生融合算子 `addcmul` 演示——**这也是个真实工业教训:Windows 做 kernel 级优化生态不全,严肃的推理优化都在 Linux + H100 上做**(正是你 AMK 环境)。
6. **CPU 太弱也会 launch-bound**:同一个模型,慢 CPU 发射跟不上快 GPU,gap 更大。所以"换更强的 GPU 却没提速"有时根源在 CPU 发射/Python 开销——这也是 CUDA Graph 特别香的场景。

### 6.3 一句话总结

> **launch-bound 的病根是"CPU 发命令太慢、GPU 空等"。三味药:CUDA Graph(少发命令,不改代码,只省启动)< 现成融合 kernel(FlashAttention 等,两笔账都省)< 手写 megakernel(终极形态,你课题的主攻)。而这一切的度量衡,就是今天量出的两笔账:每次 launch ~8µs、每次中间张量落 HBM ~几百 ns。**

---

## 7. 自测题(先合上笔记答,再翻对应节核对)

1. 一次 kernel launch,CPU 侧的 4 步分别是什么?为什么这笔开销和"kernel 算多少数据"无关?→ §2.2 / §2.3
2. 为什么 decode 特别怕 launch 开销,prefill/训练不怕?用 M 和"占比"解释。→ §1
3. 算子融合省的两笔账分别是什么?哪一笔 Day2 没讲、且在 memory-bound 场景往往更值钱?→ §3.1 / §3.2
4. 寄存器/shared memory 和 HBM 的访问延迟差几个数量级?"中间结果落 HBM"为什么是浪费?→ §3.2
5. **(核心)** megakernel 和 CUDA Graph 都治 launch-bound,根本区别是什么?各治哪笔账、各自代价?→ §4.4
6. 本机实测:单次 launch ≈ ? µs;融合省访存 ≈ ? 倍;CUDA Graph ≈ ? 倍。为什么实验③加速最夸张?→ §5.1 / §5.2
7. 把"decode → 几十个小 kernel → launch-bound + 中间张量落 HBM → megakernel 两笔账 → 课题主线 3"连成一条因果链。→ §3.4

> 讲不出来就回去重读对应节,比背答案有用。第 5、6 题是今天的题眼。

---

## 8. 与已有笔记 / 课题主线的串联

| 关联 | 关系 |
|---|---|
| [W7 Day1 · Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md) | 今天补上 Roofline **画不出的第三种 bound**:launch-bound。Roofline 判 memory/compute-bound,今天判"CPU 发射受限" |
| [W7 Day2 · 三级 Profiler](./W7_Day2_三级Profiler工具链_torchprofiler_nsys_ncu.md) | Day2 在 nsys 时间线**看见** gap 和 launch;今天把 gap **拆到 driver 底层 + 量成 8.56µs**。Day2 §6.2 提过 CUDA Graph,今天讲透它 vs 巨核 |
| W7 Day5 · torch.compile(下一天) | Day5 的 `torch.compile` = **自动做今天的融合(Inductor)+ 自动用 CUDA Graph(reduce-overhead)**,是你巨核的 baseline |
| W6 · nanoGPT / KV Cache | decode 那几十个小 kernel 就来自 nanoGPT 每层的算子;§5.3 的 nsys 就在它上面做 |
| [rnn_to_transformer_evolution](./rnn_to_transformer_evolution.md) §3.2 | 那里讲的"LSTM 四门合并成一个大矩阵乘 = 算子融合雏形",正是今天第一笔账的最小案例 |
| 小米课题主线 3(巨核算子生成) | **今天是它的动机地基**:量化了"巨核要省的两笔账"。AMK 报告 5826 tasks / attention 364µs 就是融合对象 |
| 小米课题主线 2(图级优化)| CUDA Graph + torch.compile 是"图级"的自动版,你手写巨核要打败它 |
| H100 / AMK profiling | §5.3 nsys 流程 + §5.4 AMK 真实数据,是你 AMK report 的 launch 占比一栏 |

---

## 9. 今日产出清单(对齐计划)

- [x] `kernel_launch_and_fusion.md`(本笔记正文,桌面按 `W7_Day4_*` 平铺;进仓时用计划里的文件名)
- [x] **launch overhead 实测数据**(本机 RTX 5060,可复现):单次 launch **8.56µs** / 融合省访存 **1.61×** / CUDA Graph **7.98×**
- [x] 配套脚本 `bench_launch_fusion.py`(三实验,含计时/热身/CUDA Graph 录制范例)
- [x] 打通因果链:launch 底层 4 步 → decode launch-bound → 融合两笔账 → megakernel 动机(主线 3)
- [ ] (有 H100 时)按 §5.3 用 nsys 量 nanoGPT decode 真实 launch 占比,填进 AMK report
- [ ] (Day5 衔接)对 nanoGPT 跑 `torch.compile`,看它自动融了什么、开 reduce-overhead 后 CUDA Graph 的提升,作为巨核 baseline

---

> **一句话收尾**:今天你把"融合能加速"这句正确的废话,拆成了**两笔可量化的账**(8µs/launch × N 次 + 每次中间张量几百 ns 的 HBM 往返),并从 CPU driver 的"打包→入队→敲门铃"底层看清了这笔钱到底花在哪。这就是 megakernel 的全部动机——也是你 W8 亲手写巨核、拿实测提升去回答"够不够 20%"的第一块基准石。

