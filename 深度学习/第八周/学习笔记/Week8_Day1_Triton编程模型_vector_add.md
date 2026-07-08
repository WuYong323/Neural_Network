# Week 8 · Day 1 —— Triton 编程模型:vector-add

> **今天的唯一目标:把"一个 program 处理一整块数据"这句话,刻进脑子。**
> 不追性能。今天要建立的是一种**看待 GPU 的新视角**——从"我指挥每一个工人"切换到"我指挥每一个班组长"。这个视角建对了,后面 W8 的 fused kernel、softmax、layernorm 才不会乱。

---

## 0. 先给三个学习目标问题一个"电梯答案"

在展开之前,先把三个必须脱口而出的答案放这里。读完全文后回来看,应该能自己复述出来。

1. **`tl.program_id(0)` vs CUDA `threadIdx` 的本质区别?**
   CUDA 里 `threadIdx` 是"我是第几个**线程**",你写的代码是**单个线程**的视角,一个线程干一个元素;Triton 里 `program_id` 是"我是第几个**程序实例(program)**",你写的代码是**单个 program**的视角,一个 program 干**一整块(BLOCK 个)**元素。Triton 把"块内怎么拆给线程/warp"这件事**藏起来交给编译器**了,所以叫**分块(block-level)编程**,而不是分线程。

2. **`tl.load(ptr + offs, mask=mask)` 里的 `mask` 干嘛?**
   数组长度几乎不可能被 BLOCK 整除,所以**最后一个 program 一定会算出超出数组末尾的下标**。`mask = offs < n` 是一张"哪些下标是合法的"的布尔通行证,`tl.load` 只对 `mask=True` 的位置真正去读内存,挡住越界读;`tl.store` 同理挡住越界写。它挡的就是**最后一块里那几百个"数组根本没这么长"的幽灵下标**。

3. **`BLOCK: tl.constexpr` 为什么必须是编译期常量?和 occupancy 什么关系?**
   因为 `tl.arange(0, BLOCK)` 要生成一个**形状为 `[BLOCK]` 的张量**,而张量的形状是类型的一部分,编译器要用它来分配寄存器、决定向量化和循环展开——这些都得在**编译那一刻**就知道 BLOCK 是几。Triton 是 JIT,每个不同的 BLOCK 值会**特化编译出一份独立的机器码**(和 C++ 模板一模一样)。而 BLOCK 越大,每个 program 吃的寄存器/共享内存越多,一个 SM 上能同时住下的 program 就越少 → **occupancy(占用率)越低**;反之亦然。所以 BLOCK 是一个要在"单块效率"和"并发度"之间权衡的旋钮,这也是为什么有 `@triton.autotune`。

下面把每个答案背后的"为什么"讲透。

---

## 1. 问题背景:GPU 为什么需要一种"奇怪"的编程模型

### 1.1 CPU 和 GPU 的根本分工不同

先建立一个直觉。**CPU(中央处理器)**像几个博士生:核心少(几个到几十个),但每个都极聪明,擅长处理有复杂分支、前后依赖的任务。**GPU(图形处理器,Graphics Processing Unit)**像几万个小学生:每个只会做简单的加减乘除,但人数极多,适合"同一道题换不同数字,几万份卷子同时做"。

向量加法 `c = a + b`(一百万个元素,每个位置各加各的、互不依赖)正是 GPU 的主场:一百万道"a[i] + b[i]"的加法题,彼此没有依赖,天生适合"人海战术"并行。

**所以 GPU 编程的核心问题永远是同一个:我有海量互相独立的小任务,怎么把它们铺到几万个计算单元上同时跑?** CUDA 和 Triton 只是回答这个问题的两种不同"话术"。

### 1.2 两个专业名词先立住(后面反复用)

- **kernel(核函数)**:一段在 GPU 上运行的函数。注意它和"操作系统内核"没关系,这里就是"跑在 GPU 上的那段计算代码"。你写的 `add_kernel` 就是一个 kernel。
- **launch(启动/发射)**:CPU 端下达一条命令,让 GPU 同时开出**成千上万份**这个 kernel 的"副本"去跑。你可以想象成:老师(CPU)把同一张卷子(kernel)一次性复印一百万份,发给一百万个学生(GPU 上的执行单元)同时开考。这个"一次性复印发卷"的动作就是 launch。

> **接 W7 Day4 §2 的锚点**:W7 你学的是"launch 一个 kernel 是什么意思";今天你要**第一次亲手定义这批卷子发多少份、怎么分区**——也就是定义那个 grid 的形状。

---

## 2. 核心一:program 不是 thread —— 把视角从"线程"抬到"块"

这是今天最重要、也最容易搞混的一节。请慢读。

### 2.1 CUDA 的世界:你是"线程调度员",管每一个工人

CUDA 采用 **SIMT** 执行模型。

> **SIMT(Single Instruction, Multiple Threads,单指令多线程)**:一条指令,同时驱动很多个线程去执行,但每个线程处理自己那一份数据。类比:广播体操——喇叭里喊"第八节,伸展运动"(单指令),操场上几千人同时做同一个动作(多线程),但每个人动的是自己的胳膊(各自的数据)。

在 CUDA 里,你写的代码是**一个线程的视角**。看这段经典的 CUDA C 向量加法(这就是"底层"长什么样,建议对照着读):

```cpp
// ===== CUDA C 版 vector-add(理解用,不用运行)=====
// 编译器视角:这段代码会被复制到"每一个线程"身上去跑

__global__ void add_kernel(const float* a, const float* b,
                           float* c, int n) {
    // blockIdx.x   : 我在第几个 block(线程块)里
    // blockDim.x   : 每个 block 有多少个线程
    // threadIdx.x  : 我是这个 block 里的第几个线程
    // 三者拼出"我这个线程在全局是第几个"——这就是我负责的那 1 个元素
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {          // 越界保护:第 i 个线程自己检查自己该不该干活
        c[i] = a[i] + b[i];   // 注意:一个线程只处理【一个】元素
    }
}

// CPU 端 launch:<<<多少个 block, 每个 block 多少线程>>>
int threads = 256;
int blocks  = (n + threads - 1) / threads;   // 向上取整
add_kernel<<<blocks, threads>>>(a, b, c, n);
```

关键观察:

- 代码里的主角是 **`i`**,它是**单个线程**的全局编号。你脑子里想的是"作为第 i 个线程,我该取哪个元素"。
- 一个线程 = 一个元素。一百万个元素 = 一百万个线程。
- `threadIdx / blockIdx / blockDim` 这套"三级坐标"要你自己拼。线程怎么组织成 block、block 怎么铺成 grid、块内怎么用共享内存和同步——**全是你的责任**。控制力极强,但也极容易写错。

### 2.2 Triton 的世界:你是"班组长调度员",管每一个 program

Triton 把视角**整体抬高了一层**。你不再管单个线程,你管的是 **program**。

> **program(程序实例,也叫 program instance / kernel instance)**:Triton 里被 launch 出来的一个"副本",它**负责处理一整块(BLOCK 个)连续的数据**。它约等于 CUDA 里的一个 **block(线程块)**,而不是一个线程。

> **block-level programming(分块编程)**:你写代码时的"最小操作单位"是**一整块数据**,而不是一个标量。你写 `c = a + b`,这里的 `a`、`b`、`c` 是**长度为 BLOCK 的向量**,一行代码把一整块加完。这就像写 NumPy:`c = a + b` 你从不写 for 循环去逐个加,你操作的是整个数组。

对照着看 Triton 版:

```python
# ===== Triton 版 vector-add(可运行,后面第 6 节给完整文件)=====
import triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    # 我是第几个 program?(沿 grid 的第 0 维)
    # ——注意:这问的是"第几个班组",不是"第几个工人"
    pid = tl.program_id(axis=0)

    # 我这个 program 负责的那一整块的起点
    block_start = pid * BLOCK_SIZE

    # 关键:一次生成【一整个向量】的下标,而不是一个标量
    # offsets = [block_start+0, block_start+1, ..., block_start+BLOCK-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements          # 见第 3 节

    # 一次 load 一整块;a、b 是长度 BLOCK 的向量
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    c = a + b                            # 一行,把一整块加完(NumPy 味儿)

    tl.store(c_ptr + offsets, c, mask=mask)
```

关键观察:

- 代码里的主角是 **`pid`**,它是**单个 program(≈一个班组)**的编号。你脑子里想的是"作为第 pid 个班组,我负责哪一段数据"。
- 一个 program = 一整块(比如 1024 个)元素。一百万个元素 ≈ 977 个 program。
- 你**再也没写过** `threadIdx`。"这 1024 个元素在班组内部怎么拆给 128 个工人(线程)、怎么向量化、怎么合并访存"——**全交给 Triton 编译器**了。

### 2.3 一张对照表钉死区别

| | CUDA | Triton |
|---|---|---|
| 你的视角 | 一个 **thread(线程/工人)** | 一个 **program(程序实例/班组)** |
| 一个单位处理多少数据 | 通常 1 个元素 | 一整块 BLOCK 个元素 |
| "我是第几个"怎么问 | `threadIdx / blockIdx` 自己拼 | `tl.program_id(0)` 直接给 |
| 块内线程如何分工 | **你手写**(索引、共享内存、同步) | **编译器负责**,你看不见 |
| 越界保护 | 每个线程 `if (i < n)`(标量) | 整块一张 `mask`(向量) |
| 代码风格 | 标量、逐线程 | 向量、逐块(像 NumPy) |
| 类比 | 指挥每一个工人 | 指挥每一个班组长 |

**一句话记忆**:CUDA 让你当**工人的调度员**(管到人),Triton 让你当**班组长的调度员**(管到组),组内的活儿组长(编译器)自己安排。

### 2.4 底层追问:一个 program 编译后,到底怎么变成 thread?

"编译器负责"听起来像魔法,我们把它拆开。这决定了你对 Triton 的信任程度。

Triton 编译一个 program 时,有个隐藏旋钮 **`num_warps`(用多少个 warp 来执行一个 program)**。

> **warp(线程束)**:GPU 硬件调度的最小单位,固定是 **32 个线程**捆在一起,永远执行同一条指令(这就是 2.1 说的 SIMT)。你可以把一个 warp 想成"一个必须齐步走的 32 人小队"。

假设 `BLOCK_SIZE = 1024`,`num_warps = 4`:

```
一个 program 要处理 1024 个元素
num_warps = 4  →  4 × 32 = 128 个线程 来执行这一个 program
每个线程分到  1024 / 128 = 8 个元素

于是编译器悄悄把你写的 "c = a + b"(1024 长的向量加)
翻译成大致这样的逐线程代码:

    // 每个线程(共 128 个)干这段:
    for (int k = 0; k < 8; k++) {
        int idx = my_lane_base + k * 128;   // 编译器算好的 stride
        c[idx] = a[idx] + b[idx];
    }
    // 实际还会用 128-bit 向量化 load(一次搬 4 个 float),这里简化
```

看到没?**你写的"块级向量加",最终还是落回到 CUDA 那样的"逐线程标量加"**——只不过这个翻译过程(块 → 线程、stride 怎么排、要不要向量化访存、要不要走共享内存)是 Triton 编译器替你做的。它做的事情本质上就是 2.1 里你手写的那套索引拆分,只是它做得更系统、更不容易错。

**所以 program 和 thread 不是对立的两个东西,而是两个抽象层次:program 在上(你写),thread 在下(编译器生成)。** 你搞混它们,就等于把"班组长"和"工人"当成同一个人——指挥系统当然会乱。

---

## 3. 核心二:mask —— 挡住"数组根本没这么长"的幽灵下标

### 3.1 越界是怎么发生的?用真实数字算一遍

这不是抽象的"可能越界",是**必然越界**。我们用今天的数据算死它。

- 元素总数 `n = 1,000,000`
- `BLOCK_SIZE = 1024`
- 需要多少个 program?**向上取整**:`ceil(1,000,000 / 1024) = 977` 个(program 编号 0 ~ 976)。

> 为什么向上取整?因为 `1,000,000 / 1024 = 976.56...`,976 个 program 只能覆盖 `976 × 1024 = 999,424` 个元素,还剩 576 个没人管。必须再开第 977 个 program 兜底。Triton 里用 `triton.cdiv(n, BLOCK)` 做这个向上取整除法(`cdiv` = ceil division = `(n + BLOCK - 1) // BLOCK`)。

现在看**最后一个 program(编号 976)**:

```
它负责的下标 = 976 * 1024 + [0, 1, ..., 1023]
             = [999424, 999425, ..., 1000447]

但数组合法下标只到 999999!

  999424 ~ 999999  →  576 个,合法    ✅
 1000000 ~ 1000447  →  448 个,越界    ❌  ← 幽灵下标,数组根本没这么长
```

如果不管不顾直接 `tl.load(a_ptr + offsets)`,这 448 个越界下标会去读 `a` 数组**末尾之外的内存**。后果:
- 轻则读到隔壁变量的垃圾数据,结果错误(而且是偶发、难查的错);
- 重则触碰到未映射的显存页,直接 `CUDA error: an illegal memory access was encountered`,kernel 崩溃。

### 3.2 mask 是什么:一张逐元素的"通行证"

```python
mask = offsets < n_elements
```

`mask` 是一个和 `offsets` 等长(BLOCK 个)的**布尔向量**。对最后一个 program:

```
offsets:  [999424, ..., 999999, 1000000, ..., 1000447]
mask:     [ True,  ...,  True,   False,  ...,  False  ]
           └── 前 576 个 ──┘      └──── 后 448 个 ────┘
```

然后:

```python
a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
#                             ↑           ↑
#            只对 True 的位置真的去读内存   False 的位置不读,填这个默认值
tl.store(c_ptr + offsets, c, mask=mask)
#                            ↑ 只对 True 的位置真的写回,False 的位置跳过
```

- **`mask`**:告诉 `load/store`——"这 1024 个坑位里,只有打勾的那些是真的,去动它们;打叉的别碰内存"。
- **`other`**:`load` 时,被 mask 挡掉的位置返回什么值。**对 vector-add 可以不写**(反正这些位置的结果最后也被 store 的 mask 挡住、不会写回,是多少无所谓)。但**养成写 `other=0.0` 的习惯**:一旦这个值会流进后续计算(比如求和 reduction、求最大值),不写 `other` 会让被挡的位置是**未定义值**,污染结果。这是初学者最隐蔽的 bug 之一。

> **类比**:一个货架有一百万个格子,你每次派一队 1024 只机械手去取货。最后一队的模板还是 1024 只手,但那片区域只剩 576 件货了。`mask` 就是发给这队手的名单:"1~576 号手正常取货,577~1024 号手——你们面前是货架尽头的墙(已分配内存的边界),伸手要么抓到别人的东西,要么撞墙报错,统统缩回去别动。"

### 3.3 底层追问:mask 和 CUDA 的 `if (i < n)` 是同一件事

回头看 2.1 的 CUDA 代码,那句 `if (i < n)` 其实就是 **mask 的标量版**:CUDA 里每个线程**各自**判断自己该不该干活;Triton 里一整块**用一个布尔向量一次性**判断。granularity(粒度)不同,目的完全一样。

再往下挖一层,它们在硬件上落到同一个机制:**predication(谓词执行)**。

> **predication(谓词执行)**:GPU 不用"跳转"来实现 `if`,而是让每条指令带一个"是否生效"的开关位(谓词寄存器)。指令照常发出,但只有开关为真的线程,其内存读写才真正生效;开关为假的线程"空转",结果被丢弃。为什么这么设计?因为一个 warp 的 32 个线程必须齐步走(SIMT),不能有人跳转有人不跳,所以干脆都执行、用开关决定谁的结果算数。

编译后的 PTX(CUDA 的中间汇编)大致长这样,你能直接看到那个开关 `@%p1`:

```
    setp.lt.s32   %p1, %r_i, %r_n;      // %p1 = (i < n)  —— 算出谓词开关
    @%p1 ld.global.f32  %f1, [%rd_a];   // @%p1: 仅当 p1 为真才真的 load
    @%p1 ld.global.f32  %f2, [%rd_b];
    add.f32       %f3, %f1, %f2;
    @%p1 st.global.f32  [%rd_c], %f3;   // 仅当 p1 为真才真的 store
```

**Triton 的 `mask` 编译出来,就是这套 `@%p` 谓词化的 load/store。** 所以当你理解了"mask 挡的是越界内存访问",你其实已经理解了 GPU 边界处理的最底层机制——CUDA 手写和 Triton 自动生成,殊途同归。

> 另一种消除越界的思路是把数组**padding(补齐)**到 BLOCK 的整数倍。但那要多分配显存、多一次拷贝,得不偿失。**mask 是零额外内存、近乎零开销的标准解法**,工业界一律用 mask。

---

## 4. 核心三:`BLOCK: tl.constexpr` —— 为什么它必须在编译期就定死

### 4.1 `tl.constexpr` 是什么

> **`tl.constexpr`(compile-time constant,编译期常量)**:给参数打的一个标记,意思是"这个值在**编译 kernel 那一刻**就必须是已知的确定数字,不能等到运行时才知道"。它和 C++ 的 `constexpr` / 模板参数是同一个概念。

### 4.2 为什么 BLOCK **必须**是编译期常量?两个层层递进的原因

**原因一:张量的形状是"类型"的一部分,编译器必须当场知道。**

看这行:

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

`tl.arange(0, BLOCK_SIZE)` 要生成一个**形状为 `[BLOCK_SIZE]` 的张量**。在 Triton(以及所有面向 GPU 的编译器)里,**张量的形状不是运行时才算的数据,而是编译期就固定的"类型信息"**——编译器要靠它来决定:

- 给这个向量分配**多少寄存器**;
- 循环要**展开(unroll)几次**;
- 访存要**向量化成几宽**(一次搬 4 个 float 还是 8 个);
- 要不要、用**多大的共享内存**。

这就像 C 语言里 `float arr[BLOCK];` ——`BLOCK` 必须是常量,因为编译器要在栈上**当场划出固定大小的空间**。你不可能写 `int n = 用户输入(); float arr[n];`(变长数组是另一套故事且 GPU 上不适用)。**形状不定,编译器无从下手。**

**原因二:Triton 是 JIT,不同的 BLOCK 会特化出不同的机器码。**

> **JIT(Just-In-Time compilation,即时编译)**:不是提前把所有情况都编译好,而是**等你真正调用、参数确定了,才针对这组参数现场编译**一份专用机器码,并缓存起来下次复用。

这一点和 **C++ 模板实例化**是同一个机制。对照着看就懂了:

```cpp
// ===== C++ 模板:BLOCK 作为编译期模板参数 =====
template <int BLOCK>
__global__ void add_kernel(const float* a, const float* b,
                           float* c, int n) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;   // BLOCK 编译期已知
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// 每写一个不同的 BLOCK,编译器就【生成一份独立的机器码】:
add_kernel<1024><<<g1, 1024>>>(...);   // 机器码 A(为 1024 特化)
add_kernel<512> <<<g2, 512 >>>(...);   // 机器码 B(为 512  特化,和 A 完全独立)
```

Triton 做的事一模一样:

```python
add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)  # 触发编译"BLOCK=1024 版",缓存
add_kernel[grid](a, b, c, n, BLOCK_SIZE=512)   # 触发编译"BLOCK=512 版",另一份缓存
add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)  # 命中缓存,不再重新编译
```

> **一个能自己验证的现象**:第一次用某个新 BLOCK 调用 kernel 会明显卡一下(那是在编译),第二次瞬间返回(命中缓存)。这就是为什么 benchmark **必须有 warmup**——头几次跑的是"编译时间 + 冷启动",不是真实计算时间。这直接呼应第 6 节 benchmark 里的 `warmup=30`。

### 4.3 BLOCK 和 occupancy 的关系(接 W7 Day2 ncu)

先把 occupancy 讲清楚。

> **SM(Streaming Multiprocessor,流式多处理器)**:GPU 里真正干活的"车间"。一块 H100 有 **132 个 SM**。你 launch 的那 977 个 program,会被 GPU 分批塞进这 132 个车间里滚动执行(一个 SM 上同时住着好几个 program)。

> **occupancy(占用率)**:一个 SM 上"**实际同时活跃的 warp 数**"占"**该 SM 理论最多能容纳的 warp 数**"的比例。H100 每个 SM 最多容纳 64 个 warp,如果你实际只跑起 16 个,occupancy = 16/64 = 25%。**它衡量你把车间塞得多满。**

**为什么 occupancy 重要?** 因为 GPU 靠"人多"来**掩盖访存延迟(latency hiding)**:一个 warp 去读显存(要等几百个周期),SM 立刻切换到另一个 ready 的 warp 去算,不让计算单元闲着。活跃 warp 越多,越有得可切,延迟越藏得住。占用率太低 → 一个 warp 卡在等内存时没别人可切 → 车间空转 → 慢。

**BLOCK 怎么影响 occupancy?** 一个 SM 的资源是**死的、共享的**(H100 单 SM 约):

- 寄存器:65,536 个(32-bit);
- 共享内存:可配置,最多约 228 KB;
- 最多 2048 个线程(= 64 warp);
- 最多 32 个常驻 block/program。

> (以上是 H100 SXM5 的大致规格,精确值请用 `ncu` 或 device query 查你手上那块卡确认——接 W7 Day2 的 ncu 习惯。)

BLOCK 越大,每个 program 处理的数据越多 → 它需要的**寄存器、共享内存也越多** → 一个 SM 里能同时**住下的 program 数量就越少** → 活跃 warp 变少 → **occupancy 可能下降**。反过来 BLOCK 太小,虽然能塞很多 program、occupancy 高,但每个 program 干的活太少,**launch 开销和固定成本摊不薄**,而且指令级并行(ILP)不足。

```
BLOCK 太小 ──► 每块干活少、launch/固定开销占比高、ILP 不足  ──► 慢
BLOCK 太大 ──► 单块吃资源多、SM 装不下几个、occupancy 低    ──► 可能慢
              ↑ 存在一个"甜点区",而且随 GPU 型号、算子而变
```

**结论**:BLOCK 是一个要在"单块效率"和"并发度/占用率"之间权衡的旋钮,没有放之四海皆准的最优值。这正是 **`@triton.autotune`** 存在的理由——让机器帮你把几个候选 BLLOCK(和 `num_warps`)都跑一遍,自动挑最快的(第 8 节细讲)。

> **给今天泼一盆清醒的冷水**:vector-add 是 **memory-bound(访存受限)** 的算子——瓶颈是"从显存搬数据的带宽",不是"算得快不快"。对这类算子,occupancy 到某个程度(够藏住访存延迟)就够了,再高也没用,因为你已经把 HBM 带宽跑满了。所以**今天不要为了 occupancy 调参**,理解"BLOCK 会通过资源占用影响 occupancy"这个**机制**就达标了。真正拿 occupancy 说事,是 W8 后面 compute-bound 算子和 W7 Day2 ncu 报告的事。

---

## 5. 一张图:1M 个元素是怎么被"切块 + 铺到 GPU"上的(底层理解锚点)

请把这张图画进笔记本——它是今天所有代码的"心智地图"。

```
             n = 1,000,000 个元素的数组 a、b、c(显存里一段连续内存)
 ┌──────────┬──────────┬──────────┬─────┬───────────────────────────┐
 │ 0 .. 1023│1024..2047│2048..3071│ ... │ 999424 .. 1000447(含越界) │
 └──────────┴──────────┴──────────┴─────┴───────────────────────────┘
    program0   program1   program2   ...          program976
      │          │          │                        │
   每个 program 内部:
   offsets = pid*1024 + tl.arange(0,1024)  ← 一次生成 1024 个下标(向量)
   mask    = offsets < 1000000             ← 最后一块挡掉 448 个越界下标
      │          │          │                        │
      ▼          ▼          ▼                        ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  grid = (977,)   ←── 你在 CPU 端定义的"发多少份卷子"             │
 └────────────────────────────────────────────────────────────────┘
      │
      ▼   GPU 硬件调度器把这 977 个 program 分批塞进 132 个 SM
 ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐
 │ SM0 │ │ SM1 │ │ SM2 │  ...   │SM131│   每个 SM 同时住着好几个 program,
 │p0,p3│ │p1,p4│ │p2,p5│        │ ... │   跑完一个换下一个,直到 977 个全跑完
 └─────┘ └─────┘ └─────┘        └─────┘
   └── 每个 program 内部又被拆成 num_warps×32 个线程(编译器干的,你看不见)
```

**"launch 一次到底发生了什么"的完整链条(接 W7 Day4)**:

1. 你在 CPU 端写 `add_kernel[grid](...)`,其中 `grid = (977,)`——这就是**你亲手定义的 grid 形状**。
2. Triton 检查缓存:BLOCK=1024 的机器码编过没?没有就 JIT 编译一份(第一次会卡)。
3. 驱动把这 977 个 program 的"发射指令"提交给 GPU。
4. GPU 的硬件调度器把 977 个 program 分批铺到 132 个 SM 上滚动执行。
5. 每个 program 内部,编译器生成的代码把 1024 个元素拆给 `num_warps×32` 个线程,用谓词化(mask)的 load/store 搬数据、做加法、写回。
6. CPU 端调用是**异步**的(发完命令就返回),所以计时前必须 `torch.cuda.synchronize()` 等 GPU 真干完——这也是第 6 节计时法的关键。

---

## 6. 完整可运行代码:`week8_triton/01_vector_add.py`

下面是可直接复制运行的**完整文件**,包含 kernel、封装、三方 benchmark、误差验证。把它存成 `week8_triton/01_vector_add.py`。

```python
"""
Week 8 Day 1 — Triton vector-add
可运行环境 / 依赖:
    - NVIDIA GPU(H100 / 30xx / 40xx / A100 等,需 CUDA 环境)
    - CUDA 12.x
    - Python 3.10+
    - PyTorch 2.x(自带 CUDA)     : pip install torch
    - Triton 2.1+ / 3.x           : 通常随 torch 一起装好;否则 pip install triton
运行:
    python 01_vector_add.py
"""

import torch
import triton
import triton.language as tl


# ======================================================================
# 1) Triton kernel —— 你写的是"一个 program"的视角
# ======================================================================
@triton.jit
def add_kernel(
    a_ptr, b_ptr, c_ptr,        # 三个张量在显存里的首地址(设备指针)
    n_elements,                 # 元素总数(运行时值,不是 constexpr)
    BLOCK_SIZE: tl.constexpr,   # 每个 program 处理多少元素(编译期常量)
):
    # 我是第几个 program?(沿 grid 第 0 维)
    # 为什么用 program_id 而不是 threadIdx:因为我关心的是"第几个块",
    # 块内的线程分工交给编译器,不归我管。
    pid = tl.program_id(axis=0)

    # 我负责的这一块的起始下标
    block_start = pid * BLOCK_SIZE

    # 一次生成【一整个向量】的下标(这就是"分块编程"的体现)
    # 为什么用 tl.arange:它生成 [0,1,...,BLOCK-1],形状 [BLOCK],
    # 形状必须编译期已知 —— 这就是 BLOCK 必须是 constexpr 的原因。
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 越界通行证:只有 < n 的下标是合法的。
    # 为什么必须有:n 几乎不可能被 BLOCK 整除,最后一个 program 必越界。
    mask = offsets < n_elements

    # 按块 load。mask 挡住越界读;other=0.0 让被挡位置有确定值
    # (对纯 add 其实无所谓,但这是好习惯——一旦值流入 reduction 就关键了)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # 真正的计算:一行把一整块加完(NumPy 味儿)
    c = a + b

    # 按块写回。mask 挡住越界写。
    tl.store(c_ptr + offsets, c, mask=mask)


# ======================================================================
# 2) Python 封装 —— 在 CPU 端定义 grid 并 launch
# ======================================================================
def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "输入必须在 GPU 上"
    assert a.is_contiguous() and b.is_contiguous(), "本 kernel 假设内存连续"
    c = torch.empty_like(a)
    n_elements = c.numel()

    # grid:发多少个 program。写成 lambda 依赖 meta['BLOCK_SIZE'],
    # 是为了让 @triton.autotune 换 BLOCK 时 grid 能跟着自动变。
    # triton.cdiv = 向上取整除法 = ceil(n / BLOCK)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # add_kernel[grid](...) 就是 launch;BLOCK_SIZE 作为 constexpr 传入。
    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
    return c


# ======================================================================
# 3) 计时工具 —— 用 torch.cuda.Event(W7 Day2 计时法)
# ======================================================================
def bench(fn, warmup: int = 30, iters: int = 100) -> float:
    """返回单次调用的平均耗时(毫秒)。"""
    # warmup:触发 JIT 编译、让 GPU 时钟升频、缓存预热。
    # 没有 warmup,你量到的是"编译时间",不是计算时间。
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()   # 等 GPU 把 warmup 全干完

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()   # 关键:CPU 发命令是异步的,必须等 GPU 真跑完再读时间

    # elapsed_time 返回的是 start~end 之间的【总毫秒数】,除以次数得平均
    return start.elapsed_time(end) / iters


def gbps(ms: float, n: int, dtype_bytes: int = 4) -> float:
    """有效显存带宽 GB/s:读 a、读 b、写 c = 3 次 n 个元素的搬运。"""
    moved_bytes = 3 * n * dtype_bytes
    return moved_bytes / (ms * 1e-3) / 1e9


# ======================================================================
# 4) 主程序:正确性验证 + 三方 benchmark
# ======================================================================
def main():
    torch.manual_seed(0)
    n = 1_000_000
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    # ---- 三个待测对象 ----
    torch_add = lambda: a + b                      # PyTorch 原生逐元素加
    compiled = torch.compile(lambda x, y: x + y)   # torch.compile(底层也会生成 Triton)
    compiled_add = lambda: compiled(a, b)
    my_triton_add = lambda: triton_add(a, b)

    # ---- 正确性(尺子①:elementwise 应逐元素几乎相等)----
    out_triton = triton_add(a, b)
    out_ref = a + b
    assert torch.allclose(out_triton, out_ref, rtol=1e-5, atol=1e-6), "结果不对!"
    max_err = (out_triton - out_ref).abs().max().item()
    print(f"[正确性] torch.allclose 通过,最大逐元素误差 = {max_err:.2e}")

    # ---- 三方计时 ----
    t_torch = bench(torch_add)
    t_compile = bench(compiled_add)
    t_triton = bench(my_triton_add)

    print(f"\n{'方案':<18}{'耗时(ms)':>12}{'有效带宽(GB/s)':>18}")
    print("-" * 48)
    for name, t in [("torch 原生", t_torch),
                    ("torch.compile", t_compile),
                    ("我的 Triton", t_triton)]:
        print(f"{name:<18}{t:>12.4f}{gbps(t, n):>18.1f}")

    print("\n[结论提示] 1M 元素太小,大概率三者持平、且远未跑满 HBM 带宽")
    print("           —— 因为此时是 launch/延迟受限,不是带宽受限。")
    print("           想看到带宽 roofline,把 n 调到 64M+(如 n = 1 << 26)再跑。")


if __name__ == "__main__":
    main()
```

**运行后你大概会看到的现象(以及怎么解读)**:

- 三方耗时**基本持平**,甚至 torch 原生略快。**这完全正常,也正是今天的预期**:vector-add 是 memory-bound 小算子,你写对了就已经贴着硬件极限了,"更快"无从谈起。今天的教学意义在**"我能写对"**,不在"我更快"。
- `torch.compile` 底层其实也会把这种 elementwise 融合成 Triton kernel,所以它和你手写的思路是同源的——你现在学的就是它内部在干的事。
- 有效带宽在 1M 规模下会**远低于 H100 的 ~3.35 TB/s**。别慌,这不是你的 kernel 差,是**数据太小,GPU 还没"热身"完(launch 和延迟开销占了大头)**。把 `n` 调到 `1 << 26`(约 6700 万)再跑,你会看到带宽显著上升、逼近 HBM roofline——这时才是真正的"带宽受限"状态。这个对比本身就是绝佳的 memory-bound 直觉训练。

---

## 7. 工业实践:这套东西在真实项目里怎么用

### 7.1 autotune:让机器替你选 BLOCK 和 num_warps

第 4.3 节说 BLOCK 有"甜点区"且随硬件/算子变化。工业界的标准做法不是手调,而是声明一堆候选配置,让 Triton 自动跑一遍挑最快:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],   # 当 n_elements 变化到新档位时,重新 autotune 并缓存
)
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...   # kernel 体和第 6 节一样,注意此时不要在 launch 时手传 BLOCK_SIZE
```

- **`key`**:告诉 autotune"输入形状变到什么程度算换了一档、需要重新调优"。`n` 从 1M 变到 100M,最优配置可能不同,所以以 `n_elements` 为 key。
- 代价:第一次遇到新 key 时会把所有 config 都跑一遍(慢),之后缓存复用。**别在计时循环里触发首次 autotune**,否则 warmup 里就把它消化掉。

> 你在 W6 用过的 FlashAttention、以及各家大模型推理框架(vLLM、SGLang)里的算子,大量是 Triton 写的,基本都挂着 autotune。理解这套,你才看得懂它们的 kernel 源码。

### 7.2 什么时候值得自己写 Triton kernel?

- **不值得**:单个 elementwise(如今天的 add)。PyTorch/`torch.compile` 已经做得很好,自己写只是练手。
- **值得**:**算子融合(fusion)**。比如把 `x → matmul → +bias → GELU → dropout` 一连串操作融进**一个** kernel,避免每一步都把中间结果写回显存再读出来(memory-bound 场景下,访存次数就是性能)。这是 Triton 在工业界最大的价值,也是 W8 后面几天(fused softmax / layernorm)要练的。
- **值得**:PyTorch 没有的、或形状特殊的自定义算子(如各种 attention 变体、量化 kernel)。

### 7.3 一个务实的心法

> **先用 `torch.compile` 看它给你生成的 Triton kernel(`TORCH_COMPILE_DEBUG=1`),再决定要不要手写、往哪优化。** 很多时候编译器已经融得不错了,手写只在它没融或融得不好的地方才有收益。别一上来就手搓。

---

## 8. 常见陷阱与调试技巧

1. **忘了 `mask`,或 mask 写错方向** → `illegal memory access` 或偶发错误结果。
   *自查*:任何 `tl.load/store` 都问自己"最后一个 program 会不会越界"。答案几乎永远是"会",所以几乎永远要 mask。

2. **benchmark 不 warmup / 不 synchronize** → 量到的是编译时间或异步空档,数字全错(常常快得离谱、假到不真实)。
   *自查*:计时前 `warmup` + `synchronize`,计时后再 `synchronize` 才读 `elapsed_time`。

3. **把 BLOCK 当普通参数传(忘了 `: tl.constexpr`)** → 报错,或形状推导失败。
   *记住*:凡是要决定张量形状、循环展开的量,必须 `constexpr`。

4. **`other` 不写,值又流进了 reduction** → 被 mask 挡的位置是未定义值,污染求和/求最大。
   *自查*:只要 load 的结果会参与"跨元素聚合",就老老实实写 `other`(求和用 `0.0`,求最大用 `-inf`)。

5. **输入非连续内存(`is_contiguous()` 为 False)还按连续算 offset** → 结果错乱。
   *自查*:今天的 kernel 假设连续内存;真实场景遇到转置/切片得先 `.contiguous()`,或在 kernel 里正确处理 stride。

6. **调试打印**:Triton 支持 `tl.device_print("offs", offsets)` 在 kernel 里打印,定位下标/mask 是否符合预期。别用 Python 的 `print`(kernel 里没用)。

---

## 9. Track B(~1h):H100 环境自检(接 W7 AMK 锚点)

```bash
module load CUDA/12.4                 # 加载 H100 上的 CUDA 12.4
nvidia-smi                            # 确认能看到 H100、驱动正常
python -c "import torch; print(torch.cuda.get_device_name(0))"   # 应打印 H100
amk compile small --gpu h100          # 跑通 AMK 小算例,确认环境没坏
```

目标只有一个:**确认 H100 环境可用、AMK 工具链没坏**,为后续在真实硬件上做 nsys/ncu profiling 铺路。今天不深入 AMK,别跑偏。

---

## 10. 完成标准自测(对照产出要求逐条打勾)

- [ ] **能不看资料写出 vector-add 的 kernel 骨架**:`program_id → offsets(arange) → mask → load → 计算 → store` 六步默写出来。
- [ ] **能口述"为什么 Triton 是分块编程"**:因为操作单位是 program(一整块),块内线程分工交给编译器(见 §2)。
- [ ] **能口述"mask 挡的是谁"**:挡最后一个 program 里那些 `≥ n` 的越界下标,底层是谓词化的 load/store(见 §3)。
- [ ] **能说清 constexpr 与 occupancy**:constexpr 是因为形状要编译期定死(JIT 特化,如 C++ 模板);BLOCK 大→单块吃资源多→SM 装不下几个→occupancy 低(见 §4)。
- [ ] **三方 benchmark 数据齐全**:torch 原生 / `torch.compile` / 我的 Triton 的耗时 + 有效带宽,且理解"1M 规模三者持平且未跑满带宽是正常的"。
- [ ] **`torch.allclose` 通过**:elementwise 逐元素几乎相等(尺子①)。

> 全部打勾,今天的心智模型就建对了。明天的任何 Triton 代码,都是在这六步骨架上加东西而已。

---

### 附:今日一句话总结

**CUDA 让你管每一个"工人(thread)",Triton 让你管每一个"班组长(program)";你只需说清"哪个班组负责哪段数据、用 mask 挡住越界",组内怎么把活拆给工人、怎么访存,编译器全包了。这就是"一个 program 处理一整块数据"的全部含义。**
