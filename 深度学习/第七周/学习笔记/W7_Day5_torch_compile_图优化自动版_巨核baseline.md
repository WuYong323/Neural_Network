# W7 Day5 · torch.compile:图优化的自动版——你手写巨核的 baseline

> **本笔记的唯一目标**:让你彻底搞懂一行 `torch.compile(model)` 背后到底发生了什么,并深挖到 **CPython 帧求值 hook** 和 **Inductor 生成的 Triton kernel** 这两层底层代码。读完你要能讲清三件事——(1) `torch.compile` 的三段链路 **TorchDynamo(抓图)→ TorchInductor(融合+生成 Triton)→(可选)CUDA Graph**,分别在替你做 Day4 的哪笔账;(2) 它凭什么"不改你一行代码"就能把动态的 Python 变成一张可优化的图——答案在 **PEP 523 帧求值 API**,不是魔法;(3) 为什么它是你小米课题**主线 2(图级优化)的自动版 baseline**——你手写巨核**必须打得过它**(课题指标 ≥20%),而**不先把它量出来,你根本无法证明自己更好**。
>
> **串联**:这是 [W7 学习计划](./W7_学习计划_AI_Infra主线.md) **Day5**,直接对接小米课题**主线 2(图级优化)**,并给**主线 3(巨核生成)立一根基准桩**。紧承 [W7 Day4 · Kernel Launch 与算子融合](./W7_Day4_KernelLaunch开销与算子融合_巨核动机.md)——Day4 你**亲手**把几个小算子融成一个、量出"省启动 1.61× / CUDA Graph 7.98×"两笔账;**今天你看的是机器自动做同样的事**:`torch.compile` 的 Inductor 就是"自动融合器",`mode="reduce-overhead"` 就是"自动 CUDA Graph"。也承接 [Day2 三级 Profiler](./W7_Day2_三级Profiler工具链_torchprofiler_nsys_ncu.md)(用同一套 CUDA Event 计时法量它)和 [Day1 Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md)(看它能否把 decode 从 launch-bound 里救出来)。
>
> **产出对齐**:本笔记正文即计划要求的 `tech_notes/torch_compile_baseline.md`(仓库里叫这个名;桌面按 `W7_DayX` 惯例平铺)。配套实测脚本 `bench_torch_compile.py` + before/after 加速数据 + `TORCH_LOGS=output_code` 抓到的 Inductor 生成代码观察 + `torch._dynamo.explain` 的 graph break 报告。

---

## 0. 开篇:读完你要能不看资料答出来的问题

1. `torch.compile(model)` 只加一行,凭什么能加速?它到底把你的 Python 代码变成了什么?
2. **TorchDynamo** 是怎么在**不改你代码**的前提下"偷看"到你的计算过程的?(提示:它 hook 的是 CPython 解释器本身)
3. 什么是 **graph break(图断裂)**?为什么一次 `print(x)` 或一个依赖数据的 `if` 就能让你的加速化为乌有?
4. **TorchInductor** 生成的 Triton kernel,和你 Day4 手写的融合 kernel 是**同一件事**吗?它自动融合的边界在哪?
5. `mode="default"` / `mode="reduce-overhead"` / `mode="max-autotune"` 三挡分别多做了什么?哪一挡对应 Day4 的 CUDA Graph?
6. **(灵魂题)** 为什么说"不先把 `torch.compile` 的数字量出来,你手写巨核的 ≥20% 就是一句空话"?

> 第 2、3、6 题是题眼。如果你能把第 2 题从"它捕获了计算图"讲到"CPython 3.11 起提供了 `_PyInterpreterState_SetEvalFrameFunc`,Dynamo 用它替换了帧求值函数,于是每个 Python 函数在执行前都先过一遍 Dynamo 的字节码分析器"——这一天就值了。

---

## 1. 问题背景:你 Day4 手动做的事,能不能让机器自动做?

先把 Day4 的结论接上,你才知道今天在补哪块拼图。

Day4 你已经用本机 RTX 5060 亲手证明了两件事:

- **launch 太贵**:每启动一个 GPU kernel,CPU 走"打包参数→入队→敲门铃"这条软件路径要花 ~8.56µs,**和算多少无关**。decode 阶段几十个小 kernel,大半时间 GPU 在**空等 CPU 发命令**(launch-bound)。
- **融合能省两笔账**:把 N 个小算子融成 1 个大 kernel,既省了 N-1 次 launch(省启动),又省了中间结果反复写读 HBM(省访存)。你手写融合实测 1.61×,手录 CUDA Graph 实测 7.98×。

但 Day4 那个融合是你**亲手**写的(`addcmul` 那个例子)。问题来了:

> 一个真实模型有几百上千个算子(nanoGPT 每层就有 LayerNorm、QKV 投影、attention、FFN、残差加……),难道每一处融合都要人手写 kernel?那得写到什么时候?

这正是 `torch.compile` 要解决的**工程规模问题**:

> **`torch.compile` 是什么**:PyTorch 2.0 起提供的**即时编译器(JIT compiler)**。你把模型或函数丢给它,它会**自动**分析你的计算过程、把能融的算子融成一个大 kernel、能省的 launch 省掉,**全程不需要你改一行模型代码**。一句话——**它是 Day4 那套融合手艺的自动化流水线**。

**类比(先建直觉,§3 立刻拆底层代码)**:Day4 的你像一个**手工匠人**,一个零件一个零件地打磨、焊接(手写融合 kernel)。`torch.compile` 则像一条**自动化生产线**:你把设计图(模型代码)扔进去,它自己识别"这几道工序可以合并到一个工位完成"(融合)、"这批活可以一次性下料不用来回搬运"(省访存),最后吐出成品。匠人能做出**极致**的单件(手写巨核可以更快),但生产线**又快又不知疲倦、覆盖面广**——这就是为什么它是你的 **baseline(基准线)**:一个自动、免费、人人都能开的加速,你手写巨核要是打不过它,那你这活就白干了。

**为什么它对你的课题是"绕不过去的一关"**:小米课题主线 2 就是**图级优化**,而 `torch.compile` 正是业界最成熟的**自动图级优化**方案。你的 KPI 是"相较现有算子级优化方案提升 ≥20%"——`torch.compile` 就是那个"现有方案"里最强的公开 baseline 之一。**不量它,你的 20% 没有分母。**

## 2. 全局地图:一行 `torch.compile` 背后的三级流水线

`torch.compile` 不是一个黑盒魔法,它是**三个部件接力**完成的。先把这张地图记住,后面每一节都是在放大其中一环。

```
你的 Python 模型代码
      │
      │  ①【TorchDynamo】捕获图 (graph capture)
      │     偷看 Python 字节码执行, 把张量运算抽出来, 组成一张"计算图"
      │     遇到看不懂的(print/if data/外部库)→ graph break, 断成多段
      ▼
   FX Graph (一张干净的算子依赖图, 见 Day2 讲的"算子级视角")
      │
      │  ②【AOTAutograd】(可选, 训练才需要) 提前把反向图也画出来
      │     推理(你的场景)基本不涉及, 先知道有这一层即可
      ▼
   前向(+反向)算子图
      │
      │  ③【TorchInductor】编译后端 (compiler backend)
      │     - pointwise 融合: 把逐元素算子(加/乘/relu/...)合并成一个 kernel
      │     - 自动生成 Triton kernel (GPU) / C++/OpenMP (CPU)
      │     - 可选: 套 CUDA Graph 再省 launch
      ▼
   融合后的 Triton kernel + 调度好的执行计划  →  少 launch + 少访存
```

**用一句话串起来**:

> **TorchDynamo 负责"看懂你在算什么"(捕获图),TorchInductor 负责"把它算得更快"(融合 + 生成 kernel)。中间那张 FX Graph,就是它俩交接的图纸。**

这里每个英文名词都是第一次出现,逐个拆:

- **TorchDynamo(Dynamo,图捕获器)**:`torch.compile` 的**前端**。它的唯一任务是**在你的 Python 代码运行时,把里面的张量运算"抄"下来,组成一张计算图**。关键是它**不需要你改代码**——秘密在 §3 讲(它 hook 了 CPython 解释器)。
- **FX Graph(FX 图)**:Dynamo 抄下来的结果,是一张**算子依赖图**——节点是算子(matmul、add、relu),边是数据流(谁的输出是谁的输入)。这正是 Day4 §1 说的"把推理抽象成可重写计算图",也是你小米课题**主线 2** 的核心数据结构。
- **TorchInductor(Inductor,编译后端)**:`torch.compile` 的**后端**。它拿到 FX Graph,做真正的优化:**pointwise 融合**、生成 **Triton** kernel、排执行顺序。这是**真正干活省 launch/省访存**的那一环。
- **AOTAutograd(提前自动微分)**:训练时才用到的一环,负责在编译期就把**反向传播**的图也一起画出来(AOT = Ahead-Of-Time,提前)。你现在做**推理**优化,这一层基本用不上,知道它在流水线里占个位置即可,不展开。
- **Triton**:一个用 **Python 语法**写 GPU kernel 的语言/编译器(OpenAI 出品)。Inductor 生成的 GPU kernel 就是 Triton 代码。你可以把它理解成"**不用写 CUDA C++,用 Python 就能写出接近手写水平的 GPU kernel**"。⚠️ **它只在 Linux + NVIDIA GPU 上装得起来,Windows 没有**(这个坑 §7 详说,直接决定你今天的实验在哪台机器做)。

**为什么要拆成"前端捕获 + 后端编译"两段?** 这是编译器领域的经典解耦思想(和 LLVM 的 IR 一个思路):前端只管"把各种来源的代码翻译成统一的中间表示(FX Graph)",后端只管"把中间表示编译到各种硬件"。这样前端换语言、后端换硬件,互不影响。对你的意义:**你手写巨核,本质上是在替换/超越 Inductor 这个后端的某些子图**——前端 Dynamo 抓的那张图,正是你和 Inductor 竞争的**同一张图纸**。

## 3. 前端:TorchDynamo 怎么做到"不改一行代码就抓到图"?

这是全篇第一个"反直觉"的点。你只写了 `model = torch.compile(model)`,没动 `forward` 一个字,它凭什么知道你 forward 里算了哪些张量?

### 3.1 是什么:它不是"读你的源码",而是"劫持字节码"

先破一个误解:Dynamo **不是**去分析你的 `.py` 源代码文本(那太脆弱,注释、动态生成都能搞崩)。它工作在更底层的一层——**Python 字节码(bytecode)**。

**Python 字节码是什么**:你写的 `.py` 会被 CPython(标准 Python 解释器)先编译成一串**字节码指令**(像 `LOAD_FAST`、`BINARY_OP`、`CALL`),CPython 的虚拟机再逐条执行这些指令。可以把它类比成:**你的 Python 是"菜谱文字",字节码是"翻译成后厨标准动作的分解步骤"(拿锅、放油、翻炒)**,解释器真正执行的是这些分解动作。

**类比**:Dynamo 像一个**装在后厨门口的监工**。它不看菜谱原文,而是**盯着厨师实际做的每一个标准动作**(每条字节码)。看到"这几步是在切菜炒菜(张量运算)",就悄悄记进小本本(建图);看到"厨师突然去接了个电话(`print`、依赖数据的 `if`)"这种它没法记的动作,就**在这里画一条线**——本子先合上(§5 的 graph break)。

### 3.2 底层机制:PEP 523 的 frame evaluation hook

Dynamo 能插进去,靠的是 CPython 3.6+ 提供的一个官方扩展点 **PEP 523(frame evaluation API,帧求值接口)**。一句话:**CPython 允许你替换掉"执行一个函数帧"的那个核心函数**。Dynamo 就把它替换成了自己的版本。

**frame(帧)是什么**:每次调用一个 Python 函数,解释器都会造一个 frame 对象,里面装着这次调用的一切——字节码、局部变量、执行到第几条指令。执行函数 = 解释器"求值"这个 frame。

下面用简化伪代码,把 Dynamo 这套"劫持 → 分析字节码 → 编译 → 换掉函数"的骨架露出来:

```python
# ============================================================
# 简化伪代码: TorchDynamo 的核心机制(骨架, 非真实源码)
# 目的: 看清 "不改代码就抓到图" = "劫持 CPython 的帧求值函数"
# 真实实现: torch/_dynamo/, C 侧的 _eval_frame.c
# ============================================================

# CPython 原本执行一个函数帧, 用的是内置的 _PyEval_EvalFrameDefault。
# PEP 523 允许我们替换它 —— Dynamo 装的就是下面这个。
def dynamo_frame_handler(frame, ...):
    code = frame.f_code            # 这个函数的字节码对象

    # ① 先查缓存: 这个函数 + 这批输入形状/类型, 之前编译过吗?
    guards, compiled = cache_lookup(code)
    if compiled and guards.check(frame):   # guard 全部满足 → 直接复用
        return compiled(frame)             # 走已编译的快路径, 零开销

    # ② 没命中: 符号化执行字节码, 边"假装执行"边建图
    #    遇到张量运算 → 记成 FX 图的一个节点
    #    遇到看不懂的(print / if 依赖张量值 / 未知库) → graph break
    fx_graph, guards = trace_bytecode(code, frame)  # 见 §3.3

    # ③ 把抓到的 FX 图交给后端(Inductor)编译成快 kernel
    compiled = inductor_compile(fx_graph)

    # ④ 存进缓存, 连同 "guard(什么条件下这份编译还有效)"
    cache_store(code, guards, compiled)
    return compiled(frame)
```

**读这段要抓的 3 个点:**

1. **它是"运行时即时编译(JIT)",不是提前静态分析**。第一次真正跑到这个函数时才触发编译,所以**第一次调用巨慢**(要 trace + 编译),之后才快。这就是你实测时**必须 warmup**、且第一次别计时的根本原因(呼应 Day4 §6.2 陷阱 2)。
2. **guard(守卫)是灵魂**。编译结果不是无条件复用的——Dynamo 会记下一堆假设:"输入是 float32、形状 [1,768]、这个 flag 是 True……"这些假设叫 guard。下次进来 guard 一条不满足(比如序列长度变了),缓存作废、**重新编译(recompile)**。这直接关系到 §7 的"形状一变就重编"陷阱,也解释了 decode 里为什么要"按长度分桶"。
3. **graph break 发生在字节码层**。不是你的代码"写错了",而是 Dynamo 在**逐条字节码**往下走时,撞到一条它没法符号化的指令,只能就地断开。所以 §5 用 `explain` 定位 break 时,它能精确告诉你"断在哪一行、因为哪条字节码"。

### 3.3 关键细节:`trace_bytecode` 是"假装执行",不是"真的算"

上面 ② 里那个 `trace_bytecode`,是 Dynamo 最精妙的一步,单独拆开讲。它叫 **symbolic execution(符号化执行)**:Dynamo **一条一条**地"走"你的字节码,但**不真的做张量运算**,只是**记账**。

**怎么做到"走了但不算"?** 靠 **FakeTensor(假张量)**。Dynamo 把你真实的输入张量换成"只有形状和 dtype、没有真实数据"的假张量,让代码在这堆假张量上"空跑"一遍:

- 遇到 `q = x @ wq`:它不真算矩阵乘,只**推断出**"输出是个 [1,768] 的 float32",并往 FX 图里记一个 `matmul` 节点。
- 遇到 `h = q + k`:记一个 `add` 节点,连上依赖边。
- 遇到 `if x.sum() > 0`:**傻眼了**——假张量没有真实数据,`sum()` 出不来具体值,这个 `if` 走哪个分支它不知道 → **graph break**(§5)。

**类比**:符号化执行像**排练走位**,不是**正式演出**。演员(算子)按剧本(字节码)把站位、出场顺序全走一遍(建图),但**不真的念台词、不真的打斗**(不做浮点运算)。导演(Dynamo)要的只是一张**完整的调度表**(FX 图),真正的演出(kernel 执行)交给舞台机械(Inductor 编译出的 Triton)去干。

> **为什么这个设计对你重要**:符号化执行意味着 Dynamo 抓的图**只包含张量运算的骨架**,天然剔除了 Python 的胶水逻辑。这张干净的图,就是你小米课题**主线 2** 要"重写/融合"的对象——你和 Inductor 拿到的是**同一张图**,你手写巨核就是在这张图上做比 Inductor 更激进的融合。

## 4. 后端:TorchInductor 怎么"自动做 Day4 的融合"?

图抓到了,轮到真正干活省时间的一环。**这一节是本篇和你课题最直接相关的部分**——因为 Inductor 做的事,和你 Day4 手写融合、以及 W8 要手写的巨核,是**同一件事的不同自动化程度**。

### 4.1 是什么:Inductor 是"图 → Triton kernel"的翻译+优化器

**TorchInductor** 拿到 FX 图后,大致做三步:

1. **lowering(下沉)**:把 FX 图里"框架级"的算子(`aten.add`、`aten.relu`)翻译成 Inductor 自己的**循环级中间表示(loop-level IR)**——即"对每个元素做什么"。这一步把"算子"摊平成"循环",**融合才有了操作空间**。
2. **scheduling + fusion(调度与融合)**:决定**哪些循环可以合并进同一个 kernel**。核心规则就是 Day4 讲的 **pointwise 融合**:相邻的逐元素算子(加、乘、relu、sigmoid……)因为"每个元素独立、访存模式一致",可以塞进一个 kernel 一次算完。
3. **codegen(代码生成)**:GPU 上生成 **Triton** kernel,CPU 上生成 C++/OpenMP。

### 4.2 底层代码:看一眼 Inductor 真的生成了什么

空谈"它会融合"没用。下面是 Inductor 面对一段 `y = relu(x * a + b)`(三个逐元素算子:乘、加、relu)时,`TORCH_LOGS="output_code"` 会打出来的**真实风格**的 Triton kernel(为可读性做了精简、加了中文注释):

```python
# ============================================================
# Inductor 为 relu(x*a+b) 自动生成的 Triton kernel(精简示意)
# 关键: 乘、加、relu 三个算子被融进了 ONE kernel 的一次循环里
# 对照 Day4: 这正是你手写融合想达到的效果, 现在机器自动写出来了
# ============================================================
import triton
import triton.language as tl

@triton.jit
def fused_mul_add_relu_kernel(
    in_x, in_a, in_b, out_y,   # 输入/输出张量的显存指针
    n_elements,                 # 总元素个数
    BLOCK: tl.constexpr,        # 每个 program 处理多少元素(编译期常量)
):
    pid = tl.program_id(0)                    # 我是第几个线程块
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # 我负责的元素下标
    mask = offs < n_elements                  # 越界保护(尾块可能不满)

    # —— 一次性把三个输入从 HBM 读进寄存器 ——
    x = tl.load(in_x + offs, mask=mask)
    a = tl.load(in_a + offs, mask=mask)
    b = tl.load(in_b + offs, mask=mask)

    # —— 三个算子在寄存器里连着算完, 中间结果 t1/t2 不落 HBM! ——
    t1 = x * a          # 若不融合: 这里要把 t1 写回 HBM, 下个 kernel 再读
    t2 = t1 + b         # 同理 t2 也会来回 HBM 一趟
    y  = tl.maximum(t2, 0.0)   # relu = max(·,0), 依然在寄存器里

    # —— 只有最终结果 y 写回 HBM 一次 ——
    tl.store(out_y + offs, y, mask=mask)
```

**这段代码就是本篇最值钱的 30 行**,对着它把 Day4 两笔账逐条兑现:

1. **省 launch(第一笔账)**:乘、加、relu 本该是 3 个 kernel = 3 次 launch(3 × 8.56µs)。融合后**只有 1 次 launch**。这就是 Day4 §3.1 那笔账,Inductor 自动记了。
2. **省访存(第二笔账,更值钱)**:看注释里的 `t1`、`t2`——不融合时,`x*a` 的结果 `t1` 要**写回 HBM**,下一个 kernel 再**从 HBM 读回来**加 `b`……每个中间张量一次往返 HBM(几百 ns × 元素数)。融合后 `t1`、`t2` **全程待在寄存器里**,只有最初的输入 load 一次、最终的 `y` store 一次。**这正是 Day4 §3.2"中间结果别落 HBM"的自动实现**。
3. **`tl.load`/`tl.store` 就是访存,中间的 `*`/`+`/`maximum` 就是计算**。融合的本质在代码上一目了然:**把"读一次、算好几步、写一次"塞进一个 kernel**,而不是"读-算-写、读-算-写"重复 N 遍。

> **一句话**:你 Day4 用 `addcmul` **手写**的那个"一个算子干完乘加"的融合,Inductor 在这里**自动、且对任意逐元素链条**都能生成出来。这就是"baseline"三个字的重量——它不知疲倦、覆盖全模型。

### 4.3 关键边界:Inductor 融不动的地方,正是你巨核的战场

Inductor 很强,但它的自动融合**有明确边界**。搞懂边界,你才知道手写巨核的 ≥20% 从哪来。

| Inductor **擅长**融的 | Inductor **不太融/融不动**的 | 对你的意义 |
|---|---|---|
| **pointwise 链**:一串逐元素算子(bias+激活+缩放+dropout)融成一个 kernel | **matmul / attention 这类"重算子"**:默认调 cuBLAS/cuDNN 或 FlashAttention 库,自己不生成,也**不跨它做大融合** | matmul 与它前后的 pointwise 之间那道"墙",就是你巨核要打通的缝 |
| **reduction + pointwise**:如 LayerNorm 里"求均值方差(reduction)+归一化(pointwise)"能融 | **跨 matmul 的融合**:`matmul → LayerNorm → matmul` 这种,通常被切成"matmul kernel │ 融合的 norm kernel │ matmul kernel" 三段,**段间仍要落 HBM** | 段与段之间的中间张量落 HBM,就是 Day4 第二笔账的**残余**——巨核把整条 attention/FFN 融成一个,连这些都省了 |
| 小的、形状规整的算子 | **需要 grid 级全局同步的融合**(如把整个 attention 塞进一个 kernel):Inductor **不做**,因为它按"一个算子/一段循环 = 一个 kernel"的保守边界来切 | 这正是 megakernel(AMK)干的、也是它**难**的地方(Day4 讲的跨 SM 同步),H100 上这还可能是**反主场** |

**把这张表读成一句话(务必记住)**:

> **Inductor 帮你把"松散的逐元素小算子"自动焊成中等大小的 kernel,但它不敢跨越 matmul/attention 这些大块头去做"整层级"的巨型融合——因为那需要 grid 级同步、寄存器/shared memory 的精细手工调度,超出了自动编译器的保守边界。而"整层焊成一个巨核"恰恰是你小米课题主线 3 的定位。所以你和 Inductor 的关系不是"你替代它",而是"你在它融不动的接缝处,用手写巨核再抠出 ≥20%"。**

这也解释了为什么课题指标要拿 `torch.compile` 当分母:它已经把"容易摘的果子"(pointwise 融合、省 launch)全摘了,你手写巨核的提升**必须来自它摘不到的高处**——跨大算子的融合、消除段间 HBM 往返、为固定形状定制的调度。**打不过它 = 你只做到了自动编译器免费送的那部分。**

## 5. graph break:加速为什么会"悄悄漏光"

这是使用 `torch.compile` **最常见的翻车点**,也是你调优时第一个要查的东西。

### 5.1 是什么:图被 Python 逻辑"切断"了

**graph break(图断裂)是什么**:Dynamo 符号化执行字节码时(§3.3),撞到一条它**没法放进图里**的指令,只好**当场把图切断**——前半段编译成快 kernel,断点这里**退回普通 Python 解释器**跑一下,然后**再开一张新图**继续抓后半段。

**为什么这是灾难**:一次 graph break 把一张完整的图切成两半,意味着——

1. **断点处退回 eager(逐算子解释执行)**,这里没有任何融合、没有 CUDA Graph,launch 开销全回来了。
2. **两段图各自的边界要"落地"**:前段的输出得从 GPU 同步回来、交给 Python、再喂给后段,**打断了 CUDA 的异步流水**(呼应 Day4:CPU 又开始频繁等 GPU)。
3. **断点两侧无法跨段融合**:本可融进一个 kernel 的算子,被墙隔开了。

> **类比**:你请了自动化生产线(compile),结果流水线中间**每隔几步就要停下来,把半成品搬下线、让一个老师傅手工检查一下、再搬回线上**。搬上搬下的功夫,比省下来的还多——这就是 graph break 多了之后"编译了却没多快"甚至"更慢"的原因。

### 5.2 常见触发源(背下来,查 break 先查这几样)

- **`print(x)` / 日志 / `.item()` / `.tolist()`**:任何要把张量**真实数值**取到 Python 侧的操作。因为符号化执行只有 FakeTensor(没有真值),一要真值就断。
- **依赖张量值的控制流**:`if x.sum() > 0:`、`while loss > eps:`。分支走哪边取决于运行时数据,编译期不知道 → 断。(注意:依赖**形状**的 `if x.shape[0] > 1` 通常不断,因为形状是已知的静态信息。)
- **调用 Dynamo 不认识的库**:一段 numpy、一个自定义 C 扩展、某些没被支持的 Python 内置。
- **数据依赖的动态形状**:输出形状取决于数据内容(如 `x[x > 0]` 这种 boolean mask 索引,结果长度不定)。

### 5.3 底层代码:一个 `.item()` 是怎么断图的

看一段"看似人畜无害、实则把图切成两半"的代码,以及 Dynamo 内部大致怎么处理它:

```python
# 你写的 forward(简化)——中间偷偷取了个标量做判断
def forward(self, x):
    a = self.norm(x)          # ← 图 A 开始:norm 能进图
    a = torch.relu(a @ self.w)# ← 还在图 A:matmul + relu 能进图
    s = a.sum().item()        # ← 【断!】.item() 要真实数值 → graph break
    if s > 0:                 #     这个 if 依赖上面的真值, 也进不了图
        a = a * 2             # ← 图 B 开始:这里之后是一张全新的图
    return self.proj(a)       # ← 还在图 B
```

Dynamo 内部逢到 `.item()` 那条字节码时,做的事等价于:

```python
# ============================================================
# 简化伪代码: Dynamo 撞到无法符号化的操作时怎么办
# ============================================================
def trace_bytecode(code, frame):
    graph = FxGraph()
    for instr in bytecode_of(code):        # 逐条字节码往下走
        if is_tensor_op(instr):
            graph.add_node(instr)          # 张量运算 → 记进图
        elif needs_concrete_value(instr):  # 如 .item()/.tolist()/print
            # ↓ 关键: 没法用 FakeTensor 符号化 → 就地封口
            compile_and_emit(graph)        # ① 把已抓的"图 A"编译掉
            emit_eager_fallback(instr)     # ② 断点这条退回普通解释器跑
            graph = FxGraph()              # ③ 开一张空图, 继续抓"图 B"
        # ... 依赖数据的分支同理
    compile_and_emit(graph)                # 收尾编译"图 B"
```

**关键认知**:图从 1 张变 3 段(图 A │ eager 的 `.item()`+`if` │ 图 B),每个"│"都是一次异步流水的中断。**修法**:把 `.item()`、`print`、数据依赖的 `if` 尽量挪出热路径(hot path);实在要判断,能改成"用张量运算表达"就别取标量(如用 `torch.where` 代替 `if`)。这就是工业界"写 compile-friendly 模型"的核心手艺。

### 5.4 怎么查:`torch._dynamo.explain`

知识点里点名的工具。它**不真跑编译**,而是告诉你"这段代码会被切成几张图、每次 break 断在哪、为什么断":

```python
import torch
import torch._dynamo as dynamo

def forward(x, w):
    a = torch.relu(x @ w)
    s = a.sum().item()        # 故意埋一个 break
    if s > 0:
        a = a * 2
    return a

# explain 返回一份报告: 图的段数、break 数、每个 break 的原因和位置
explanation = dynamo.explain(forward)(torch.randn(4, 4), torch.randn(4, 4))
print(explanation)
# 你会看到类似:
#   Graph Count: 2          ← 被切成 2 张图
#   Graph Break Count: 1    ← 1 次断裂
#   Break Reasons: ['Tensor.item() ...']  ← 断因: 调了 .item()
#   Break at: forward line 3               ← 断在第 3 行
```

> **工业习惯**:上线 `torch.compile` 前,先跑 `explain` 看 **Graph Break Count**。理想是 **0**(整个 forward 一张图)。生产里 LLM 推理框架(vLLM 等)都会刻意把模型写成"无 graph break",甚至设 `torch._dynamo.config.error_on_graph_break = True` 让一有 break 就报错,逼你消灭它。对你课题的意义:**你要优化的那张图,必须是"完整一张图",否则你和 Inductor 比的根本不是同一个东西。**

## 6. 动手实测:before / after 加速 + 抓 Inductor 生成代码

到了知识点要求的三件动手事:(1) `torch.compile(model)` 实测 before/after;(2) `TORCH_LOGS="output_code"` 看它融了什么;(3) `torch._dynamo.explain` 找 graph break。下面给一份**可直接跑的完整脚本**,同时附上**在哪台机器跑**的关键提醒。

### 6.1 ⚠️ 先说环境:这个实验必须在 Linux + GPU 上跑

这是 Day4 §6.2 陷阱 5 你已经踩过的坑,今天必须正面说清,否则你在本机 RTX 5060 上会白忙半天:

> **`torch.compile` 的 GPU 加速依赖 Triton,而 Triton 官方不支持 Windows。** 在你本机 Windows 上 `torch.compile(model)` 跑 GPU,大概率报 `TritonMissing` / 编译后端拿不到、或悄悄退回 eager 不加速。**真正能看到 Inductor 生成 Triton kernel、量出 before/after 的地方,是你那台 H100 的 Linux 环境**(`~/YSQ/AutoMegaKernel`,`module load CUDA/12.4` 那套)。

所以本节脚本的定位:**在 H100 Linux 上跑**。你本机可以先读懂脚本逻辑;真实数据去 H100 采。这本身也是一条工业真相——**严肃的 kernel 级 / 编译级优化都在 Linux + NVIDIA 上做**,和你 AMK 的主战场完全一致。

### 6.2 完整脚本 `bench_torch_compile.py`

```python
# ============================================================
# bench_torch_compile.py
# 目的: 量出 torch.compile 的 before/after 加速, 并抓 Inductor 生成代码
# 运行环境: Linux + NVIDIA GPU(H100), PyTorch>=2.1, 已装 Triton
#   H100 上: module load CUDA/12.4 后, python bench_torch_compile.py
# 看生成代码: TORCH_LOGS="output_code" python bench_torch_compile.py
#   (或 TORCH_LOGS="output_code,graph_breaks,recompiles" 看更全)
# ============================================================
import torch
import torch.nn as nn

torch.manual_seed(0)                    # 固定 seed, 结果可复现(计划的产出规范)
assert torch.cuda.is_available(), "这个实验要 GPU; Windows 上没有 Triton, 去 H100 跑"
dev = "cuda"

# ---- 一个"像 Transformer 一层 FFN"的小模型: 多个 pointwise 算子相邻 ----
# 特意这样设计: LayerNorm→Linear→GELU→Linear→残差, 里面一堆逐元素算子,
# 正好给 Inductor 的 pointwise 融合表演的舞台(呼应 Day4 融合两笔账)。
class FFNBlock(nn.Module):
    def __init__(self, d=1024, hidden=4096):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.ln(x)
        h = self.act(self.fc1(h))       # Linear + GELU: GELU 是逐元素, 可被融进来
        h = self.fc2(h)
        return x + h                    # 残差加: 又一个逐元素算子, 可被融

model = FFNBlock().to(dev).eval()

# decode 场景: batch=1, 序列=1(一次一个 token), 最能暴露 launch-bound
# (呼应 Day1: decode M=1, 算术强度≈1, memory/launch-bound)
x = torch.randn(1, 1, 1024, device=dev)

# ---- 计时工具: 必须用 CUDA Event, 不能用 time.time()(Day2/Day4 反复强调) ----
def bench(fn, x, warmup=30, iters=100):
    for _ in range(warmup):             # 热身: 躲开首次 JIT 编译/选 kernel/显存池初始化
        fn(x)
    torch.cuda.synchronize()            # 等 GPU 把热身干完, 再开始计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()            # 等所有 kernel 真正执行完(异步!)
    return start.elapsed_time(end) / iters  # 每次迭代的毫秒数

with torch.no_grad():                   # 推理: 关梯度, 省显存也更贴真实部署
    # baseline: 原生 eager
    t_eager = bench(model, x)

    # after: 默认 compile(Dynamo 抓图 + Inductor 融合生成 Triton)
    compiled = torch.compile(model)                       # mode="default"
    t_default = bench(compiled, x)

    # after++: reduce-overhead(默认 + 自动套 CUDA Graph, 专治 launch-bound)
    compiled_ro = torch.compile(model, mode="reduce-overhead")
    t_ro = bench(compiled_ro, x)

print(f"eager                : {t_eager:.4f} ms/iter  (baseline)")
print(f"compile default      : {t_default:.4f} ms/iter  "
      f"({t_eager/t_default:.2f}x)")
print(f"compile reduce-overhd : {t_ro:.4f} ms/iter  "
      f"({t_eager/t_ro:.2f}x)  ← 加了自动 CUDA Graph")
```

**预期现象(H100,数量级示意,别死记具体值):**

- `compile default` 相对 eager 有加速,主要来自 **Inductor 把 LayerNorm/GELU/残差这些 pointwise 融进更少的 kernel**(省访存 + 省一部分 launch)。
- `reduce-overhead` 通常**更快**,因为它在融合之上又套了 **CUDA Graph**,把 decode 那串固定的 launch 序列**一次性录制回放**,直接干掉 Day4 量的那 8µs/launch × N。**这一挡就是你 Day4 手录 CUDA Graph(7.98×)的自动版**。
- batch=1、seq=1 这种极端 launch-bound 场景,`reduce-overhead` 的收益往往最夸张——正好呼应 Day4"decode 最怕 launch"。

### 6.3 三挡 `mode` 分别多做了什么(对着 Day4 的两笔账看)

`torch.compile(model, mode=...)` 有三挡常用模式,**它们的区别正好对应 Day4 那两笔账 + autotune**:

| mode | 做了什么 | 对应 Day4 的账 | 代价 / 适用 |
|---|---|---|---|
| `"default"` | Dynamo 抓图 + Inductor **pointwise 融合** + 生成 Triton | 主要省**访存**(融合)+ 一部分**启动** | 编译快、最稳,默认先用它 |
| `"reduce-overhead"` | default 之上**自动套 CUDA Graph** | 再狠狠省**启动**(把固定 launch 序列录制回放) | 吃额外显存(要存图的静态输入);形状一变要重录。**decode 首选** |
| `"max-autotune"` | default 之上,对每个 kernel **实测多个候选实现挑最快**(含 Triton 版 matmul、CUDA Graph) | 省访存 + 启动 + **调 kernel 本身** | **编译极慢**(要跑一堆 benchmark),适合"编译一次、部署很久"的生产模型 |

**给你的实操建议**:量 baseline 时,**三挡都量**。因为你手写巨核要打的,不是最弱的 `default`,而是这台机器上 `torch.compile` 能开出的**最强一挡**(通常是 `max-autotune`)。拿最弱的比会虚高你的提升,自己骗自己。

> **一个反直觉但重要的点**:`reduce-overhead` 不一定总比 `default` 快。如果模型不是 launch-bound(比如大 batch 的 prefill,kernel 本身就很大),CUDA Graph 省的那点 launch 无关痛痒,反而因为静态输入拷贝等开销可能持平甚至略慢。**所以"哪挡最快"是量出来的,不是想出来的**——这正是你 Day2/Day4 建立的"一切用数字说话"的延续。

## 7. 灵魂:为什么"不量 torch.compile,你的 ≥20% 就是空话"

这一节回答开篇第 6 题,也是整篇最该记住的一句。

你小米课题的硬指标是:**相较现有算子级优化方案,整体推理性能提升 ≥20%**。拆开这句话里的每个词对你意味着什么:

- **"现有方案"是谁?** 不是没优化的 eager。业界任何人拿到一个 PyTorch 模型,第一件事就是加一行 `torch.compile`。所以**真正的 baseline 是开了 `torch.compile`(且是最强 mode)的版本**,不是裸模型。拿裸 eager 当分母,你随便就有 2-8× 的"提升",但这是**自欺**——评审一句"你和 torch.compile 比过吗"就把你问倒。
- **"提升 ≥20%"的分母是什么?** 是 `torch.compile` 的那个数字。**你今天不把它量出来、量准(固定 seed、warmup、CUDA Event、三挡全测),你后面 W8 手写巨核跑出任何数字,都无法算出"提升了百分之几"——因为没有分母。**
- **你手写巨核的"合法性"从哪来?** 就从 §4.3 那张**融合边界表**来:Inductor **只做 pointwise 自动融合,不敢跨 matmul、跨 reduction 做激进的全 attention 融合**。你手写巨核(像 AMK 那样把整个 attention/FFN 塞进一个 kernel)正是**吃掉 Inductor 不敢吃的那部分**——这就是那 ≥20% 的**理论来源**。你要能指着这张表说:"Inductor 在这里断成了 5 个 kernel,我融成 1 个,省下的就是我的提升空间。"

> **一句话钉死**:`torch.compile` 是你课题的**标尺**,不是你的对手的"随便一个版本"。**先量准标尺(今天),才能证明你的巨核(W8)真的更长。** 这也是你 AMK 任务的直接延续——AMK 论文里 H100 上打不过 cuBLAS/vLLM,靠的也是同一套"和最强 baseline 比、用真实硬件数字说话"的诚实。

## 8. 工业实践:常见陷阱与调试技巧

这些是把 `torch.compile` 从"demo 能跑"用到"生产/科研可信数据"之间的坑,每一条你之后测 baseline 都会撞到。

1. **第一次调用巨慢是正常的,别把它算进去**。`torch.compile` 是 JIT——首次触发时要 trace + 编译(`max-autotune` 还要跑一堆 benchmark),可能慢几秒到几十秒。**必须 warmup**(脚本里 `warmup=30`),第一次的时间**绝不能**计入 baseline。这是 Day4 §6.2 陷阱 2 的升级版:那里 warmup 躲的是 cuBLAS 选 kernel,这里躲的是**整个编译过程**。

2. **形状一变就 recompile(重编译),是隐形的性能杀手**。§3.2 说过编译结果带 guard,输入形状/dtype 变了 guard 失效就重编。decode 里序列长度每步都在变(KV Cache 越来越长),**朴素写法会导致每步都重编,比不 compile 还慢**。工业解法:① `torch.compile(model, dynamic=True)` 让它编译支持动态形状的版本;② 或"按长度分桶",每桶一份编译(和 Day4 CUDA Graph 分桶同一个思路)。**用 `TORCH_LOGS="recompiles"` 能看到它到底重编了几次**——数字大说明你踩坑了。

3. **graph break 静默吃掉你的加速**。有 break 时它不会报错,只是**默默退回 eager 跑那一段**,你看到的"没怎么加速"往往就是这个。上线前务必 `torch._dynamo.explain` 看 break 数(§5),理想是 0。

4. **`reduce-overhead`(CUDA Graph)对"就地改写输入"很敏感**。CUDA Graph 假设输入张量的**显存地址固定**,如果你的代码每次传进去的是新分配的张量,可能出错或悄悄失效。表现是"结果对但没加速"或"偶发数值错"。调试时先退回 `default` 排除是不是 CUDA Graph 的锅。

5. **别用 `time.time()` 测,永远用 CUDA Event**。CUDA 异步,`time.time()` 测到的是"塞队列"时间(接近 0),不是真实执行时间(Day2/Day4 反复强调)。脚本里 `bench()` 就是标准范例:warmup → synchronize → Event 计时 → synchronize。

6. **`TORCH_LOGS` 是你的 X 光机**,记住这几个开关:
   - `output_code`:看 Inductor 生成的 Triton kernel 长啥样(§4.2)——**看它融了什么**。
   - `graph_breaks`:每次 break 打一行,定位断点。
   - `recompiles`:每次重编打一行 + 原因,抓形状抖动。
   - 组合用:`TORCH_LOGS="output_code,graph_breaks,recompiles" python bench_torch_compile.py`

7. **数值一致性要主动验**。融合/CUDA Graph/autotune 可能换了算子实现或累加顺序,导致和 eager 有**微小数值差异**。你课题指标明确要求"数值一致或误差可控可解释",所以量加速的同时,顺手 `torch.allclose(eager_out, compiled_out, rtol=1e-3, atol=1e-3)` 验一下,别只看速度不看对错——这也是 AMK 里 `matches_eager:true` 那一栏的同款检查。

## 9. 自测题(先合上笔记答,再翻对应节核对)

1. `torch.compile` 的三级流水线是哪三段?各自负责什么?中间交接的数据结构叫什么?→ §2
2. **(核心)** TorchDynamo 凭什么"不改一行代码"就能抓到图?讲到 PEP 523 帧求值 hook 这一层。→ §3.2
3. 什么是符号化执行 + FakeTensor?为什么"假张量空跑一遍"就能建出图、却在 `if x.sum()>0` 处断掉?→ §3.3 / §5
4. Inductor 的 pointwise 融合,和你 Day4 手写融合是同一件事吗?它的**融合边界**在哪(哪些它不敢融)?→ §4.2 / §4.3
5. `default` / `reduce-overhead` / `max-autotune` 各多做了什么?哪挡对应 Day4 的 CUDA Graph?→ §6.3
6. 什么是 graph break?为什么一个 `print` 就能腰斩你的加速?怎么定位它?→ §5
7. **(灵魂)** 为什么说"不先量 `torch.compile`,你手写巨核的 ≥20% 就是空话"?你的提升空间理论上来自 Inductor 的哪个"不敢"?→ §7 / §4.3
8. 把"Day4 手写融合两笔账 → Inductor 自动做同样的事 → 但融合边界有限 → 手写巨核吃掉它不敢融的部分 → 课题 ≥20%"连成一条因果链。→ §4 / §7

> 讲不出来就回去重读对应节。第 2、4、7 题是今天的题眼——尤其第 4 题的"融合边界",直接决定你 W8 巨核的价值定位。

## 10. 与已有笔记 / 课题主线的串联

| 关联 | 关系 |
|---|---|
| [W7 Day4 · Kernel Launch 与融合](./W7_Day4_KernelLaunch开销与算子融合_巨核动机.md) | **最直接的前置**。Day4 你**手写**融合、手录 CUDA Graph、量出两笔账;今天看**机器自动做同一件事**:Inductor=自动融合器,`reduce-overhead`=自动 CUDA Graph |
| [W7 Day2 · 三级 Profiler](./W7_Day2_三级Profiler工具链_torchprofiler_nsys_ncu.md) | 用同一套 **CUDA Event 计时法**量 `torch.compile` 的 before/after;`TORCH_LOGS` 是继 nsys/ncu 之后第三种"看内部"的 X 光机 |
| [W7 Day1 · Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md) | `reduce-overhead` 能不能把 decode 从 **launch-bound** 救出来,正是 Roofline 判的那个 bound;融合能不能把访存压下来,对应算术强度 |
| W6 · nanoGPT / KV Cache | 你要 compile 的真实对象就是 nanoGPT;§8 陷阱 2 的"序列变长就重编"直接命中 KV Cache decode 场景 |
| [rnn_to_transformer_evolution](./rnn_to_transformer_evolution.md) | 那里的"KV Cache 用 `torch.cat` 浪费显存"和这里"形状抖动触发 recompile"是同一类"动态形状在推理里的代价" |
| **小米课题主线 2(图级优化)** | `torch.compile` 就是**自动图级优化的公开最强 baseline**;你抓的 FX Graph 就是主线 2 要"重写"的对象 |
| **小米课题主线 3(巨核生成)** | 今天给主线 3 **立了基准桩**:§4.3 融合边界表 = 你手写巨核的**提升空间理论来源**;§7 = 你 ≥20% 的分母 |
| H100 / AMK profiling | 本实验必须在 H100 Linux 跑(§6.1);AMK 的 `matches_eager` 检查 = §8 陷阱 7 的数值一致性;AMK 打不过 cuBLAS 的诚实 = §7 "和最强 baseline 比" |

## 11. 今日产出清单(对齐计划)

- [x] `torch_compile_baseline.md`(本笔记正文,桌面按 `W7_Day5_*` 平铺;进仓时用计划里的文件名)
- [x] 讲透三级链路:**TorchDynamo(PEP 523 帧求值 hook 抓图)→ Inductor(pointwise 融合 + 生成 Triton)→ CUDA Graph**,并对应到 Day4 两笔账
- [x] 底层代码级理解:Dynamo 的帧求值 hook 伪码(§3.2)+ 符号化执行/FakeTensor(§3.3)+ Inductor 生成的 Triton kernel 逐行读(§4.2)
- [x] 配套脚本 `bench_torch_compile.py`(before/after + 三挡 mode + CUDA Event 计时 + 数值验证)
- [x] 打通因果链:自动融合有边界(§4.3)→ 手写巨核吃掉边界 → 课题 ≥20% 的分母(§7)
- [ ] (在 H100 上)实跑脚本,采 eager / default / reduce-overhead / max-autotune 四组真实数字,**这就是你巨核的 baseline 分母**
- [ ] (在 H100 上)`TORCH_LOGS="output_code"` 抓 nanoGPT / AMK small 模型的 Inductor 生成代码,记录"它融了哪几段、在哪断开",填进 AMK report
- [ ] (在 H100 上)`torch._dynamo.explain` 跑一遍目标模型,确认 graph break = 0,否则先消灭 break 再比

---

> **一句话收尾**:今天你把 `torch.compile(model)` 这行"魔法",拆成了**Dynamo 劫持 CPython 帧求值抓图 → Inductor 做 pointwise 融合并吐出 Triton kernel → CUDA Graph 省启动**这条透明的流水线,并看清它就是 Day4 你手工活的**自动版**。更关键的是,你搞懂了它的**融合边界**——它不敢跨 matmul/reduction 做激进融合,而**那道边界之外的空间,正是你 W8 手写巨核要吃下的 ≥20%**。所以今天不是"学个 API",是**为你整个课题立好那把要超越的标尺**:先量准它,你的"更好"才有分母。
