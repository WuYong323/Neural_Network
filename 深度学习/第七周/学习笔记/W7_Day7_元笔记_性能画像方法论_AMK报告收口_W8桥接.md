# W7 Day7 · 元笔记:把一周的单点串成网 —— 性能画像方法论 + AMK 报告收口 + W8 桥接

> **本笔记的唯一目标**:不是学新知识点,而是**收口**——把本周 5 天的单点(Roofline → 三级 profiler → launch/融合 → torch.compile → 量化)**串成一张能用的方法论网**,并产出你科研生涯**第一份可见痕迹**(AMK profiling 报告 v0)。读完你要能做到一件事:**面对"给我一个模型和一张 GPU,你怎么判断它的推理瓶颈在哪、为什么、该往哪优化",你能在 5 分钟内用本周的方法、工具、真实数字讲清楚**,而不再是"我以为 decode 是 memory-bound"。这一步跨过去,你就从"学 DL 的学生"变成"能做 AI Infra 性能画像的人"。
>
> **串联**:这是 [W7 学习计划](./W7_学习计划_AI_Infra主线.md) **Day7 收口**,延续你 W4/W5/W6 的元笔记格式(每周把单点串成网)。它把本周全部笔记 [Day1 Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md) / [Day2 三级 Profiler](./W7_Day2_三级Profiler工具链_torchprofiler_nsys_ncu.md) / [Day4 launch 与融合](./W7_Day4_KernelLaunch开销与算子融合_巨核动机.md) / [Day5 torch.compile](./W7_Day5_torch_compile_图优化自动版_巨核baseline.md) / [Day6 精度与量化](./W7_Day6_数值精度与量化地基_FP32_FP16_BF16_INT8.md) 收进一张总表,每项标注对应**小米课题主线**。并把 [project-amk-h100] 的真实编译报告数字整理成课题的第一份产出。
>
> **产出对齐**:本笔记 = 计划要求的 `tech_notes/week7_industrial_view.md`(本周元笔记)。§4 = `tech_notes/amk_profiling_report_v0.md`(Track B 收口,科研第一份产出雏形)。§5 = 更新后的 `inference_optimization_landscape.md`(5 术语理解程度表)。§6 = 代码清理 + ≥3 次 commit 清单。

---

## 0. 本周一句话主线

> **从"会写模型"升级到"会科学地量化模型的推理瓶颈"。**

W6 结束时,你能**定性**说"decode 是 memory-bound"。本周结束,你能**定量**地说:"我用 Roofline 算过它的算术强度≈1、脊点是 295;我用 nsys 测过 decode 一步有几十个 kernel、gap 占了多少;我知道融合能省两笔账(实测 launch 8.56µs、融合 1.61×);我知道 torch.compile 自动做了这些、是我的 baseline;我知道量化是第二把杀器、且必须配误差度量。"——**每一句话背后都有数字、有图、有代码**。这就是本周的全部价值。

---

## 1. 性能画像方法论总表:一条从"判断"到"优化"的流水线

这是本周最该沉淀下来的东西——**一套遇到任何推理负载都能套用的固定流程**。别把本周五天看成五个孤立知识点,它们其实是**一条流水线的五道工序**,前一道的输出正好是后一道的输入。

### 1.1 先看这张总表(本周的骨架)

| 阶段 | 工具/方法 | 回答什么问题 | 本周实测/关键数字 | 对应课题主线 |
|---|---|---|---|---|
| **① 判 bound** | **Roofline**(Day1) | 这个负载卡在**算力**还是**带宽**? | decode 算术强度 ≈ **1**,H100 脊点 ≈ **295** → 远低于脊点 → **memory-bound** | 主线 1(性能画像) |
| **② 逐级定位** | **三级 profiler**(Day2):torch.profiler → nsys → ncu | 到底**哪个算子/哪段时间线/哪个 kernel** 在拖后腿? | decode 一步几十个小 kernel + 大片 gap(望远镜→显微镜逐级放大) | 主线 1 |
| **③ 找到病根** | **launch 开销 + 融合两笔账**(Day4) | 时间花在"算"还是"等 CPU 发命令"? | 单次 launch **8.56µs**、融合省访存 **1.61×**、CUDA Graph **7.98×** | 主线 3(巨核动机) |
| **④ 自动优化 baseline** | **torch.compile**(Day5):Dynamo 抓图 → Inductor 融合 → CUDA Graph | 机器能自动省到什么程度?我的手写巨核要打的是谁? | 自动 pointwise 融合 + 自动 CUDA Graph;**这是主线2/3 的 ≥20% 分母** | 主线 2(图级优化) |
| **⑤ 第二把杀器** | **精度 / 量化**(Day6):FP16/BF16/INT8 + 三把误差尺子 | 能不能让"每次搬的字节"变小?省了多少、错了多少? | 精度减半 → 搬运减半 → 提速;误差用 allclose/cosine/PPL 度量 | 主线 4(误差可控可解释) |

### 1.2 为什么是这个顺序?(流水线的内在逻辑)

这个顺序不是随便排的,它是一条**严格的因果链**,每一步都为下一步提供"没有它就没法做"的前提:

> **判 bound(①)→ 才知道该往哪优化。定位(②)→ 才知道具体改哪里。找病根(③)→ 才知道为什么慢(launch/访存)。量 baseline(④)→ 才有分母去证明"更好"。上第二把杀器(⑤)→ 在融合之外再省一层,且必须证明误差可控。**

**类比(把这套方法论想成医院看病)**:

- **① Roofline = 分诊台**:先判断你是"心脏问题(算力)"还是"血管问题(带宽)",挂错科后面全白费。
- **② 三级 profiler = 三级检查**:先体检报告(torch.profiler 看哪个器官分数低)→ 再动态心电图(nsys 看时间线上什么时候出问题)→ 最后造影(ncu 钻进单个 kernel 看堵在哪)。**永远从粗到细,不能上来就造影**(ncu 极慢,没定位等于拿显微镜扫全身)。
- **③ launch/融合 = 确诊病根**:查出来是"血管里塞了太多小血栓(几十个小 kernel 的 launch 开销 + 中间张量反复往返 HBM)"。
- **④ torch.compile = 标准疗法**:医院现成的标准治疗方案能治到什么程度——你的"新疗法(手写巨核)"必须和它比,不能和"不治疗(eager)"比。
- **⑤ 量化 = 第二种药**:换一种机理的药(减小数据本身),但**必须做药物副作用监测(误差度量)**,证明"疗效有、副作用可控"。

> **一句话钉死方法论**:**判 bound → 逐级定位 → 找病根 → 量 baseline → 上优化并验证误差**。这五步,是你以后拿到任何一个新模型/新硬件,都能照着走一遍的固定套路。本周就是把这条套路,在 nanoGPT + H100 + AMK 上**走了一整遍**。

## 2. 完整回答:"nanoGPT 在 H100 上瓶颈在哪、为什么、怎么优化?"

这是本周的**毕业考**——计划的完成标准是"5 分钟内对懂 PyTorch 但不懂 Infra 的同学讲清"。下面给一份**用本周数字支撑**的完整答案,你要能脱稿讲出来(不是背,是理解后复述)。

### 2.1 瓶颈在哪:decode 阶段,memory-bound

先分两阶段(W6 §5),因为它俩瓶颈完全不同:

- **prefill(预填充,一次并行处理整段 prompt)**:算术强度 ≈ batch×序列长度(Day1),很高 → **compute-bound(受算力限制)**,GPU 算力吃得满,不是主要矛盾。
- **decode(解码,自回归一次生成一个 token)**:算术强度 ≈ **1**(Day1),因为**每个权重只被当前这 1 个 token 用一次、用完即弃、零复用** → 远低于 H100 脊点 295 → **memory-bound(受内存带宽限制)**。

> **结论**:**nanoGPT 在 H100 上的推理瓶颈,主要在 decode 阶段,是 memory-bound**——GPU 的算力大把闲着,时间几乎全花在"把权重和 KV Cache 从显存搬进来"上。

### 2.2 为什么:两个可量化的原因

不是"我觉得",是本周量出来的两笔账:

1. **权重零复用导致的访存瓶颈(Day1 数学根源)**:decode 每步搬整个模型的权重,只算 1 个 token,算术强度恒为 1。这是**数学决定的**,换多大的矩阵都改不了。
2. **几十个小 kernel 的 launch 开销 + 中间张量反复往返 HBM(Day2 看到 + Day4 量化)**:nsys 时间线上 decode 一步是**几十个小 kernel + 大片 gap**(GPU 空等 CPU 发命令,launch-bound)。本机实测单次 launch 固定开销 **8.56µs**,几十个 kernel 光启动就几百 µs;加上每两个算子间中间结果落 HBM 又读回(每趟几百 ns)。

> 一句话:**慢在"搬得多(权重零复用)+ 搬得碎(几十个小 kernel,launch 和中间访存都是浪费)"**。

### 2.3 怎么优化:四把递进的杀器(附本周量化的效果)

按"性价比 / 从自动到手写"排序,正好是本周②③④⑤的应用:

| 优化手段 | 治的是哪个病 | 本周依据 | 属于课题哪条主线 |
|---|---|---|---|
| **① 融合 / CUDA Graph** | 少发命令、中间结果不落 HBM | Day4:融合 1.61×、CUDA Graph 7.98× | 主线 3 / 2 |
| **② torch.compile(先开)** | 自动做①,免费拿走"容易摘的果子" | Day5:Inductor 自动融合 + reduce-overhead 自动 CUDA Graph | 主线 2(baseline) |
| **③ 手写 megakernel(AMK)** | 把整层融成一个大核,吃掉 torch.compile 融不动的(跨 matmul/attention) | Day4 §4.3 融合边界 = 提升空间来源 | 主线 3(主攻,≥20%) |
| **④ 量化 KV Cache/权重(FP16/INT8)** | 减小每次搬运的字节数 | Day6:精度减半→搬运减半;误差用三尺子验证 | 主线 4 |

> **优化的总策略**:**先用 torch.compile 拿走自动收益(建 baseline)→ 再手写巨核吃掉它融不动的接缝(争 ≥20%)→ 叠加量化把 KV Cache/权重的搬运字节再砍半 → 全程用 Roofline/nsys 量提升、用三把尺子验证误差**。这就是把本周五天串成的**一条完整优化路线**,也正好覆盖小米课题的四条主线。

### 2.4 一张因果链图(把上面全串起来)

```
decode 自回归(M=1)
   │  权重零复用 → 算术强度≈1  (Day1)
   ▼
memory-bound + launch-bound
   │  几十个小 kernel(Day2 nsys)+ 每次 launch 8.56µs + 中间张量落 HBM(Day4)
   ▼
优化:少搬 + 搬得整 + 搬得轻
   ├─ 融合/CUDA Graph:少发命令、中间不落 HBM  (Day4)   ← 主线3/2
   ├─ torch.compile:自动做上面 = baseline        (Day5)   ← 主线2
   ├─ 手写巨核 AMK:吃掉自动融不动的接缝          (→W8)   ← 主线3
   └─ 量化 FP16/INT8:每次搬的字节减半            (Day6)   ← 主线4
        │  必须配三把尺子(allclose/cosine/PPL)验证误差
        ▼
   提升用 Roofline+nsys 量(主线1)、误差可控可解释(主线4)
```

## 3. 串联表:本周内容与已有笔记的血脉(延续 W4/W5/W6 元笔记格式)

| 本周内容 | 关联的已有笔记 | 关系 |
|---|---|---|
| Roofline 定量画像(Day1) | W5 `flops_vs_latency.md` 的 Roofline 雏形 | 从"知道有这张图"到"能用它算脊点、判 bound" |
| 三级 profiler 工具链(Day2) | W6 §8 nanoGPT profiler(玩具版)、W5D6 chrome trace | 从"跑一次 profiler 看看"到"望远镜→显微镜逐级放大的方法论" |
| kernel launch & 融合(Day4) | W6 §5.4 KV Cache 访存代价、Day3 fused QKV"省搬运" | 从"融合能加速"到"量化成两笔账:8µs/launch + 中间张量 HBM 往返" |
| torch.compile 图优化(Day5) | Day4 §4.4"torch.compile 是巨核 baseline" | 兑现:它 = 自动融合 + 自动 CUDA Graph,主线2/3 的分母 |
| 量化数值一致性(Day6) | **W6 §4.4** 优化版 vs 朴素版数值对比 | 正式升级:从 `torch.equal` 到"允许误差但用三把尺子度量并解释" |
| KV Cache / 两阶段(全周背景) | W6 全篇 + `rnn_to_transformer_evolution.md` | 本周所有"decode memory-bound"的讨论都建立在 W6 的 KV Cache 之上 |

## 4. 【Track B 收口】AMK profiling 报告 v0 —— 你科研的第一份可见痕迹

这是本周**最有分量**的产出:把你在 H100 上对 AMK(AutoMegaKernel)的观察,整理成一份**别人能看懂、能复现**的报告。它是你简历"第一段科研经历"的第一个实体。

> **写这份报告的心法(最重要,决定它专不专业)**:**诚实**。你手上的 AMK 编译报告明确标注了"latency 是 cost-model 预测、GPU 路径未完全接通、correctness 判 FAIL"。一份**诚实标注了这些边界**的报告,比一份吹嘘"H100 上快了 N 倍"的报告**专业一百倍**——这正是 AMK 论文自己的态度("We do not claim numbers we did not measure"),也是你相较师姐工作的**独特价值**:你有真 H100 + nsys 权限,能把预测数换成实测数。

下面是可以直接落成 `amk_profiling_report_v0.md` 的正文骨架(数字来自你桌面那份真实编译报告 `...Llama-3.1-8B-Instruct.h100.report.md`):

### 4.1 报告正文(可直接拷贝为 amk_profiling_report_v0.md)

```markdown
# AMK Profiling Report v0 — Llama-3.1-8B on single H100

## 1. 硬件与环境配置(可复现前提)
- GPU: NVIDIA H100-80GB-HBM3(学校集群,登录节点 hn33)
- 环境: uv + `module load CUDA/12.4`(提供 nvcc),torch 2.11+cu128
- 代码: ~/YSQ/AutoMegaKernel(AMK,arXiv 2606.09682,Llama-only megakernel 合成器)
- 模型: Llama-3.1-8B-Instruct(weights ≈ 16060 MB)
- 关键提醒: 每个新终端必先 `module load CUDA/12.4`

## 2. 编译产物规模(megakernel 要编排的量)
- schedule id: sch_2f8b213192,IR/ABI: 0.2.0 / 0.2
- **tasks: 5826  buffers: 905  counters: 611**
  → 一个 8B 模型 decode 展开成 ~5826 个 task。若每个独立 launch,
    按 Day4 实测 8.56µs/launch 估算,光启动就是天文数字 → 这正是巨核存在的理由。

## 3. 延迟画像(⚠️ 预测值,非实测,见 §5 诚实声明)
- value: 719.80 µs/token(PREDICTED,analytic cost model)
- region breakdown(µs):
  | region    | µs      | 占比   |
  |-----------|---------|-------|
  | attention | 364.33  | ~50%  |  ← 融合主战场之一
  | mlp       | 251.76  | ~35%  |  ← 融合主战场之二
  | lm_head   | 4.61    | ~1%   |
  | other     | 99.10   | ~14%  |
- HBM-bandwidth roofline floor: 4794.19 µs(当前 15% of bound,666% HBM 利用率)
  → 呼应 Day1 Roofline:这是带宽下限的量化坐标。

## 4. 正确性(reference VM vs eager PyTorch)
- verdict: **FAIL**
- max abs err: 9.219e-01   top-1 agreement: 1.0000
  → 解读(接 Day6 三把尺子):逐元素误差大(尺子①判 FAIL),
    但 top-1 一致率 = 1.0(尺子③替身:每步选的 token 都没变)。
    这正是 Day6 §6.2 说的"allclose 判死刑不判无罪"的真实案例——
    需进一步看 logits cosine / 端到端 PPL 才能判断是否"误差可控可解释"。

## 5. 诚实声明(报告可信度的核心)
- Correctness 由 CPU reference VM(bit-exact 调度语义)对 eager 证明。
- Latency 是 cost-model 预测;该模型的 GPU 端到端路径尚未完全接通(gpu_mismatch)。
- 不声称任何未实测的 H100/B200 数据。

## 6. 下一步(我的独特贡献点)
- 用真 H100 + nsys 抓 `.nsys-rep`(多 iteration),把 §3 的**预测延迟换成实测**。
- 按 Day2 三级流程:torch.profiler 定位大头 → nsys 数 kernel 数/gap/launch 占比 → ncu 解剖热点 kernel。
- 定位论文点名的 future work:megakernel 每 tile 一次 grid 级跨 SM 同步瓶颈(H100 反主场根源)。
```

### 4.2 这份报告为什么这么写(方法论落地)

- **§2 tasks=5826** 不是随便贴的数字——它直接对应 Day4"几十上千个算子若独立 launch,开销爆炸"的论断,是**巨核存在理由的实证**。
- **§4 correctness FAIL 但 top-1=1.0** 是全报告最有教学价值的一行:它是 Day6 §6"三把尺子各有盲区"的**真实翻车现场**——只看 max abs err(0.92,很大)会判死刑,但 top-1 一致率 1.0 说明"决策没变"。到底能不能用?**必须补 logits cosine 和 PPL 才能下结论**。这就是你报告里能写出的、比别人深一层的分析。
- **§5 诚实声明** 是这份报告的"专业度签名"。科研诚信不是空话,它是你和"跑个数就吹"的区别。

## 5. 更新 `inference_optimization_landscape.md`:6 个术语的理解程度

这是你追踪"推理优化全景里,每个关键术语我到什么理解程度"的表。诚实打分——**能解释动机 < 能讲原理 < 能实现+量化**,别高估自己。

| 术语 | 中文 | W6 末 | **W7 末(现在)** | 目标(W8+) |
|---|---|---|---|---|
| **KV Cache** | 键值缓存 | 已完整实现+解释 | **已实现+解释,且有 Roofline 定量佐证**(算术强度≈1 的数学根源) | 保持,叠加量化(FP16 KV) |
| **quantization** | 量化 | (未列) | **新增:能解释位布局(FP16/BF16/INT8)+ 误差度量(三把尺子)** | 在真实模型上量 PPL |
| **PagedAttention** | 分页注意力 | 能解释动机 | 能解释动机(KV Cache 显存碎片化催生) | 读 vLLM 源码级 |
| **continuous batching** | 连续批处理 | 能解释动机 | 能解释动机(decode memory-bound → 拼 batch 提吞吐) | 动手跑 vLLM |
| **speculative decoding** | 投机解码 | 仅知名字 | 仅知名字 | W8+ 补原理 |
| **tensor parallelism** | 张量并行 | 仅知名字 | 仅知名字 | W8+ 补原理 |

> **本周这张表的两个实质进步**:① `quantization` 从"没列"到"能讲位布局 + 误差度量"(Day6);② `KV Cache` 从"实现了"到"知道它为什么慢的**数学根源**"(Day1 算术强度)。其余待 W8+。**这张表就是你的'能力地图',面试/汇报时照着讲,清楚知道自己边界在哪。**

## 6. 代码清理 + commit(计划要求 ≥3 次有意义提交)

本周产出了几个脚本,收口时清理并提交。**"有意义的 commit"= 一次提交对应一个能独立说清的成果**,不是把一堆文件一次性 `git add .`。

```bash
# 环境: 本机 git bash。⚠️ 你的 home 目录是 git 仓库且有大量无关文件,
#   务必只 add 本周的产出文件, 不要 git add .(会把整个用户目录卷进来)。

# —— commit 1: Roofline 脚本(Day1 产出)——
git add tech_notes/roofline_nanogpt.md scripts/roofline.py
git commit -m "W7: add Roofline analysis for nanoGPT decode (arithmetic intensity vs H100 ridge point)"

# —— commit 2: profiler 工具链 + launch/融合微基准(Day2/Day4)——
git add tech_notes/profiling_toolchain.md tech_notes/kernel_launch_and_fusion.md scripts/bench_launch_fusion.py
git commit -m "W7: three-level profiler workflow + launch overhead & fusion microbench (8.56us/launch, 1.61x fusion)"

# —— commit 3: torch.compile baseline + 量化(Day5/Day6)——
git add tech_notes/torch_compile_baseline.md tech_notes/precision_and_quantization.md scripts/bench_torch_compile.py scripts/bench_precision.py
git commit -m "W7: torch.compile baseline + FP16/INT8 quantization notes with error metrics"

# 查看提交历史确认
git log --oneline -5
```

> **常见陷阱**:① 在 home 目录直接 `git add .` 会把 `AppData/`、`.ssh/` 等敏感/无关内容提交——**只精确 add 产出文件**;② commit message 写"update""fix"这种没信息量的——**写清"加了什么、量出了什么数字"**,以后回看和给导师展示都清楚。

## 7. W8 桥接:从"会测"到"会改"

本周你学会了**量化瓶颈**(判 bound、定位、找病根、量 baseline、验误差)。W8 该**正式动手优化**了——从"能测"跨到"能改",这是质变。

**W8 建议的第一步(二选一,按 H100 可用性定)**:

1. **Triton 入门,手写第一个融合 kernel**:在 H100 Linux 上(Triton 不支持 Windows,Day5 §6.1),照着 Day6 的 `relu(x*a+b)` 或一个 RMSNorm,手写一个 Triton 融合 kernel,用 Day2 的 CUDA Event 计时法量它 vs eager vs torch.compile 三方对比——**这是你第一次生产"比 baseline 快"的证据**。
2. **深入 AMK,融一个真实算子**:接 §4 的报告,先用 nsys 把 AMK 的**预测延迟换成实测**(你的独特价值),再挑 attention 或 mlp(占比最大的两块)里的一个子结构,尝试理解/改进它的融合,用本周方法论证明提升。

**无论哪条,W8 的验收都是同一句话**(本周方法论的自然延续):

> **"我改了什么 → 用 Roofline/nsys 量出快了 X%(相对 torch.compile baseline,不是 eager)→ 用三把尺子证明误差可控(cosine>0.99 / PPL 涨<1%)。"** —— 快多少 + 错多少,两个数字都要有,这才是课题 ≥20% + 误差可控可解释的完整答卷。

## 8. 自测题(本周毕业考,合上笔记答)

1. 把本周的性能画像方法论**五步流水线**背出来,并说清"为什么必须是这个顺序"。→ §1
2. **(毕业题)** 5 分钟讲清"nanoGPT 在 H100 上瓶颈在哪、为什么、怎么优化",每个论断配一个本周的数字。→ §2
3. decode 的算术强度为什么恒等于约 1?这和"权重零复用"什么关系?脊点 295 又是怎么来的?→ Day1 / §2.1
4. 融合省的两笔账是什么?本机实测各是多少?torch.compile 自动做了哪几笔?→ Day4/Day5 / §1
5. 你的手写巨核为什么要和 **torch.compile** 比而不是和 eager 比?≥20% 的分母是谁?→ Day5 / §2.3
6. AMK 报告里 correctness FAIL 但 top-1=1.0,这说明了 Day6 三把尺子的什么道理?下一步该看什么?→ §4.2
7. 写这份 AMK 报告为什么"诚实标注边界"比"报个漂亮数字"更专业?→ §4

> 第 2 题是本周的命脉——讲不流畅就说明方法论还没真串起来,回 §1/§2 重读。

## 9. 本周产出总清单(对齐计划)

- [x] `week7_industrial_view.md`(本元笔记):性能画像方法论总表(§1)+ 完整回答毕业题(§2)+ 串联表(§3)
- [x] `amk_profiling_report_v0.md`(§4):硬件配置 / 编译规模(5826 tasks)/ 延迟画像(719.8µs 预测)/ 正确性分析 / 诚实声明 / 下一步——**科研第一份可见产出**
- [x] 更新 `inference_optimization_landscape.md`(§5):新增 quantization、KV Cache 补 Roofline 佐证,6 术语理解程度
- [x] commit 清单(§6):Roofline / profiler+融合 / torch.compile+量化 三次有意义提交
- [x] W8 桥接(§7):从"会测"到"会改",第一个 Triton kernel 或 AMK 真实算子融合
- [ ] (待做)真跑本周脚本采数据填进各笔记:`bench_launch_fusion.py`(已有)、`bench_torch_compile.py`(H100)、`bench_precision.py`(本机)
- [ ] (待做)在 H100 上按 §4.1 报告里的"§6 下一步"用 nsys 把 AMK 预测延迟换成实测,更新报告到 v1
- [ ] (周日复盘)按你的习惯写 `W7_review.md`:本周最有效动作 / 最大阻塞点 / 下周要砍什么

---

> **一句话收尾**:本周你把散落的五个知识点,焊成了一条"**判 bound → 逐级定位 → 找病根 → 量 baseline → 上优化并验证误差**"的方法论流水线,并用它在 nanoGPT + H100 + AMK 上走了完整一遍,产出了你科研生涯第一份诚实、可复现的 profiling 报告。你已经不是"学 DL 的学生"了——你是**能拿着数字和工具、科学地回答'这个模型该往哪优化'的 AI Infra 入门者**。W8,开始动手改。
