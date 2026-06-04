# W5 Day6 学习笔记：profiler chrome trace 实战 + Course 4 Week 3-4 概念扫盲

> 日期：2026-05-30（周六） · 对应计划：第5周 Day6「AI Infra 主线本周高潮」
> 本周定位：把卷积从「会调用 `nn.Conv2d`」升级到「能用 profiler 找出任何 PyTorch 模型的 hotspot」。
> 上承 W4D3「第一次 profiler（CPU only，文本表格）」，今天升级到 **chrome trace 可视化 + CUDA 时间线**。

---

## 0. 今天要回答的问题（带着这些问题往下读）

1. 目标检测、人脸识别、风格迁移这三类任务，**本质上各自在解决什么**？它们在推理部署里有什么特别的工程难点？
2. 我 W4 已经用过 profiler 输出文本表格了，**为什么还要费劲导出一个 trace 文件、再拖进浏览器看时间线**？这两者差在哪？
3. `schedule(wait=1, warmup=2, active=3)` 这三个数字到底在干嘛？不写会怎样？
4. trace 时间线里的 **hotspot（热点）/ launch gap（启动间隙）/ H2D（主机到设备拷贝）** 长什么样？怎么一眼认出来？分别说明什么问题？
5. profiler 看到的真实数字，和我 D3 笔记里用 FLOPs/MAC 算出来的理论预测，为什么会对不上？

---

## 1. Course 4 Week 3-4 概念过完（只建立意识，不实现）

> 计划要求：这两周内容**只过概念、不实现**。但「过概念」不是看个名字就完事——每一类任务背后都有一个 AI Infra 工程师该关心的部署特性。下面每个任务我都补一条「这玩意儿推理时贵在哪」。

### 1.1 目标检测（Object Detection）

**是什么。** 图像分类（Classification）只回答「这张图是不是猫」；目标检测要回答「猫在**哪**、有**几只**、每只框在哪个矩形里」。输出从一个标签，变成「一堆带坐标的框 + 每个框的类别」。

**几个绕不开的英文名词（首次出现，逐个解释）：**

- **Bounding Box（边界框）**：框住物体的矩形，用 4 个数描述，比如「左上角坐标 (x, y) + 宽高 (w, h)」。可以理解成你在照片上用鼠标拖出来的那个选框。
- **Anchor（锚框）**：预先在图上密密麻麻铺好的一堆「候选框模板」（不同大小、不同长宽比）。**类比**：与其让模型凭空画框（无从下手），不如先发给它一沓不同尺寸的「相框模板」，它只需要回答「哪个模板最贴近真实物体，再微调一下」。这把「回归一个任意框」的开放问题，变成了「在固定候选里挑 + 小修正」的好解问题。
- **IoU（Intersection over Union，交并比）**：两个框「重叠面积 ÷ 合并面积」，取值 0~1。1 表示完全重合，0 表示毫不相干。它是判断「预测框准不准」的尺子。
- **NMS（Non-Maximum Suppression，非极大值抑制）**：同一个物体常被好几个框同时框住，NMS 负责「同一堆高度重叠的框里，只留置信度最高的那个，其余删掉」。**类比**：一群人都举手说「猫在这」，NMS 让喊得最响（置信度最高）的那个人留下，其他举重复手的全部放下。

**为什么这样设计。** 直接让网络吐出「任意数量的任意框」非常难训（输出长度不固定、没有对齐目标）。Anchor 把它转成「每个锚框做一次二分类（有没有物体）+ 一次坐标回归（微调）」，输出长度固定、可批量算——这正是工程上最爱的「把不规则问题规则化」。

> **YOLO（You Only Look Once）**：把整张图一次性切成网格，每个网格直接预测框+类别，一次前向出结果（不像更早的两阶段方法先选区域再分类）。「You Only Look Once」就是在强调「单次前向、端到端」，所以快、能实时。

**🏭 AI Infra 锚点。**
- **NMS 是后处理（post-processing），通常跑在 CPU 上，且含循环+排序，不是规整的矩阵乘**——它经常成为检测模型上线后的隐藏延迟瓶颈。TensorRT 专门提供 `EfficientNMS` 插件把它搬上 GPU，就是为了消掉这块 CPU 尾巴。这呼应你今天后面要观察的「H2D/D2H 拷贝」——检测模型典型地会在前向算完后把结果拷回 CPU 做 NMS，trace 上能看到这一刀。
- Anchor 数量极多（几万个），大部分是背景，**算力浪费在没物体的区域**——这是后来 anchor-free 方法（如 FCOS、DETR）想解决的事，也是「FLOPs 不等于有效计算」的一个例子。

### 1.2 人脸识别（Face Recognition）

**是什么。** 注意区分两个词：
- **Face Verification（人脸验证，1:1）**：给两张脸，判断「是不是同一个人」。手机解锁就是这个。
- **Face Recognition（人脸识别，1:N）**：给一张脸，在一个库里找「他是谁」。

**核心难点叫 One-shot Learning（单样本学习）**：公司门禁库里每个员工通常只有 1 张照片，你不可能为每个人收集几千张来训分类器。**类比**：传统分类像「背熟全班同学的脸再考试」，但人脸库每人只给你看一眼——你得学会一种「通用的比脸能力」，而不是「记住具体某个人」。

**怎么做。**
- **Embedding（嵌入向量）**：把一张脸喂进网络，输出一个固定长度的向量（比如 128 维）。**关键设计**：让「同一个人的不同照片」向量挨得很近，「不同人」的向量离得很远。这样识别就退化成「算两个向量的距离」。
  - （呼应 W4D1 `embedding_as_lookup.md`：那里 embedding 是「查表取一行」；这里 embedding 是「网络把高维输入压成一个语义向量」。同一个词，两种用法，别混。）
- **Triplet Loss（三元组损失）**：一次拿三张图——锚点 A（某人）、正样本 P（同一人另一张）、负样本 N（别人）。损失逼着「A 到 P 的距离 + 间隔 < A 到 N 的距离」。**类比**：训练时不断告诉模型「你得让自己人贴得比外人近，至少近一个安全距离」。

**🏭 AI Infra 锚点。** 识别阶段的本质是**向量检索（近邻搜索）**：把待查向量和库里几百万个向量比距离。库一大，这就是个独立的系统工程问题——对应工业界的 **向量数据库 / ANN（Approximate Nearest Neighbor，近似最近邻）**，如 Faiss、Milvus。这跟 RAG（检索增强生成）里给 LLM 做知识检索是**同一套技术栈**，你未来做推理系统一定会再遇到。

### 1.3 神经风格迁移（Neural Style Transfer）

**是什么。** 把「一张照片的内容」和「另一张画的风格」合成——比如把你的自拍画成梵高《星空》的笔触。

**怎么做（这个最反直觉，值得记）。** 它**不训练网络的权重**，而是**把一张随机噪声图当作「待优化的变量」，反向传播去改这张图的像素**，直到它同时满足两个目标：
- **Content Loss（内容损失）**：生成图在某个中间层的特征，要接近「内容图」的特征（保证还认得出是你）。
- **Style Loss（风格损失）**：用 **Gram Matrix（格拉姆矩阵，特征通道两两做内积得到的相关性矩阵）** 描述「风格」。直觉是——风格 = 「哪些纹理/颜色特征经常一起出现」，而 Gram 矩阵恰好抓的是「特征之间的相关性、丢掉空间位置」。让生成图的 Gram 矩阵接近风格图的 Gram 矩阵，就「染上」了那种笔触。

**🏭 AI Infra 锚点。** 这是你第一次见到「**梯度不更新权重，而更新输入**」的范式。记住这个心智模型——它和后面两类东西同源：
- **对抗样本（adversarial examples）**：也是固定权重、优化输入像素，去骗过模型。
- **Prompt 优化 / soft prompt**：固定 LLM 权重，只优化输入端的向量。

「优化输入而非权重」这条线，会一路通到你以后关心的安全和提示工程。原始风格迁移要对每张图迭代几百步（慢），工业上后来改成「训一个前馈网络一次出图」——这又是一次「把迭代优化蒸馏成单次前向」的经典推理优化套路，和上面 YOLO「单次前向」的思路一脉相承。

---

## 2. 为什么需要 chrome trace：从「报表」到「监控录像」

### 2.1 先理解 profiler 在测什么

**Profiler（性能剖析器）**：一个测「程序的时间和内存花在哪」的工具。**类比（呼应你 feedback 里提过的「部门人力支出报表」）**：你是部门经理，月底想知道钱花哪了。

- **W4D3 你用的 `key_averages().table()`**，相当于一张**汇总报表**：「Conv 这个部门总共花了 200ms、占 40%」。它告诉你**谁花得多**，但不告诉你**什么时候花、谁在等谁**。
- **今天的 chrome trace**，相当于调出**整层楼的监控录像**：你能看到每个员工（CPU 线程、GPU 流）在每一毫秒在干嘛，谁在干活、谁在发呆等别人。

为什么报表不够？因为深度学习的性能问题，**很多是「时序问题」而非「总量问题」**：

- GPU 明明很闲，却在「等 CPU 发指令」（launch gap）——报表上 GPU 总忙碌时间看起来正常，但墙钟时间（wall-clock）被拖长了。
- 数据在 CPU 和 GPU 之间来回搬运（H2D/D2H），这部分在按算子汇总的报表里容易被淹没，但在时间线上是一道显眼的「断层」。

> **一句话**：报表回答「**算力花在哪类算子**」，时间线回答「**为什么墙钟时间这么长**」。AI Infra 优化的目标通常是后者——降低用户实际等待的延迟。

### 2.2 三个核心概念：CPU 时间线、GPU 时间线、它们之间的「缝」

PyTorch 跑在 GPU 上时，有一个关键事实**必须先建立**：

> **CPU 负责「发号施令」，GPU 负责「干活」，两者异步（asynchronous）。**

**类比**：CPU 是厨房里喊单的主管，GPU 是炒菜的厨师。主管喊「下一道宫保鸡丁」（这叫 **kernel launch，内核启动**），喊完**不等菜做完**就接着喊下一道（异步）。理想情况下主管喊单的速度快过厨师炒菜，厨师永远有活干、灶台不熄火。

由此推出 trace 上的三类典型现象（也就是计划里要求你标注的三个观察）：

| 现象 | 在 trace 上长什么样 | 厨房类比 | 说明什么问题 |
|---|---|---|---|
| **Hotspot（热点）** | GPU 时间线上一根**特别长**的色块 | 某道菜炒了特别久 | 这个 kernel 是耗时大头，优化要先从它下手 |
| **Launch gap（启动间隙）** | GPU 时间线上算子之间**反复出现的空白缝隙**，CPU 行却在忙 | 厨师炒完一道菜，干等主管喊下一道 | CPU 发指令太慢（launch overhead 高），GPU 被饿着，吞吐上不去 |
| **H2D / D2H（主机↔设备拷贝）** | 出现 `Memcpy HtoD`/`DtoH` 或 `aten::to` 色块，尤其在训练主循环内 | 临时跑去仓库搬原料，全员等他 | 数据在 CPU/GPU 间来回搬，常是写法 bug（如循环里 `.cpu()`/`.item()`/`.to(device)`） |

- **H2D = Host to Device**：Host（主机）指 CPU + 内存，Device（设备）指 GPU + 显存。H2D 就是「把数据从内存搬进显存」，D2H 反过来。这步走 PCIe 总线，比 GPU 内部计算慢一两个数量级——**能不搬就别搬，要搬就一次搬够**。

### 2.3 `schedule` 三个数字：为什么不能从第一步就开始测

直接对训练的**每一步**都开 profiler，会有两个问题：（1）trace 文件爆炸大，浏览器打不开；（2）前几步的数字不准。所以用 `schedule` 控制「只采样中间几步」：

```python
schedule(wait=1, warmup=2, active=3, repeat=1)
```

- **wait=1（等待）**：跳过第 1 步完全不记。
- **warmup=2（预热）**：第 2~3 步开着 profiler 但**丢弃数据**。**为什么要预热**——第一次跑某个 kernel 时，cuDNN 要做 **benchmark（自动挑最快的卷积算法）**、显存分配器要建缓存池、CUDA context 要初始化。这些**一次性开销**会让前几步异常慢，记进去会严重误导你。**类比**：餐厅刚开门，灶具要预热、厨师要找趁手的锅，头几单慢是正常的，不能拿来算「平均出餐速度」。
- **active=3（采样）**：第 4~6 步**真正记录**，这才是干净的稳态数据。
- **repeat=1**：上述循环只做一轮。

> **关键直觉**：性能测量永远要测**稳态（steady state）**，不是冷启动。这条规则不止用于 profiler——你以后 benchmark 推理延迟、测 `torch.compile` 加速比，全都要先 warmup 再计时。漏掉 warmup 是新手最常见的「假数据」来源。

---

## 3. 完整可运行代码：导出 ResNet18 的 chrome trace

> **运行环境 / 依赖**：
> - Python 3.9+，PyTorch ≥ 2.0（profiler API 在 2.x 稳定）
> - `pip install torch torchvision`
> - 有 NVIDIA GPU 最好（能看到 CUDA 时间线）；**没 GPU 也能跑**——脚本会自动退回 CPU，只是看不到 H2D 和 kernel 异步现象（计划风险表里也提到这点）。
> - 保存为 `week5_cnn/src/profile_resnet.py`，产物落到 `week5_cnn/logs/resnet18_trace.json`

```python
# week5_cnn/src/profile_resnet.py
# 用 torch.profiler 给 ResNet18 跑一遍 train step，导出 chrome trace。
# 运行: python src/profile_resnet.py
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

# --- 0. 固定随机种子 + 选设备 ---
# 为什么固定 seed: 计划的"质量约束"要求里程碑可复现。性能数会有波动，
# 但至少保证模型权重、输入数据每次一致，排查问题时不会引入额外变量。
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] running on: {DEVICE}")

# --- 1. 模型 / 优化器 / 损失 ---
# 直接用 torchvision 的工业级实现做"标准答案"，不自己搭——今天目的是 profile,不是复现结构。
model = models.resnet18(num_classes=10).to(DEVICE)
model.train()  # train 模式: BN 用 batch 统计、Dropout 生效 —— 呼应 W4 batchnorm_inference.md
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
criterion = nn.CrossEntropyLoss()

# --- 2. 造一批假数据 ---
# 用随机张量代替真实 CIFAR-10: profile 关心的是"算子怎么跑",不是"学得准不准",
# 没必要真去读数据集拖慢脚本。shape 对齐 CIFAR(3通道) 但放大到 224 让卷积负载更明显。
BATCH = 64
x = torch.randn(BATCH, 3, 224, 224, device=DEVICE)
y = torch.randint(0, 10, (BATCH,), device=DEVICE)

# --- 3. 把一步训练封成函数,方便在 profiler 循环里反复调 ---
def train_step():
    optimizer.zero_grad(set_to_none=True)  # set_to_none=True 比置零省一次 kernel,工业默认写法
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

# --- 4. 配置 profiler ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# activities: CPU 看"发指令", CUDA 看"GPU 干活"。两条线都要,才能看出它们之间的"缝"。
activities = [ProfilerActivity.CPU]
if DEVICE == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),  # 见 §2.3: 跳过冷启动,只记稳态
    record_shapes=True,       # 记录每个算子的输入张量形状 —— 排查"哪个 shape 慢"必备
    profile_memory=True,      # 记录显存分配 —— 能看到激活值占多少(呼应 D5 显存账)
    with_stack=True,          # 记录 Python 调用栈 —— 能点到具体哪行代码,但会让 trace 变大变慢
) as prof:
    for step in range(6):     # wait1+warmup2+active3 = 需要至少 6 步
        train_step()
        prof.step()           # 关键: 告诉 profiler"又过了一步",它据此推进 schedule 状态机

# --- 5. 导出 chrome trace + 打印文本汇总 ---
trace_path = os.path.join(LOG_DIR, "resnet18_trace.json")
prof.export_chrome_trace(trace_path)
print(f"[ok] chrome trace 已导出: {trace_path}")

# 文本汇总仍然有用: 它给"按算子总耗时排序"的全局视角,和时间线互补。
sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
```

**预期输出（CPU 版示意，GPU 版会多出 CUDA 列）：**

```
[info] running on: cuda
[ok] chrome trace 已导出: .../logs/resnet18_trace.json
-------------------------------  ------------  ------------  ------------
                           Name    Self CUDA    CUDA total    # of Calls
-------------------------------  ------------  ------------  ------------
        aten::cudnn_convolution      12.3ms        12.3ms            20
   aten::cudnn_convolution_backward 18.1ms        18.1ms            20
              aten::batch_norm        3.2ms         8.5ms            20
                     aten::add_       2.1ms         2.1ms            ...
-------------------------------  ------------  ------------  ------------
```

> 注意 `convolution_backward` 通常比 forward 还贵（反向要算两份梯度：对输入、对权重）——这解释了「训练比推理慢」的一部分。

### 3.1 怎么打开 trace（两条路，任选）

1. **Chrome 自带（计划指定的方式）**：地址栏输入 `chrome://tracing/` → 点 `Load` → 选 `resnet18_trace.json`。用 `W/A/S/D` 缩放平移时间线（W 放大、S 缩小、A/D 左右移）。
2. **Perfetto（更现代，推荐）**：打开 `https://ui.perfetto.dev` → `Open trace file`。同一个 json 直接能开，界面更顺手，大文件也更稳。（`chrome://tracing` 是老工具，新版 Chrome 实际也会引导到 Perfetto。）

> **如果 trace 文件太大打不开**：说明 `active` 步数设太多或 `with_stack=True` 拖累。把 `active` 调到 2、或临时关掉 `with_stack`。这正是计划风险表里写的应急策略。

### 3.2 常见错误与调试（贴近真实踩坑）

```python
# ❌ 错误 1: 忘了调 prof.step(),schedule 永远停在 wait 状态,导出的 trace 是空的
with profile(schedule=schedule(...)) as prof:
    for step in range(6):
        train_step()
        # 漏了 prof.step()  ← trace 文件几乎没内容

# ❌ 错误 2: GPU 上计时不 synchronize 就读时间
#   GPU 是异步的,CPU 代码跑到这行时 GPU 可能还没算完。
#   profiler 内部已正确处理同步,但你若自己用 time.time() 掐表测 GPU,必须:
torch.cuda.synchronize()   # 等 GPU 把活干完,再读时间,否则测出来的是"发指令的时间",假快

# ❌ 错误 3: 在 active 区间内不小心触发 H2D —— 这反而是"特性",正好让你在 trace 里抓现行
loss_value = loss.item()   # .item() 会把标量从 GPU 拷回 CPU(D2H)并强制同步!
                           # 训练循环里每步都 .item() 打印 loss,是真实项目里常见的隐藏性能杀手
```

> 第 3 条特别值得记：很多人写 `print(f"loss={loss.item()}")` 放在每个 step，**每次 `.item()` 都强制 GPU↔CPU 同步 + 一次 D2H 拷贝**，把异步流水线打断。工业做法是攒几步记一次，或用 `loss.detach()` 累积、循环外再 `.item()`。这正是你今天要在 trace 上找的「意外 H2D」的最常见来源。

---

## 4. 三个 trace 观察标注（今天的核心产出）

> 计划完成标准：**至少标注 3 个具体观察**。下面是填写模板 + 一份「读图时该看什么」的对照，跑完脚本后把你 trace 里的真实数字填进去，存进 `tech_notes/profiler_chrome_trace.md`。
> （如果你是 CPU 跑的，观察 2/3 可能看不到典型现象——照实记录「CPU 环境下无 CUDA 时间线，未观察到 launch gap」，这本身就是一条诚实的结论，不要编。）

### 观察① Hotspot（最慢的算子）

- **怎么找**：在文本汇总里看 `cuda_time_total` 排第一的算子；在时间线里找 GPU 行最长的那段色块。
- **预期**：通常是 **stage3/stage4 的 `3×3 cudnn_convolution`**（channel 数大、空间分辨率虽小但通道乘起来 FLOPs 高），或它的 `_backward`。
- **填写模板**：
  ```
  最慢算子: __________________ (如 aten::cudnn_convolution_backward)
  自身耗时: ______ ms, 占单步总时间 ______ %
  对应层  : ______________ (如 layer4 的第二个 3x3 conv)
  判断    : 这是 compute-bound 还是 memory-bound? 依据是? (回看 D3 flops_vs_latency.md)
  ```

### 观察② Launch gap（启动间隙）

- **怎么找**：放大到单步内部，看 GPU 行的色块**之间有没有反复的空白**，同时 CPU 行此刻在忙（在发下一条指令）。
- **预期**：ResNet18 单个 kernel 不算大，**小 batch 时容易看到密集的小 gap**——CPU 发指令的速度跟不上 GPU 吞吃的速度。
- **填写模板**：
  ```
  是否观察到 launch gap: 是 / 否
  典型 gap 宽度       : 约 ______ us
  出现位置           : (如每个 conv kernel 之间普遍存在)
  推断               : launch overhead 偏高 / GPU 利用率未打满
  工业解法           : torch.compile 做 CUDA graph capture, 把多次 launch 合并; 或加大 batch
  ```

### 观察③ H2D / D2H（意外的主机↔设备拷贝）

- **怎么找**：搜 `Memcpy`、`aten::to`、`aten::item`、`aten::_to_copy`，看它们**是否出现在训练主循环内**。
- **预期**：本脚本数据是一次性 `device=DEVICE` 造好的、循环里也没 `.item()`，所以**理想情况下 active 区间内不该有 H2D**。如果你故意加一行 `loss.item()` 再跑，就能在 trace 上抓到这道「同步刀」。
- **填写模板**：
  ```
  active 区间内是否有 H2D/D2H: 是 / 否
  若有, 来源算子: __________ (如 aten::item)
  触发代码行    : __________ (如 train_step 里某次 .item()/.cpu())
  修复          : 移出循环 / 用 detach 累积 / 攒 N 步记一次
  ```

> **给自己加个小实验**（计划里 D6 末尾的「串联」要求）：故意在 `train_step()` 里加一行 `_ = loss.item()`，重跑，对比两份 trace——你会直观看到「一行无害的打印」如何在时间线上凿出一道同步缝。把这个对比写进笔记，比抄十遍定义都管用。

---

## 5. 把 trace 串回 D3 的理论预测（计划要求的「串联」）

> 计划：把今天 trace 看到的现象，串到 D3 `flops_vs_latency.md`，**真实数字和理论预测会有出入，记录差异原因**。

你 D3 用 FLOPs/MAC 算过「哪层理论上最重」。今天对照 trace，大概率发现**对不上**，常见原因：

1. **FLOPs 高 ≠ 耗时长**。1×1 conv FLOPs 不高却可能 memory-bound（带宽卡脖子）；理论只数乘加，没算访存。这正是 D3「arithmetic intensity」的核心——**延迟由 FLOPs 和 MAC 里更紧的那个决定**。
2. **cuDNN 算法选择**。同一个卷积，cuDNN 有好几种实现（implicit GEMM、Winograd、FFT），实测选哪个取决于 shape，理论公式预测不到。
3. **kernel launch 开销**。小算子多时，发指令的固定开销占比变大——这部分理论 FLOPs 模型里**根本不存在**。
4. **BN / ReLU / add 这些「便宜」算子的累加**。单看每个不贵，但数量多 + 都要读写整个激活张量（memory-bound），加起来不可忽视。这也是「fused Conv-BN-ReLU」（W4 学的）能提速的原因——少几趟访存。

> **一句话收口**：FLOPs 是「纸面工资」，profiler 是「实发到手」。AI Infra 工程师信后者。两者差多少、为什么差，就是你的优化空间。

---

## 6. 自测题（先自己答，答案锚点见括号）

1. 同样是 profiler，`key_averages().table()` 和 chrome trace 各自回答什么问题？为什么时序问题必须看后者？（§2.1）
2. `schedule(wait=1, warmup=2, active=3)` 里 warmup 那两步为什么要「记录但丢弃」？不 warmup 会得到什么假象？（§2.3）
3. 在时间线上，怎么一眼区分「hotspot」和「launch gap」？它们各自的优化方向是什么？（§2.2 表 + §4）
4. 为什么训练循环里每步 `loss.item()` 会拖慢速度？它在 trace 上留下什么指纹？（§3.2 + §4 观察③）
5. 你 D3 算的「理论最重的层」和 profiler 实测的 hotspot 对不上，列 2 个可能原因。（§5）
6. （连 W4）推理时 BN 能 fuse 进 Conv 而「消失」，这件事如果反映到 trace 上，推理 trace 比训练 trace 会少掉哪些色块？（W4 `batchnorm_inference.md` §3 + 本篇 §5.4）

---

## 7. 与已有笔记的串联

| 今天内容 | 关联笔记 | 关系 |
|---|---|---|
| profiler chrome trace | `tech_notes/first_profiler.md`（W4D3） | 从「CPU 文本表格」升级到「CPU+GPU 时间线」 |
| hotspot 是 compute 还是 memory bound | `tech_notes/flops_vs_latency.md`（W5D3） | 理论预测 ↔ 实测对照，§5 专门串 |
| H2D / .item() 同步刀 | `tech_notes/conv_anatomy.md`（W5D1）激活显存 | 数据搬运成本，访存视角 |
| 推理时 BN 消失 → trace 变化 | `tech_notes/batchnorm_inference.md`（W4D5） | fused BN 的 trace 级证据 |
| 人脸 embedding / 向量检索 | `tech_notes/embedding_as_lookup.md`（W4D1） | 同名不同义，提醒区分两种 embedding |
| launch gap → torch.compile | `tech_notes/inference_optimization_landscape.md`（W5D7 待写） | 图优化层，明天元笔记承接 |

---

## 8. 当日完成标准 checklist（对齐计划 Day6）

- [ ] Course 4 Week 3-4 概念过完（目标检测 / 人脸识别 / 风格迁移，能各用一句话说清「是什么 + 推理难点」）
- [ ] `profile_resnet.py` 跑通，产出 `logs/resnet18_trace.json`
- [ ] 用 `chrome://tracing/` 或 Perfetto 成功打开 trace
- [ ] 在 `tech_notes/profiler_chrome_trace.md` 写满**至少 3 个观察**（hotspot / launch gap / H2D，用 §4 模板）
- [ ] （加分）做 `loss.item()` 对比小实验，记录两份 trace 差异
- [ ] （加分）把实测 hotspot 串回 D3 理论预测，写差异原因
- [ ] 写 `W5_day6_log.md`

> **今天的「胜利条件」不是「我会用 profiler」，而是**：随手给你一个陌生的 PyTorch 模型，你能 5 分钟内 profile 出它的 hotspot，并基于 FLOPs/MAC 直觉判断它卡在算力还是带宽。这是 AI Infra 工程师的「听诊器」基本功。明天（D7）把它写进 `week5_industrial_view.md` 元笔记。

---
*笔记生成：2026-05-30 · 配合 `week5_cnn/src/profile_resnet.py` 使用 · 风格遵循 W4D2 三段式标杆*

