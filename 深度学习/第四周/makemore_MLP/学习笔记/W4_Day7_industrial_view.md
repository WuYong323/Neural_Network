# W4 Day 7（2026-05-24，周日）—— 本周元笔记 + profiler 完整实战 + 工程收尾

> **今天的定位**：W4 这周你学了 embedding / MLP / 初始化 / BatchNorm / Adam——单看每一块都不难，但工业界的厉害之处恰恰不是"懂某一块"，是**能把这些零件串起来回答一个问题：训练和推理这两件事，到底差在哪里？为什么差这么多？**
>
> 今天的核心产出是 `tech_notes/week4_industrial_view.md`——这是 W4 的"元笔记"，相当于你给未来的自己（或者将来面试官）讲一遍："经过这一周，我能从工业视角看懂什么"。
>
> 另外三件配套事：**真正跑一次 `torch.profiler` 导出 chrome trace**（W4D3 那次只是文本表格热身，今天要看可视化时间线）、**清理项目 + 写 README**（让别人能复现你的成果）、**GitHub commit ≥ 3 次**（每一步都留痕迹）。
>
> **学习目标（看完笔记你应该能回答）**：
> 1. 元笔记到底是什么？为什么要花一整天写？
> 2. `torch.profiler` 的三个时间口径（CPU time / CUDA time / Self time）分别在说什么？
> 3. 什么叫 chrome trace？它和文本表格的差异，等价于 X 光片和病历本的差异，怎么理解？
> 4. README 写到什么程度算"工业级"？怎么避免"写完一周就没人能跑通"？
> 5. 什么叫"有意义的 commit"？为什么 GitHub 上的提交历史是一种简历？

---

## §1 元笔记是什么——以及为什么不是"周记"

### §1.1 是什么

**元笔记（meta-note）**：英文 *meta-* 这个前缀表示"关于自身的"，所以**元笔记 = 关于笔记的笔记**。换个说法——你 W4 这周写了一堆 `embedding_as_lookup.md`、`batchnorm_inference.md`、`optimizer_memory.md`，元笔记就是把这些**单点**笔记串成一张**网**的东西。

### §1.2 类比：单词本 vs 词族图

想象你学英语：
- 你的 `tech_notes/*.md` 每篇就像是**单词本**的一页：记住了 `embedding`、`batchnorm`、`adam` 各自是什么意思。
- 元笔记就像**词族图**：把 `embed-`、`-norm`、`-ize` 这些词根关联起来——一旦你看到 `tokenizer` / `quantize`，立刻能猜到大概是什么含义。

工业界做研发也是同一个模式：
- 单点笔记是**只读凭证**（reference）——遇到具体问题时翻出来抄一遍。
- 元笔记是**思维模型**（mental model）——遇到新问题时，能本能地知道"这事大概属于哪一类问题"。

> 大厂里资深工程师和应届生最大的差距，常常不在"知道多少细节"，而在"有没有形成这种思维模型"。一个 7B 模型显存爆了，资深的人第一反应是"先想想 Adam state、激活、梯度三大块谁出问题"，新手第一反应是去翻 Stack Overflow。

### §1.3 为什么必须花一整天写

三个理由，每个理由都能省你后面几十小时：

**理由 1：知识在串联的瞬间才真正变成你的。**
单看 BN 的 fused 推理优化、单看 Adam state 占显存，你可能都"觉得自己懂了"——但当我问"训推显存为什么差 4 倍"，要求你用 30 秒答完，单点知识是接不住这种问题的。串联过一次，这个答案就是肌肉记忆。

**理由 2：元笔记是你下一次复习时的入口。**
两个月后你回头看 W4，没人会再读 7 篇 `tech_notes/*.md`。**只会读这一篇**。所以这篇必须是"3 分钟刷完就能回忆起整周内容"的密度。

**理由 3：这是你未来简历/博客/面试的弹药库。**
"训推显存为什么差 4 倍"这个问题，**字节、阿里、Meta、英伟达 AI Infra 岗的面试都问过**。今天你把它整理成元笔记里的一节，半年后面试时不用重新组织语言。

---

## §2 W4 元笔记：训练 vs 推理——这周教会我的事

> 这一节就是元笔记的主体内容。最终要把它存为 `tech_notes/week4_industrial_view.md`。

### §2.1 一句话总览

> **训练和推理的差异，不是"反向传播打开/关闭"那么简单——它是显存、算子行为、优化目标的全方位不同。这周学的每一块知识，本质上都在说这件事的一个侧面。**

### §2.2 训练 vs 推理的四大差异（核心表格）

| 维度 | 训练态（`model.train()`） | 推理态（`model.eval()`） | 本周对应知识点 |
|---|---|---|---|
| **显存构成** | 参数 + 梯度 + 优化器状态 + 激活值 | 仅参数（+ KV Cache，第6周讲） | `optimizer_memory.md`（W4D6） |
| **BN 行为** | 用 batch 统计 + 更新 running stats | 用 running stats，与 batch 无关 | `batchnorm_inference.md`（W4D5） |
| **Dropout** | 随机置零 + 缩放 | 完全关闭（identity） | Course 2 Week 1 |
| **算子图** | 完整 forward + backward | 仅 forward，且可融合（fused Conv-BN-ReLU） | `batchnorm_inference.md` §3 |
| **优化目标** | loss 最小化、收敛速度 | 延迟（latency）、吞吐（throughput）、显存峰值 | `flops_vs_latency.md`（W5 预告） |

> **记忆锚点**：训练态像"做菜的厨房"——锅碗瓢盆全摆出来；推理态像"上菜的传菜口"——只保留最终成品 + 必要工具。

### §2.3 一道经典面试题的完整答案

**问**：用 Adam 训练一个 7B 参数的 FP32 模型，需要多少显存？同样的模型推理需要多少？

**答**：

```
训练态（FP32，batch_size=1，未做任何优化）：
  参数             : 7B × 4 bytes = 28 GB
  梯度             : 7B × 4 bytes = 28 GB    （和参数同形状）
  Adam state (m,v) : 7B × 8 bytes = 56 GB    （两份 momentum）
  激活值           : 取决于网络深度，对 LLM 通常 10-20 GB（batch_size=1 时）
  ──────────────────────────────────────────
  合计             ≈ 120-130 GB

推理态（FP32）：
  参数             : 28 GB
  KV Cache         : 几百 MB 到几 GB（取决于上下文长度，第6周细讲）
  ──────────────────────────────────────────
  合计             ≈ 28-30 GB

差距            ≈ 4-5 倍
```

**这就是为什么"训练 7B 要 A100 80G ×2，推理 70B 也能跑 A100 80G ×1"——本质是上面这张表。**

### §2.4 W4 五个知识点的串联图

```
              ┌─────────────────────────────────┐
              │   核心命题：训练 ≠ 推理         │
              └────────────┬────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
  显存视角            算子视角            优化目标视角
       │                   │                   │
   Adam state           BN双行为            lr-range test
   梯度                  Dropout            初始化稳定性
   激活                  fused算子          FP16/INT8伏笔
       │                   │                   │
   W4D6                W4D5               W4D2/D4
   optimizer_memory    batchnorm_inference init_and_stability
```

### §2.5 与你已有笔记的串联

| W4 节点 | 关联笔记 | 怎么连 |
|---|---|---|
| Embedding lookup | W3 `autograd_explained.md` §Q9 | one-hot @ W 的稀疏更新 → 工业里直接退化成 indexing |
| MLP 训练稳定性 | W3 §Q4 发散回路 + §Q10 loss 选择 | bigram 的"三张优惠券"在 MLP 里失效，需要 Adam + BN + 初始化补回 |
| BN 训练/推理双行为 | W3 `autograd_explained.md` §5.1 | `model.eval()` 关掉的最显眼的算子就是 BN |
| Adam state 显存 | W3 `autograd_explained.md` §5.2 | "训推显存差 4 倍"今天有了具体数字 |
| 初始化 + FP16 伏笔 | （W5 D1 `flops_vs_latency.md` 会接） | 初始化决定能不能 FP16/INT8 量化 |

---

## §3 torch.profiler 实战——从文本表格升级到 chrome trace

### §3.1 是什么 + 为什么

**profiler（性能分析器）**：英文 *profile* 这里取"画像"的意思——给程序的运行时间画一张"人体测温图"，告诉你每个部位（每个算子）烧了多少 CPU/GPU 时间。

**类比**：
- 你跑步喘不过气，去医院做心电图——profiler 就是给模型做心电图。
- W4D3 你第一次跑的文本表格是**血常规化验单**：每个指标一个数字，简洁但不直观。
- 今天要做的 chrome trace 是**24 小时动态心电图**：能看到每个时刻 CPU 和 GPU 各自在干什么，谁在等谁，哪段时间空转。

**工业现实**：在大厂 AI Infra 团队，**任何一个"模型推理慢"的工单，第一步永远是 profile**。没看过 trace 就上手优化，等于不看 CT 直接动刀。

### §3.2 三个时间口径——务必分清

profiler 输出里你会看到三个时间，初学者最容易搞混：

| 名称 | 直觉解释 | 部门 KPI 类比 |
|---|---|---|
| **CPU time** | 这个算子的 Python/C++ 调用栈花的时间 | 部门**直接员工**工资 |
| **CUDA time** | 这个算子在 GPU 上实际跑的时间 | 部门**外包**工资（GPU 是异步执行的"外包团队"） |
| **Self time** | 这个算子**自己**的时间，不含子调用 | 部门**自己人**工资，不含子团队 |

举个例子：`nn.Linear(1024, 1024)` 内部会调用 `aten::addmm`：
- Linear 的 CPU time 包括 `aten::addmm` 的时间（包子调用）
- Linear 的 Self time **不包括** `aten::addmm`（只算自己 Python wrap 那一层）

> 看 hotspot 找瓶颈时，**优先按 Self time 排序**——否则永远是最外层的 `forward` 排第一，没有意义。

### §3.3 可运行代码（save 到 `week4_makemore_mlp/src/run_profiler.py`）

```python
"""
W4D7: 用 torch.profiler 跑 makemore MLP 完整训练，导出 chrome trace。

依赖：
    pip install torch torchvision
    Python ≥ 3.9
运行：
    python src/run_profiler.py
输出：
    logs/makemore_trace.json   <- 拖进 chrome://tracing/ 查看
    控制台会打印 Top-15 hotspot
"""

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

# ───── 1. 复用 W4 已经写好的 MLP（这里给最简版本，方便单文件运行）─────
class MakemoreMLP(torch.nn.Module):
    def __init__(self, vocab_size=27, emb_dim=10, block_size=3, hidden=200):
        super().__init__()
        self.C = torch.nn.Embedding(vocab_size, emb_dim)
        # 关键：用 nn.Linear 而不是裸 W/b，是为了让 profiler 能把它识别成"算子"
        self.fc1 = torch.nn.Linear(emb_dim * block_size, hidden)
        self.bn = torch.nn.BatchNorm1d(hidden)        # 训练/推理双行为，W4D5
        self.fc2 = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x):
        emb = self.C(x).view(x.size(0), -1)            # embedding lookup（不是矩阵乘）
        h = torch.tanh(self.bn(self.fc1(emb)))
        return self.fc2(h)

# ───── 2. 造一批 dummy 数据（无需真实 names.txt，省事）─────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MakemoreMLP().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam，W4D6 知道它带 m/v 两份 state
batch_size = 128
X = torch.randint(0, 27, (batch_size, 3), device=device)
Y = torch.randint(0, 27, (batch_size,), device=device)

# ───── 3. profile 配置 ─────
# 为什么这么写：训练前几个 step 有 JIT 编译开销，profile 全程会被冷启动污染。
# schedule 是工业标准做法——前面 wait/warmup 跳过，只 active 几步采样。
prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

# activities 同时开 CPU 和 CUDA，否则只能看到一半。
# record_shapes=True 能让你在 trace 里看到 tensor 形状（debug 形状不匹配神器）。
# profile_memory=True 能记录每个算子的显存峰值——回答"激活显存花在哪"的核心数据。
activities = [ProfilerActivity.CPU]
if device == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    schedule=prof_schedule,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,             # True 会很慢，调试时再开
) as prof:
    for step in range(10):        # 1 wait + 2 warmup + 3 active = 6，跑 10 步留余量
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        prof.step()               # 必须调，告诉 profiler 一步结束了

# ───── 4. 输出结果 ─────
# Self time 排序：找真正的 hotspot，不被外层 wrapper 干扰
print(prof.key_averages().table(
    sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total",
    row_limit=15,
))

# chrome trace：拖进 chrome://tracing/ 看可视化时间线
prof.export_chrome_trace("logs/makemore_trace.json")
print("Trace saved to logs/makemore_trace.json")
print("打开方式：浏览器访问 chrome://tracing/ ，点 Load，选择该文件")
```

### §3.4 trace 怎么看——三个必看观察点

打开 `chrome://tracing/` 加载 `makemore_trace.json` 后，你会看到上下两条时间轴：上面是 CPU，下面是 GPU（如果有）。

**观察点 1：hotspot 是哪个算子？**
- 在 trace 里找最长的彩色块——99% 情况下是 `aten::addmm`（矩阵乘）或 `aten::mm`。
- **结论应该是**：W4 这个小 MLP 的 hotspot 是 `fc1` / `fc2` 的矩阵乘，embedding lookup 占比极小（呼应 W4D1 `embedding_as_lookup.md`）。

**观察点 2：CPU 和 GPU 之间有没有"缝"？**
- 如果 GPU 时间轴上经常空着——说明 GPU 在等 CPU 喂数据/调度 kernel，这叫 **kernel launch overhead（核启动开销）**。
- 工业里解决方案：`torch.compile` 把多个 kernel 融合成 CUDA graph，一次 launch 跑完。这是 W8 推理优化的重点之一。

**观察点 3：有没有意外的 host-device 拷贝（`Memcpy HtoD` / `aten::to`）？**
- 如果训练中途出现这些算子，**几乎一定是 bug**——比如不小心把 tensor 放在了 CPU 上，每步都要拷贝到 GPU。
- 这是新手在 LLM 训练里最常见的"显存够，但跑得比别人慢 10 倍"的元凶。

### §3.5 把观察写到 log

照着模板写到 `W4_day7_log.md`：

```markdown
## profiler 实战观察（2026-05-24）

设备：CPU / CUDA（填实际）
batch_size：128
采样步数：3 active steps

### Top-3 self time 算子
1. aten::addmm        — 占总 self time XX%
2. aten::native_batch_norm — 占 XX%
3. aten::embedding    — 占 XX%

### 三个观察
1. hotspot：__________（应是矩阵乘相关）
2. launch gap：__________（CPU/GPU 时间线之间是否有大段空白）
3. 意外拷贝：__________（理论上不应出现 Memcpy）
```

---

## §4 项目清理 + README 工业级写法

### §4.1 清理代码：减法是难的

**原则**：**今天的代码量应该比昨天少，但功能更完整。**

清理 checklist（直接对 `week4_makemore_mlp/` 操作）：
- [ ] 删除所有 `print("debug:", x.shape)` 类调试语句。
- [ ] 删除 `# TODO`、`# XXX`、`# old version` 注释段；如果还需要它，commit 到 git 历史里就够了。
- [ ] 函数命名统一：`build_dataset` 不要混着写 `buildDataset` / `make_data`。
- [ ] 文件结构整理：
  ```
  week4_makemore_mlp/
  ├── src/
  │   ├── dataset.py
  │   ├── model.py
  │   ├── train.py
  │   ├── batchnorm.py
  │   └── run_profiler.py
  ├── tech_notes/
  │   ├── embedding_as_lookup.md
  │   ├── init_and_stability.md
  │   ├── batchnorm_inference.md
  │   ├── optimizer_memory.md
  │   └── week4_industrial_view.md       ← 今天的元笔记
  ├── logs/
  │   ├── lr_range_test.png
  │   ├── activation_histogram.png
  │   ├── bn_vs_nobn.png
  │   └── makemore_trace.json
  ├── checkpoints/
  │   └── best.pt
  ├── configs/
  │   └── default.yaml
  └── README.md
  ```

### §4.2 README 的工业级模板

**判断标准**：把你的笔记本电脑寄给一个陌生人，他**只看 README**能不能跑通你的训练。

模板（直接抄到 `week4_makemore_mlp/README.md`）：

```markdown
# Week 4: makemore MLP + BatchNorm + Profiler

PyTorch 实现 Karpathy makemore MLP，含 BatchNorm 训练/推理双行为、kaiming 初始化、Adam 优化、torch.profiler 完整 trace。

## 环境
- Python 3.10
- PyTorch 2.x
- (可选) CUDA 11.8+ 用于 GPU 训练

安装：
\`\`\`bash
pip install torch matplotlib pyyaml
\`\`\`

## 数据
- `names.txt`：32k 英文人名，从 Karpathy/makemore 仓库下载
- 放到 `data/names.txt`

## 快速开始
\`\`\`bash
# 1. 训练
python src/train.py --config configs/default.yaml

# 2. 跑 profiler 导出 chrome trace
python src/run_profiler.py

# 3. (可选) 在 chrome://tracing/ 加载 logs/makemore_trace.json
\`\`\`

## 结果

| 配置 | dev loss | 备注 |
|---|---|---|
| baseline (随机初始化) | 27.0 | 初始 loss 异常 |
| + kaiming init | 3.3 | 接近理论 log(27) ≈ 3.30 |
| + BatchNorm | 2.17 | dev 收敛速度 +30% |
| + Adam (lr=1e-3) | 2.08 | 当前最佳 |

## 关键截图
- ![lr-range test](logs/lr_range_test.png)
- ![激活分布](logs/activation_histogram.png)
- ![BN 对比](logs/bn_vs_nobn.png)

## 技术笔记
- [embedding 是 lookup 不是矩阵乘](tech_notes/embedding_as_lookup.md)
- [初始化与稳定性](tech_notes/init_and_stability.md)
- [BatchNorm 训练/推理双行为 + fused BN](tech_notes/batchnorm_inference.md)
- [Adam 优化器的显存代价](tech_notes/optimizer_memory.md)
- [**本周元笔记：训练 vs 推理的全方位差异**](tech_notes/week4_industrial_view.md)

## 复现性
- 随机种子：42（在 `configs/default.yaml`）
- 已固定 `torch.manual_seed` + `torch.cuda.manual_seed_all`
```

### §4.3 README 的常见踩坑

| 错误 | 工业版应该怎么写 |
|---|---|
| "运行 train.py" | `python src/train.py --config configs/default.yaml` |
| "需要 PyTorch" | `PyTorch 2.x，CUDA 11.8+`，给具体版本范围 |
| "Adam 优化器训练" | 给一张配置 → 结果对照表 |
| 没说怎么获取数据 | 显式给数据来源链接 + 放置路径 |

---

## §5 GitHub commit：每一步都留痕迹

### §5.1 什么叫"有意义的 commit"

**反面教材**：
```
commit: "update"
commit: "fix bug"
commit: "more changes"
commit: "wip"
```
→ 这种历史等于没历史。半年后你自己都看不懂。

**工业惯例**（Conventional Commits 规范）：
```
<type>(<scope>): <subject>

type 常见值：
  feat     新功能
  fix      bug 修复
  refactor 重构（行为不变）
  docs     文档
  test     测试
  perf     性能优化
  chore    杂项（依赖、配置）
```

### §5.2 W4D7 的三次推荐 commit

```bash
# Commit 1: 元笔记完成（最重要的产出）
git add tech_notes/week4_industrial_view.md
git commit -m "docs(tech_notes): add week4 industrial view meta-note

- 串联 embedding/BN/Adam/初始化四大主题
- 训练 vs 推理四大差异表
- 7B 模型显存账实例计算
- 与 W3 autograd 笔记的串联"

# Commit 2: profiler 完整实战
git add src/run_profiler.py logs/makemore_trace.json
git commit -m "feat(profiler): add full torch.profiler integration

- 同时采样 CPU + CUDA activities
- 用 schedule(wait,warmup,active) 跳过冷启动
- 导出 chrome trace 供时间线分析
- 标注 Top-3 hotspot 与 launch gap 观察"

# Commit 3: 项目结构清理 + README
git add README.md src/ configs/
git commit -m "refactor: clean up week4 project structure

- 统一命名为 snake_case
- 删除调试 print
- 整理 src/ tech_notes/ logs/ 目录
- 写完整 README（含环境、运行、结果表、笔记索引）"
```

### §5.3 GitHub 提交历史 = 简历

实习/求职时，HR 看不懂你的代码，但**任何人都看得懂 git log**。一个仓库展示的不只是"做了什么"，更是"怎么做的"——

- 半年后回头看，干净的 commit 历史 = 你能向别人讲清楚成长路径。
- 面试时被问"你用 profiler 做过什么优化"，你可以直接 `git show <hash>` 给面试官看真实代码。
- 这就是 W4 D7 计划里"GitHub 至少 3 次有意义 commit"的真正含义——不是数量要求，是**让你的工作变得可追溯**。

---

## §6 自测题（带参考答案位置）

1. **元笔记和单点笔记的区别用一个类比来说？**
   - 参考：§1.2（单词本 vs 词族图）

2. **训练 7B FP32 模型显存约 120GB，推理只要 28GB——这 90GB 差额主要来自哪三块？**
   - 参考：§2.3（梯度 28G + Adam state 56G + 激活约 10G）

3. **profiler 的 Self time 和 CPU time 有什么区别？为什么找 hotspot 要用 Self time？**
   - 参考：§3.2（部门 KPI 类比）

4. **chrome trace 里看到 GPU 时间线有大段空白说明什么？工业里怎么解决？**
   - 参考：§3.4 观察点 2（kernel launch overhead → `torch.compile`）

5. **为什么 BN 在推理时可以"完全消失"进 Conv 权重？**
   - 参考：W4D5 `batchnorm_inference.md` §3（fused BN-Conv）
   - 用一句话：推理时 BN 的 mean/var 都是常数 → 整个 BN 退化成仿射 → 可以并入前面 Conv 的 weight/bias

6. **下面这段 commit message 哪里不规范？**
   ```
   commit: "更新了一些东西"
   ```
   - 参考：§5.1，缺 type、缺 scope、subject 完全无信息

---

## §7 与下周（W5 CNN/ResNet）的衔接预告

> 这一节是"未来视角的元笔记"——告诉自己 W4 的哪些东西下周还要用。

| W4 学到的 | W5 怎么用 |
|---|---|
| BatchNorm1d 训练/推理双行为 | W5 升级到 `BatchNorm2d`，统计是 per-channel（shape 从 `(B,H)` 变 `(B,C,H,W)`） |
| fused BN 数学推导 | W5 D4 残差块里 `Conv-BN-ReLU` 三合一融合的真实例子 |
| Adam state 显存账 | W5 D5 在 ResNet18 + CIFAR-10 上**实测**四份开销 |
| Kaiming 初始化（fan_in = hidden） | W5 D1 升级到 `fan_in = C_in × k × k` |
| 第一次 profiler（文本表格 + chrome trace） | W5 D6 用同样方法分析 ResNet18，看 Conv stage3/4 的 hotspot |
| 元笔记结构 | W5 D7 用同样的模板写 `week5_industrial_view.md`（姐妹篇） |

**如果今天 §3 的 profiler trace 没跑通，下周 W5 D6 一定会卡住——今天必须搞定。**

---

## §8 完成标准 Checklist

主计划 W4 D7 的四个清单项：

- [ ] **DL**：完成 `tech_notes/week4_industrial_view.md`（本周核心元笔记）——本笔记 §2 的内容
- [ ] **DL**：用 `torch.profiler` 跑完整训练并导出 chrome trace——`logs/makemore_trace.json` 存在且能被 `chrome://tracing/` 打开
- [ ] **DL**：清理代码 + 写最终 README（含所有截图）——README 含运行命令、结果表、笔记索引
- [ ] **DL**：GitHub 至少 3 次有意义 commit——参考 §5.2 的模板

**追加的"工业级"标准**（让今天的产出真的能进简历）：

- [ ] 能脱稿在 3 分钟内回答"训推显存为什么差 4 倍"
- [ ] chrome trace 里能口头标注至少 3 个观察（hotspot / launch gap / 拷贝）
- [ ] README 里至少 1 张配置 → 结果对照表
- [ ] commit message 全部符合 Conventional Commits 规范

---

## §9 一句话总结

> **W4 不是教你"BN 是什么"或"Adam 是什么"——它在教你一种思维方式：每学一个新算子/新概念，先问"它训练时长什么样、推理时长什么样、显存代价多少、能不能融合"。这种条件反射就是 AI Infra 工程师和普通 PyTorch 用户的分水岭。**

今天写完元笔记的那一刻，W4 才算真正结束。
