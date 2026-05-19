# Week 4 · Day 3（2026-05-20，缓冲日）：train/dev/test split + 三轴调参 + 首次 torch.profiler

> **覆盖任务**（计划 line 1172-1175）：
> - [ ] DL：看 EP3 第 1:00-end
> - [ ] DL：实现 train/dev/test split，dev loss < 2.2
> - [ ] DL：完成三轴调参实验（embedding / hidden / context），至少 6 组
> - [ ] DL：完成第一次 `torch.profiler` 实验，写 `tech_notes/first_profiler.md`
>
> **阅读对象**：你自己——已经完成 Day 2 的 MLP 前向 + mini-batch SGD + lr-range test，现在 dev loss 卡在 2.3 左右，需要"会调"才能继续往下推。
>
> **本笔记的设计**：每节按"为什么 → 怎么做 → 工业怎么用"三段式。读完应该能独立把 dev loss 调到 2.2 以下，并且看懂 chrome trace 时间线。

---

## 0. 学习目标（看完应能回答）

1. 为什么必须切 train / dev / test 三份？为什么不能只用 train + test？
2. dev loss 和 test loss 的角色完全不一样——具体不一样在哪？
3. 三轴调参（embed_dim / hidden_size / block_size）应该按什么顺序调？为什么不能"全部一起穷举"？
4. 一次"实验"的最小可复现单元是什么？该记录哪些字段？
5. `torch.profiler` 输出的表里那些 `CPU time` / `Self CPU time` 到底是哪段时间？
6. **为什么 embedding lookup 在 profiler 里只占 1-3%，而矩阵乘法占 60%+？这跟"GPU 的 roofline"有什么关系？**
7. 缓冲日的本质：今天为什么不学新东西？

---

## 1. EP3 第 1:00-end 视频内容速览

这 15 分钟 Karpathy 做了**调参方法论**的演示，没有新的网络结构：

### 1.1 加上 dev / test split
之前 Day 2 你只有一个数据集，所有训练都看 train loss。Karpathy 把名字按 80/10/10 切成 train / dev / test：
- **train**：用来更新参数（梯度下降只看这部分）
- **dev**（也叫 validation）：用来**选超参**（看哪个 embed_dim/hidden/lr 组合最好）
- **test**：**最后只用一次**，报告最终结果，不能用它来调参

### 1.2 三轴消融
他依次试了：
1. embed_dim：2 → 10（最大单步改善，dev loss 从 2.3 降到 2.2）
2. hidden_size：100 → 200/300（小幅改善）
3. block_size：3 → 4/5/8（看上下文更长是否帮助）
4. 加入 lr decay（最后 1/10 步把 lr 缩 10×）—— Karpathy 视频里最后调到 dev loss ≈ 2.17

### 1.3 一个关键直觉：超参之间有耦合
他演示了"embed_dim=10 + lr=0.1 训练不动，但 embed_dim=10 + lr=0.01 就行"——**改一个超参，可能要重新调另一个**。这是为什么不能"全部一起穷举"的根本原因。

---

## 2. 为什么要切 train / dev / test：一个新生的常见误解

### 2.1 一个真实场景：餐厅厨师学做菜

把训练比作"厨师学新菜"：
- **train set**：师傅教学时用的 100 道试做菜——厨师反复尝试、调整
- **dev set**：师傅手里 20 道"考核菜"——每周给厨师一次评分，让他知道哪里要改
- **test set**：开业当天来的真实客人——他们的反馈才是真实水平

**只有 train + test 会怎么样？**
你为了让 test 分数好看，会反复调超参 → 等于让厨师**反复用考核菜练习**——分数高，但客人来了照样不会做。这就是**用 test 调参 = test 泄露 = 假精度**。

### 2.2 dev 和 test 的核心区别

| | dev set | test set |
|---|---|---|
| 用途 | 选超参（哪个配置好） | 报告最终性能 |
| 频率 | 可以反复看（每个实验跑完看一次） | **整个项目只看 1-2 次** |
| 数据来源 | 与 train 同分布 | 与 train 同分布 |
| 是否能用来"再调一次" | 可以 | **不可以**，看了就废了 |

> **工业纪律**：实际项目里，team leader 会把 test set 加密放在一个 secret 文件里，只有"快上线了"才解密评一次。Kaggle 比赛把 test set 设为 private leaderboard 就是这个意思。

### 2.3 为什么必须随机切，不能按字母序切

如果你直接 `words[:80%]` 当 train，剩下的当 dev/test：
- names.txt 大概率是按某种顺序排的（首字母、长度、来源）
- train 看到全是 A-S 开头的名字，dev 全是 T-Z → **分布不一致**
- dev loss 会比真实情况差很多，你以为模型不行，其实是 dev 不公平

**正确做法**：先 `torch.randperm(len(words))` 打乱再切。**且必须固定 seed**——否则下次跑实验，train/dev/test 都换了，前后实验不可比。

```python
g = torch.Generator().manual_seed(42)
perm = torch.randperm(len(words), generator=g).tolist()
words = [words[i] for i in perm]
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
train_words = words[:n1]
dev_words   = words[n1:n2]
test_words  = words[n2:]
```

> 这段代码 Day 2 的 `src/train.py::load_data` 已经写好了，今天直接复用。

---

## 3. 三轴调参：从"瞎试"升级为"消融实验"

### 3.1 一个错误示范：穷举法

新手最容易这样想：
> "我有 3 个超参（embed_dim, hidden, block_size），每个试 4 个值，那就是 4×4×4=64 组实验，全跑一遍取最好。"

64 组实验 × 每组 5 万步 × 每步几毫秒 = **几个小时甚至一天**。而且最后你也不知道：
- 为什么这组好？是哪个超参起作用？
- 如果再加一个超参（如 lr），要扩展到 256 组吗？

### 3.2 正确做法：消融实验（Ablation Study）

**核心思想：一次只动一个变量，其他全部固定。**

类比体检：医生不会同时让你"减肥 + 戒烟 + 吃药"再看血压——他会一项一项改，看每项的贡献。

具体到 makemore，按以下顺序：

| 顺序 | 改什么 | 固定什么 | 为什么这个顺序 |
|---|---|---|---|
| 1 | embed_dim: 2 → 10 → 30 | hidden=100, block=3, lr=0.1 | 这是 Karpathy 视频里**最大单步改善**的轴，先打主力 |
| 2 | hidden_size: 100 → 200 → 300 | embed_dim 用上一步最优值 | hidden 是"模型容量"的核心，第二重要 |
| 3 | block_size: 3 → 4 → 5 → 8 | 前两个用最优 | block_size 改了之后 `W1.shape` 也变（`B*E → H`），影响最深 |

每改一个超参，**也要重新做一次 lr-range test**（Day 2 §4），因为最佳 lr 跟模型大小耦合：模型变大，lr 通常要调小。

### 3.3 实验记录的最小字段（写到 `logs/exp_compare.csv`）

| 字段 | 含义 | 为什么记 |
|---|---|---|
| `exp_id` | 实验编号（如 W4D3_01） | 出 bug 时能精确定位 |
| `seed` | 随机种子 | **不记这个=结果不可复现**，这是工业第一守则 |
| `embed_dim` | E | 改的超参 |
| `hidden_size` | H | 改的超参 |
| `block_size` | B | 改的超参 |
| `lr` | 学习率 | 改的超参 |
| `batch_size` | 默认 32 | 固定值也要记，方便日后对照 |
| `steps` | 训练步数 | 默认 50000 |
| `num_params` | 总参数量 | 不同配置参数量差很多，模型容量横向对比要看这个 |
| `train_loss_final` | 最后 100 步的平均 train loss | 看是否过拟合（train 远低于 dev） |
| `dev_loss_final` | 最后一次 eval 的 dev loss | **选超参的依据** |
| `wall_time_sec` | 训练总耗时 | 工业上很关键：dev loss 改善 0.01 但训练慢 10× 不值得 |

**为什么不记 test_loss？** 因为 **test 只在最后选完所有超参后跑一次**。在调参阶段就记 test 等于用 test 选超参，犯了第2节的错。

### 3.4 至少 6 组实验的最小方案

```
基线（baseline）：    B=3, E=2,  H=100, lr=0.1   → 期望 dev ~2.30
exp_01：              B=3, E=10, H=100, lr=0.1   → 期望 dev ~2.20  ← 最大改进
exp_02：              B=3, E=10, H=200, lr=0.1   → 期望 dev ~2.18
exp_03：              B=3, E=10, H=300, lr=0.05  → 期望 dev ~2.17（lr 减半因为模型变大）
exp_04：              B=5, E=10, H=200, lr=0.05  → 期望 dev ~2.15  ← 长上下文红利
exp_05：              B=8, E=10, H=200, lr=0.05  → 期望 dev ~2.13
```

> **目标**：至少一组 dev loss < 2.2。视频里 Karpathy 调到 2.17 是经过 lr decay 的——你这周还没学 lr decay，能到 2.15-2.20 已经达成。

### 3.5 一个实战代码片段：批量跑实验

复用 Day 2 的 `src/train.py`，外层加一个调度循环：

```python
"""批量跑消融实验，结果写入 logs/exp_compare.csv。

运行：python -m src.run_ablation
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model import MakemoreMLP
from src.train import eval_loss, load_data, train


EXPERIMENTS = [
    # (exp_id, block_size, embed_dim, hidden_size, lr)
    ("W4D3_00_baseline", 3, 2,  100, 0.10),
    ("W4D3_01_emb10",    3, 10, 100, 0.10),
    ("W4D3_02_h200",     3, 10, 200, 0.10),
    ("W4D3_03_h300",     3, 10, 300, 0.05),
    ("W4D3_04_b5",       5, 10, 200, 0.05),
    ("W4D3_05_b8",       8, 10, 200, 0.05),
]


def run_one(exp_id, B, E, H, lr, Xtr, Ytr, Xdv, Ydv, steps=50_000, seed=42):
    model = MakemoreMLP(block_size=B, embed_dim=E, hidden_size=H, seed=seed)
    t0 = time.perf_counter()
    history = train(model, Xtr, Ytr, Xdv, Ydv,
                    steps=steps, batch_size=32, lr=lr,
                    eval_every=steps, seed=seed)
    wall = time.perf_counter() - t0
    final_step, train_loss, dev_loss = history[-1]
    return {
        "exp_id": exp_id, "seed": seed,
        "block_size": B, "embed_dim": E, "hidden_size": H,
        "lr": lr, "batch_size": 32, "steps": steps,
        "num_params": model.num_params(),
        "train_loss_final": round(train_loss, 4),
        "dev_loss_final": round(dev_loss, 4),
        "wall_time_sec": round(wall, 1),
    }


def main():
    Path("logs").mkdir(exist_ok=True)
    # 注意：不同 block_size 的数据要重新构造，这里偷懒用最大的 8 先建一份再切
    # 实际上更严谨的做法是每个 block_size 各跑一次 load_data
    rows = []
    for exp_id, B, E, H, lr in EXPERIMENTS:
        print(f"\n=== {exp_id}: B={B} E={E} H={H} lr={lr} ===")
        Xtr, Ytr, Xdv, Ydv, _, _, _, _ = load_data(block_size=B)
        row = run_one(exp_id, B, E, H, lr, Xtr, Ytr, Xdv, Ydv)
        rows.append(row)
        print(f"→ dev_loss={row['dev_loss_final']:.4f}, "
              f"params={row['num_params']:,}, wall={row['wall_time_sec']}s")

    # 写 CSV
    csv_path = "logs/exp_compare.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()
```

### 3.6 怎么读 `exp_compare.csv`

跑完后，在 Excel/jupyter 里按 `dev_loss_final` 排序——但**不要只看最小**，还要看：
- **每参数量的改善**：`(baseline_dev - this_dev) / num_params` 单位换成 "每 1k 参数能降多少 loss"，挑性价比最高的
- **wall_time 性价比**：如果一个配置 dev loss 只低 0.005 但慢了 3 倍，**不值得**——这就是工业里的"diminishing returns"
- **train_loss vs dev_loss 的 gap**：差值 > 0.3 说明过拟合，模型容量过剩了（要么减小，要么加正则/Dropout）

---

## 4. torch.profiler：第一次摸到"代码到底慢在哪"

### 4.1 一个直觉问题：你猜代码 60% 的时间花在哪？

新手通常猜"训练循环里 forward 最慢"——其实**取决于模型规模和硬件**：
- makemore MLP（极小模型）在 CPU 上跑：**数据准备 + Python 循环开销可能占 50%+**，真正的矩阵乘法占不到 30%
- ResNet50 在 GPU 上跑：矩阵乘法占 70%+，数据准备能用流水线掩盖
- LLM 推理：**几乎 100% 时间都在矩阵乘法和 attention 上**，数据 IO 几乎可忽略

**不实测，你永远猜不准。** 这就是为什么工业界永远先 profile 再优化——盲目优化是程序员最大的浪费。

### 4.2 profiler 基础概念：三个时间口径

| 口径 | 中文 | 含义 | 一句话理解 |
|---|---|---|---|
| `CPU total` | CPU 总时间 | 包括子调用 | "这一行代码连同它调的所有函数总共花了多久" |
| `Self CPU` | 自身 CPU 时间 | **不**包括子调用 | "这一层函数自己干活花了多久（子函数另算）" |
| `# Calls` | 调用次数 | 这个 op 被触发几次 | 一次 forward 触发 `aten::addmm` 2 次（W1 一次，W2 一次） |

**优化策略**：
- 看 **Self CPU 排名第一**的算子 → 优化它 = 优化代码热点
- 看 **CPU total 排名第一**的高层函数 → 知道大块时间花在哪个模块

类比：餐厅经营报表
- `CPU total` = 整个部门总人力支出（含外包）
- `Self CPU` = 部门内部员工的工资（不含外包给别人的）
- `# Calls` = 这个部门今天接了几单

### 4.3 最小可运行的 profiler 代码（`src/profile_one_step.py`）

```python
"""用 torch.profiler 跑一次 forward + backward，打印 top 算子时间分布。

运行：python -m src.profile_one_step
输出：
    1) 控制台 top-20 算子表
    2) logs/profiler_trace.json（chrome://tracing 打开看时间线）
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from src.model import MakemoreMLP
from src.train import load_data


def main():
    # 准备数据 + 模型
    Xtr, Ytr, _, _, _, _, _, _ = load_data(block_size=3)
    model = MakemoreMLP(block_size=3, embed_dim=10, hidden_size=200)

    # 预热：第一次 forward 会触发各种 lazy init，时间不准
    Xb, Yb = Xtr[:32], Ytr[:32]
    for _ in range(3):
        logits = model.forward(Xb)
        loss = F.cross_entropy(logits, Yb)
        for p in model.parameters():
            p.grad = None
        loss.backward()

    # 正式 profile：跑 10 步，取平均
    Path("logs").mkdir(exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,           # 记录每个 op 的输入 shape，方便排查"为啥这个算子调用 100 次"
        profile_memory=False,         # CPU 上内存追踪意义不大，先关
        with_stack=False,             # 加 Python 栈信息会慢 10×，初次 profile 关掉
    ) as prof:
        for step in range(10):
            with record_function("step"):                  # 自定义区域名，方便在 chrome trace 里找
                with record_function("forward"):
                    logits = model.forward(Xb)
                    loss = F.cross_entropy(logits, Yb)
                with record_function("backward"):
                    for p in model.parameters():
                        p.grad = None
                    loss.backward()
                with record_function("update"):
                    with torch.no_grad():
                        for p in model.parameters():
                            p.data -= 0.1 * p.grad

    # 1) 打印 top-20 算子（按 Self CPU 排）
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    ))

    # 2) 导出 chrome trace（在 Chrome 浏览器打开 chrome://tracing 后 Load 这个文件）
    trace_path = "logs/profiler_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nchrome trace saved to {trace_path}")
    print("打开 Chrome → 输入 chrome://tracing → Load → 选这个文件")


if __name__ == "__main__":
    main()
```

### 4.4 你大概率会看到的输出（CPU 上，B=3, E=10, H=200）

```
---------------------  ------------  ------------  ------------  ------------
Name                   Self CPU %    Self CPU      CPU total %   CPU total
---------------------  ------------  ------------  ------------  ------------
aten::addmm              35.21%      18.456ms      37.50%        19.654ms      ← W1/W2 矩阵乘法
aten::tanh                8.43%       4.418ms      8.43%          4.418ms
aten::cross_entropy       6.12%       3.207ms      14.85%         7.785ms
aten::randint             5.88%       3.082ms      5.88%          3.082ms      ← 数据采样开销
aten::index               4.21%       2.207ms      4.21%          2.207ms      ← C[X] embedding lookup
aten::view                2.15%       1.127ms      2.15%          1.127ms      ← flatten
...
```

### 4.5 关键观察（写到 `tech_notes/first_profiler.md`）

**观察 1：embedding lookup 占比远小于矩阵乘法**
- `aten::index`（C[X]）：~4%
- `aten::addmm`（@矩阵乘法）：~35%
- 比例约 1:9

为什么？
- **embedding lookup 是 memory-bound**：只读 N×B=96 个内存位置，几乎不算东西
- **矩阵乘法是 compute-bound**：`(32, 6) @ (6, 200)` 要做 32×200×6 = 38400 次乘加，纯算力消耗
- 这是你 Day 1 `embedding_as_lookup.md` 的 profiler 工业实证——**理论说 lookup 比 onehot@W 快，profiler 给数字**

**观察 2：随机采样 `randint` 占 5-6% 也不小**
这是 Python 调度开销 + 张量创建的固定 overhead。工业训练里会用 `DataLoader + num_workers` 把这部分挪到独立进程，让 GPU 一直忙——这就是为什么 LLM 训练你看到 GPU 利用率 95%+ 的秘密。

**观察 3：cross_entropy 的 self vs total 差很多**
`Self CPU 6%, CPU total 14%` 说明它自身只算了一点，其余时间在调子函数（log_softmax + nll_loss）。这是看"高层 API 是不是黑盒"的好方法——`Self CPU << CPU total` 的算子，里面有更小的 op 可以单独优化。

### 4.6 chrome trace 是 profiler 的高级形态

打开 `chrome://tracing` → Load `logs/profiler_trace.json`，会看到：
- 横轴：时间（毫秒）
- 纵轴：调用栈（你的 `step` / `forward` / `backward` / `update` 块清晰可见）
- 每个长方形：一个 op 的执行时长

**怎么用**：
- 看哪个 step 异常长（数据准备 spike）
- 看 forward 和 backward 的时间比例（理想 1:2，因为 backward 计算量是 forward 的两倍）
- 看是否有"空白时间"（CPU/GPU 空转 = 数据没准备好）—— 工业训练常用 trace 找数据 pipeline 瓶颈

### 4.7 工业延伸：profile 是优化的第一步

| 场景 | 用 profiler 找的瓶颈 | 对应优化 |
|---|---|---|
| 小模型 CPU 训练（你今天） | Python overhead + 数据采样 | torch.compile / 异步 DataLoader |
| ResNet50 GPU 训练 | 数据加载（CPU→GPU 传输） | 多 worker + pin_memory + non_blocking |
| LLM 推理 | KV cache 重复计算 | PagedAttention / FlashAttention |
| LLM 推理（小 batch） | matmul 太小，没打满 SM | continuous batching / dynamic batching |

> **第8周推理优化主线的核心技能**：会用 profiler 找瓶颈、读 chrome trace。今天就是这条主线的入口。

---

## 5. 把 Day 3 串成一条可运行的命令

```bash
# 0. 先确保 Day 2 的 src/model.py, src/train.py 都能跑
python -m src.model
python -m src.train --steps 5000   # 用小步数先验证训练循环没 bug

# 1. 看 EP3 第 1:00-end（视频任务，无代码产出）

# 2. 跑批量消融实验（约 30-60 分钟，看 CPU 速度）
python -m src.run_ablation
# 期望：logs/exp_compare.csv 至少 6 行，至少 1 行 dev_loss < 2.2

# 3. 跑首次 profiler 实验
python -m src.profile_one_step
# 期望：
#   控制台输出 top-20 算子，aten::addmm 排第一
#   logs/profiler_trace.json 生成

# 4. 用 Chrome 打开 chrome://tracing，Load profiler_trace.json，截图存到 logs/

# 5. 写 tech_notes/first_profiler.md，至少包含：
#    - 你机器的 top-5 算子表
#    - embedding vs matmul 的占比对比
#    - chrome trace 截图 + 你的 3 条观察
```

---

## 6. 缓冲日的本质：今天为什么不学新东西？

回看计划 V2 的执行优化第 2 条："每周固定 1 次缓冲日，只做查漏补缺，不开新任务。"

为什么？因为深度学习的学习曲线**不是线性的**：

```
进度 │                              ←── 没缓冲日：知识塌方
     │      🚀
     │     /
     │    /         ⚠️
     │   /         /
     │  /        /
     │ /        /
     │_______/_____________ 时间
       新内容   忘了/没消化
```

加上缓冲日：

```
进度 │                                    ←── 有缓冲日：每周稳定上升
     │              🚀
     │             /
     │           ✓ 缓冲日：调参、profile、整理
     │          /
     │         /
     │        /
     │       ✓ 缓冲日：复盘 Day 1-2
     │      /
     │_________________________ 时间
```

**今天的任务 = "用旧知识把上半周的代码打磨成可复现实验"**——这比学新概念更能产生面试时能讲出来的"工程产出"。

---

## 7. 自测题（合上文档默答）

1. 一个新人说"我把 80% 数据当 train，20% 当 test 就行了"——你怎么劝他？
2. 如果你跑出来 dev_loss < train_loss（dev 比 train 还低），可能是什么原因？
3. embed_dim=2 → 10 dev loss 改善 0.1；embed_dim=10 → 30 dev loss 几乎不变。这说明什么？
4. profiler 表里 `aten::addmm` Self CPU 35% 但 CPU total 也是 35%（几乎一样），说明什么？
5. 如果 chrome trace 看到 forward block 之间有大块空白时间，是 CPU 还是数据 pipeline 的问题？
6. **为什么"用 dev 选最好的超参，然后报告 dev loss"是错的？应该报告什么？**

> 参考答案位置：1→§2.1 厨师比喻；2→dev 集太小或刚好命中容易样本，统计涨落；3→模型容量饱和，embed_dim 不再是瓶颈，该升级 hidden 或 block_size；4→该算子是叶子算子（没有子调用），优化它就是直接收益；5→数据 pipeline 没跟上，CPU 在等数据；6→dev 已经被你"用过"了，跟 train 一样污染了，必须报告 test loss（最后跑一次）。

---

## 8. 与已有笔记的串联

| 今天的内容 | 关联点 |
|---|---|
| train/dev/test split | Day 2 §3.3 训练脚本里其实已经切好了，今天给它配上"为什么这样切"的方法论 |
| 三轴消融 | Day 2 §5 三张优惠券（每改一个超参，可能要重新做 lr-range test） |
| profiler 看到 embedding 占比小 | Day 1 `embedding_as_lookup.md` 速度对比的实证（理论 → 数字） |
| profiler 看到 matmul 占大头 | Day 2 §5.4 工业延伸里"GPU 的 roofline 模型"——matmul 是 compute-bound 的代表 |
| chrome trace 找空白时间 | 第8周推理优化主线的入门工具，今天先建立"trace 思维" |

**明天（Day 4）的入口**：EP4 上半。你今天 profile 出来矩阵乘法占 35%——明天会发现初始化错误会让这个 35% **白做工**（tanh 饱和后梯度近 0，权重不更新）。Day 4 一开始 Karpathy 就会演示"初始 loss=27 vs 期望 3.3"的诊断过程，那就是你今天 `aten::tanh` 那一行 op 在白干活的证据。

**Day 5（最关键的一天）的伏笔**：BatchNorm 是给"梯度有界"那张优惠券打的补丁——今天你跑完 6 组实验如果发现 hidden=300 训练不稳，那就是 BN 出场的时机。

---

## 9. 完成标准检查清单

- [ ] `logs/exp_compare.csv` 至少有 6 行实验记录（带完整字段）
- [ ] 至少一组配置 dev loss < 2.2
- [ ] `tech_notes/first_profiler.md` 包含：
  - 你机器上的 top-5 算子表（截图或文本）
  - embedding lookup 占比 vs 矩阵乘法占比
  - chrome trace 截图 + 至少 3 条观察
- [ ] `W4_day3_log.md` 记录今天最大发现 + 卡壳点 + 明天的入口数值（初始 loss）
- [ ] **能口头解释**："为什么 embedding lookup 在 profiler 里只占 4%，但在 LLM 推理里却是瓶颈？"
  （提示：今天 N×B×E = 32×3×10=960 太小；LLM 时是 batch×seq×4096×50000 的 GB 级 lookup → memory-bound 才显现）

---

*笔记生成日期：2026-05-20（W4 Day 3，缓冲日）*
