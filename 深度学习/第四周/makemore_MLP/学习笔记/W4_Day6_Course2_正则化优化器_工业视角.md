# Week 4 · Day 6（2026-05-23，周六）：Andrew Ng Course 2 全 3 周整合 · 正则化/优化器/超参 + Adam 显存账

> **覆盖任务**（计划 line 1204-1207）：
>
> - [ ] DL：Andrew Ng Course 2 Week 1 完成（正则化、Dropout、梯度消失/爆炸）
> - [ ] DL：Course 2 Week 2 完成（Momentum / RMSProp / Adam / lr scheduling）
> - [ ] DL：Course 2 Week 3 完成（超参调优、BN、Softmax）
> - [ ] DL：完成 `tech_notes/optimizer_memory.md`（含 Adam state 显存分析）
>
> **阅读对象**：你自己——Day 1-5 已经把 makemore MLP（含 Kaiming + BN）训出 dev loss ≈ 2.1；Day 5 的 `batchnorm_inference.md` 已经讲清"训推双行为"。Course 2 这三周内容里，**BN 和 Softmax 你已经会了**（Day 5 + 第3周 bigram），所以今天的重点是 W1 的正则化/Dropout/梯度消失 和 W2 的优化器演化 + 显存账——这两块是 AI Infra 主线绕不过去的。
>
> **本笔记的设计**：每节"现象 → 直觉 → 数学 → 代码 → 工业锚点"五段式。读完应该能：（1）口述 L2/Dropout 各自的"训推双行为"差异（呼应 Day 5 BN 那张表）；（2）从 SGD → Momentum → RMSProp → Adam 一条线推导每一步加了什么、为什么加；（3）**算清楚 7B 模型用 FP32+Adam 训练为什么要 112GB 显存**——这是 `tech_notes/optimizer_memory.md` 的核心；（4）解释清楚 Course 2 W3 的"调参优先级"为什么把 lr 排第一。

---

## 0. 学习目标（看完应能回答）

1. L2 正则的"权重衰减"和"高斯先验"是什么关系？为什么 PyTorch 的 `weight_decay` 不是 Andrew Ng 视频里那个 `λ`？
2. Dropout 训练时随机置零 + 缩放 1/(1-p)，推理时关闭——这和 BN 一样是"训推双行为"吗？为什么 `model.eval()` 会同时关掉它俩？
3. 梯度消失 / 梯度爆炸的本质是什么？为什么深网络中 $W^T$ 连乘会出问题？Kaiming 初始化、BN、残差连接、Adam 这四个"补丁"分别堵了哪个口子？
4. 为什么 mini-batch SGD 比 full-batch 工业上更好用？除了"内存装不下"之外还有什么理由？
5. **从 SGD → Momentum → RMSProp → Adam 这条演化线**：每一步加了什么新机制？为什么 Adam 几乎是工业默认选择？
6. **Adam 维护的 m 和 v 各占多少显存**？为什么"7B FP32 训练 ≈ 112GB"这个账要拆成"参数 + 梯度 + Adam state + 激活"四份？
7. 学习率调度（lr decay / warmup / cosine）解决了什么问题？为什么 transformer 训练几乎一律用 warmup？
8. Course 2 W3 给的调参优先级 `lr → momentum/Adam β → batch_size → layers/units → lr decay` 是怎么排出来的？这个排序背后的逻辑是什么？
9. **本周和你 Day 5 / `autograd_explained.md` / `batchnorm_inference.md` 怎么串成完整链条**？（这是周末元笔记 `week4_industrial_view.md` 的脚手架）

---

## 1. 正则化：L2 与 Dropout 的"训推双行为"

### 1.1 一个比喻：复习 vs 考试

```
训练（学生复习）        推理（考试）
─────────────────────────────────────
L2 正则：    每天背书前先把书包重 1kg     考试时背原书包（轻装上阵）
Dropout：    复习时随机蒙住 30% 内容    考试时所有内容都看（关掉随机性）
BN（Day 5）：用今天班级的平均分校准    用历史 EMA 分数校准
```

三个机制有个共同点：**训练时启用，推理时关闭/退化**——这就是为什么 `model.eval()` 一行能同时关掉它们。**这是工业代码里最常见的 bug 来源之一**：忘了 `.eval()`，推理结果时好时坏。

### 1.2 L2 正则：原理 + 代码

**数学形式**：在 loss 后加一项 $\frac{\lambda}{2} \sum_w w^2$。求导后变成：

$$\nabla L_{\text{total}} = \nabla L_{\text{data}} + \lambda \cdot w$$

更新规则：$w \leftarrow w - \eta(\nabla L_{\text{data}} + \lambda w) = (1 - \eta\lambda) w - \eta \nabla L_{\text{data}}$

**注意**：每一步都把 w 乘以 $(1 - \eta\lambda)$——这就是"权重衰减"（weight decay）这个名字的由来。

**直觉**：你在告诉模型"参数越大越罚你"，等价于给参数加了一个零均值高斯先验（$p(w) \propto \exp(-\lambda w^2/2)$）——贝叶斯视角下，L2 正则的最大后验估计 = 朴素 MLE + 先验。

**为什么有效**：
- 防止权重无限增长 → 间接控制函数的 Lipschitz 常数 → 决策边界更平滑
- 让模型"被迫"用多个小权重组合，而不是依赖少数大权重 → 鲁棒性更好

```python
"""L2 正则的两种实现方式（结果一致，工业代码常用方式 B）"""
import torch
import torch.nn as nn

model = nn.Linear(100, 10)
optimizer_a = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer_b = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 方式 A：手动加到 loss 里（教科书写法）
def loss_with_l2(logits, y, model, lam=1e-4):
    base = nn.functional.cross_entropy(logits, y)
    l2 = sum((p ** 2).sum() for p in model.parameters())
    return base + 0.5 * lam * l2

# 方式 B：优化器内建 weight_decay（工业默认）
# 等价于"每步把 w 乘以 (1 - lr * weight_decay)"
# PyTorch 的 weight_decay = Andrew Ng 视频里的 λ，但已经吸收了 lr，所以数值上对不上
```

**易错点**（呼应 Day 5 `model.eval()` 那条因果链）：
- L2 正则**没有训推双行为**——它修改的是参数本身，推理时 w 就是训练后的 w
- 真正的"双行为"是 Dropout 和 BN
- 但 `weight_decay` 是工业训练几乎必开的，Transformer 训练里典型值 0.01 / 0.1

### 1.3 Dropout：训推双行为的"亲兄弟"

**训练时**：以概率 p 随机把每个神经元的输出置零，**剩下的乘以 1/(1-p)** 补偿（保持期望不变）。

```python
def dropout_train(x, p=0.5):
    """训练时：随机置零 + inverse scaling"""
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)  # 缩放保证 E[output] = E[input]

def dropout_eval(x, p=0.5):
    """推理时：直接返回 x，完全关闭"""
    return x
```

**直觉（用类比解释 inverse scaling）**：

```
假设 hidden 层 100 个神经元，每个均值 1，p=0.3
训练时：平均 70 个还在，每个变成 1/(1-0.3) ≈ 1.43
       总输出期望 = 70 × 1.43 ≈ 100  ← 和不 dropout 时一样
推理时：100 个全在，每个还是 1
       总输出期望 = 100 × 1 = 100   ← 对得上
```

**没有这个缩放会怎样**：训练时输出期望 70，推理时变 100——下一层看到的输入分布**完全错位**。这是上古版本 Dropout 的 bug，叫 "inverted dropout" 之前的写法。**今天所有框架默认都是 inverted（即上面这种）**。

**为什么有效（两个流派解释）**：
1. **集成视角**：每次 forward 都是一个不同的子网络，相当于训了 $2^N$ 个网络的隐式集成
2. **正则化视角**：随机置零强迫每个神经元不能依赖特定上游 → 学到的特征更冗余、更鲁棒

**工业现状（重要）**：
- CNN 时代 Dropout 几乎标配（VGG / ResNet 全连接层都用）
- **Transformer 时代 Dropout 用得很少**——LLaMA-2、GPT-3 都把 Dropout 设为 0 或极小（0.1）
- 原因：大模型的训练数据量足够大，过拟合本来就轻；BN/LN 已经提供了足够的正则化；Dropout 对训练吞吐有损（要算 mask）
- 但 fine-tune 小数据集时还是常用

**工业锚点（呼应 Day 5）**：
- BN 关掉时切到 running stats；Dropout 关掉时直接 identity
- **这两个就是 `model.eval()` 改变行为的"两兄弟"**——再加 LayerNorm（LN 训推完全一致，所以不在此列）
- 推理服务里忘了 `.eval()`：每次请求 Dropout 还在随机置零 → 同一输入两次结果不同 → 监控告警

### 1.4 梯度消失 / 梯度爆炸：四个"补丁"分别堵了什么

回忆 Day 5 §1：训练深网络时，反向传播会让梯度从最后一层"流"到第一层。这个流动是一连串的矩阵乘法：

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_n} \cdot \prod_{k=2}^{n} W_k^T \cdot \text{(激活导数)}$$

**核心问题**：如果每个 $W_k^T$ 的"放大系数"（更严谨说是奇异值）平均是 0.5，连乘 50 层后梯度变成 $0.5^{50} \approx 10^{-15}$——梯度消失；如果是 1.5，连乘 50 层后变成 $1.5^{50} \approx 6 \times 10^8$——梯度爆炸。

这就是 RNN 时代为什么训深网络几乎不可能、Transformer 时代为什么能训 100+ 层的核心矛盾点。**四个工业补丁**：

| 补丁 | 堵哪个口子 | 数学动作 | 你的笔记 |
|---|---|---|---|
| Kaiming/Xavier 初始化 | 起步时方差守恒 | $W \sim \mathcal{N}(0, g^2/\text{fan\_in})$ | Day 4 |
| BatchNorm | 训练中持续校准激活分布 | 每层强制 0 均值/单位方差 | Day 5 |
| 残差连接（ResNet） | 让梯度有"直通路" | $y = F(x) + x$，梯度 = $F'(x) + 1$ | 第5周 |
| Adam | 自适应学习率抹平梯度尺度差异 | 每参数有自己的有效 lr | 本笔记 §2.5 |

**关键洞察**：这四个补丁是**叠加生效**的，不是"二选一"。现代 Transformer 是 Kaiming 初始化 + LayerNorm + 残差 + Adam 四件套**全部用上**，缺一个都训不好。**Karpathy EP4 你已经体验过前两个**——下周 ResNet 会补第三个，本笔记 §2 讲第四个。

---

## 2. 优化器演化线：SGD → Momentum → RMSProp → Adam

### 2.1 演化线全景图（先看这张表再读细节）

| 版本 | 维护的状态 | 更新规则核心 | 解决的痛点 |
|---|---|---|---|
| SGD | 无 | $w \leftarrow w - \eta g$ | 基线 |
| SGD + Momentum | $v$（速度） | $v \leftarrow \beta v + g$, $w \leftarrow w - \eta v$ | 振荡、平坦区慢 |
| RMSProp | $s$（梯度平方 EMA） | $w \leftarrow w - \eta g / \sqrt{s+\epsilon}$ | 不同参数梯度尺度差异大 |
| Adam | $m$（一阶动量）+ $v$（二阶动量） | 上面两者合体 + 偏差修正 | 综合所有问题 |

**理解的关键**：每个新版本都是在前一个上加一个"工程补丁"，不是从零设计。

### 2.2 SGD：为什么 mini-batch 是工业默认

**Full-batch SGD**：每步用全部 N 个样本算梯度 → 一步走一次。**Mini-batch SGD**：每步用 B 个样本（B 通常 32-512）。

**Andrew Ng W2 的关键论点**：mini-batch 比 full-batch 工业上更好，**不仅因为内存装不下**，还有三个深层原因：

1. **每个 epoch 走更多步** → 同样时间预算下走得更远
2. **梯度噪声当成隐式正则** → 噪声能帮你跳出尖锐的局部极小值（sharp minima），落到平坦极小值（flat minima），后者泛化更好
3. **GPU 算力利用率最优**：太小（B=1）GPU 跑不满，太大（B=N）显存爆——B 在 32-512 是甜区

**用一个比喻**：

```
Full-batch  =  一个人扛 100kg 上楼，一次走一步（精确但慢）
Mini-batch  =  10 个人扛 10kg 上楼，10 次走 10 步（方向有抖动但总效率高）
SGD (B=1)   =  100 个人各扛 1kg，乱跑（方向噪声太大，可能反复横跳）
```

**工业 batch size 选择**（呼应 Day 2 lr-range test）：
- 视觉任务（CNN）：B=32 ~ 256，受显存限制
- LLM 训练：B 物理上 32（per GPU），但通过 gradient accumulation 等效到 4M tokens
- 推理（不在本笔记范围）：B 越大吞吐越高，但延迟也越高 → 服务工程的 tradeoff

### 2.3 Momentum：给 SGD 装一个"惯性"

**问题**：纯 SGD 在 loss 平面是椭圆形（不同方向曲率差异大）时，会沿狭窄方向振荡，沿平坦方向爬得很慢。

**核心机制**：维护一个"速度" $v$，每步用速度更新参数（不是直接用梯度）：

$$v_t = \beta v_{t-1} + g_t$$
$$w_t = w_{t-1} - \eta v_t$$

**直觉**：把参数空间想象成山地，纯 SGD 像每秒重新决定走哪——容易被局部地形误导；Momentum 像一辆滑下山的车——惯性帮你"穿过"小坑、"压住"震荡。

```python
def sgd_momentum_step(w, g, v_state, lr=0.01, beta=0.9):
    """单参数单步更新"""
    v_state = beta * v_state + g            # 更新速度
    w = w - lr * v_state                    # 用速度更新 w
    return w, v_state
```

**关键参数 β**：
- β=0：退化为纯 SGD
- β=0.9：等价于"平均过去 10 步的梯度"（$1/(1-0.9)=10$）—— **工业最常用**
- β=0.99：等价于"平均过去 100 步" —— 太慢响应不及时

**显存代价**：每个参数多一份 $v$ —— **额外 1× 参数大小的显存**（记住这个数，§3 显存账要用）。

### 2.4 RMSProp：给每个参数发"自适应学习率"

**问题**：不同参数的梯度尺度差异巨大——比如 embedding 层梯度是 $10^{-3}$，最后 logits 层是 $10^{0}$，用同一个 lr 必然出问题：要么 embedding 学不动，要么 logits 炸。

**核心机制**：维护"梯度平方的 EMA" $s$，用它给每个参数发独立的有效学习率：

$$s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2$$
$$w_t = w_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t$$

**直觉**：$\sqrt{s_t}$ 大致是梯度的"近期典型幅度"，除以它相当于把每个参数的梯度都缩放到差不多的量纲——**所有参数都有公平的"步长预算"**。

```python
def rmsprop_step(w, g, s_state, lr=0.001, beta2=0.999, eps=1e-8):
    s_state = beta2 * s_state + (1 - beta2) * (g ** 2)
    w = w - lr * g / (s_state.sqrt() + eps)
    return w, s_state
```

**显存代价**：每个参数多一份 $s$ —— **额外 1× 参数大小的显存**。

### 2.5 Adam：Momentum + RMSProp 合体 + 偏差修正

Adam = **A**daptive **M**oment **Estimation**，"自适应矩估计"。它做了两件事：

1. 同时维护 $m$（一阶动量，来自 Momentum）和 $v$（二阶动量，来自 RMSProp）
2. 在训练初期对 $m, v$ 做"偏差修正"——因为初始化为 0，前几步估计偏小

完整更新规则：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$w_t = w_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

**典型超参**：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \eta=10^{-3}$（这套是 PyTorch 默认值，**也是 Adam 论文给的推荐值，工业上几乎不调**）。

```python
"""Adam 单参数单步（教学版，工业版用 torch.optim.Adam）"""
def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g ** 2)
    m_hat = m / (1 - b1 ** t)      # 偏差修正
    v_hat = v / (1 - b2 ** t)
    w = w - lr * m_hat / (v_hat.sqrt() + eps)
    return w, m, v
```

**为什么 Adam 几乎是工业默认**：
- 鲁棒：默认超参就能跑大多数任务（不像 SGD 要精调 lr + momentum）
- 自适应：每个参数有独立有效 lr，对参数尺度不敏感
- 论文级：Transformer、BERT、GPT 系列、Stable Diffusion 全部用 Adam/AdamW

**Adam 的代价（接 §3 显存账）**：
- **每个参数多 2 份状态**（m 和 v）—— 额外 2× 参数大小的显存
- 对比 SGD（0×）和 Momentum（1×），Adam 是优化器里**显存最贵的常见选择**

**AdamW vs Adam（一句话）**：AdamW 把 weight decay 从"加到梯度里"改成"直接乘到 w 上"——**解耦了 weight decay 和自适应 lr**，效果更好，**现代 Transformer 训练几乎一律用 AdamW**。

---

## 3. 显存账：为什么 7B 模型 FP32 训练 ≈ 112GB（`optimizer_memory.md` 核心）

这一节是本周 AI Infra 主线的"压轴产出"——一个能在面试里 5 分钟讲清楚的显存模型。**这就是你 `tech_notes/optimizer_memory.md` 应该写进去的内容**。

### 3.1 训练显存的四份开销

把模型参数总数记作 $P$（比如 7B = $7 \times 10^9$），每个数用 FP32（4 字节）。**训练时显存分四份**：

| 分项 | 大小（FP32） | 7B 模型时 | 原因 |
|---|---|---|---|
| **参数 $W$** | 4P 字节 | 28 GB | 必须有 |
| **梯度 $\nabla W$** | 4P 字节 | 28 GB | backward 算出来的，要保存到 step 后才能释放 |
| **优化器状态** | 8P 字节（Adam）| **56 GB** | Adam 的 m + v，**两份** |
| **激活值（activation）** | 视序列长度/batch 而定 | 视情况 | 反向传播要用前向中间值 |
| **总计** | $\geq 16P$ 字节 | **≥ 112 GB** | 还不算激活 |

**关键洞察**：**优化器状态是最大的一份**——比参数本身还大。这就是为什么 LLM 训练显存优化第一个砍的就是 optimizer state。

### 3.2 推理显存：只剩参数

推理时三件事**全部消失**：
- 梯度不需要算（`torch.no_grad()`）
- Adam state 不存在（没在训练）
- 激活值不需要保存（forward 完就丢）

**所以推理只需要 4P 字节 = 28 GB**——比训练省 4 倍。

```
训练 7B FP32 ≈ 112 GB（A100 80G 单卡装不下，必须 ZeRO/FSDP 分片）
推理 7B FP32 ≈ 28 GB （A100 80G 单卡舒服跑）
推理 7B INT8 ≈ 7 GB  （消费级显卡 RTX 4090 24GB 都能跑）
```

**这就是为什么"训练用 A100 集群，推理用 4090"——不是因为推理"简单"，而是因为推理的显存模型完全不同**。

### 3.3 各优化器的显存账（速查表）

| 优化器 | 额外状态 | 7B 模型 FP32 总训练显存 |
|---|---|---|
| SGD | 0× | 28 + 28 + 0 = **56 GB** |
| SGD + Momentum | 1× | 28 + 28 + 28 = **84 GB** |
| RMSProp | 1× | 28 + 28 + 28 = **84 GB** |
| **Adam/AdamW** | **2×** | 28 + 28 + 56 = **112 GB** |
| Adafactor | ~0.5×（分解 v）| 28 + 28 + 14 ≈ **70 GB** |
| 8-bit Adam | 2× 但每个 8-bit | 28 + 28 + 14 = **70 GB** |

**工业延伸**：
- **bitsandbytes 8-bit Adam**：把 m 和 v 量化到 8-bit 存储，反量化后参与计算 → 优化器状态显存减半。Tim Dettmers 这一手让 24GB 消费级显卡也能 fine-tune 大模型
- **Adafactor**（Google T5 团队）：把 v 分解成行均值和列均值的外积 → 优化器状态从 2× 降到 ~0.5×，**牺牲少量精度换显存**
- **ZeRO（DeepSpeed）/ FSDP（PyTorch）**：把这三份开销**分片到多卡**——ZeRO-1 分片 optimizer state，ZeRO-2 加梯度，ZeRO-3 加参数。**这是现代 LLM 训练能跑 70B+ 的核心技术**

### 3.4 代码：用 PyTorch 实测优化器状态显存

```python
"""验证 Adam 的 optimizer state 真的是参数的 2 倍"""
import torch
import torch.nn as nn

# 一个 1M 参数的小模型（验证完原理就行，不必跑 7B）
model = nn.Sequential(nn.Linear(1000, 1000), nn.Linear(1000, 1000))
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量：{n_params:,}")
print(f"参数 FP32 显存：{n_params * 4 / 1024 / 1024:.2f} MB")

# SGD：optimizer state = 0
opt_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
opt_sgd.step()  # 没有 grad 也能 step
sgd_state = sum(s.numel() for group in opt_sgd.state.values() for s in group.values() if isinstance(s, torch.Tensor))
print(f"SGD 状态大小：{sgd_state * 4 / 1024 / 1024:.2f} MB")  # 0

# Adam：跑一步触发状态分配
opt_adam = torch.optim.Adam(model.parameters(), lr=0.001)
for p in model.parameters():
    p.grad = torch.zeros_like(p)
opt_adam.step()
adam_state = sum(s.numel() for group in opt_adam.state.values() for s in group.values() if isinstance(s, torch.Tensor))
print(f"Adam 状态大小：{adam_state * 4 / 1024 / 1024:.2f} MB")  # ≈ 2× 参数大小

# 预期输出：
# 参数量：2,002,000
# 参数 FP32 显存：7.64 MB
# SGD 状态大小：0.00 MB
# Adam 状态大小：15.27 MB  ← 正好 2× 参数显存
```

**这段代码你今天可以亲手跑一遍**——眼见为实比任何笔记都有说服力。

### 3.5 串到你已有的笔记

呼应 `autograd_explained.md §5.2` 那张"训推显存差 4 倍"的表，本笔记 §3.1-3.2 把它**升级成具体公式 + 具体数字**：

```
你的 autograd_explained §5.2：训练显存 ≈ 4× 推理显存
本笔记 §3.1-3.2 升级版：
  训练 = 4P（参数）+ 4P（梯度）+ 8P（Adam）+ 激活 ≥ 16P
  推理 = 4P
  比值 = 4×（不算激活）/ 更大（算激活）
```

---

## 4. 学习率调度：warmup + decay 的工业意义

### 4.1 为什么固定 lr 不够好

训练动力学的两个阶段：
- **早期**：参数离最优解远，需要大 lr 快速接近
- **后期**：参数接近最优解，大 lr 会跨过去，需要小 lr 精修

固定 lr 必然在某个阶段"不合适"。**学习率调度**就是让 lr 随训练进程变化。

### 4.2 三种常见调度策略

**Step decay**：每 N epoch 把 lr 减半。CNN 时代经典做法，简单粗暴。

```python
# 训练 90 epoch，每 30 epoch lr /= 10
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
```

**Cosine decay**：lr 按余弦曲线平滑下降到 0。**Transformer 训练标配**。

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
```

**Warmup + Cosine**：前 X 步 lr 线性增长到峰值，之后 cosine 降到 0。**LLaMA / GPT / BERT 全用这个**。

### 4.3 为什么 Transformer 一定要 warmup（关键工业细节）

**直觉**：Transformer 训练初期，Adam 的 v（二阶动量）估计极不准——前几步 v 还接近 0，导致有效 lr = $\eta/\sqrt{v}$ 巨大无比 → loss 直接 NaN。

**warmup 的作用**：让模型先用很小的 lr 跑几百步，等 Adam 的 v 稳定下来再放大 lr。**没有 warmup 的 Transformer 训练 99% 直接发散**。

```python
"""GPT-2 / LLaMA 经典 warmup + cosine 调度"""
import math
def get_lr(step, warmup_steps=2000, total_steps=600000, max_lr=6e-4, min_lr=6e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps  # 线性 warmup
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

**工业锚点**：这段代码几乎一字不差地出现在 nanoGPT、LLaMA 训练脚本、Mistral 训练脚本里——**第6周看 nanoGPT 时会再遇到**。

---

## 5. 超参调优优先级（Course 2 W3 核心）

Andrew Ng 给的优先级（**强烈推荐你按这个顺序调，不要乱来**）：

```
1. learning rate（α）            ← 影响最大，先调这个
2. momentum β / Adam β1 β2       ← 通常不调，用默认 0.9/0.999
3. mini-batch size               ← 受显存约束，能多大就多大
4. hidden units                  ← 容量调整
5. layers 数                     ← 改架构的大事
6. lr decay schedule             ← 锦上添花
```

### 5.1 为什么 lr 排第一

回顾你 Day 2 的 lr-range test 经验：**同一个模型，lr=1e-4 慢得跑不动，lr=1e-3 收敛漂亮，lr=1e-2 直接发散**——三个数量级，loss 差几十倍。其他超参不可能有这么大的灵敏度。

### 5.2 为什么 β1/β2 几乎不调

Adam 论文的默认值（β1=0.9, β2=0.999）是在几百个任务上验证过的，工业界几乎所有 Transformer 训练都用这套。**唯一例外**：β2=0.95 在某些大模型训练里（LLaMA、Mistral）用过，理由是"加快 v 的响应速度"。

### 5.3 调参搜索策略

**坏方法**：grid search（网格搜索）—— 如果有 5 个超参，每个 5 个值，要跑 $5^5 = 3125$ 次实验。

**好方法**（Andrew Ng W3 推荐）：random search（随机搜索）—— 在每个超参的合理区间内随机采样。**为什么？**

```
假设只有 lr 真正重要，其他 4 个超参影响很小：
- Grid search：每个 lr 值只采样到 1 次（因为其他 4 个超参也在变）
- Random search：每个 lr 值附近能采样到 N/5 次
```

直观说：**grid search 把搜索预算浪费在不重要的维度上，random search 不会**。

**工业现状**：
- 小模型：random search 够用
- 大模型：Bayesian optimization（如 Optuna）+ 早停（pruner）
- 超大模型（>10B）：基本不调超参，靠"小模型 sweep + 经验外推"

---

## 6. Course 2 W3 的 BN 和 Softmax：你已经会了

**BN**：参见 Day 5 `batchnorm_inference.md`——训推双行为、fused BN、running stats、Frozen BN。**Course 2 W3 的 BN 内容你已经覆盖 + 超额完成**。

**Softmax**：参见第3周 bigram 笔记——softmax + CE 的合并梯度 dZ = A - Y、数值稳定的 log-sum-exp 技巧。Course 2 W3 的 Softmax 内容你已经会了。

**今天唯一新增的认知**：Andrew Ng W3 给的"调参优先级"和"random search 优于 grid search"这两条——上面 §5 已经覆盖。

---

## 7. 本周笔记串联表（为周日元笔记 `week4_industrial_view.md` 铺垫）

| 本笔记 章节 | 串联到 | 串联方式 |
|---|---|---|
| §1.2 L2 正则 | `autograd_explained.md` §5.1 | L2 不是双行为，但 BN/Dropout 是，凑齐"`model.eval()` 三兄弟" |
| §1.3 Dropout 训推双行为 | Day 5 BN 双行为 | 同一个 `model.eval()` 关掉的两个最重要算子 |
| §1.4 梯度消失/爆炸四补丁 | Day 4 Kaiming + Day 5 BN | 把"为什么 Kaiming + BN"放进更大的框架 |
| §2 优化器演化线 | 第3周 micrograd | micrograd 你只写了纯 SGD，今天补完整个家族 |
| §3 显存账 | `autograd_explained.md` §5.2 | 把"训推显存差 4 倍"升级成具体公式 + 7B 数字 |
| §4.3 warmup | 第6周 nanoGPT | 下下周看 Karpathy nanoGPT 训练脚本时会再遇到 |
| §5 调参优先级 | Day 2 lr-range test | 你 Day 2 的实验是"lr 排第一"的实证 |

---

## 8. 自测题（不要看答案直接做）

1. 你的 hidden 层有 1000 个神经元，p=0.3 的 Dropout 训练时，你期望输出的"等效有效神经元数"是多少？乘以 1/(1-p) 之后，平均输出强度是多少？
2. 一个 13B 参数的模型，用 FP32 + Adam 训练，**不考虑激活**时总显存是多少？换成 BF16 + 8-bit Adam 呢？
3. 解释为什么 Transformer 训练前几百步必须 warmup？如果你跳过 warmup 直接用 max_lr 训练，最先炸的是 Adam 的 m 还是 v？为什么？
4. 你训练时 acc 一路涨，但推理时 acc 突然掉到 50%——三个可能原因，按工程上"最可能"排序。
5. 同一个 ResNet，把 SGD+Momentum 换成 Adam，训练显存大约多多少？给具体百分比。
6. 为什么 LLaMA-2 训练用 weight_decay=0.1 但 Dropout=0？两个机制理论上都是正则化。

参考答案位置（不要先翻）：
1. 答案在 §1.3 inverse scaling 的比喻里
2. 答案：FP32+Adam = 16P = 208GB；BF16+8bit Adam = 2P（参数）+ 2P（梯度）+ P（8bit Adam）= 5P = 65GB —— 数量级差异
3. 答案在 §4.3，关键词"v 估计不准"
4. 答案：(1) 忘了 `model.eval()`，BN 和 Dropout 还在训练态；(2) 数据预处理不一致；(3) 真的过拟合了
5. 答案：Adam 多 2× 参数大小的状态 = 在 SGD+Momentum（1×）基础上多 1× —— 总训练显存大约多 11%-15%（看激活占比）
6. 答案：weight_decay 几乎免费（不影响吞吐），且解耦后效果好；Dropout 要算 mask、训练吞吐损失明显；大模型数据量大，本来过拟合就轻

---

## 9. 完成标准 checklist

- [ ] 能口述 L2 / Dropout / BN 各自的"训推双行为"差异，能写出哪些是 `model.eval()` 切换的
- [ ] 能从 SGD → Momentum → RMSProp → Adam 推导每一步加了什么，不看资料能写出 Adam 完整公式（带偏差修正）
- [ ] **能 5 分钟内给一个完全不懂 DL 的人讲清"7B 模型为什么训练要 112GB 显存，推理只要 28GB"**
- [ ] 跑通 §3.4 的代码，亲眼看到 Adam 状态显存是参数的 2×
- [ ] 完成 `tech_notes/optimizer_memory.md`，把 §3 的内容整理进去（这是计划 line 1207 的硬交付物）
- [ ] 能解释 Transformer 训练为什么必须 warmup，且能写出 nanoGPT 风格的 warmup+cosine lr schedule 代码
- [ ] 能讲清 Andrew Ng 调参优先级 1-6 的逻辑，特别是为什么 lr 排第一、random search 优于 grid search

---

## 10. 与下周（第5周 CNN + ResNet）的衔接

- 本笔记 §1.4 的"四个补丁"里，**残差连接是下周的核心**——ResNet 怎么让 100 层网络也能训
- 本笔记 §1.3 Dropout 现状里提到"CNN 时代标配，Transformer 时代弃用"——下周你会在 ResNet 实现里再看到 Dropout（虽然实际作用越来越小）
- 本笔记 §3 显存账的"激活值"那一栏今天故意没展开——下周用 ResNet50 实际跑训练时，会算清楚激活值在显存里占多少（**这是 `torch.profiler` + 第8周 KV Cache 的伏笔**）
- 本笔记 §2.5 的 AdamW 下周也会用——ResNet 训练标配 SGD+Momentum，但 Transformer 视觉模型（ViT、Swin）都用 AdamW

---

*笔记完成时间：2026-05-23（W4 Day 6）*
*下一篇：W4 Day 7 — 周日元笔记 `week4_industrial_view.md` + `torch.profiler` chrome trace*
