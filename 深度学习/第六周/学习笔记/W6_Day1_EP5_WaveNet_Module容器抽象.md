# Week 6 · Day 1（2026-06-01）：EP5 WaveNet + Module 容器抽象

> **覆盖任务**（计划 line 1607-1620 区段 / W6 Day1 checklist）：
> - [ ] DL：看 EP5，实现 `FlattenConsecutive` + `Sequential` 容器，WaveNet makemore **dev loss ≈ 1.99**
> - [ ] DL：完成 `tech_notes/module_abstraction.md`
> - [ ] 完成标准：能口述"`Sequential` 容器怎么把多个 Module 串成一个 forward"，并解释 WaveNet 层次化感受野的直觉
>
> **阅读对象**：你自己——W3 你已经手写过 micrograd（理解了 autograd），W4 你写过 makemore MLP（embedding + 一层隐藏层），W5 你刚啃完 CNN/ResNet 还第一次用了 profiler。今天 EP5 是 Karpathy makemore 系列的第 5 集，**它表面在讲 WaveNet，真正在教的是"怎么把一坨平铺的代码,重构成像 PyTorch 那样的模块化框架"**。这是你明天（Day2）读懂 nanoGPT 源码、以及未来读任何工业框架代码的钥匙。
>
> **本笔记的设计**：沿用 [[W5_Day2_LeNet5复现_经典CNN对比]] 的三段式——每节先讲"原理 + 直觉"，再给可直接拷进 `src/` 的可运行代码，最后写工业锚点。EP5 你说看不太懂，所以这篇我**讲得比平时更慢、更碎**，把每个卡点拆开。读完应该能从 0 把今天的任务全部跑通。

---

## 0. 学习目标（看完应能脱口而出）

1. EP5 这一集，表面在讲 WaveNet，**真正的主线任务**到底是什么？（提示：和"代码长得像不像 PyTorch"有关）
2. 什么叫"扁平的代码"（flat code）？Karpathy 为什么说 W4 那种写法不可持续？
3. `Sequential` 容器是什么？为什么有了它，加一层网络就只是"在列表里多塞一个对象"？
4. `FlattenConsecutive` 在做一件什么事？为什么 WaveNet 不一次性把 8 个字符全拍平，而要"两两合并、慢慢长大"？
5. 什么叫 **感受野**（receptive field）？为什么说 WaveNet 是"层次化地扩大感受野"，而 W4 的 MLP 是"一步到位"？
6. 为什么 dev loss 能从 W4 的 ~2.1 降到 ~1.99？是模型变"深"了还是变"聪明"了？
7. 一个隐藏大坑：把 BatchNorm 放进 `Sequential` 后，为什么它对"批次维度"的处理会出 bug？（呼应你 W4 的 `batchnorm_inference.md`）

---

## 1. 先把 EP5 的"剧情"讲清楚：它到底在干嘛

### 1.1 一句话剧情

EP5 的真实主线**不是 WaveNet**，是这件事：

> **把 W4 那种"所有代码平铺在一起、手动一行行调用"的写法，重构成 PyTorch 风格的"乐高积木 + 流水线"——然后顺便用这套新积木，搭一个比 MLP 更深的 WaveNet 模型，把 loss 再压低一点。**

WaveNet 只是"用来验证新积木好不好用"的例子。Karpathy 真正想喂给你的，是**工程能力**：怎么组织一个能扩展、能复用、不容易出错的神经网络代码库。

为什么这件事对你（AI Infra 方向）特别重要？因为 **AI Infra 工程师每天打交道的不是"写一个新模型"，而是"读懂、改造、优化别人写好的大型模型代码库"**（PyTorch 源码、nanoGPT、vLLM、llm.c）。这些代码库无一例外都是"模块化"的。今天这一集，就是你从"会写玩具脚本"跨到"看得懂工业代码"的第一步。

### 1.2 为什么 W4 的写法"不可持续"——先看痛点

回忆你 W4 写 makemore MLP 时，代码大概长这样（这叫 **flat code，扁平代码**：所有变量、所有计算步骤,一股脑平铺在主流程里，没有分层、没有封装）：

```python
# W4 风格：扁平代码（痛点演示，不要照抄）
C  = torch.randn((27, 10))           # embedding 表
W1 = torch.randn((30, 200)) * 0.01   # 第一层权重（30 = 3个字符 × 10维）
b1 = torch.randn(200)
W2 = torch.randn((200, 27)) * 0.01   # 第二层权重
b2 = torch.randn(27)
parameters = [C, W1, b1, W2, b2]

# 前向：手动一行行写
emb = C[Xb]                          # 查表
emb_flat = emb.view(emb.shape[0], -1)  # 手动拍平
h = torch.tanh(emb_flat @ W1 + b1)   # 第一层 + 激活
logits = h @ W2 + b2                 # 第二层
```

这段代码本身没错，能跑。但它有三个会越来越疼的痛点：

1. **想加一层?痛苦。** 你得手动新建 `W3, b3`，手动塞进 `parameters` 列表，手动在前向里多写一行 `@ W3 + b3`。改三个地方，漏一个就报错。
2. **维度全靠脑算。** `30 = 3 × 10` 这种数字散落在代码里，网络一深，shape 算错是头号 bug（你 W5 的 `400 = 16×5×5` 就栽过同款）。
3. **没法复用。** 想再搭一个结构不同的网络？整段重写。

**生活类比**：flat code 就像你把"煮一顿饭"的所有动作——洗菜、切菜、开火、放油、下锅、翻炒、装盘——全部写成一条没有分段的流水账。能照着做出来，但想在中间插一道菜、或者换个做法，你得在这坨流水账里小心翼翼找位置改。而模块化（接下来要讲的）就是把它拆成"备菜模块、烹饪模块、装盘模块"，每个模块自己管自己,想换菜就换一个模块。

---

## 2. 核心积木一：`Module` 是什么——一块"自带说明书的乐高"

### 2.1 是什么

> **Module（模块）**：神经网络里一个"自包含的计算单元"。它自己管两件事：① **自己的参数**（比如一层的权重和偏置）；② **自己的前向计算**（给我输入 `x`，我返回输出）。PyTorch 里所有网络组件（`Linear`、`Conv2d`、`BatchNorm`）本质都是 Module。

**类比**：Module 就是一块**自带说明书的乐高积木**。每块积木知道两件事——"我内部有哪些零件"（参数）和"我能怎么和别人拼接"（输入输出接口）。你不需要关心积木内部怎么造的，只要知道它的接口，就能往上拼。

### 2.2 怎么做——手写几个最小 Module

EP5 里 Karpathy 不直接用 PyTorch 的 `nn.Linear`，而是带你**亲手写一遍**这些积木——因为亲手写一遍，你才知道 `nn.Linear` 内部到底藏了什么。我们照做：

```python
# 运行环境：Python 3.10+ / PyTorch 2.x（CPU 即可，char-level 很省）
# 这几个类是 EP5 的核心积木，建议拷进 week6_nanogpt/src/modules.py
import torch

class Linear:
    """全连接层：out = x @ W (+ b)。等价于 PyTorch 的 nn.Linear。"""
    def __init__(self, fan_in, fan_out, bias=True):
        # 为什么除以 fan_in**0.5：这是 Kaiming 初始化的思想（呼应 W4 init_and_stability.md）
        # 目的是让输出方差≈输入方差，前向不爆不消，反向梯度才稳定
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        # __call__ 让这个对象能像函数一样被调用：layer(x) 实际触发这里
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        # 把自己所有可学习参数收集成一个列表——这是"自己管自己参数"的体现
        return [self.weight] + ([] if self.bias is None else [self.bias])


class Tanh:
    """激活函数层。注意它没有参数，所以 parameters() 返回空列表。"""
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []   # 激活函数没有可学习参数
```

**这里有两个新手必须吃透的点：**

- **`__call__` 是什么**：Python 里如果一个类定义了 `__call__` 方法，那么这个类的对象就能"像函数一样被调用"。即写 `layer = Linear(30, 200)` 后，`layer(x)` 这一下，Python 自动去执行 `__call__` 里的代码。**为什么这么设计**：这样每个 Module 用起来都像一个函数 `输出 = 层(输入)`，拼接起来特别自然——这正是 PyTorch 的 `forward` 背后的机制。
- **`parameters()` 为什么重要**：训练时你需要把"所有可学习的张量"收集起来交给优化器更新。每个 Module 自己提供 `parameters()`，上层容器只要把每个子模块的 `parameters()` 拼起来,就拿到了全网参数。这就是"自己管自己参数"——不用你再手动维护那个全局 `parameters` 列表了。

> **关键认知**：你现在手写的 `Linear`/`Tanh`，和 PyTorch 的 `nn.Linear`/`nn.Tanh` 是**同一个东西的两个版本**。明天读 nanoGPT 源码时，你看到的 `nn.Linear(n_embd, 4*n_embd)`，脑子里要能立刻浮现出今天这段 `__init__` + `__call__`——这就是 Karpathy 让你手写一遍的全部意义。

---

## 3. 核心积木二：`Sequential` 容器——把积木串成流水线

### 3.1 是什么

> **Sequential（顺序容器）**：一个"装 Module 的盒子"。你把若干个 Module 按顺序放进去，它的 forward 就是"让数据依次穿过每一个 Module"——前一个的输出，是后一个的输入。

**类比**：`Sequential` 就是一条**工厂流水线**。你把一台台机器（Module）按顺序排好，原料（输入 `x`）从第一台进去，每台机器加工一下传给下一台，最后从尾部出来成品（输出）。你想加一道工序？在流水线上插一台机器就行,不用重排整条线。

### 3.2 怎么做

```python
class Sequential:
    """顺序容器：数据依次流过 layers 里的每一层。等价于 nn.Sequential。"""
    def __init__(self, layers):
        self.layers = layers   # layers 是一个 Module 列表

    def __call__(self, x):
        # 这就是整个抽象的精髓：一个 for 循环，让 x 穿过每一层
        for layer in self.layers:
            x = layer(x)       # 上一层的输出 x，喂给下一层
        self.out = x
        return self.out

    def parameters(self):
        # 把每个子层的参数全部收集起来——列表推导式拍平
        return [p for layer in self.layers for p in layer.parameters()]
```

看 `__call__` 里那个 `for` 循环——**整个模块化抽象的核心，就是这 3 行**。无论你的网络有 3 层还是 300 层，前向传播永远是"让 x 依次穿过 layers"。

现在回头看 §1.2 的痛点，全解决了：

```python
# 模块化风格：搭一个 W4 同款 MLP，但现在"加层"只是往列表里塞对象
model = Sequential([
    Linear(30, 200), Tanh(),
    Linear(200, 27),
])
# 想加一层？在列表里多塞 Linear(200, 200), Tanh() 即可——改一个地方，不碰前向逻辑
logits = model(emb_flat)        # 前向：一行搞定
params = model.parameters()     # 收集全部参数：一行搞定
for p in params:
    p.requires_grad = True
```

**对比 §1.2 的痛点表：**

| 痛点 | flat code | Sequential |
|---|---|---|
| 加一层 | 改 3 处（建权重 / 塞列表 / 写前向） | 列表里塞 1 个对象 |
| 收集参数 | 手动维护全局列表 | `model.parameters()` 自动拍平 |
| 复用 | 整段重写 | 换个 layers 列表即可 |

---

## 4. 核心积木三：`FlattenConsecutive`——WaveNet 的灵魂

> 这一节是 EP5 最容易看懵的地方，慢慢来。先讲清"WaveNet 想干嘛"，再讲这个层怎么实现。

### 4.1 先讲 WaveNet 的核心思想：不要一口吃成胖子

> **WaveNet**：DeepMind 2016 年提出的、原本用来生成原始音频波形的网络。它的招牌思想是 **层次化地、逐步地融合信息**——而不是一次性把所有输入怼进一个大全连接层。EP5 借用的就是这个"逐步融合"的思想。

回忆 W4 的 MLP：你把 **3 个字符** 的 embedding 直接 `view` 拍平成一个长向量（3×10=30 维），一把喂给第一层 `Linear(30, 200)`。这叫**一步到位地融合**——3 个字符的信息在第一层就被全部揉在一起了。

WaveNet 不这么干。假设现在上下文变长到 **8 个字符**，WaveNet 的做法是：

```
8 个字符： [c1 c2 c3 c4 c5 c6 c7 c8]
第1层：两两合并 → [(c1c2) (c3c4) (c5c6) (c7c8)]   还剩 4 组
第2层：再两两合并 → [(c1c2c3c4) (c5c6c7c8)]        还剩 2 组
第3层：最后合并 → [(c1c2c3c4c5c6c7c8)]             1 组，看全了
```

**生活类比**：一步到位的 MLP 像"把 8 个人塞进一个小会议室同时七嘴八舌讨论"——信息是融合了，但很混乱，模型很难学到"哪两个字符的搭配特别重要"。WaveNet 像"先两两组队私聊，再小组合并，再大组汇总"的**金字塔式开会**——每一层只融合相邻的一点点信息,结构清晰，模型更容易学到局部规律（比如英文里 `th`、`ing` 这种常见字母组合）。

### 4.2 怎么做：FlattenConsecutive 就是"两两打包"

> **FlattenConsecutive（连续展平 / 分组展平）**：把序列里**连续的 n 个元素**打包合并成一个，从而让序列长度变短、每个元素的维度变厚。它是实现上面"两两合并"的那把工具。

代码（这是 EP5 最关键的一个类，慢慢看注释）：

```python
class FlattenConsecutive:
    """把连续 n 个时间步的特征拼到一起。n=2 就是'两两合并'。"""
    def __init__(self, n):
        self.n = n   # 每次合并几个连续元素

    def __call__(self, x):
        # x 的形状：(B, T, C) = (批大小, 序列长度, 每个位置的特征维度)
        B, T, C = x.shape
        # 核心一行：把 T 维度按 n 切分，多出来的 n*C 拼进特征维
        # 例：(4, 8, 10) 且 n=2 → (4, 4, 20)，序列减半，特征翻倍
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            # 当序列长度被压到 1 时，把多余的中间维度挤掉，回到 (B, C)
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []   # 纯形状变换，没有可学习参数
```

**吃透那一行 `view`**：`view(B, T // n, C * n)` 在做的事，就是"把序列长度 T 砍成 T/n，砍掉的那部分信息没丢，而是塞进了特征维度 C"。信息守恒——`T × C = (T/n) × (C×n)`，总量不变，只是重新排布。这就是"两个相邻字符的 embedding 被拼成了一个更厚的向量"。

> **呼应 W4**：你 W4 用的 `emb.view(N, -1)` 是 `FlattenConsecutive` 的**极端特例**——一次性把整个序列全拍平（相当于 n = 整个序列长度）。EP5 只是把"一次拍平"拆成了"分多层、每层拍一点"。本质同源。

### 4.3 把三块积木拼成 WaveNet

```python
n_embd = 10        # 每个字符的 embedding 维度
n_hidden = 68      # 隐藏层宽度
block_size = 8     # 上下文长度：用前 8 个字符预测第 9 个

model = Sequential([
    Embedding(vocab_size, n_embd),                          # 查表：8个字符 → (B,8,10)
    FlattenConsecutive(2), Linear(n_embd*2, n_hidden), BatchNorm1d(n_hidden), Tanh(),  # 8→4
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden), BatchNorm1d(n_hidden), Tanh(),# 4→2
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden), BatchNorm1d(n_hidden), Tanh(),# 2→1
    Linear(n_hidden, vocab_size),                           # 输出 27 类 logits
])
# 三次 FlattenConsecutive(2)，序列 8→4→2→1，正好对应 §4.1 的金字塔
```

看出结构了吗？**每一层都是"先两两合并(FlattenConsecutive)，再过一个全连接+归一化+激活"**。三次合并，把 8 个字符的序列一步步收拢成 1 个总结向量，最后分类。这就是"层次化扩大感受野"的代码长相。

---

## 5. 感受野：理解 WaveNet "为什么更聪明"

> **感受野（receptive field）**：网络中某个输出，能"看到"多少个原始输入。这个词来自视觉神经科学——指一个神经元能感知到的视野范围。在序列模型里，它指"当前这个特征，融合了前面多少个字符的信息"。

**类比**：感受野就像你站在不同楼层往下看。一楼（浅层）只能看清门口几个人（相邻 2 个字符）；爬到顶楼（深层），整个广场（全部 8 个字符）尽收眼底。WaveNet 的每一层 `FlattenConsecutive(2)` 都让你往上爬一层，视野翻倍。

- **W4 的 MLP**：感受野"一步到位"——第一层就把全部 3 个字符拍平看完。简单粗暴。
- **WaveNet**：感受野"层次化增长"——第1层只看 2 个、第2层看 4 个、第3层看 8 个，**指数级扩大**。

### 5.1 为什么层次化更好？dev loss 为什么能降到 ≈1.99

EP5 最后把 dev loss 从 W4 MLP 的 ~2.1 压到了 ~1.99。**注意：主要功劳不是"WaveNet 结构本身"，而是两件事叠加**（Karpathy 在视频里特意澄清过这点，是个容易误解的地方）：

1. **上下文变长了**：从 block_size=3 加到了 8。能看到更多历史字符，预测自然更准——这是最大的功劳。
2. **层次化融合**：让模型更容易学到"局部字母组合"的规律，相比一次性拍平，结构上更有归纳偏置（inductive bias，指模型结构自带的先验假设）。

> **诚实的认知（重要）**：Karpathy 在 EP5 里坦白——把 MLP 简单加长上下文，效果其实和 WaveNet 差不太多。**WaveNet 结构本身带来的增益是"锦上添花"，不是"质变"**。这一集真正的价值在于"代码模块化"和"理解层次化感受野这个思想"，而不是"WaveNet 多神"。别被"刷低 0.1 loss"带偏了注意力——这是初学者最容易误解的点。

---

## 6. 隐藏大坑：BatchNorm 放进 Sequential 后的维度 bug

> 这一节呼应你 W4 的 `batchnorm_inference.md`。EP5 里 Karpathy 真的踩了这个坑并当场 debug，是全集含金量很高的一段。

### 6.1 bug 是什么

你 W4 写的 `BatchNorm1d`，是按"输入形状是 `(B, C)`（二维）"来算统计量的——它对第 0 维（batch 维）求均值方差。

但在 WaveNet 里，数据穿过 `FlattenConsecutive` 后，中间形状是 **`(B, T, C)`（三维）**，比如 `(32, 4, 68)`。这时候 BatchNorm 如果还只对第 0 维求统计，就**只统计了 batch，漏掉了 T 维度**——相当于把 4 个时间步当成 4 套独立的统计，维护出来的 `running_mean` 形状会变成 `(1, 4, 68)` 而不是期望的 `(1, 1, 68)`。

**生活类比**：这就像你统计"全班数学平均分"，本该把所有人的所有次考试一起算，结果你按"第几次考试"分了组，算出 4 个独立的平均分——口径错了。

### 6.2 怎么修

```python
class BatchNorm1d:
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0          # (B, C)：只对 batch 维求统计
            elif x.ndim == 3:
                dim = (0, 1)     # (B, T, C)：对 batch 和时间维一起求统计 ← 关键修复
            mean = x.mean(dim, keepdim=True)
            var  = x.var(dim, keepdim=True)
        # ... 其余和 W4 一致（running stats 更新、归一化）
```

**这个坑的工业意义**：它精确呼应了你 `batchnorm_inference.md` 的核心结论——**BatchNorm 对"在哪个维度上求统计"极其敏感**。维度搞错，模型照样能跑、loss 照样下降，但统计口径是错的，结果会悄悄变差。这种"不报错但结果错"的 bug，是工业训练里最难查的一类。也正是这种麻烦，催生了下周你要学的 **LayerNorm**——它在特征维归一化，天然不依赖 batch 和序列维度，从根上绕开了这个坑。

---

## 7. 完整可运行训练脚本（拷进 `src/wavenet.py` 直接跑）

```python
# 运行环境：Python 3.10+ / PyTorch 2.x / CPU 即可
# 数据：names.txt（Karpathy makemore 仓库，约 32k 英文名字）
# 跑法：python src/wavenet.py，约几分钟，dev loss 应收敛到 ≈1.99
import torch, torch.nn.functional as F
torch.manual_seed(42)   # 固定种子，保证可复现（你的产出规范要求）

# ---------- 1. 读数据 + 建字符表 ----------
words = open('names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}; stoi['.'] = 0   # '.' 作起止符
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)   # 27

block_size = 8           # 上下文长度
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context); Y.append(ix)
            context = context[1:] + [ix]   # 滑动窗口
    return torch.tensor(X), torch.tensor(Y)

import random; random.seed(42); random.shuffle(words)
n1, n2 = int(0.8*len(words)), int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])      # 训练
Xdev, Ydev = build_dataset(words[n1:n2])  # 验证（调超参看它）

# ---------- 2. 积木（精简版，省略 parameters() 细节见上文） ----------
class Linear:
    def __init__(self, fi, fo, bias=True):
        self.weight = torch.randn((fi, fo)) / fi**0.5
        self.bias = torch.zeros(fo) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None: self.out += self.bias
        return self.out
    def parameters(self): return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps, self.momentum, self.training = eps, momentum, True
        self.gamma, self.beta = torch.ones(dim), torch.zeros(dim)
        self.running_mean, self.running_var = torch.zeros(dim), torch.ones(dim)
    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)   # ← §6 的关键修复
            mean, var = x.mean(dim, keepdim=True), x.var(dim, keepdim=True)
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
                self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        self.out = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return self.out
    def parameters(self): return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x): self.out = torch.tanh(x); return self.out
    def parameters(self): return []

class Embedding:
    def __init__(self, num, dim): self.weight = torch.randn((num, dim))
    def __call__(self, ix): self.out = self.weight[ix]; return self.out
    def parameters(self): return [self.weight]

class FlattenConsecutive:
    def __init__(self, n): self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1: x = x.squeeze(1)
        self.out = x; return self.out
    def parameters(self): return []

class Sequential:
    def __init__(self, layers): self.layers = layers
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        self.out = x; return self.out
    def parameters(self): return [p for l in self.layers for p in l.parameters()]

# ---------- 3. 搭模型 ----------
n_embd, n_hidden = 10, 68
model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd*2,  n_hidden, False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])
with torch.no_grad(): model.layers[-1].weight *= 0.1   # 缩小最后一层，初始 loss 接近 log(27)
parameters = model.parameters()
for p in parameters: p.requires_grad = True
print("参数量：", sum(p.nelement() for p in parameters))

# ---------- 4. 训练 ----------
for i in range(20000):
    ix = torch.randint(0, Xtr.shape[0], (32,))      # mini-batch
    logits = model(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])         # 这就是你数学在学的 MLE / NLL
    for p in parameters: p.grad = None
    loss.backward()
    lr = 0.1 if i < 15000 else 0.01                 # 后期降学习率
    for p in parameters: p.data += -lr * p.grad
    if i % 5000 == 0: print(f"{i:5d} | loss {loss.item():.4f}")

# ---------- 5. 评估（切 eval 模式，BN 用 running stats！） ----------
for layer in model.layers:
    if isinstance(layer, BatchNorm1d): layer.training = False   # ← 呼应 batchnorm_inference.md
with torch.no_grad():
    dev_loss = F.cross_entropy(model(Xdev), Ydev)
    print(f"dev loss: {dev_loss.item():.4f}")        # 目标 ≈1.99
```

**两个最容易踩的运行坑（注释里已埋）：**

1. **评估前必须把所有 BatchNorm 切到 `training=False`**——否则它用当前 batch 的统计而非 running stats，dev loss 会虚高且不稳定。这就是你 `batchnorm_inference.md` 写的"训练/推理双行为"，**今天第一次在多层网络里亲手触发**。
2. **`Linear` 在 BN 前要 `bias=False`**——因为 BN 自带 `beta` 偏移，前面再加 bias 是冗余的（会被 BN 减均值时抵消）。这正是你 W5 `lenet_vs_modern.md` 提过的"有 BN 时 bias 可省"，同一条规则。

---

## 8. 工业锚点：`tech_notes/module_abstraction.md` 的核心（今天第二个任务）

> 这一节就是你今天要产出的 `tech_notes/module_abstraction.md` 的主体内容。它回答一个问题：**今天手写的这套 Module/Sequential，在真实工业代码里长什么样、为什么所有框架都这么设计？**

### 8.1 你手写的版本 vs PyTorch 真身

| 你今天写的 | PyTorch 对应 | 区别 |
|---|---|---|
| `class Linear` + `__call__` | `nn.Linear` + `forward` | PyTorch 用 `forward`，靠 `nn.Module.__call__` 自动调用它（还顺手处理 hook） |
| `class Sequential` | `nn.Sequential` | 几乎一模一样，就是个 for 循环 |
| 手动收集 `parameters()` | `nn.Module` 自动注册 | PyTorch 在 `__setattr__` 里自动发现子模块和参数，不用你手拼列表 |
| `FlattenConsecutive` | `nn.Flatten` / `view` / `einops.rearrange` | 工业里形状变换常用 `einops`，可读性更高 |

**核心认知**：PyTorch 的 `nn.Module` 不是魔法，它就是你今天手写的这套东西 + 一堆自动化糖（自动注册参数、自动管理 train/eval 状态、自动支持 `.to(device)` 搬到 GPU、自动支持 hook）。**你今天手写一遍，等于把 PyTorch 最核心的抽象拆开看了一次。**

### 8.2 为什么"模块化"是 AI Infra 的地基

这套抽象对你目标方向（推理优化）有三个直接的工业意义：

1. **可优化的边界**：因为每个算子被封装成独立 Module，推理引擎（TensorRT / ONNX）才能"识别出 Conv-BN-ReLU 这三块积木，把它们融合成一个 kernel"——你 W5 学的**算子融合**，前提就是模块化的清晰边界。flat code 是融合不了的，因为引擎看不出哪里到哪里是一个可融合单元。
2. **可替换的组件**：明天你看 nanoGPT，会发现把 `nn.LayerNorm` 换成 `RMSNorm`、把标准 Attention 换成 FlashAttention，**只需要替换一个 Module，其余代码不动**。这种"热插拔"能力，是模块化给的。vLLM 之所以能给各种模型做优化，靠的就是大家都遵守同一套 Module 接口。
3. **可遍历的结构**：profiler（你 W5 用过的）能逐层报时间，靠的是它能遍历 `model.modules()`。量化工具能逐层换成 INT8，也靠这个。**"网络是一棵 Module 树"这个数据结构，是几乎所有 AI Infra 工具的工作前提。**

> **一句话锚点**：今天你以为在学"怎么写整齐的代码"，其实你在学的是**AI Infra 全套工具链（融合、量化、profiling、并行）赖以工作的那个底层数据结构**。没有模块化，就没有可优化的对象。

### 8.3 一个真实案例

PyTorch 的 `torch.fx` 和 `torch.compile` 做的第一件事，就是把你的 `nn.Module` 树**追踪（trace）成一张计算图**，然后在图上做融合、常量折叠等优化。你今天写的 `Sequential.__call__` 那个 for 循环，在 `torch.compile` 眼里就是一串可以被分析、重排、融合的节点。**模块化是"可编译"的前提**——这是你 W8 会正面接触的内容，今天先埋下根。

---

## 9. 自测题（合上笔记，能答出来才算过）

1. 用一句话说清：EP5 表面讲 WaveNet，真正的主线任务是什么？
2. `__call__` 方法让一个对象能干什么？为什么 Module 要用它？
3. `Sequential.__call__` 的核心是哪几行？为什么说"加 300 层和加 3 层，前向代码一样"？
4. `FlattenConsecutive(2)` 把 `(32, 8, 10)` 变成什么形状？信息丢了吗？为什么？
5. 什么是感受野？WaveNet 和 W4 MLP 在"扩大感受野"上的区别是什么？
6. EP5 dev loss 从 2.1 降到 1.99，**主要**功劳是 WaveNet 结构吗？（陷阱题）
7. 为什么 BatchNorm 进了 `(B,T,C)` 三维数据会出 bug？怎么修？
8. 评估前为什么必须把 BatchNorm 切到 `training=False`？（呼应哪篇旧笔记？）
9. 为什么 BN 前面的 Linear 可以设 `bias=False`？
10. 模块化抽象为什么是"算子融合""量化""profiling"这些 AI Infra 工具的前提？

> 参考答案分散在：Q1→§1.1，Q2→§2.2，Q3→§3.2，Q4→§4.2，Q5→§5，Q6→§5.1（注意是陷阱），Q7→§6，Q8→§7 坑1 + `batchnorm_inference.md`，Q9→§7 坑2，Q10→§8.2。

---

## 10. 与已有笔记的串联

| 今天的内容 | 关联到你已有的 | 关系 |
|---|---|---|
| Module / Sequential 抽象 | 明天 nanoGPT 的 `Block`/`GPT` 类 | 同一套抽象，明天直接复用 |
| `FlattenConsecutive` 的 view | W4 `emb.view(N,-1)` | 后者是前者的极端特例（一次拍平） |
| Linear 的 `/fan_in**0.5` 初始化 | W4 `init_and_stability.md` | 同一个 Kaiming 初始化思想 |
| BatchNorm 三维 bug + eval 切换 | W4 `batchnorm_inference.md` | 训练/推理双行为第一次在多层网络触发 |
| BN 前 Linear 省 bias | W5 `lenet_vs_modern.md` | 同一条"有 BN 则 bias 冗余"规则 |
| 模块化 = 可优化边界 | W5 算子融合、profiler | 融合/profiling 的前提是模块化 |
| `cross_entropy` = MLE/NLL | 本周数学 MLE + W3 bigram 起 | 你一直在做 MLE，本周补理论闭环 |
| 模块化 = 可编译 | W8 `torch.compile` | 今天埋根，W8 正面接触 |

---

## 11. 完成标准 checklist（对齐计划）

- [ ] 看完 EP5（≈56min，1.5× 速可），能复述"主线是代码模块化，WaveNet 是验证例子"
- [ ] `src/wavenet.py` 跑通，WaveNet makemore **dev loss ≈ 1.99**（固定 seed，可复现）
- [ ] 能口述"`Sequential` 容器怎么把多个 Module 串成一个 forward"（§3.2 那 3 行 for 循环）
- [ ] 能解释 WaveNet 层次化感受野的直觉（§5 爬楼层类比）
- [ ] 完成 `tech_notes/module_abstraction.md`（主体即 §8：手写版 vs PyTorch、模块化为何是 AI Infra 地基）
- [ ] 自测题 §9 能合上笔记答出，尤其 Q6 陷阱题和 Q7 BN 维度坑
- [ ] `W6_day1_log.md` 记录：今天最大卡点 + 是否踩了 BN eval 坑

> **今天的一句话总结**：你不是在学一个叫 WaveNet 的模型，你是在学"神经网络代码该怎么组织成乐高积木"——这套组织方式，是你明天读 nanoGPT、未来读 vLLM、做算子融合和量化的**共同地基**。
