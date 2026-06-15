# W4 学习笔记:组装完整 GPT —— FeedForward、Pre-Norm Block、初始 Loss 验证与残差流

> **本笔记覆盖的四个任务:**
> - [ ] 实现 FeedForward + Block(pre-norm)+ 组装完整 GPT
> - [ ] 验证初始 loss ≈ -log(1/vocab)
> - [ ] 完成 `tech_notes/residual_stream_and_layernorm.md`(残差流 + LN vs BN)
> - [ ] 通读 nanoGPT `model.py` 对照
>
> **建议学习顺序:** 先读第 1~3 章理解原理 → 跑通第 4 章完整代码 → 用第 5 章验证初始 loss → 照第 6 章写你自己的 tech_note → 最后带着第 7 章的"寻宝清单"去读 nanoGPT。

---

## 0. 你现在在整个拼图的哪个位置?

回顾一下你已经走过的路:你已经实现了 **Self-Attention(自注意力)**——它解决的问题是"让序列中的每个 token 能看到前面的 token,并从中收集信息"。

但只有 Attention 的模型是残缺的。打个比方:

> **Attention 像开会,FeedForward 像회后独立思考。**
> 开会(Attention)时,每个人(token)听取其他人的发言,把有用的信息记到自己笔记上。但光开会不出活——会后每个人还得**回到自己工位,独立消化、加工这些信息**(FeedForward)。一轮"开会 + 独立思考"就是一个 **Block(块)**。把这样的 Block 摞 12 层、24 层、96 层,再加上输入输出的"接口",就是完整的 GPT。

本周你要做的,就是补上"独立思考"这一半,然后把整栋楼盖起来。

---

## 1. FeedForward:每个 token 的"独立思考时间"

### 1.1 是什么

**FeedForward Network(前馈网络,简称 FFN,在 GPT 代码里常叫 MLP)**:就是一个最朴素的两层全连接神经网络,对序列里的**每个 token 独立地、相同地**做一次非线性变换。

> **MLP(Multi-Layer Perceptron,多层感知机)**:深度学习里最古老的结构——"线性变换 → 非线性激活 → 线性变换"。你可以把它理解成一个"查表 + 加工"的机器:输入一个向量,它根据自己学到的知识(权重)对这个向量做加工,输出一个新向量。在 Transformer 里,MLP 和 FFN 指的是同一个东西。

它的结构只有三步:

```
输入 (n_embd 维) → 线性层放大到 4×n_embd → GELU 激活 → 线性层缩回 n_embd 维
```

### 1.2 为什么需要它?Attention 不是已经很强了吗?

这是初学者最容易跳过、但面试和科研中最关键的问题。两个原因:

**原因一:Attention 本质上只是"加权平均",它不会"计算"。**

仔细想 Attention 在干嘛:算出一组权重,然后对 value 向量做加权求和。加权求和是**线性操作**——它只能"搬运和混合"已有的信息,不能产生新的非线性特征。就像开会只能交换大家已有的观点,不能凭空产生新知识。FFN 里的非线性激活函数才是模型"思考加工"的地方。

**原因二:FFN 是模型的"知识库"。**

可解释性研究(如 Anthropic 的 Transformer Circuits 系列)发现:模型记住的事实性知识(比如"巴黎是法国的首都")主要存储在 FFN 的权重里。Attention 负责"把相关信息搬运到当前位置",FFN 负责"根据搬来的信息查询知识库并输出结论"。这也是为什么 FFN 占了 GPT 总参数量的约 **2/3**——知识库就得大。

> **一个工业界视角:** 正因为 FFN 参数多但每个 token 的计算相互独立,它成了推理优化的主战场——比如 MoE(Mixture of Experts,混合专家)就是把一个大 FFN 拆成多个小 FFN 按需激活;推理引擎(vLLM、TensorRT-LLM)里 FFN 的两个大矩阵乘也是 GPU 利用率最高的算子。你做 AI Infra 方向,以后 profile 一个 LLM,会发现大部分 FLOPs 都花在这里。

### 1.3 两个设计细节的"为什么"

**① 为什么中间层是 4 倍宽(`4 * n_embd`)?**

诚实的回答:这是从原始 Transformer 论文(Attention Is All You Need, 2017)沿用下来的**经验值**,后续大量实验证明 2~8 倍之间差别不大,4 倍是性价比甜点。直觉理解:先把向量投影到一个更高维的空间,让特征"摊开"以便非线性函数分别处理,再压缩回来——就像解一道几何题时先把图形放大画清楚,解完再缩回原比例。

**② 为什么用 GELU 而不是 ReLU?**

> **ReLU(Rectified Linear Unit,修正线性单元)**:`max(0, x)`,负数全部砍成 0。简单粗暴,像一个"非黑即白"的门卫:负的一律不放行。
>
> **GELU(Gaussian Error Linear Unit,高斯误差线性单元)**:可以理解为"软化版 ReLU"——在 0 附近不是一刀切,而是平滑过渡(轻微负值会被保留一小部分)。像一个"讲人情"的门卫:明显不行的拦下,边缘情况酌情少放一点。

实际差异:ReLU 在 0 处不可导、且会造成"神经元死亡"(dead neuron,某神经元输入恒为负→输出恒为 0→梯度恒为 0→永远学不到东西)。GELU 处处平滑,梯度流动更顺畅。GPT-2、BERT 用 GELU;新一代模型(LLaMA、Qwen)用 SwiGLU——趋势是激活函数越来越"平滑且带门控"。**作为学习者记住:在 GPT-2 复现里用 GELU 就是工业标准。**

---

## 2. Block 与 Pre-Norm:残差流是整个 Transformer 的灵魂

### 2.1 残差连接:一条贯穿全模型的"传送带"

> **Residual Connection(残差连接)**:不让数据被层"完全替换",而是让层的输出**加到**原输入上:`x = x + layer(x)`。
>
> **类比:文档审阅模式。** 想象一份 Word 文档在公司里流转:不用残差连接,相当于每个部门把文档**重写一遍**再传给下一个部门——传到第 10 个部门,原文早就面目全非,而且哪个部门改错了都无法追溯。用残差连接,相当于开了**修订模式**:原文一直都在,每个部门只是"批注 + 增量修改"。文档主干始终畅通,每个部门只贡献自己的增量。

这条贯穿所有层的主干,有一个在可解释性研究中非常重要的名字:

> **Residual Stream(残差流)**:把 `x` 看成一条从输入 embedding 一直流到输出层的"信息高速公路"或"共享内存总线"。每个 Attention 层、每个 FFN 层都不是"串联的加工站",而是挂在总线上的"外设":各自从总线**读取**(read)信息,加工后把结果**写回**(write)总线。层与层之间不直接对话,全靠这条总线通信。

这个视角为什么重要?两个层面:

**① 优化层面:梯度的"超级高速公路"。**

反向传播时,`x = x + f(x)` 的梯度是 `dL/dx = dL/dout * (1 + f'(x))`。注意那个 **`+1`**:无论 `f'(x)` 多小(哪怕层学得很差、梯度接近 0),总有一条"恒等通路"让梯度原封不动地传回浅层。没有残差连接,梯度要连乘几十个层的雅可比矩阵,极易消失或爆炸——这正是 2015 年之前深网络训不动、ResNet 之后突然能训 100+ 层的根本原因。**Transformer 能堆到 96 层(GPT-3),残差连接是第一功臣。**

**② 表示层面:模型行为变得可分析。**

因为每层只做"加法",最终输出 = embedding + 所有层写入量的总和。可解释性研究(induction heads、logit lens 等)都建立在这个视角上:你可以问"第 5 层的第 3 个注意力头往残差流里写了什么",这在"每层完全重写"的架构里是没法问的。你以后做推理优化,看 Anthropic/OpenAI 的可解释性论文,"residual stream" 是出现频率最高的词之一。

### 2.2 Pre-Norm vs Post-Norm:LayerNorm 放哪儿,差别巨大

> **LayerNorm(层归一化,LN)**:对**单个 token 的特征向量**(n_embd 维)做归一化——减去这个向量自己的均值、除以自己的标准差,再用两个可学习参数(gain `γ` 和 bias `β`)缩放平移。作用:把每个 token 的数值分布拉回"均值 0、方差 1"的标准范围,防止数值在深层网络中越滚越大或越缩越小。
>
> **类比:** 每个加工站(层)开工前,先把来料"校准到标准规格"再加工。来料忽大忽小,机器(后续的矩阵乘)就容易出废品(数值溢出、梯度爆炸)。

原始 Transformer(2017)的做法是 **Post-Norm**(后归一化),GPT-2(2019)改成了 **Pre-Norm**(前归一化),这个改动是 GPT-2 论文里明确提到的少数架构变化之一:

```text
Post-Norm(原始 Transformer):      Pre-Norm(GPT-2 及之后几乎所有 LLM):
x = LN(x + Attn(x))                 x = x + Attn(LN(x))
x = LN(x + FFN(x))                  x = x + FFN(LN(x))
        ↑                                   ↑
LN 卡在主干道上,                    LN 挪到支路入口,
残差通路被 LN "打断"                 主干道是一条纯净的加法链
```

**为什么 Pre-Norm 胜出?** 看主干道:

- **Post-Norm**:梯度从输出流向输入,每过一层都要穿过一次 LN。LN 的梯度依赖当前激活值的统计量,几十层 LN 叠加后,浅层梯度变得很不稳定。所以原始 Transformer 必须配合 **learning rate warmup(学习率预热,训练初期用极小的学习率慢慢升上去)** 才能训得动,且层数难以堆深。
- **Pre-Norm**:主干道是"纯加法",从 loss 到第一层 embedding 存在一条**不经过任何 LN 和权重矩阵**的恒等梯度通路。训练从第一步起就稳定,对学习率、warmup 不敏感,可以轻松堆几十层。

> **工业界现状:** GPT-2/3、LLaMA、Qwen、DeepSeek……清一色 Pre-Norm(LLaMA 系把 LN 换成了更省计算的 RMSNorm,但位置不变)。Post-Norm 理论上最终效果略好(主干信号更"干净"),所以也有少数工作(如部分多模态模型)尝试混合方案,但**默认选 Pre-Norm 是行业共识**。
>
> **一个容易忽略的细节:** Pre-Norm 架构在最后一个 Block 之后、输出层之前,还要加**一个收尾的 LayerNorm**(`ln_f`)。因为残差流一路只加不减,数值方差会随层数增长,最后必须校准一次再去做预测。初学者复现 GPT-2 忘加 `ln_f` 是高频 bug。

### 2.3 一个 Block 的完整数据流

把上面的元素拼起来,一个 Pre-Norm Block 内部是这样:

```
        残差流 x (B, T, C)
            │
            ├──→ LN1 → Multi-Head Attention ──┐
            │            (开会:token 间通信)   │
            x ←──────────── + ←───────────────┘   ① x = x + attn(ln1(x))
            │
            ├──→ LN2 → FeedForward ───────────┐
            │            (独思:逐 token 加工)  │
            x ←──────────── + ←───────────────┘   ② x = x + ffn(ln2(x))
            │
            ▼ 流向下一个 Block
```

记住这个节奏:**通信(communicate)→ 计算(compute)**,交替进行。Attention 负责 token 之间横向搬运信息,FFN 负责每个 token 纵向深度加工。这是 Karpathy 在视频里反复强调的心智模型。

---

## 3. 组装完整 GPT:从 token 进到 logits 出

整体架构(对照 nanoGPT 的命名,方便你第 7 章对照阅读):

```
输入 token ids (B, T)                          B=batch, T=序列长度
   │
   ├─ wte: Token Embedding   → (B, T, C)       查表:每个 token id 换成 C 维向量
   ├─ wpe: Position Embedding → (T, C)         查表:每个位置一个 C 维向量
   │        两者相加 → 进入残差流
   ▼
   Block × n_layer                             残差流穿过 N 个块
   ▼
   ln_f: 最终 LayerNorm                        收尾校准(见 2.2 末尾)
   ▼
   lm_head: Linear(C → vocab_size)             把 C 维向量翻译回"词表上的打分"
   ▼
logits (B, T, vocab_size) → cross_entropy → loss
```

> **Logits(对数几率/未归一化打分)**:模型输出的"原始分数",每个 token 位置上有 vocab_size 个分数,还没经过 softmax 变成概率。类比:考试的**原始分**(可以是任意实数,可正可负),softmax 之后才变成"百分比排名"(概率,加和为 1)。PyTorch 的 `F.cross_entropy` 内部自带 softmax,所以模型只需输出 logits——**不要自己先 softmax 再传给 cross_entropy,这是经典错误**(数值不稳定且等于做了两次)。

两个值得专门解释的工业实践:

**① Weight Tying(权重绑定):** `wte`(把 token 变向量)和 `lm_head`(把向量变回 token 打分)做的是互逆的事,GPT-2 让它们**共享同一个权重矩阵**。好处:省掉一大块参数(GPT-2 small 里约 38M,占总参数 30%+),且实验上效果更好——因为"理解一个词"和"生成一个词"理应使用同一套语义表示。

**② 初始化的讲究(`_init_weights`):** 线性层权重用 `std=0.02` 的正态分布初始化,而**写回残差流的投影层**(attention 的输出投影、FFN 的第二层)要额外乘 `1/sqrt(2*n_layer)`。为什么?残差流上有 `2*n_layer` 个"写入口",每个都写入方差相近的量,总方差会随层数线性增长;按 `1/sqrt(N)` 缩小每个写入量,正好让总方差不随深度爆炸。这是 GPT-2 论文里一句话带过、但 nanoGPT 代码里专门处理的细节(搜 `NANOGPT_SCALE_INIT` 或 `c_proj`)。

---

## 4. 完整可运行代码

> **运行环境:** Python ≥ 3.9,只依赖 PyTorch(`pip install torch`)。CPU 可跑(训练慢一点),有 GPU 自动加速。整个文件保存为 `gpt.py`,直接 `python gpt.py` 即可,自带数据(代码内置一段文本,无需下载)。
>
> 结构刻意对齐 nanoGPT 的 `model.py`(类名、变量名一致),你读完这份代码再去读 nanoGPT 会非常顺。

```python
"""
最小完整 GPT:decoder-only Transformer,字符级语言模型。
对齐 nanoGPT model.py 的结构与命名,便于对照阅读。
运行: python gpt.py   (CPU 约 1-2 分钟,GPU 数秒)
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)  # 固定随机种子:复现实验是工程素养,debug 时尤其重要

# ---------------- 超参数(刻意缩小规模,保证 CPU 能跑) ----------------
batch_size = 16      # B: 一次并行处理多少个序列
block_size = 64      # T: 上下文长度(模型最多能看多远)
n_embd     = 128     # C: 残差流宽度(每个 token 的向量维度)
n_head     = 4       # 注意力头数。注意 n_embd 必须能被 n_head 整除
n_layer    = 4       # Block 层数
dropout    = 0.0     # 小模型小数据,不需要正则;大模型预训练一般也是 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- 数据:字符级 tokenizer ----------------
# 实际工业界用 BPE(如 tiktoken),这里用字符级是为了让 vocab 小、聚焦架构本身
text = ("To be, or not to be, that is the question: Whether 'tis nobler "
        "in the mind to suffer The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles. " * 200)
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    # 随机切 batch_size 段长度为 block_size 的序列
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # 目标 = 输入右移一位
    return x.to(device), y.to(device)

# ---------------- 模型组件 ----------------
class CausalSelfAttention(nn.Module):
    """多头因果自注意力(W3 已实现,这里给出 nanoGPT 风格的合并写法)"""
    def __init__(self):
        super().__init__()
        # 把 Q、K、V 三个投影合并成一个大矩阵:1 次大 GEMM 比 3 次小 GEMM 快
        # (GPU 喜欢大矩阵乘——这是 AI Infra 的第一课:kernel 启动有开销,算子要做大做粗)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)   # 输出投影:写回残差流的"闸门"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)                       # (B, T, 3C)
        q, k, v = qkv.split(n_embd, dim=2)         # 各 (B, T, C)
        # 拆多头:(B, T, C) -> (B, n_head, T, head_dim)。head 维放到 batch 旁边,
        # 让每个头的注意力计算彼此独立、可并行
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        # PyTorch 2.x 内置的融合注意力算子,自动调用 FlashAttention 类 kernel:
        # 不会显式生成 (T, T) 注意力矩阵,省显存且快。is_causal=True 自动做因果掩码。
        # nanoGPT 中这条路径叫 "flash";手写 masked_fill + softmax 的旧路径仅作教学
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头
        return self.dropout(self.c_proj(y))

class MLP(nn.Module):
    """FeedForward:逐 token 的两层全连接。本周新内容①"""
    def __init__(self):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd)  # 升维 4 倍:经验甜点(见笔记 1.3)
        self.gelu   = nn.GELU()                      # 平滑激活,GPT-2 标准配置
        self.c_proj = nn.Linear(4 * n_embd, n_embd)  # 降维写回残差流
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意:这里完全没有 token 之间的交互——FFN 对 (B, T) 个 token 独立同等处理
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    """Pre-Norm Transformer Block。本周新内容②"""
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp  = MLP()

    def forward(self, x):
        # Pre-Norm 的精髓:LN 在支路入口,主干 x 是纯净的加法链(残差流)
        x = x + self.attn(self.ln_1(x))  # 通信:token 间交换信息
        x = x + self.mlp(self.ln_2(x))   # 计算:逐 token 独立加工
        return x
        # 高频错误写法 x = self.attn(self.ln_1(x)) —— 丢了 "x +",残差流断裂,
        # 模型立刻退化成训不动的深层网络(loss 下降极慢甚至发散)

class GPT(nn.Module):
    """组装完整模型。本周新内容③"""
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, n_embd),   # token embedding(查表)
            wpe  = nn.Embedding(block_size, n_embd),   # 位置 embedding(GPT-2 用可学习的)
            h    = nn.ModuleList([Block() for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),               # 收尾 LN:Pre-Norm 架构必需,勿漏!
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight Tying:输入查表和输出打分共享同一矩阵(省 30%+ 参数,效果更好)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # 写回残差流的投影层,按 1/sqrt(2*n_layer) 缩小初始化:
            # 残差流上有 2*n_layer 个写入口,不缩小则深层激活方差线性膨胀(见笔记第 3 章)
            if hasattr(module, '_is_residual_proj'):
                std *= (2 * n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.transformer.wte(idx)                            # (B, T, C)
        pos_emb = self.transformer.wpe(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb        # 进入残差流(广播相加:每个 batch 共用同一套位置向量)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                       # (B, T, vocab)

        loss = None
        if targets is not None:
            # cross_entropy 要求 (N, classes) + (N,),所以把 B、T 两维摊平
            # 直接传 (B, T, vocab) 会报错——这是新手必踩的 shape 坑
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()  # 关闭 dropout(本例 dropout=0 无影响,但养成习惯)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 截断到上下文窗口,否则 wpe 查表越界报错
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]        # 只取最后一个位置的预测
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样(而非贪心)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ---------------- 训练 ----------------
if __name__ == '__main__':
    model = GPT().to(device)
    # 给残差投影层打标记(配合 _init_weights;nanoGPT 用变量名 c_proj 判断,异曲同工)
    for block in model.transformer.h:
        block.attn.c_proj._is_residual_proj = True
        block.mlp.c_proj._is_residual_proj = True
    model.apply(model._init_weights)  # 打完标记后重新初始化一次

    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab_size = {vocab_size}, 参数量 = {n_params/1e3:.0f}K, device = {device}")

    # ======== 本周新内容④:验证初始 loss ≈ -log(1/vocab),详见笔记第 5 章 ========
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    expected = math.log(vocab_size)
    print(f"初始 loss = {loss.item():.4f}, 理论值 -log(1/{vocab_size}) = {expected:.4f}")
    assert abs(loss.item() - expected) < 0.3, "初始 loss 偏离理论值,检查初始化/最后一层!"

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW + 3e-4:LLM 默认起手式
    for step in range(2000):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)  # set_to_none 比置零省一次显存写,工业惯例
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- 生成样例 ---")
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
```

**预期现象:** 初始 loss 非常接近 `log(vocab_size)`(本例 vocab≈30+,约 3.4);训练 2000 步后 loss 降到 1 以下,生成的文本明显呈现莎翁片段的模式。

---

## 5. 验证初始 loss ≈ -log(1/vocab):一个被低估的"工程体检"

### 5.1 原理:为什么初始 loss 应该是这个值?

**Cross-Entropy Loss(交叉熵损失)** 对单个样本就是 `-log(模型分给正确答案的概率)`。

一个刚初始化、什么都没学的模型,理应对词表里所有 token "一视同仁"——每个 token 的预测概率都是 `1/vocab_size`。于是:

```
初始 loss = -log(1/vocab_size) = log(vocab_size)
```

| 场景 | vocab_size | 理论初始 loss |
|---|---|---|
| 本笔记的小例子 | ~30 | ~3.4 |
| Karpathy 莎士比亚字符级 | 65 | **4.17** |
| GPT-2(BPE 词表) | 50257 | **10.82** |

> **类比:** 一个全新的多选题答题机,面对 65 个选项,如果它的"初始自信度"在 65 个选项上完全平均,那它的"惊讶程度"(交叉熵)就恰好是 log 65 ≈ 4.17。如果初始就明显低于 4.17,说明机器**出厂前被人动过手脚**(初始化让它偏爱某些选项);明显高于 4.17,说明它出厂时**自信地押错宝**(某些 logit 初始值过大,概率分布过尖)。两种都不健康。

### 5.2 为什么工业界把它当作开训前的"必查项"?

这是 Karpathy 在《A Recipe for Training Neural Networks》里列的第一类检查:**verify loss @ init**。原因很实际:

- **训练大模型一次烧几万到几百万美元。** 任何能在第 0 步发现的 bug,都比训了三天再发现便宜一万倍。初始 loss 检查是成本最低的 sanity check(健全性检查)。
- 它能一次性暴露一串高频 bug:
  - 初始 loss **偏高**(比如 65 词表测出 20+):最后一层初始化方差太大,logits 过尖 → 模型一开始就"自信地胡说",前期训练全花在"消除错误自信"上,白白浪费算力。这就是为什么有的实现刻意把 `lm_head` 初始化得很小甚至置零。
  - 初始 loss **偏低**:数据泄漏(target 混进了 input)、或忘了 shuffle 导致退化分布。
  - 初始 loss = **NaN**:学习率/初始化爆炸,或 label 越界(`targets` 里出现 ≥ vocab_size 的 id)。
- **延伸技巧:** 第二个常用体检是"先在一个 batch 上 overfit 到 loss≈0"——能过拟合说明模型有学习能力、数据流没断;不能过拟合必有 bug。建议你在自己的代码上也做一次。

### 5.3 独立的最小验证脚本

```python
# 环境:仅需 PyTorch。独立运行,不依赖上面的 gpt.py
import math, torch, torch.nn.functional as F

vocab_size, B, T = 65, 8, 32
# 模拟"完全均匀"的模型输出:所有 logit 相等 → softmax 后每个 token 概率 = 1/65
logits = torch.zeros(B, T, vocab_size)
targets = torch.randint(0, vocab_size, (B, T))
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
print(f"均匀 logits 的 loss = {loss.item():.4f}")        # 4.1744
print(f"log(vocab_size)    = {math.log(vocab_size):.4f}") # 4.1744 完全一致

# 反面案例:把 logits 初始化方差调大,看 loss 怎么飘
bad_logits = torch.randn(B, T, vocab_size) * 10  # 故意用过大的 std
bad_loss = F.cross_entropy(bad_logits.view(-1, vocab_size), targets.view(-1))
print(f"过尖 logits 的 loss = {bad_loss.item():.4f}")     # 远大于 4.17,通常 10+
```

> **注意:** 实际模型的初始 loss 不会精确等于理论值(权重是随机的,不是恒零),在 ±0.1~0.3 范围内浮动都正常。要的是**量级正确**,不是小数点后对齐。

---

## 6. 残差流 + LN vs BN:写 `tech_notes/residual_stream_and_layernorm.md` 的素材与框架

残差流部分第 2 章已经讲透(写 note 时直接复用"总线/修订模式"两个类比 + 梯度 `+1` 通路的推导)。这一章补齐 **LN vs BN** 的对比——这是面试高频题,也是理解"为什么 Transformer 选了 LN"的钥匙。

### 6.1 BatchNorm 是什么,它和 LayerNorm 差在哪一根轴上

> **BatchNorm(批归一化,BN)**:对一个 batch 里**所有样本的同一个特征通道**做归一化。即:统计的是"这一批数据在某个特征上的均值和方差"。

两者做的事一模一样(减均值除标准差),**唯一的区别是沿哪个维度统计**。以形状 `(B, T, C)` 的激活为例:

```
                 B(样本) T(位置) C(特征)
BatchNorm:  对每个特征 c,跨 [B, T] 统计  → 归一化依赖"同批的其他样本"
LayerNorm:  对每个 (b, t),跨 [C] 统计    → 归一化只看"这个 token 自己"
```

> **类比:考试成绩标准化。**
> - **BN = 按全班排名**:你的标准分取决于同班同学考得怎样。全班是学霸,你就显得差;全班划水,你就显得强。——**依赖"同批次的别人"**。
> - **LN = 按你自己各科的均衡度**:只看你一个人的语数外分布,把它们拉到同一尺度。别人考多少与你无关。——**只依赖"自己"**。

### 6.2 为什么 Transformer/LLM 几乎不用 BN?四个硬伤

1. **训练/推理行为不一致。** BN 训练时用 batch 统计量,推理时 batch 可能只有 1 个样本,只能用训练期累积的滑动平均(running mean/var)。这套"双轨制"是大量诡异 bug 的来源(经典症状:模型 `train()` 模式下效果很好,`eval()` 一开就崩,多半是 running stats 没积累好)。LN 训练推理完全同一套计算,无状态、无双轨。

2. **变长序列 + padding 污染统计量。** NLP 的 batch 里序列长短不一,要补 padding(填充占位符)。BN 跨样本统计时会把这些"假 token"也算进去,均值方差全被污染。LN 每个 token 自理,padding 影响不了别人。

3. **自回归生成时"batch"概念失效。** 推理时逐 token 生成,batch size 可能为 1,且每步只有 1 个新位置——根本凑不出有统计意义的 batch。LN 无此问题。

4. **分布式训练的通信代价(AI Infra 视角,建议写进你的 note)。** 大模型训练时一个 batch 切分到几百张 GPU 上(数据并行),BN 要算"全 batch"统计量就必须做跨卡同步(SyncBatchNorm,一次 all-reduce),在每一层、每一步都通信——延迟不可接受。LN 的统计只在单个 token 的特征维度内完成,**天然不需要任何跨卡通信**。这是 LN 在大规模训练时代胜出的工程原因,和算法原因同样重要。

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| 统计维度 | 跨样本(B,T) | 单 token 特征维(C) |
| 依赖 batch size | 是(小 batch 统计噪声大) | 否 |
| 训练 vs 推理 | 双轨(running stats) | 同一套计算 |
| 变长序列/padding | 被污染 | 不受影响 |
| 分布式训练 | 需跨卡同步 | 零通信 |
| 主战场 | CNN/视觉(ResNet 等) | Transformer/NLP、大模型 |

### 6.3 手写 LayerNorm 验证理解(可直接放进 tech_note)

```python
# 环境:仅需 PyTorch。验证:手写 LN == nn.LayerNorm
import torch

def my_layernorm(x, eps=1e-5):
    # 关键:dim=-1,只在特征维 C 上统计——这一行就是 LN 和 BN 的全部区别
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, keepdim=True, unbiased=False)  # 注意 unbiased=False!
    # PyTorch 的 var 默认 unbiased=True(除以 N-1),而 LN 定义用有偏方差(除以 N)
    # 忘了这个参数,结果会有微小但确实存在的偏差——典型的"看文档才知道"的坑
    return (x - mean) / torch.sqrt(var + eps)  # eps 防止方差为 0 时除零

x = torch.randn(4, 8, 32)                      # (B, T, C)
ln = torch.nn.LayerNorm(32, elementwise_affine=False)  # 关掉 γ/β,纯归一化对比
print(torch.allclose(my_layernorm(x), ln(x), atol=1e-5))  # True
```

### 6.4 你的 tech_note 建议结构

```markdown
# residual_stream_and_layernorm.md 建议骨架
1. 残差流是什么 —— 总线/修订模式类比 + ASCII 数据流图(可参考本笔记 2.3)
2. 为什么需要 —— 梯度 +1 恒等通路推导;ResNet 的历史教训
3. Pre-Norm vs Post-Norm —— 两行伪代码对比 + 为什么 GPT-2 改了 + ln_f 的必要性
4. LN vs BN —— 一根轴的区别 + 考试类比 + 四个硬伤 + 对比表
5. AI Infra 视角 —— BN 跨卡同步 vs LN 零通信;LN→RMSNorm 的演化(LLaMA 为什么换)
6. 手写验证代码 + 我踩过的坑(unbiased=False 等)
```

---

## 7. 通读 nanoGPT `model.py`:带着"寻宝清单"去读

仓库:`github.com/karpathy/nanoGPT`,文件 `model.py`(全文 ~330 行)。**不要从第一行平铺直叙地读**,带着下面这张清单去"寻宝"——每找到一个,就和你自己的实现对一次:

### 7.1 结构对照表(你的代码 ↔ nanoGPT)

| 本笔记代码 | nanoGPT `model.py` | 读时注意 |
|---|---|---|
| `CausalSelfAttention` | 同名类 | 它保留了手写 attention 的旧路径(`self.flash` 为 False 的分支),对照看 FlashAttention 替代了哪几行 |
| `MLP` | 同名类 | 一模一样的 fc→gelu→proj 三件套 |
| `Block` | 同名类 | 验证 pre-norm 的两行公式和你写的是否一致 |
| `GPT.transformer` 字典 | 同结构(wte/wpe/h/ln_f) | 注意 `drop`(embedding 后的 dropout)我们省略了 |
| weight tying | `self.transformer.wte.weight = self.lm_head.weight` | 搜这一行,看注释里引用的论文 |
| `_is_residual_proj` 标记 | 搜 `NANOGPT_SCALE_INIT` 思想对应的 `c_proj` 特殊初始化 | 在 `__init__` 里 `for pn, p in self.named_parameters()` 那段 |
| 初始 loss 验证 | 不在 model.py(在训练脚本/讲解中) | — |

### 7.2 八个"寻宝点"(每个都值得停下来想一分钟)

1. **`GPTConfig` dataclass**——所有超参集中在一个配置类。工业惯例:模型代码不写死任何数字,全部走 config(对照 HuggingFace 的 `config.json` 同一思想)。
2. **自定义 `LayerNorm` 类**——nanoGPT 自己包了一层,只为支持 `bias=False`(PyTorch 旧版 `nn.LayerNorm` 不支持关 bias)。GPT-2 原版有 bias,新模型趋势是去掉(参数更少、略快、效果不降)。
3. **`assert config.n_embd % config.n_head == 0`**——多头拆分的前提。想想如果不整除会发生什么。
4. **`self.flash = hasattr(F, 'scaled_dot_product_attention')`**——运行时检测 PyTorch 版本特性、新旧路径兼容。工业代码常见模式。
5. **`crop_block_size()`**——加载预训练权重后裁短上下文窗口(位置 embedding 直接切片)。体会"权重是可以手术的"。
6. **`from_pretrained()`**——加载 HuggingFace GPT-2 权重时,有一段 `transposed` 列表特殊处理:因为 OpenAI 原版用 TF 风格的 `Conv1D` 存权重,和 PyTorch `Linear` 互为转置。**权重格式转换是 infra 工程师的日常**,这是绝佳的真实案例。
7. **`configure_optimizers()`**——把参数分成两组:二维参数(矩阵)做 weight decay(权重衰减,一种防过拟合的正则),一维参数(bias、LN 的 γ/β)不做。为什么?衰减 LN 的缩放参数会直接干扰归一化的校准功能。这是训练 LLM 的标准做法,几乎所有框架都这么干。
8. **`generate()` 里的 `temperature` 和 `top_k`**——对照你写的朴素采样,看推理服务里真实使用的两个采样旋钮:temperature 控制"随机性温度",top_k 砍掉长尾低概率 token。

### 7.3 读完后的自测(费曼检验)

合上代码,你应该能不查资料回答:

1. 为什么 FFN 要先升维 4 倍再降回来?Attention 和 FFN 的分工用一句话怎么概括?
2. `x = x + attn(ln(x))` 中,如果把 `x +` 删掉会发生什么?为什么?
3. Pre-Norm 比 Post-Norm 好训的根本原因是什么?`ln_f` 为什么不能省?
4. 65 词表的字符级模型,初始 loss 应该是多少?如果实测是 20,最可能是哪里出了问题?
5. LN 和 BN 的区别用"一根轴"怎么说清?为什么大模型分布式训练天然排斥 BN?
6. 残差流上的投影层为什么要按 `1/sqrt(2*n_layer)` 缩小初始化?
7. weight tying 绑的是哪两个矩阵?为什么合理?

---

## 8. 常见陷阱速查表(踩坑成本从高到低)

| 陷阱 | 症状 | 修复 |
|---|---|---|
| 忘写 `x = x + ...`(残差断裂) | loss 下降极慢/不降 | 检查 Block.forward 的两行 |
| 漏掉 `ln_f` | loss 能降但明显偏高,深层模型更糟 | Block 堆叠后补最终 LN |
| cross_entropy 直接喂 (B,T,V) | 报 shape 错误 | `.view(-1, vocab_size)` / `.view(-1)` |
| 先 softmax 再 cross_entropy | loss 数值诡异、训练慢 | cross_entropy 只吃原始 logits |
| `var(unbiased=True)` 手写 LN | 和 nn.LayerNorm 对不上 | `unbiased=False` |
| generate 不截断上下文 | wpe 查表越界 IndexError | `idx[:, -block_size:]` |
| weight tying 写反顺序 | 静默错误,权重没真正共享 | 先建 lm_head 再赋值给 wte(或反之保持一致),用 `id()` 验证同一对象 |
| 推理忘了 `model.eval()` | 有 dropout 时输出随机抖动 | generate 前 eval(),训练前 train() |

---

## 9. 收尾:把四个任务串成一条线

这四个任务其实是一个完整的工程闭环:

```
实现(FFN + Block + GPT) → 验证(初始 loss 体检) → 沉淀(tech_note) → 对标(nanoGPT)
   写出来的代码              证明它没病              讲明白原理         向工业实现看齐
```

这个"实现 → 验证 → 沉淀 → 对标"的循环,正是你以后做 AI Infra(比如小米项目里改一个 kernel、提一个优化)的标准工作流。本周你是在一个 100K 参数的小模型上,排练将来在 100B 参数模型上要做的事。

**下一步衔接(W4 后半):** 模型已经能跑,profiler 进场——用 `torch.profiler` 看看时间都花在哪个算子上(剧透:大头是 FFN 和 Attention 的矩阵乘),那才是推理优化故事的开始。
