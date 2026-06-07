# Week 6 · Day 2（2026-06-02）：EP6 nanoGPT 通看 + 单头 Self-Attention

> **覆盖任务**（计划 line 1656-1697 区段 / W6 Day2 checklist）：
> - [ ] DL：EP6 第一遍通看（不写代码），建立整体架构心智模型
> - [ ] DL：实现单头 self-attention（含 causal mask + 缩放）
> - [ ] DL：完成 `tech_notes/attention_complexity.md`（O(T²) 手算 + 缩放因子作用）
> - [ ] 完成标准：能不看资料口述单头 attention 的 5 行计算流程，解释"为什么是 O(T²)"和"缩放因子 1/√d 的作用"
>
> **阅读对象**：你自己——昨天（Day1）你刚把 EP5 的"代码模块化"吃透，知道了 `nn.Module` + `Sequential` 是怎么把网络拼成乐高积木的。今天 EP6 就是用同一套乐高，搭一个**全新的积木：Attention（注意力）**，再堆成 GPT。
>
> **你说 EP6 看不太懂，所以这篇我换一种讲法**：Karpathy 视频是"边写代码边解释"，信息密度很高，第一遍听容易被代码淹没。这篇笔记**反过来**——先把"这一集到底在干嘛"的大故事讲清楚（§1），再把唯一的新东西 Attention 用大白话+类比拆开（§2-§6），最后才落到代码（§4、§7）。读完这篇再去看视频，你会发现"哦，原来他那段是在讲这个"。
>
> **本笔记的设计**：沿用 [[W6_Day1_EP5_WaveNet_Module容器抽象]] 的三段式——每节先讲"原理 + 直觉"，再给可直接拷进 `src/` 的可运行代码，最后写工业锚点（AI Infra 视角）。今天是**整个学期 DL 的收口之战的开端**，attention 这一个概念吃透了，后面 KV Cache、FlashAttention、长上下文全都有根。

---

## 0. 学习目标（看完应能脱口而出）

1. EP6 这一集，从头到尾在搭一个什么东西？它的**数据流主干**是哪一条线？（提示：5 个站点，从字符到预测）
2. "Attention（注意力）"这个词别被唬住——用一句大白话，它到底在让每个词干什么？
3. Q、K、V 三个英文字母分别代表什么？为什么一个词要同时变出这三样东西？
4. 单头 self-attention 的核心就 **5 行代码**，这 5 行每一行在算什么？为什么是这个顺序？
5. 什么叫 **causal mask（因果掩码）**？为什么语言模型必须把"未来"挡住？它为什么长成一个"下三角"？
6. 那个 `* head_size**-0.5`（即 1/√d）的**缩放因子**到底防的是什么灾难？不加会怎样？
7. 为什么大家都说 Attention 是 **O(T²)**？这个"平方"是算力问题还是显存问题，还是两个都是？
8. 长上下文（GPT-4 的 128K）为什么会"爆显存"？给我手算一个具体数字。

---

## 1. 先把 EP6 的"剧情"讲清楚：它到底在搭什么

### 1.1 一句话剧情

EP6 的全名是 *Let's build GPT: from scratch, in code, spelled out*。一句话：

> **从零搭一个能"续写莎士比亚"的迷你 GPT——喂它一段字符，它一个字一个字地预测下一个字符，越写越像莎士比亚。**

它和你 W3 写过的 bigram（二元模型）是**同一个任务**："给定前面的字，猜下一个字"。区别只在于：
- **bigram**：猜下一个字时，只看**前一个字**（记忆只有 1 格）。
- **GPT**：猜下一个字时，能看**前面所有字**，而且能"挑重点看"——这个"挑重点"的能力，就是今天唯一的新东西：**Attention**。

所以你可以把今天理解成：**给 bigram 装上一个"能回看全文并抓重点"的大脑**。

### 1.2 整体数据流：记住这 5 个站点

EP6 代码看着多，但数据从输入到输出只走一条主干道，**5 个站点**。先背下这条线，看视频时你就知道每段代码在修哪个站点：

```
   一串字符的编号                    每个字都带上"我是谁"+"我在第几位"
   [18, 47, 56, ...]                            │
        │                                       ▼
        ▼                          ┌─────────────────────────┐
  ① token embedding   ──加──▶ ②   │  token_emb + pos_emb    │
  （查表：每个字 → 一个向量）        │ （内容信息 + 位置信息）  │
                                   └────────────┬────────────┘
                                                ▼
                                   ③ N 个 Block 串联（核心！）
                                   每个 Block = Attention + 前馈网络
                                   （Attention 让每个字回看全文抓重点）
                                                │
                                                ▼
                                   ④ final LayerNorm（最后归一化一下）
                                                │
                                                ▼
                                   ⑤ lm_head（一个 Linear）
                                   把每个位置的向量 → 27/65 个字符的分数
                                                │
                                                ▼
                                          logits（下一个字的打分）
                                                │
                                          softmax → 采样 → 吐出下一个字
```

用大白话复述这条线：

1. **① token embedding（词嵌入）**：把每个字符（其实是它的编号）查表换成一个向量。"查表换向量"你 W4 就干过——`C[Xb]`，呼应 [[W6_Day1_EP5_WaveNet_Module容器抽象]] 里提的 embedding。**这一步在说"这个字的含义是什么"。**
2. **② 加上 position embedding（位置嵌入）**：再查一张"位置表"，第 0 个位置一个向量、第 1 个位置一个向量……加到上面。**为什么要加？** 因为 Attention 本身"分不清前后顺序"（后面 §2.4 解释），所以得手动告诉模型"这个字在第几位"。
3. **③ N 个 Block（块）串联**：这是整个 GPT 的肉。每个 Block 里有两件事：**Attention（回看全文抓重点）** + **FeedForward（前馈网络，单独消化一下）**。堆 N 个（nanoGPT 里 N=6），信息一层层加工得越来越深。**今天只攻 Block 里的 Attention，FFN 和怎么堆是明天/后天的事。**
4. **④ final LayerNorm（最终层归一化）**：出 Block 后做最后一次归一化，把数值整理整齐再送出去。LayerNorm 是什么、为什么不用 BatchNorm，是 Day4 的核心（呼应你 W4 `batchnorm_inference.md`），今天先知道它"是个归一化"就行。
5. **⑤ lm_head（语言模型头）**：一个普通的 `nn.Linear`，把每个位置的向量映射成"词表里每个字的分数（logits）"。分数最高的那个字，就是模型猜的"下一个字"。

> **看视频的正确姿势**：Karpathy 是从最简单的 bigram 起步，一步步往上加东西，最后才拼出上面这张图。如果你第一遍跟丢了，就**对照这 5 个站点**，每听一段问自己："他现在在修哪个站点？" 90% 的时间他在修 ③ 里的 Attention。

### 1.3 为什么说今天"只要攻下 Attention，这一集就破了"

看上面的图：①查表、②查表、④归一化、⑤一个 Linear——这四个站点你**全都见过**（W4 embedding、W5 归一化、micrograd 里的 Linear）。整张图里**唯一全新的概念，就是 ③ Block 里的 Attention**。

所以今天的策略很明确：**把单头 self-attention 这一个东西彻底吃透，EP6 就从"看不懂"变成"就这？"**。多头、Block、堆叠都是后面几天在这个地基上摞砖头。下面我们就用整篇剩下的篇幅，只讲这一个东西。

---

## 2. Attention 是什么：先建直觉，别碰公式

### 2.1 一个生活场景：开会时你怎么"分配注意力"

想象一个 8 个人的小组会，轮到**你**发言总结。你要说的内容，取决于前面 7 个人说了啥。但你不会平均地记住每个人——你会**挑重点**：

- 你心里有个**问题/诉求**：比如"我现在要总结一下**预算**问题"。这就是你的 **Query（查询，记作 Q）——"我想找什么"**。
- 在场每个人头上都挂着一个**标签**，写着他刚才主要讲了什么主题。"老王讲的是预算""小李讲的是排期"。这些标签就是每个人的 **Key（键，记作 K）——"我是关于什么的"**。
- 你拿你的诉求（Q="找预算"）去和每个人的标签（K）比对，**匹配度高的人，你就多分一点注意力**：老王（讲预算）你给 0.7 的注意力，小李（讲排期）你给 0.1，其他人分剩下的。
- 最后你**按这个注意力比例**，把大家**实际说的话**揉成你的总结。每个人实际说的具体内容，就是他的 **Value（值，记作 V）——"我携带的真实信息"**。

把"你"换成序列里的**每一个 token（词元/字符）**，把"开会"换成"处理这句话"，这就是 self-attention（自注意力）。"self（自）"的意思是：Q、K、V 全都来自**同一句话内部**的这些 token，它们自己跟自己开会。

> **token（词元）**：模型处理文本的最小单位。在 nanoGPT 字符级模型里，一个 token 就是一个字符；在真实 LLM 里，一个 token 大约是 0.75 个英文单词（"attention"可能被切成"atten"+"tion"两个 token）。下文"token""字""词"在本笔记里基本同义，都指序列里的一个位置。

### 2.2 Q、K、V：为什么一个 token 要变出三样东西

关键点：**Q、K、V 不是天生的，是同一个 token 向量乘三个不同的权重矩阵"投影"出来的**。

```
              ┌──乘 Wq──▶  Q（我想找什么）
token 向量 x ─┼──乘 Wk──▶  K（我是关于什么的，给别人看的标签）
              └──乘 Wv──▶  V（我携带的真实信息）
```

为什么要拆成三个、而不是直接拿 x 自己跟自己比？用开会类比：

- 你**对外宣传的标签（K）**和你**心里真正想找的（Q）**往往不是一回事。老王挂的标签是"预算专家"（K），但他此刻心里想找的是"谁懂税务"（Q）。同一个人，对外身份和对内需求是两套，所以要两个矩阵 Wk、Wq 分开学。
- 你**实际能贡献的信息（V）**，又和你的"标签"不一样——标签是给人**检索**用的（短、概括），实际内容是给人**取用**的（详细）。就像图书馆：书脊上的**索书号/标题**是 K（帮你找到书），书里的**正文**是 V（你真正要读的）。所以 V 又是第三个矩阵 Wv。

> **investigate before asserting**：这三个矩阵（Wq/Wk/Wv）就是 attention 里**唯一要学习的参数**（再加一个输出投影，明天多头才有）。"学习 attention"本质就是学这三个投影怎么把 token 映射成合适的"诉求/标签/内容"。

### 2.3 把类比翻译成矩阵：QKᵀ 是一张"谁该看谁"的表

现在把"每个 token 拿 Q 去和所有 token 的 K 比对匹配度"这件事，变成矩阵运算。

设这句话有 T 个 token（T = 序列长度，比如 8），每个 Q、K 是 d 维向量（d = head_size，比如 16）。

- 第 i 个 token 的诉求 `q_i`，和第 j 个 token 的标签 `k_j`，它俩的**匹配度**用**点积**（dot product，两个向量对应位置相乘再求和）来算：`q_i · k_j`。点积越大 = 两个向量方向越一致 = 越"对味"。
- 把**所有 i 对所有 j** 的匹配度算出来，就得到一张 **T×T 的表**，第 i 行第 j 列 = "第 i 个 token 该给第 j 个 token 多少关注"。这张表用一个矩阵乘法一次算完：

```
wei = Q @ Kᵀ     # Q 形状 (T, d)，Kᵀ 形状 (d, T)，相乘得 (T, T)
```

`@` 是矩阵乘法，`Kᵀ` 是 K 的转置（把 (T,d) 翻成 (d,T) 好相乘）。**记住这个 (T, T) 形状——它就是后面 O(T²) 的元凶，整篇笔记的高潮都在它身上。**

### 2.4 一个反直觉的点：Attention 自己"看不见顺序"

注意上面整个过程：每个 token 跟每个 token 比匹配度，**完全没用到"谁在前谁在后"的信息**。把"the cat sat"打乱成"sat the cat"，算出来的匹配度表里的数值是一样的（只是行列换了位置）。

这就是 §1.2 ② 为什么要**单独加 position embedding（位置嵌入）**：Attention 是个"无序的集合操作"，你必须在输入阶段就把"我在第几位"这个信息**塞进 token 向量里**，它才能区分语序。这是个常考点，也是初学者最容易漏的直觉。


---

## 3. Causal Mask（因果掩码）：为什么要把"未来"挡住

### 3.1 问题背景：自回归模型不能"偷看答案"

GPT 是 **autoregressive（自回归）**模型——一个字一个字往外吐，每一步用前面已经生成的字去预测下一个字。训练时也一样：第 t 个位置的任务是"根据第 1 到第 t 个字，预测第 t+1 个字"。

这里藏着一个致命陷阱。训练时我们把**整句话**一次性喂进去（为了并行，快）。但第 3 个 token 在算它的输出时，如果 Attention 让它去看了第 5、第 6 个 token——那就等于**考试时偷看了后面的答案**。模型会学到"看第 5 个字就能猜第 4 个字"这种作弊捷径，一到真实生成（那时根本没有未来的字）就全废。

所以必须有个规则：**第 i 个 token 只能关注第 1 到第 i 个（含自己），第 i+1 之后的一律不许看。** 这个规则的实现，就叫 causal mask（因果掩码，"causal=因果"指信息只能从过去流向未来）。

### 3.2 怎么实现：把"未来"那一格填成负无穷

回到 §2.3 那张 T×T 的匹配度表 `wei`。第 i 行代表"第 i 个 token 对所有人的关注分数"。我们要做的就是：**把第 i 行里属于"未来"（列号 j > i）的格子，全部抹掉。**

技巧很巧妙——不是直接删，而是**填成 `-inf`（负无穷）**，然后交给 softmax：

```
原始 wei（T=4，随便编的分数）        mask 后（上三角填 -inf）
       j=0   j=1   j=2   j=3              j=0   j=1   j=2   j=3
i=0 [ 0.9   0.3   0.5   0.1 ]      i=0 [ 0.9  -inf  -inf  -inf ]
i=1 [ 0.2   0.8   0.4   0.7 ]  ──▶ i=1 [ 0.2   0.8  -inf  -inf ]
i=2 [ 0.6   0.1   0.9   0.3 ]      i=2 [ 0.6   0.1   0.9  -inf ]
i=3 [ 0.4   0.5   0.2   0.8 ]      i=3 [ 0.4   0.5   0.2   0.8 ]
```

留下来的那个**下三角（含对角线）**，就是"每个 token 只能看自己和过去"。**这就是为什么 causal mask 永远长成一个下三角形状**——这个形状你要刻进脑子，明天 Day3 还要再写一遍。

**为什么填 `-inf` 而不是 0？** 因为下一步是 softmax。softmax 会算 `e^x`，而 `e^(-inf) = 0`。所以填 `-inf` 的格子经过 softmax 后**精确变成 0 权重**——那个未来 token 一点注意力都分不到。如果你填 0，`e^0 = 1` 反而还有权重，等于没挡住。**这是个高频面试/笔试坑：mask 的值要填到 softmax 之前、且填负无穷。**

### 3.3 工业锚点：下三角结构 = KV Cache 能成立的根本前提

记住这个下三角，它不只是个训练技巧。**正因为"历史 token 永远看不到未来"，所以一个 token 的 K 和 V 算出来之后就永远不会变。**

- 朴素生成：每吐一个新字，把整句话重新 forward 一遍——但前面那些字的 K/V 明明没变，却被反复重算，纯浪费。
- KV Cache（你 W8 和暑假的主攻点）：**把历史 token 的 K、V 缓存起来，每步只算新 token 的**。O(n²) 的重复计算降到每步 O(n)。

**而 KV Cache 之所以正确，根基就是这个下三角**——因果性保证了"过去不会因为未来而改变"。这条线（causal mask → 历史不变 → KV Cache）是你目标方向 AI Infra 的第一性原理，今天先埋下，呼应计划 line 1614-1616。

---

## 4. 把 §2、§3 合起来：单头 self-attention 的 5 行核心

现在把直觉翻译成代码。这就是 Day2 完成标准要你"不看资料口述"的 5 行。先看代码，逐行讲为什么：

```python
# x: (B, T, C)  ——  B=batch（几句话），T=序列长度（几个token），C=每个token的向量维度
q = self.query(x)                                  # (B, T, hs)  每个token算出"我想找什么"
k = self.key(x)                                    # (B, T, hs)  每个token算出"我的标签"
v = self.value(x)                                  # (B, T, hs)  每个token算出"我的真实信息"
wei = q @ k.transpose(-2, -1) * head_size**-0.5    # (B, T, T)   ① 匹配度表 + ② 缩放
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # ③ causal mask（下三角）
wei = F.softmax(wei, dim=-1)                       # (B, T, T)   ④ 归一化成"注意力比例"
out = wei @ v                                      # (B, T, hs)  ⑤ 按比例加权汇总 V
```

逐行翻译成"开会"类比，背的时候按这个故事背：

| 行 | 代码 | 一句话 | 开会类比 |
|---|---|---|---|
| 1-3 | `q/k/v = self.xxx(x)` | 每人各自生成诉求/标签/发言内容 | 开会前每人准备好三样东西 |
| 4 | `q @ kᵀ * d**-0.5` | 算"谁该看谁"的匹配度表，再缩放 | 你拿诉求挨个比对别人标签，打分 |
| 5 | `masked_fill(...-inf)` | 把未来的格子抹成负无穷 | 不许偷看还没发言的人 |
| 6 | `softmax` | 把分数变成加起来=1 的比例 | 把打分换算成"注意力百分比" |
| 7 | `wei @ v` | 按比例把大家的发言揉成总结 | 按注意力比例汇总大家说的话 |

> **关于 `head_size**-0.5`（即 1/√d）这个缩放**：它就藏在第 4 行末尾，看着不起眼，但删了它训练会崩。它防的是什么灾难，专门留到 §6 讲，因为它直接对接你 W4 的 `init_and_stability.md`。

> **关于 `transpose(-2, -1)`**：k 的形状是 `(B, T, hs)`，要算 `q @ kᵀ` 得把最后两维转置成 `(B, hs, T)`，这样 `(B,T,hs) @ (B,hs,T) = (B,T,T)`。`-2, -1` 是指"倒数第二维和倒数第一维"，不动 batch 维 B。


---

## 5. 完整可运行代码：`src/attention_single.py`

下面是可以直接拷进 `week6_nanogpt/src/attention_single.py` 跑的完整版本，含正常使用、形状打印调试、以及一个验证 causal mask 正确性的小测试。

```python
# src/attention_single.py
# 运行环境：Python 3.10+，PyTorch 2.x（CPU 即可，无需 GPU）
# 运行：python src/attention_single.py
# 依赖：pip install torch
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)  # 固定种子，结果可复现（呼应你 tech_notes 的"可复现"产出规范）


class Head(nn.Module):
    """单头 self-attention。教学实现，对照 5 行核心逐块认领。"""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # 三个投影：把 C 维的 token 向量 → head_size 维的 q/k/v
        # bias=False 是 nanoGPT 的惯例：后面有 LayerNorm，bias 基本冗余（呼应 Day1 §7 坑2）
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 下三角矩阵注册成 buffer：它是常量、不是要学习的参数，所以不进梯度、不被优化器更新
        # 但它要跟着模型一起 .to(device)/save，所以用 register_buffer 而不是普通属性
        # 这是个高频坑：写成 self.tril = torch.tril(...) 在 GPU 上会报 device 不匹配
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # B=batch, T=序列长度, C=n_embd
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)    # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # ① 匹配度表 + ② 缩放（1/√d，防 softmax 饱和，见 §6）
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5   # (B, T, T)
        # ③ causal mask：上三角（未来）填 -inf。只取 [:T,:T] 以兼容比 block_size 短的序列
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # ④ softmax 沿最后一维：把每一行变成"加起来=1 的注意力比例"
        wei = F.softmax(wei, dim=-1)                              # (B, T, T)
        wei = self.dropout(wei)
        # ⑤ 加权汇总 V
        out = wei @ v                                             # (B, T, hs)
        return out


if __name__ == "__main__":
    # ---- 正常使用：一个 batch，2 句话，每句 8 个 token，每个 token 32 维 ----
    B, T, C, head_size, block_size = 2, 8, 32, 16, 8
    x = torch.randn(B, T, C)
    head = Head(n_embd=C, head_size=head_size, block_size=block_size)
    out = head(x)
    print("输入  x  形状:", tuple(x.shape))       # (2, 8, 32)
    print("输出 out 形状:", tuple(out.shape))      # (2, 8, 16) —— 每个token聚合后的新表示
    assert out.shape == (B, T, head_size)

    # ---- 调试技巧 1：把注意力权重 wei 抠出来看，验证它是下三角且每行和为 1 ----
    q, k = head.query(x), head.key(x)
    wei = q @ k.transpose(-2, -1) * head_size ** -0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    print("\n第 0 句话的注意力权重矩阵（应是下三角，上三角=0）:")
    print(wei[0].round(decimals=2))
    print("每行之和（应全为 1）:", wei[0].sum(dim=-1).round(decimals=3).tolist())

    # ---- 调试技巧 2（关键验证）：第 0 个 token 的输出只能依赖它自己 ----
    # 改掉未来 token 的输入，第 0 个 token 的输出应该纹丝不动（证明没偷看未来）
    x2 = x.clone()
    x2[:, 1:, :] = torch.randn(B, T - 1, C)   # 把第 1..T 个 token 全换掉
    out2 = head(x2)
    diff = (out[:, 0, :] - out2[:, 0, :]).abs().max().item()
    print(f"\n改掉未来 token 后，第0个token输出的最大变化: {diff:.2e}  (应≈0，证明 causal mask 生效)")
    assert diff < 1e-6, "causal mask 失效！第0个token偷看了未来"
    print("[OK] causal mask 验证通过：信息只从过去流向未来")
```

**预期输出（节选）**：

```
输入  x  形状: (2, 8, 32)
输出 out 形状: (2, 8, 16)

第 0 句话的注意力权重矩阵（应是下三角，上三角=0）:
tensor([[1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.43, 0.57, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.31, 0.28, 0.41, 0.00, 0.00, 0.00, 0.00, 0.00],
        ...                                              ])
每行之和（应全为 1）: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

改掉未来 token 后，第0个token输出的最大变化: 0.00e+00  (应≈0，证明 causal mask 生效)
✅ causal mask 验证通过：信息只从过去流向未来
```

> **这个"改掉未来、看第0个token变不变"的测试，就是 Day3 计划里要写进 `tests/test_attention.py` 的核心断言**（计划 line 1696）。第一个 token 注意力权重必然是 `[1, 0, 0, ...]`——它只能看自己，所以输出 100% 等于自己的 V，符合直觉。


---

## 6. 缩放因子 1/√d：那个不起眼的 `* head_size**-0.5` 到底防什么

这是今天**第二个完成标准**，也是最容易被忽略却最能体现"懂行"的细节。

### 6.1 是什么：在算完点积后，除以 √d

第 4 行那个 `* head_size**-0.5`，head_size 就是 d（每个头的维度），`head_size**-0.5` 就是 `1/√d`。所以这行干的事是：

```
wei = (Q @ Kᵀ) / √d
```

### 6.2 为什么要除：不除的话，softmax 会"饱和"

核心问题在点积本身。回忆 `q_i · k_j` 是 d 个数对应相乘再相加。

**关键的统计直觉**：假设 q、k 的每个分量都是均值 0、方差 1 的随机数（初始化时大致如此）。那么：
- 它俩的点积 = d 项之和。每一项方差约为 1，d 个独立项相加，**总方差 ≈ d**，标准差 ≈ √d。
- 也就是说，**d 越大，点积的数值就摆得越开**——d=16 时点积大概在 ±4 晃，d=64 时在 ±8 晃，d=128 时能到 ±11。

现在把这些"摆得很开"的大数喂进 softmax 会发生什么？softmax 是 `e^x / Σe^x`，它对大数极其敏感。如果一行里有个数特别大（比如 11），`e^11` 会把其它项彻底压成 0：

```
softmax([2, 1, 0.5])         ≈ [0.59, 0.24, 0.17]   ← 健康，注意力分散，能学
softmax([11, 5, 3])          ≈ [0.998, 0.002, 0.000] ← 饱和！几乎全压给一个
```

**饱和（saturation）**的后果：注意力退化成"几乎只看一个 token"的硬选择（接近 one-hot）。而 softmax 在这种极端区域**梯度几乎是 0**——一旦饱和，这部分就学不动了，这就是**梯度消失（vanishing gradient）**。

### 6.3 怎么解决：除以 √d，把方差拉回 1

既然点积的标准差是 √d，那就除以 √d，**把方差从 d 重新归一化回 1**。这样不管 d 是 16 还是 128，喂进 softmax 的数值范围都稳定在 ±2 左右，softmax 工作在"健康、梯度充足"的区间。

做个对照实验（可以拷进上面的 `__main__` 里跑）：

```python
d = 64
q = torch.randn(8, d)
k = torch.randn(8, d)
print("不缩放，点积标准差:", (q @ k.T).std().item())        # ≈ 8  (=√64)
print("缩放后，点积标准差:", (q @ k.T * d**-0.5).std().item())  # ≈ 1
# 看 softmax 后第一行的"尖锐程度"（最大权重）：
print("不缩放 softmax 最大权重:", F.softmax(q @ k.T, dim=-1)[0].max().item())       # 往往 >0.9，饱和
print("缩放后 softmax 最大权重:", F.softmax(q @ k.T * d**-0.5, dim=-1)[0].max().item())  # 温和得多
```

### 6.4 工业锚点：这是"控制方差不让数值爆"思想的又一次出现

这个 1/√d **不是 attention 独有的魔法，而是你已经见过两次的同一招**：

| 出现位置 | 在控制什么的方差 | 手段 |
|---|---|---|
| W4 `init_and_stability.md`（Kaiming 初始化） | 让每层激活的方差保持 ≈1，不随层数爆炸/消失 | 权重乘 `1/√fan_in` |
| W5 BatchNorm / LayerNorm | 让每层输出重新归一化到均值0方差1 | 减均值除标准差 |
| **今天 attention 的 1/√d** | 让点积的方差保持 ≈1，softmax 不饱和 | 除以 `√d` |

**一句话收口**：从初始化、到归一化、到 attention 缩放，深度学习里反复出现的同一个工程母题就是——**"把流经网络的数值的方差死死摁在 1 附近"**。数值一爆（方差过大），要么梯度消失要么 NaN；一塌（方差过小），信息传不动。这个直觉是你做 AI Infra（尤其 FP16/混合精度训练稳定性）时天天要用的，今天它在 attention 里又露了一次脸。呼应计划 line 1690-1691。


---

## 7. O(T²)：为什么 Attention 是长上下文的"显存黑洞"

这是今天**第三个完成标准**，也是你 AI Infra 方向的第一性命题。这一节的精华会单独沉淀到 `tech_notes/attention_complexity.md`（§8 给全文）。

### 7.1 平方从哪来：那张 (T, T) 的表

回到 §2.3。`wei = Q @ Kᵀ` 的形状是 **(T, T)**。意思是：序列里有 T 个 token，**每个 token 都要和所有 T 个 token 算一次匹配度**，所以是 T×T = T² 次点积。

- 序列翻倍（T→2T），计算量和这张表的大小都变成 **4 倍**（不是 2 倍）。这就是"平方爆炸"。
- 对比你昨天的 WaveNet：那是 O(T) 线性的——每个 token 只看固定窗口的邻居。Attention 用 O(T²) 的代价换来了"一步到位的全局感受野"（任何 token 一层就能看到任何其它 token）。**这是一笔明确的交易：用平方的算力/显存，买全局视野。**

### 7.2 这个"平方"是算力问题还是显存问题？答案是：两个都是

很多人只记得"O(T²) 算力贵"，漏了更要命的另一半。

- **算力（compute）层面**：`Q @ Kᵀ` 是 T² 个点积，每个 d 维 → O(T²·d) 次乘加。T 大时这是实打实的 FLOPs 负担。
- **显存（memory）层面**：那张 (T, T) 的 `wei` 矩阵**要在显存里实际存下来**（前向要存它做 softmax，反向传播还要用它算梯度）。**它的大小本身就是 O(T²)**。

这第二点才是长上下文真正的杀手。下面手算一个具体数字，把"爆显存"从一句口号变成你能写在简历里的数字。

### 7.3 手算：T=1024 时一个注意力矩阵多大

单头、FP32（每个数 4 字节）：

```
单个注意力矩阵大小 = T × T × 4 字节
T=1024:   1024 × 1024 × 4  = 4 MB / 头
```

4MB 看着不大？乘上真实模型的规模：

```
总注意力显存 ≈ batch × layer 层数 × head 头数 × (T × T × 4)

举例 batch=8, 12 层, 12 头, T=1024:
   8 × 12 × 12 × 4MB ≈ 4.6 GB   ← 仅仅是注意力矩阵，还没算权重、激活、KV、优化器状态
```

再把 T 拉到长上下文，看平方的恐怖（固定其它，只看单头单层单 batch 的那张表）：

```
T=1024   (GPT-2):       4 MB
T=8192   (8K 上下文):    8192² × 4 ≈ 256 MB    ← T×8，显存×64
T=131072 (GPT-4 128K):  131072² × 4 ≈ 64 GB    ← 一张 A100(80G) 单卡装不下一个头的注意力矩阵
```

**结论**：上下文长度每翻一倍，注意力矩阵显存翻 4 倍。128K 上下文下，单张注意力矩阵就要 64GB——**这就是"长上下文爆显存"的精确来源**，也是为什么 FlashAttention、PagedAttention 这些技术存在的根本原因。

### 7.4 工业锚点：这条线串起你 AI Infra 的整张地图

今天这个 O(T²) 是个枢纽，往后所有方向都从它分叉出去（呼应计划 line 1692）：

| 问题 | 解法 | 你什么时候学 |
|---|---|---|
| 注意力矩阵 O(T²) 显存，存不下 | **FlashAttention**：不把整张 (T,T) 写进显存，分块在 SRAM 里算，边算边丢 | W8 / 暑假 |
| 生成时每步重算历史 K/V，O(T²) 算力浪费 | **KV Cache**：缓存历史 K/V，每步只算新 token，O(T²)→每步 O(T)（§3.3） | W6 Day6 + W8 |
| KV Cache 本身又占 O(T) 显存，碎片化 | **PagedAttention（vLLM）**：像 OS 分页一样管 KV 显存 | 暑假 vLLM 源码 |
| decode 阶段受 KV 读带宽限制 | 这是 **memory-bound**，呼应 W5 Roofline（`flops_vs_latency.md`） | W6 Day6 |

**一句话**：你目标方向（推理优化）里几乎每一个名词，根都扎在今天这张 (T, T) 的表上。今天把"为什么是 T²、它既费算力又费显存"想透，后面所有优化技术你都能问对"它到底在省哪一个 T"。


---

## 8. 常见陷阱与调试技巧（踩过才记得住）

1. **mask 填 0 而不是 -inf** —— softmax 后未来 token 还有权重，等于没挡住。必须填 `float('-inf')`，且在 softmax **之前**填。（§3.2）
2. **忘了缩放 `* head_size**-0.5`** —— 小模型可能看不出，d 大了训练 loss 不降或 NaN。（§6）
3. **`tril` 用普通属性而非 `register_buffer`** —— 模型 `.to('cuda')` 后报 device 不匹配，因为普通 tensor 不会跟着搬。（§5 代码注释）
4. **softmax 维度搞错** —— 必须 `dim=-1`（沿"被关注的 token"那一维归一化），写成 `dim=-2` 或 `dim=0` 注意力比例就全错。调试时永远 `print(wei.sum(dim=-1))` 看是不是全 1。
5. **`transpose` 转错维度** —— `transpose(-2,-1)` 只转后两维。写成 `.T`（转所有维）在带 batch 的 3D 张量上会把 B 维也转了，形状直接崩。（§4）
6. **训练时忘了切 dropout/eval 模式** —— 呼应你 Day1 §7 的 BN eval 坑，attention 里的 dropout 同理，推理前要 `model.eval()`。

> **黄金调试习惯**：写完 attention，永远先跑 §5 那个"改掉未来 token 看第0个输出变不变"的断言。这一个测试能一次性抓住 mask 错、维度错、缩放漏等大半的 bug。

---

## 9. 自测题（合上笔记，能脱口而出才算过）

1. 用一句大白话说清 self-attention 在让每个 token 干什么？（→ §2.1）
2. Q、K、V 各代表什么？为什么不能直接拿 token 向量自己跟自己比，非要拆三个矩阵？（→ §2.2）
3. 不看代码，口述单头 attention 的 **5 行计算流程**（q/k/v → 匹配 → mask → softmax → 加权）。（→ §4，**Day2 硬完成标准**）
4. causal mask 为什么长成下三角？为什么填 -inf 而不是 0？（→ §3.2）
5. `* head_size**-0.5` 防的是什么灾难？不加会怎样？它和 W4 Kaiming 初始化是同一个什么思想？（→ §6，**Day2 硬完成标准**）
6. 为什么 Attention 是 O(T²)？这个平方是算力问题还是显存问题？（→ §7.2，**Day2 硬完成标准**）
7. 手算：单头、T=1024、FP32，一个注意力矩阵多大？T 翻 8 倍显存翻几倍？（→ §7.3）
8. 下三角结构和 KV Cache 能成立之间，是什么逻辑关系？（→ §3.3）
9. Attention 为什么"看不见顺序"？这导致必须额外做什么？（→ §2.4）
10. 第 0 个 token 的注意力权重一定长什么样？为什么？（→ §5 末尾）

> 参考答案位置已在每题后标注。Q3/Q5/Q6 是计划明文要求"不看资料口述"的三条硬标准，反复练到张口就来。

---

## 10. 与已有笔记的串联

| 今天的内容 | 关联到你已有的 | 关系 |
|---|---|---|
| token / pos embedding（§1.2 ①②） | W4 `embedding_as_lookup.md`、Day1 embedding | 同一个"查表换向量"，pos 是新增的一张表 |
| 单头 attention 的 Head 类（§5） | Day1 `nn.Module` / `Sequential` 抽象 | 同一套乐高，Attention 是新积木 |
| 缩放 1/√d（§6） | W4 `init_and_stability.md` Kaiming | 同一个"控方差≈1"母题 |
| final LayerNorm（§1.2 ④） | W4 `batchnorm_inference.md` | Day4 正面讲"LM 为何弃 BN 用 LN" |
| O(T²) 既费算力又费显存（§7） | W5 Roofline `flops_vs_latency.md` | decode 是 memory-bound 的根 |
| 下三角 → 历史不变 → KV Cache（§3.3） | 计划 W6 Day6 + W8 推理优化 | 今天埋根，Day6 亲手实现 KV Cache |
| 残差流 `x + Attn(LN(x))` | W5 `residual_grad_flow.md` 的 +1 直觉 | Day4 把 Attention 装进 Block 时复用 |
| 多头 = 多个 Head 并排 | 明天 Day3 `MultiHeadAttention` | 今天的 Head 类明天直接 ×N |

---

## 11. 完成标准 checklist（对齐计划 line 1656-1697）

- [ ] EP6 第一遍完整看完（≈1h56min，1.5× 速，不写代码），能对照 §1.2 的 **5 个站点**复述数据流
- [ ] `src/attention_single.py` 跑通，§5 的 causal mask 断言通过（固定 seed=1337，可复现）
- [ ] 能**不看资料口述单头 attention 的 5 行计算流程**（§4，硬标准）
- [ ] 能解释**为什么是 O(T²)**、它既费算力又费显存（§7，硬标准）
- [ ] 能解释**缩放因子 1/√d 的作用**及与初始化的同源思想（§6，硬标准）
- [ ] 完成 `tech_notes/attention_complexity.md`（§12 给了全文，直接拷）
- [ ] 自测题 §9 合上笔记能答，尤其 Q3/Q5/Q6 三条硬标准
- [ ] `W6_day2_log.md` 记录：EP6 通看最大卡点 + attention 实现踩了哪个 §8 的坑

> **今天的一句话总结**：EP6 的 5 个站点你见过 4 个，唯一的新东西就是 Block 里的 Attention；而 Attention 的核心又只是 5 行——"每个 token 拿 Q 比所有人的 K，挡住未来，softmax 成比例，再按比例汇总 V"。把这 5 行连同 1/√d 缩放和 O(T²) 想透，你 AI Infra 主线（KV Cache、FlashAttention、长上下文）的地基就打好了。


---

## 12. 产出文件清单（本日交付）

今天的代码和技术笔记已落到项目目录（呼应你的"可复现"产出规范）：

```
week6_nanogpt/
├── src/
│   └── attention_single.py          # §5 完整可运行单头 attention（含 causal mask 验证）
├── tests/                           # （Day3 写 test_attention.py，今天先建好目录）
└── tech_notes/
    └── attention_complexity.md      # O(T²) 手算 + 1/√d 缩放，AI Infra 锚点
```

- `attention_complexity.md` 是 Day2 明文要求的技术笔记，主体即本笔记 §6 + §7 的浓缩速查版，已写好。
- `attention_single.py` 即 §5 代码，可直接 `python src/attention_single.py` 跑，看到 causal mask 断言通过。

> **下一步（Day3 缓冲日）**：今天的 `Head` 类明天原样 ×N 个并排，`torch.cat` 拼接再过一个输出投影，就是 `MultiHeadAttention`——多头不是新东西，是今天单头的复制粘贴。届时再把 §5 那个 causal 断言扩成正式的 `tests/test_attention.py`。
