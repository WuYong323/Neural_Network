# Week 6 · Day 3（2026-06-03，缓冲日）：Multi-Head Attention + Fused QKV

> **覆盖任务**（计划 line 1682-1709 区段 / W6 Day3 checklist line 1846-1850）：
> - [ ] DL：实现 `MultiHeadAttention` + `tests/test_attention.py`
> - [ ] DL：理解 **fused QKV**（为什么一次大 GEMM 比循环快）
> - [ ] 当日输出：`src/multihead_attention.py` + `W6_day3_log.md`
> - [ ] 完成标准：能画出多头注意力的数据流（切头 → 各自 attention → 拼接 → 投影），解释"fused QKV 为什么比循环快"和"causal mask 为什么是下三角"
>
> **阅读对象**：你自己——昨天（Day2）你已经把**单头 self-attention 的 5 行核心**吃透了（q/k/v → 匹配 → mask → softmax → 加权汇总），还手算了 O(T²) 的显存账。今天是缓冲日，**不学新概念**，只做两件事：① 把单头**复制粘贴成多头**（多头不是新东西，是单头并排）；② 借多头这个场景，正式攻下你 AI Infra 主线的一块硬骨头——**fused QKV（融合 QKV 投影）**：为什么"把一堆小矩阵乘法塞进一个大矩阵乘法"能快好几倍。
>
> **为什么这天对你（AI Infra 方向）含金量特别高**：多头本身是 10 分钟的事，但"fused QKV 为什么快"这个问题，是你第一次真正触碰 **GPU 性能的第一性原理**——kernel 启动开销、内存带宽、算力利用率。这套直觉你 W5 在 Conv-BN-ReLU 融合里见过雏形（计划 line 1545、1564 明确点名："本周 ConvBNReLU 融合思想 → 下周 fused QKV"），今天把它在 Transformer 里讲透，后面 FlashAttention、`torch.compile`、CUDA kernel 全都建在这块地基上。
>
> **本笔记的设计**：沿用 [[W6_Day2_EP6_nanoGPT_单头SelfAttention]] 的三段式——每节先讲"原理 + 直觉"，再给可直接拷进 `src/` 的可运行代码，最后写工业锚点（AI Infra 视角）。§2-§3 讲"怎么写多头"，§4 是今天的主菜（fused QKV 为什么快），§5-§6 是可运行代码 + 你今天的硬产出 `tests/test_attention.py`。

---

## 0. 学习目标（看完应能脱口而出）

1. 单头 attention 已经能"回看全文抓重点"了，**为什么还要搞多头**？多头多出来的能力，用一句大白话怎么说？
2. 多头是把模型"变大"了吗？`head_size = n_embd / n_head` 这个除法在说什么？（陷阱：多头几乎不增加总算力）
3. 多头的数据流四步是什么？（**切头 → 各自 attention → 拼接 → 投影**）最后那个"输出投影"为什么不能省？
4. **教学版**（`nn.ModuleList` 循环）和**工业版**（fused QKV）写出来的多头，**结果一样吗**？差别在哪？
5. 什么叫 **GEMM**？为什么说"GPU 本质就是一台 GEMM 机器"？
6. **今天的核心**：为什么把 Q/K/V 三个投影**拼成一个大矩阵乘一次**，比分开乘三次（甚至按头循环乘 N 次）快？给出**三个层面**的原因。
7. 手算：3 次独立投影 vs 1 次融合投影，**内存读取量**差多少？为什么这件事和你 W5 的 Roofline 是同一回事？
8. fused QKV 是"横向融合"，Conv-BN-ReLU 是"纵向融合"——这两种融合**省的是不一样的东西**，分别省什么？
9. 形状变换 `view → transpose → contiguous → view` 里，那个 `.contiguous()` 不写会怎样？为什么？

---

## 1. 先承上：从单头到多头，到底缺了什么

### 1.1 单头的天花板：一个"诉求"只能追一种关系

回忆昨天 §2.1 的**开会类比**：每个 token 拿着一个 **Query（查询，"我想找什么"）**，去比对所有人的 **Key（标签，"我是关于什么的"）**，按匹配度加权汇总大家的 **Value（值，"我携带的真实信息"）**。

但这里有个隐藏的局限：**一个头，只有一套 Q/K/V，也就只能追踪"一种关系"**。

举个具体例子。读这句话：

> "**The animal** didn't cross **the street** because **it** was too tired."

"**it**"（它）这个词，在算自己该看谁的时候，其实需要**同时追好几种不同的关系**：
- **指代关系**：it 指的是 animal 还是 street？（语义层面，得回看 "animal"）
- **语法关系**：it 是这个从句的主语，谓语是 "was"（句法层面，得看 "was tired"）
- **位置关系**：紧挨着 it 的前后词是什么（局部层面）

一个单头，**只能学会盯其中一种**。你让它盯指代，它就顾不上句法；让它学一个"折中的平均关系"，结果哪种都没学好。这就像开会时你**只带了一个问题**进场——你没法同时既追"预算"又追"排期"又追"人手"。

### 1.2 多头的直觉：同时开 N 个"分论坛"

**Multi-Head Attention（多头注意力）** 的解法简单粗暴：**那就同时开 N 个会（N 个头），每个头追一种关系，最后把各组的结论汇总。**

> **Multi-Head Attention（多头注意力，多头自注意力）**：把注意力机制并行地跑 N 份，每一份叫一个 **head（头）**，各自有**独立的** Q/K/V 投影矩阵，因此各自能学到一种不同的关注模式。N 个头算完后，把它们的输出**拼接（concatenate）**起来，再过一个**输出投影**揉成最终结果。

**类比**：单头 = 你一个人带一个问题去开大会，只能记一条线索。多头 = 公司开**并行分论坛**：A 组专门讨论"指代谁"，B 组专门盯"语法主谓"，C 组只关心"相邻词"……各组独立讨论完，最后到主会场把 N 份会议纪要**拼到一起**，再由一个主持人（输出投影）综合成一份总报告。每个头**专精一个视角**，合起来视角就全了。

真实模型里 N 不小：GPT-2 是 **12 个头**，GPT-3 有 **96 个头**。研究者做可视化时真的能看到不同头学出了不同分工——有的头专门连接代词和它的指代对象，有的头专门盯标点，有的头盯相邻词。这不是玄学，是可观测的现象。

### 1.3 关键认知（陷阱题）：多头不是"变大"，是"分工"

这是初学者最容易误解的点，也是 §0 第 2 问的答案。**多头几乎不增加总的参数量和算力**，它只是把同样大小的"表示空间"**切成 N 份让它们分工**。

看这个除法——它是整个多头机制的灵魂：

```
head_size（每个头的维度）= n_embd（总维度）/ n_head（头数）
例：n_embd = 384，n_head = 6  →  head_size = 64
```

也就是说，**不是**"每个头都用满 384 维、N 个头就 N 倍开销"。而是**把 384 维切成 6 段、每段 64 维，每个头只在自己那 64 维的子空间里干活**。算一下总账：

```
单个大头（384维）的 Q 投影参数：384 × 384
6 个小头（各64维）的 Q 投影参数：6 ×（384 × 64）= 384 × 384  ← 一模一样！
```

**总参数、总 FLOPs 几乎不变**（只多了最后一个输出投影矩阵）。多头买到的不是"更大的模型"，而是"**同样的预算下，把注意力拆成多个能各自专精的子空间**"。

> **一句话收口**：单头是"一个 384 维的大眼睛看一种关系"，多头是"六个 64 维的小眼睛各看一种关系，再汇总"。预算一样，视角更多。这就是多头唯一的、也是全部的价值。

---

## 2. Multi-Head Attention 实现（一）：教学版（ModuleList 循环）

> 先写最直观、最好懂的版本——Karpathy 在 EP6 里就是这么教的。它的好处是"一眼看穿多头就是单头并排"；它的坏处（慢）正是 §4 fused QKV 要解决的，所以这一版我们**故意先写慢的**，好对比。

### 2.1 数据流：记住四步

多头的数据流就四步，对应你完成标准要画的那张图：

```
   x: (B, T, C)   C = n_embd = 384
        │
        ▼  ①「切头」：复制给 N 个独立的 Head，每个头在 64 维子空间算
   ┌────────┬────────┬──── ... ──┬────────┐
   │ Head 0 │ Head 1 │           │ Head 5 │   ← ② 每个头各自跑昨天的单头 attention
   │(B,T,64)│(B,T,64)│           │(B,T,64)│      （各自的 q/k/v、各自的 causal mask）
   └────┬───┴───┬────┴──── ... ──┴───┬────┘
        └───────┴─── ③「拼接」cat ────┘
                     │  (B, T, 6×64) = (B, T, 384)
                     ▼
              ④「输出投影」proj: Linear(384, 384)
                     │
                     ▼
                  (B, T, 384)  最终输出
```

1. **切头**：同一个输入 `x`，分发给 N 个**互相独立**的 Head（每个 Head 内部有自己的 Wq/Wk/Wv，互不共享）。
2. **各自 attention**：每个 Head 就是你昨天写的那个单头 `Head` 类，原封不动，只是 head_size 从 384 变成 64。
3. **拼接（concatenate）**：N 个头各输出 `(B, T, 64)`，沿最后一维拼成 `(B, T, 384)`——相当于把 6 份会议纪要订在一起。
4. **输出投影（output projection）**：过一个 `Linear(384, 384)`，把"6 份拼起来的纪要"揉合成一份总结。

### 2.2 代码：复用昨天的 `Head`

昨天的 `Head` 类直接拿来用，多头只是把它 `×N` 装进一个 `nn.ModuleList`：

```python
# 教学版：直观但慢。对应 Karpathy EP6 的写法。
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """单头 self-attention——昨天 Day2 §5 的类，原样搬过来。"""
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)        # 各 (B,T,hs)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5     # (B,T,T) 匹配+缩放
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask（下三角）
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v                                             # (B,T,hs)


class MultiHeadAttentionLoop(nn.Module):
    """多头注意力·教学版：N 个独立 Head 并排，结果拼接再投影。"""
    def __init__(self, n_head, n_embd, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        head_size = n_embd // n_head                  # ← §1.3 的灵魂除法：切，不是加
        # ModuleList：N 个互相独立的 Head（各有自己的 Wq/Wk/Wv）
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        # ④ 输出投影：把拼接后的 (B,T,n_embd) 再揉一次。这一步不能省，原因见 §2.3
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ②③ 每个头各自算，再沿最后一维 cat 拼起来
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B,T,n_embd)
        # ④ 输出投影 + dropout
        out = self.dropout(self.proj(out))                    # (B,T,n_embd)
        return out
```

**注意那个 `for h in self.heads` 的列表推导**——它就是"慢"的根源：N 个头**一个一个排队算**，每个头内部又是 q、k、v **三次独立的小矩阵乘法**。一个 `MultiHeadAttentionLoop` 跑一次，要启动 `N × 3` 次以上的矩阵乘法 kernel。这正是 §4 要干掉的东西。先记住这个数字：**N 头 × 3 投影 = 一大堆小矩阵乘法**。

### 2.3 那个"输出投影"为什么不能省（§0 第 3 问）

很多人第一次写多头会想："拼接完不就齐了吗，最后那个 `self.proj` 是不是多余？" **不是，它是必需的，而且有明确职责。**

拼接（`torch.cat`）只是把 6 个头的输出**物理上排在一起**——像把 6 份纪要订成一摞，但它们之间还**没有任何信息交流**。第 0 维（前 64 维）完全来自 Head 0，第 1 段完全来自 Head 1，井水不犯河水。

输出投影 `Linear(384, 384)` 干的事，是**让这 6 个头的信息互相混合、加权**：输出的每一维，都是 384 个输入维（即 6 个头全部输出）的加权组合。**类比**：拼接是"把 6 个组的纪要订在一起"，输出投影是"主持人读完 6 份纪要，综合写出一份真正融会贯通的总报告"。没有这一步，下游拿到的只是 6 份各说各话的草稿，没人替它们做整合。

> **工业细节**：在 nanoGPT / GPT-2 里这个投影叫 `c_proj`，而且它的初始化会被**特意缩小**（乘 `1/√(2·n_layer)`），目的是控制残差通路上累加的方差——这又是昨天 §6、以及 W4 `init_and_stability.md` 那个"把方差摁在 1 附近"母题的延续。今天先知道"它管混合、不能省"，方差细节 Day4 装进 Block 时再碰。

---

## 3. Multi-Head Attention 实现（二）：工业版（一次切出所有头）

> 教学版好懂但慢。工业代码（nanoGPT 的 `CausalSelfAttention`、HuggingFace、vLLM）**没有人用 `for` 循环跑头**。它们用一个技巧：**所有头一次性算出来，靠 reshape 把"头"这一维变出来，再用一次批量矩阵乘法让所有头并行。** 这一版是 §4 fused QKV 的载体，务必看懂形状怎么变。

### 3.1 核心思想：头不是循环出来的，是 reshape 出来的

教学版把"头"实现成 N 个**对象**（ModuleList 里 N 个 Head）。工业版把"头"实现成**张量的一个维度**。

一句话：**先用一个大投影一次性算出全部 384 维的 Q（K、V 同理），再把这 384 维 `view` 成 (6 头 × 64 维)，把"头"这一维 `transpose` 提到前面当成 batch 的一部分，这样 PyTorch 的批量矩阵乘法会自动让 6 个头并行算 attention——一次 kernel，干掉循环。**

形状变换是这一版唯一的难点，画出来：

```
q: (B, T, 384)                          ← 一次大投影算出的完整 Q
   │ view(B, T, n_head=6, head_size=64)
   ▼
   (B, T, 6, 64)                        ← 把最后 384 维拆成 6 头 × 64
   │ transpose(1, 2)   把"头"维提到"序列"维前面
   ▼
   (B, 6, T, 64)                        ← 现在前两维 (B,6) 都被当批量维
   │
   ▼  对 (T,64) 这后两维做 attention，B×6 个 (T,64) 矩阵全部并行
   q @ k.transpose(-2,-1) → (B, 6, T, T)   ← 6 个头的注意力矩阵一次算完
```

关键洞察：PyTorch 的 `@`（`matmul`）对 **>2 维的张量，会把除最后两维之外的所有维都当成"批量维"并行处理**。所以一旦把形状摆成 `(B, 6, T, 64)`，那个 `(B, 6)` 就成了"一共 B×6 个独立的 (T,64) 小矩阵"，一条 `q @ kᵀ` 指令就让**所有头、所有 batch 全部并行**——这就是工业版"快"的第一层来源（消灭了 Python 的 for 循环）。

### 3.2 代码：工业版 `CausalSelfAttention`（含 fused QKV）

```python
class CausalSelfAttention(nn.Module):
    """工业版多头注意力（nanoGPT 同款）：fused QKV + reshape 出头维 + 批量 attention。
    和教学版 MultiHeadAttentionLoop 在数学上完全等价，但快得多（原因见 §4）。"""
    def __init__(self, n_head, n_embd, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        # ★ fused QKV：一个 Linear 同时算出 Q、K、V —— 输出维是 3*n_embd（关键！见 §4）
        #   不是三个 Linear(n_embd, n_embd)，而是一个 Linear(n_embd, 3*n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # 输出投影（§2.3 那个不能省的混合层）
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape          # C = n_embd
        # ① 一次大 GEMM 算出 Q/K/V 三件套，再切成三块
        qkv = self.c_attn(x)                       # (B, T, 3C) ← 唯一一次 QKV 投影
        q, k, v = qkv.split(self.n_embd, dim=-1)   # 各 (B, T, C)，沿最后一维三等分
        # ② reshape 出头维：(B,T,C) → (B,T,nh,hs) → (B,nh,T,hs)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)   # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # ③ 批量 attention：B×nh 个头全部并行（@ 自动把前两维当批量维）
        wei = (q @ k.transpose(-2, -1)) * hs ** -0.5        # (B, nh, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v                                       # (B, nh, T, hs)
        # ④ 合头：把头维 transpose 回去再 view 合并 → (B,T,C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # ← .contiguous() 不能少，见 §3.3
        # ⑤ 输出投影
        out = self.resid_dropout(self.c_proj(out))          # (B,T,C)
        return out
```

对照教学版，三个变化值得标注：
- **三个 `nn.Linear(n_embd, head_size)` × N 个头**，变成了**一个 `nn.Linear(n_embd, 3*n_embd)`**——这就是 **fused QKV**，§4 的主角。
- **`for h in self.heads`** 那个 Python 循环**没了**，头变成了张量里的 `nh` 维，靠 `@` 的批量语义并行。
- 多了一句 **`.contiguous()`**，§3.3 解释它为什么是个必踩的坑。

### 3.3 `.contiguous()` 这个坑（§0 第 9 问）

`out.transpose(1, 2)` 之后形状是对的 `(B,T,nh,hs)`，但**它在内存里并不是连续排列的**——`transpose` 只是改了"怎么解读这块内存"的索引（stride），没真的搬数据。紧接着的 `.view(B,T,C)` 要求**内存连续**才能工作，直接 view 会报错：

```
RuntimeError: view size is not compatible with input tensor's size and stride ...
              use .reshape(...) instead.
```

`.contiguous()` 的作用就是**真正按当前逻辑顺序把数据在内存里重排成连续的**，之后 `view` 才合法。

> **类比**：`transpose` 像图书馆给你一张"换了排序规则的索引卡"——书还在原来的架子上没动，只是你查的顺序变了。`view` 这个操作却要求"书必须按新顺序真的摆在连续的一排架子上"。`.contiguous()` 就是雇人**真的把书按新顺序搬到连续货架**。不搬（不写 contiguous），view 就找不到连续的一排，报错。
>
> **工业惯例**：要么像这里显式 `.contiguous().view(...)`，要么直接用 `.reshape(...)`（reshape 会在需要时自动帮你 contiguous）。两者等价，nanoGPT 用前者写得更显式。这是写多头 100% 会撞一次的坑，撞一次就记住了。

---

## 4. 【今日主菜】Fused QKV：为什么一次大 GEMM 比循环快

> 这是计划 line 1847 明文要求"理解"的知识点，也是你 AI Infra 直觉的**第一块基石**。前面都是铺垫，这一节要讲透。我们分三层讲：先讲清 GEMM 是什么（§4.1），再讲"融合"省了什么（§4.2-§4.4），最后手算一个数字（§4.5）+ 区分两种融合（§4.6）。

### 4.1 先认识 GEMM：GPU 的"母语"

> **GEMM（General Matrix Multiply，通用矩阵乘法）**：就是矩阵乘矩阵 `C = A @ B`（外加可选的缩放和加偏置）。名字唬人，本质就是你线代里的矩阵乘法。之所以单独给它起个大写缩写，是因为**它是深度学习里绝对的计算主体**——Linear 层是 GEMM，attention 的 `q@kᵀ` 和 `wei@v` 是 GEMM，卷积也能转化成 GEMM。

**关键事实：GPU 本质上就是一台为 GEMM 极致优化的机器。** NVIDIA 的 Tensor Core（张量核心）就是专门用来狂算矩阵乘法的硬件单元。一块 H100 的算力高达每秒近千万亿次浮点运算（接近 1000 TFLOPS），**但这个恐怖算力只有喂给它"又大又规整的矩阵乘法"时才发挥得出来**。喂一堆零碎小矩阵，它大部分时间在空转。

记住这句话，它是理解今天一切的钥匙：**GPU 不怕算得多，怕的是任务太碎、太小。**

### 4.2 "循环 N 个头 + 分开算 QKV" 到底慢在哪：三层开销

回到 §2 教学版那个 `for h in self.heads`，每个头里又 `q/k/v` 分三次算。假设 N=6 个头，那么光是 QKV 投影，就要发起 **6 头 × 3 个矩阵 = 18 次**小矩阵乘法（`(B,T,384) @ (384,64)`）。慢在三个层面：

**① kernel 启动开销（kernel launch overhead）——"开工费"**

> **kernel（核函数）**：一段在 GPU 上跑的小程序。每做一次矩阵乘法，CPU 都要"叫" GPU 启动一个 kernel——这个"叫一声"的动作本身有固定开销（微秒级），与算多少无关，**纯粹是开工的手续费**。

发起 18 次小 GEMM = 交 18 次开工费。融合成 1 次大 GEMM = 只交 1 次。**类比**：你要复印 18 份不同的文件，每次都得"走到复印机前、刷卡、登录"——这套手续做 18 遍。如果能把 18 份拼成 1 个大任务一次提交，手续只走 1 遍。任务越小、数量越多，这笔"手续费"占比越离谱。

**② 内存带宽浪费（memory bandwidth）——"重复搬同一批货"**

这是更深、更值钱的一层，也是和你 W5 Roofline 直接接轨的地方。

Q、K、V 三个投影，**输入是同一个 `x`**（都是那个 `(B,T,384)`）。分开算三次，意味着这个 `x` 要**从显存（HBM）里被读取三遍**：

```
分开算：读 x（算Q）→ 读 x（算K）→ 读 x（算V）   x 被搬了 3 趟
融合算：读 x 一遍 → 一次大乘法同时产出 Q|K|V    x 只搬 1 趟
```

> **为什么"少搬几趟"这么关键？** 你 W5 的 Roofline（`flops_vs_latency.md`）讲过：现代 GPU 的**算力远远过剩，瓶颈往往是显存带宽**——数据从显存搬到计算单元的速度，赶不上计算单元的胃口。所以很多操作不是"算得慢"，是"**喂得慢**"，专业词叫 **memory-bound（受内存带宽限制）**。QKV 投影正是这种：矩阵本身不大，算起来很快，**搬运输入的开销占比反而高**。融合后输入只读一遍，省的就是这笔搬运。

**③ 算力利用率低（GPU 利用率）——"大厨切土豆丝"**

GPU 有成千上万个核心。一个 `(B,T,384)@(384,64)` 的小矩阵乘法，可能只喂饱了一小部分核心，剩下的在围观。算 18 个这种小活，每个都喂不满，整体利用率上不去。

把 QKV 融合成 `(B,T,384)@(384,1152)`（1152 = 3×384），矩阵变"宽"了，一次就能铺满更多核心，Tensor Core 也更容易进入高效模式。**类比**：让米其林大厨（GPU）切一根土豆丝，他大材小用、还得反复洗手换砧板（启动开销）；给他一筐土豆一次切，他的刀工才真正跑满。GPU 同理——**喂大块、规整的活，它才高兴。**

### 4.3 融合到底是什么操作：把三个矩阵"横着拼"

直觉上 fused QKV 就一句话：**把 Wq、Wk、Wv 三个 `(384, 384)` 的权重矩阵，沿输出维拼成一个 `(384, 1152)` 的大矩阵，一次乘法同时算出 Q、K、V，再切三刀分开。**

```
分开（3 次小 GEMM）：
   Q = x @ Wq      x:(B,T,384)  Wq:(384,384) → Q:(B,T,384)
   K = x @ Wk                   Wk:(384,384) → K:(B,T,384)
   V = x @ Wv                   Wv:(384,384) → V:(B,T,384)

融合（1 次大 GEMM）：
   W_qkv = [Wq | Wk | Wv]                    拼成 (384, 1152)
   QKV   = x @ W_qkv          → (B,T,1152)   一次算完
   Q, K, V = QKV.split(384, dim=-1)          切三刀
```

数学上**完全等价**——拼起来乘再切，和分开乘，每个数字一模一样（§6 的测试会用 `allclose` 严格验证这一点）。变的只是**计算的组织方式**：从"三趟小活"变成"一趟大活"，省下的是 §4.2 那三层开销。代码里就是 §3.2 的那个 `nn.Linear(n_embd, 3*n_embd)`——一个 Linear 顶三个。

### 4.4 一张表收束：fused QKV 快在哪

| 维度 | 分开算 Q/K/V（×N 头循环） | fused QKV（1 次大 GEMM） | 省了什么 |
|---|---|---|---|
| kernel 启动次数 | 多（N×3+ 次开工费） | 1 次 | **启动开销** |
| 输入 x 读取次数 | 3 遍（每个投影各读一遍） | 1 遍 | **内存带宽** |
| 单次矩阵大小 | 小、瘦 | 大、宽 | **算力利用率** |
| GPU 满意度 | 任务碎，核心喂不饱 | 任务大，跑满 Tensor Core | **吞吐** |

### 4.5 手算：内存读取量差多少（§0 第 7 问）

把"省内存带宽"变成一个具体数字，写进你的 log。设 `B=8, T=1024, n_embd=384`，FP32（每数 4 字节）。输入 `x` 的大小：

```
x 大小 = B × T × n_embd × 4 = 8 × 1024 × 384 × 4 ≈ 12 MB
```

- **分开算 Q/K/V**：x 被读 3 遍 → 搬运 `3 × 12 = 36 MB`
- **fused QKV**：x 读 1 遍 → 搬运 `12 MB`
- **仅输入这一项，融合就省下 2/3 的读取量（24 MB）**

这还只算了输入。再叠加 §4.2① 的 kernel 启动开销（教学版按头循环要发 N×3 次，融合 1 次）、以及更宽矩阵带来的算力利用率提升，**真实加速通常是 2-4 倍**（取决于硬件、序列长度、是否被其它部分掩盖）。

> **和 Roofline 接轨（呼应 W5 `flops_vs_latency.md`）**：QKV 投影的"计算量 / 读取量"比值（arithmetic intensity，算术强度）本来就偏低，是个偏 memory-bound 的操作。fused QKV 没有减少计算量（FLOPs 一点没省），它**减少的是分母——内存读取量**，于是算术强度上升，把操作从"被带宽卡住"往"被算力卡住"那一侧推。这正是你 W5 Roofline 模型预测的优化方向：**memory-bound 的操作，优化重点是减少数据搬运，而不是减少计算。** 今天 fused QKV 就是这条原理的第一个活例子。

### 4.6 两种融合别搞混：横向 vs 纵向（§0 第 8 问）

你 W5 学过 **Conv-BN-ReLU 融合**（`lenet_vs_modern.md`），今天又学 fused QKV。它们都叫"融合"，但**省的是不一样的东西**，这是个高频混淆点，必须分清：

| | **fused QKV（横向融合）** | **Conv-BN-ReLU（纵向融合）** |
|---|---|---|
| 融合方向 | **横向**：三个**并列、互相独立**的操作（Q/K/V 谁也不依赖谁）拼成一个 | **纵向**：三个**串行、前后依赖**的操作（Conv→BN→ReLU 一个接一个）合成一个 |
| 为什么能融 | 它们**输入相同**（都吃 x），所以能共享输入读取、合并成一个大矩阵 | 后一个的输入是前一个的输出，融合后**中间结果不落显存**，直接在寄存器里传给下一步 |
| 主要省 | 省**输入的重复读取** + 启动开销 + 提利用率 | 省**中间结果的写出与读回**（intermediate activation）+ 启动开销 |
| 一句话 | "**几个人吃同一份原料，合并采购**" | "**流水线上工序首尾相接，半成品不下线**" |

**共同的母题**：两种融合，省的都是**内存搬运（memory traffic）和 kernel 启动开销**，而**不是浮点计算量（FLOPs 不变）**。这再次印证 §4.5 的 Roofline 直觉——深度学习的推理优化，**大量工作是在省"搬运"而非省"计算"**。你把这两种融合的区别讲清楚，就摸到 AI Infra 算子融合（operator fusion）的门了；这也是 `torch.compile`、TensorRT 这些编译器在图上自动找的两类机会，W8 会正面接触。

## 5. 完整可运行代码：`src/multihead_attention.py`

下面是可直接拷进 `week6_nanogpt/src/multihead_attention.py` 跑的完整文件。它做三件事：① 教学版 + 工业版两个实现并存；② 一个**数值等价性检查**（证明两种写法结果一致）；③ 一个**实测加速 benchmark**（亲手量出 fused 比循环快多少，这就是你简历里"实测加速"的同款方法）。

```python
# src/multihead_attention.py
# 运行环境：Python 3.10+，PyTorch 2.x（CPU 可跑；有 CUDA 则自动测 GPU 加速更明显）
# 运行：python src/multihead_attention.py
# 依赖：pip install torch
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)  # 固定种子，可复现（延续你的产出规范）


class Head(nn.Module):
    """单头 self-attention（Day2 §5 原样搬来，多头复用它）。"""
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttentionLoop(nn.Module):
    """教学版：N 个独立 Head 循环 + cat + 投影。直观但慢。"""
    def __init__(self, n_head, n_embd, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class CausalSelfAttention(nn.Module):
    """工业版：fused QKV + reshape 出头维 + 批量 attention（nanoGPT 同款）。"""
    def __init__(self, n_head, n_embd, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)  # ★ 一个 Linear 顶三个
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)   # 1 次大 GEMM 出 QKV
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)     # (B,nh,T,hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) * hs ** -0.5          # (B,nh,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v                                         # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 合头（contiguous 必需）
        return self.resid_dropout(self.c_proj(out))
```

紧接着同一个文件里放 `__main__`：先验证形状、再做一个 fused-vs-loop 的加速实测。

```python
if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, n_embd, n_head, block_size = 8, 256, 384, 6, 256
    x = torch.randn(B, T, n_embd, device=device)

    # ---- 1. 形状自检：输入输出形状必须一致 (B,T,n_embd) ----
    mha = CausalSelfAttention(n_head, n_embd, block_size).to(device)
    out = mha(x)
    print("输入形状:", tuple(x.shape), "→ 输出形状:", tuple(out.shape))
    assert out.shape == (B, T, n_embd), "多头不应改变 (B,T,C) 形状"

    # ---- 2. 加速实测：fused（工业版）vs loop（教学版）----
    # 注意：CPU 上差距偏小；GPU 上 kernel 启动 + 带宽优势才充分显现
    loop = MultiHeadAttentionLoop(n_head, n_embd, block_size).to(device).eval()
    fused = CausalSelfAttention(n_head, n_embd, block_size).to(device).eval()

    def bench(model, x, iters=200):
        with torch.no_grad():
            for _ in range(20):           # 预热（warmup）：让 GPU 进入稳定频率、缓存就绪
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()  # GPU 是异步的，计时前必须同步，否则量到的是假数
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()  # 同上：等所有 kernel 真跑完再停表
            return (time.perf_counter() - t0) / iters * 1e3   # 毫秒/次

    t_loop  = bench(loop, x)
    t_fused = bench(fused, x)
    print(f"\n设备: {device}")
    print(f"教学版(循环) : {t_loop:.3f} ms/次")
    print(f"工业版(fused): {t_fused:.3f} ms/次")
    print(f"加速比       : {t_loop / t_fused:.2f}x  ← fused QKV 的实测收益")
```

**预期输出（数字随机器变，趋势一致；fused 一定更快）**：

```
输入形状: (8, 256, 384) → 输出形状: (8, 256, 384)

设备: cuda
教学版(循环) : 0.812 ms/次
工业版(fused): 0.241 ms/次
加速比       : 3.37x  ← fused QKV 的实测收益
```

> **两个计时硬规矩（写错就量到假数据，工业 benchmark 必踩）**：
> 1. **预热（warmup）**：头几次跑包含 GPU 升频、cuDNN 选算法、缓存冷启动等一次性开销，不能算进计时。先空跑 20 次再开表。
> 2. **`torch.cuda.synchronize()`**：GPU 调用是**异步**的——CPU 发完指令立刻返回，kernel 还在后台跑。不同步就停表，你量到的是"发指令的时间"而非"算完的时间"，数字会假得离谱（看着飞快其实没算完）。计时的开始和结束都要 sync。这个坑你 W5 用 profiler 时其实已经间接碰过，今天明确写下来。

---

## 6. 今日硬产出：`tests/test_attention.py`

> 这是计划 line 1846 明文要求的交付物。昨天 §5 那个"改掉未来 token 看第 0 个输出变不变"的断言，今天**升级成正式的 pytest 测试文件**。一个好的 attention 测试要守住三条底线：**形状对、因果性对（不偷看未来）、两种实现数值等价**。

把下面文件拷进 `week6_nanogpt/tests/test_attention.py`：

```python
# tests/test_attention.py
# 运行环境：Python 3.10+，PyTorch 2.x，pytest
# 运行：在 week6_nanogpt/ 下执行  pytest tests/test_attention.py -v
# 依赖：pip install torch pytest
import torch
from src.multihead_attention import (
    MultiHeadAttentionLoop, CausalSelfAttention,
)

torch.manual_seed(1337)
B, T, N_EMBD, N_HEAD, BLOCK = 4, 16, 64, 4, 16


def test_output_shape():
    """底线1：多头不改变 (B,T,C) 形状——下游残差连接 x+Attn(x) 才能相加。"""
    x = torch.randn(B, T, N_EMBD)
    mha = CausalSelfAttention(N_HEAD, N_EMBD, BLOCK)
    assert mha(x).shape == (B, T, N_EMBD)


def test_head_divides_embd():
    """底线2：n_embd 不能被 n_head 整除时必须显式报错，别静默算错。"""
    import pytest
    with pytest.raises(AssertionError):
        CausalSelfAttention(n_head=5, n_embd=64, block_size=BLOCK)  # 64/5 不整除


def test_causal_no_peeking():
    """底线3（最重要）：因果性——第 t 个 token 的输出只依赖 0..t，改未来不影响过去。
    这是 attention 正确性的命门，也是 KV Cache 能成立的前提（Day2 §3.3）。"""
    mha = CausalSelfAttention(N_HEAD, N_EMBD, BLOCK).eval()
    x = torch.randn(B, T, N_EMBD)
    with torch.no_grad():
        out = mha(x)
        # 把第 t+1..T 的输入全换掉，第 t 个位置的输出必须纹丝不动
        t = 5
        x2 = x.clone()
        x2[:, t + 1:, :] = torch.randn(B, T - t - 1, N_EMBD)
        out2 = mha(x2)
        diff = (out[:, t, :] - out2[:, t, :]).abs().max().item()
    assert diff < 1e-5, f"因果性被破坏！第{t}个token偷看了未来 (diff={diff:.2e})"


def test_attention_weights_are_probabilities():
    """注意力权重每一行必须是合法概率分布：非负、和为 1。"""
    mha = CausalSelfAttention(N_HEAD, N_EMBD, BLOCK).eval()
    x = torch.randn(B, T, N_EMBD)
    with torch.no_grad():
        q, k, v = mha.c_attn(x).split(N_EMBD, dim=-1)
        hs = N_EMBD // N_HEAD
        q = q.view(B, T, N_HEAD, hs).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) * hs ** -0.5
        wei = wei.masked_fill(mha.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
    assert torch.all(wei >= 0), "注意力权重出现负数"
    row_sums = wei.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "每行未归一化到1"


def test_loop_equals_fused():
    """fused QKV 是性能优化，不能改变数学结果。共享同一套权重时，
    教学版(循环)和工业版(fused)的输出必须逐元素相等。"""
    loop  = MultiHeadAttentionLoop(N_HEAD, N_EMBD, BLOCK).eval()
    fused = CausalSelfAttention(N_HEAD, N_EMBD, BLOCK).eval()
    _copy_weights_loop_to_fused(loop, fused)   # 把循环版的权重塞进 fused 版
    x = torch.randn(B, T, N_EMBD)
    with torch.no_grad():
        assert torch.allclose(loop(x), fused(x), atol=1e-5), "两种实现结果不一致！"


def _copy_weights_loop_to_fused(loop, fused):
    """把教学版 N 个头的 Wq/Wk/Wv 按 [全部Q | 全部K | 全部V] 拼进 fused 的 c_attn。
    这一步本身就是 fused QKV '把多个小矩阵横向拼成大矩阵' 的代码化演示（§4.3）。"""
    qs = [h.query.weight for h in loop.heads]   # 每个 (head_size, n_embd)
    ks = [h.key.weight   for h in loop.heads]
    vs = [h.value.weight for h in loop.heads]
    # fused 的 c_attn.weight 形状 (3*n_embd, n_embd)，按 Q块|K块|V块 纵向堆叠
    w = torch.cat([torch.cat(qs, 0), torch.cat(ks, 0), torch.cat(vs, 0)], dim=0)
    with torch.no_grad():
        fused.c_attn.weight.copy_(w)
        fused.c_proj.weight.copy_(loop.proj.weight)
```

**预期输出**：

```
tests/test_attention.py::test_output_shape PASSED
tests/test_attention.py::test_head_divides_embd PASSED
tests/test_attention.py::test_causal_no_peeking PASSED
tests/test_attention.py::test_attention_weights_are_probabilities PASSED
tests/test_attention.py::test_loop_equals_fused PASSED
========================= 5 passed in 0.4s =========================
```

> **为什么 `test_loop_equals_fused` 是今天最有价值的测试**：它用代码**钉死**了 §4.3 的论断——fused QKV 只是"换了组织方式的同一笔运算"，结果必须逐元素相等。那个 `_copy_weights_loop_to_fused` 里把 N 个头的小权重 `torch.cat` 成大矩阵的动作，本身就是 fused QKV 的本质演示。**写优化代码时，这类"优化版 vs 朴素版数值等价"的回归测试是工业标配**——你优化 kernel、做量化、写 CUDA 时，第一件事永远是先有个"朴素正确版"当对照，确保优化没改变语义。这是 AI Infra 的基本职业素养，今天第一次亲手写。

## 7. 工业延伸：fused QKV 之后，这条优化线通向哪

> 缓冲日不展开新内容，但给你的 AI Infra 地图标几个路标——今天 fused QKV 是这条线的起点，让你知道往后在优化什么。

- **MQA / GQA（多查询 / 分组查询注意力）**：fused QKV 是"省投影开销"，而推理时真正的显存大头是 **KV Cache**（Day5-6 主攻）。Llama-2 70B 用 **GQA**：让多个 Q 头**共享同一组 K/V 头**，K/V 的数量从"每头一份"砍到"每组一份"，KV Cache 显存直接砍掉数倍。它和 fused QKV 是同一个家族的优化——都在"Q/K/V 怎么投影、存多少"上做文章。
- **FlashAttention（W8/暑假）**：fused QKV 融的是**投影**那一步；FlashAttention 融的是后面 `q@kᵀ → softmax → @v` 那三步，让中间那张 O(T²) 的注意力矩阵**根本不写进显存**（昨天 §7 的显存黑洞，FlashAttention 就是来堵它的）。两者一前一后，是 attention 上最重要的两类融合。
- **`torch.compile`（W8）**：你今天手动把三个投影拼成一个，是**手工融合**。`torch.compile` 会在计算图上**自动**找这类融合机会。今天手动做一遍，是为了让你 W8 看 `torch.compile` 的 fusion 日志时，能认出"哦，它自动干的就是我 Day3 手动干的事"。

**一句话**：今天你学会了第一个"算子融合"的真实案例。从这里出发，GQA、FlashAttention、编译器自动融合，都是同一套"省搬运、省启动、喂大块给 GPU"的思想在不同位置的展开。

---

## 8. 常见陷阱与调试技巧（踩过才记得住）

1. **`n_embd` 不能被 `n_head` 整除** —— `head_size = n_embd // n_head` 会向下取整，悄悄丢维度。必须 `assert n_embd % n_head == 0`，让它当场报错而不是静默算错。（§3.2、`test_head_divides_embd`）
2. **合头时漏掉 `.contiguous()`** —— `transpose` 后内存不连续，直接 `.view()` 抛 stride 错误。要么 `.contiguous().view()`，要么用 `.reshape()`。（§3.3）
3. **fused QKV 的 split 维度搞错** —— `qkv.split(n_embd, dim=-1)` 必须沿最后一维、且按 `n_embd` 等分。写成 `dim=1` 或切错大小，Q/K/V 会被切串，结果全错且不报错。
4. **benchmark 没预热 / 没 synchronize** —— GPU 异步 + 冷启动，不处理这两点量到的全是假数。计时前 warmup、计时首尾 `torch.cuda.synchronize()`。（§5）
5. **以为多头能 N 倍提升表达力，于是猛加头** —— 头数翻倍，每个头的 head_size 就减半（总维度不变），头太多反而每个头维度太小、学不动。GPT 系列 head_size 一般稳定在 64 左右，是经验最优区。（§1.3）
6. **输出投影 `c_proj` 忘了写** —— 模型能跑、loss 也降，但少了跨头信息融合，效果偷偷变差。这类"不报错但变差"的 bug 最难查（呼应 Day1 §6 那个 BN 维度坑，同一类阴险 bug）。（§2.3）

> **黄金调试习惯**：写完多头，先跑 §6 的 `test_causal_no_peeking`（因果性）和 `test_loop_equals_fused`（等价性）。前者守正确性命门，后者守"优化没改变语义"——这两条绿了，多头基本就对了。

---

## 9. 自测题（合上笔记，能脱口而出才算过）

1. 单头已经能抓重点了，为什么还要多头？多出来的能力用一句大白话怎么说？（→ §1.1-§1.2）
2. `head_size = n_embd / n_head` 这个除法说明多头是"变大"还是"分工"？多头增加总算力吗？（→ §1.3，**陷阱**）
3. 不看代码，画出多头数据流四步，并说清最后那个输出投影为什么不能省。（→ §2.1、§2.3，**完成标准**）
4. 什么是 GEMM？为什么说"GPU 是一台 GEMM 机器"、它"怕碎不怕多"？（→ §4.1）
5. fused QKV 为什么比循环快？给出**三个层面**的原因（启动开销 / 内存带宽 / 利用率）。（→ §4.2，**计划硬要求**）
6. 手算：B=8,T=1024,n_embd=384,FP32，分开算 vs fused，输入 x 的读取量各多少？省了多少？（→ §4.5）
7. fused QKV（横向融合）和 Conv-BN-ReLU（纵向融合）分别省什么？它们的共同点是"不省什么"？（→ §4.6）
8. 工业版里 `transpose` 之后为什么必须 `.contiguous()` 才能 `.view()`？（→ §3.3）
9. 为什么 fused QKV 不改变数学结果？你打算用什么测试来钉死这一点？（→ §4.3、§6 `test_loop_equals_fused`）
10. benchmark GPU 时，warmup 和 `synchronize` 各防的是什么假数据？（→ §5）

> 参考答案位置已标注。Q3/Q5 是计划明文要求的完成标准，练到张口就来。

---

## 10. 与已有笔记的串联

| 今天的内容 | 关联到你已有的 | 关系 |
|---|---|---|
| 多头 = N 个单头并排（§2） | Day2 `attention_single.py` 的 `Head` 类 | 昨天的 Head 今天原样 ×N，多头不是新东西 |
| `nn.ModuleList` / 输出投影 Module | Day1 `nn.Module` / `Sequential` 抽象 | 同一套乐高，多头是新积木 |
| fused QKV 省内存搬运（§4.2、§4.5） | W5 Roofline `flops_vs_latency.md` | memory-bound：优化重点是减搬运不是减计算 |
| fused QKV（横向）vs Conv-BN-ReLU（纵向）（§4.6） | W5 `lenet_vs_modern.md` 算子融合 | 两类融合对照，共同母题=省搬运+启动 |
| 输出投影初始化缩放（§2.3 注） | Day2 §6 的 1/√d、W4 `init_and_stability.md` | 同一个"把方差摁在 1"母题 |
| causal mask 下三角（§3.2、§6 测试） | Day2 §3 单头 causal mask | 多头每个头各自 mask，下三角不变 |
| 因果性 → KV Cache 前提（§6 测试） | Day2 §3.3、计划 W6 Day6 | 今天用测试钉死因果性，Day6 实现 KV Cache |
| GQA / FlashAttention / torch.compile（§7） | 计划 W8 + 暑假 vLLM/CUDA | 今天 fused QKV 是这条融合优化线的起点 |
| 多头数据流四步 | 明天 Day4 把 Attention 装进 Block | 多头是 Block 两条残差通路之一 |

---

## 11. 完成标准 checklist（对齐计划 line 1682-1709 / 1846-1850）

- [ ] `src/multihead_attention.py` 跑通：形状自检通过 + 实测打印出 fused 比 loop 的加速比（固定 seed=1337，可复现）
- [ ] `tests/test_attention.py` 写完，`pytest -v` **5 条全绿**（形状 / 整除断言 / 因果性 / 概率分布 / 循环=fused 等价）——计划 line 1846 硬产出
- [ ] 能**不看资料画出多头数据流四步**（切头 → 各自 attention → 拼接 → 投影），讲清输出投影为什么不能省（§2，完成标准）
- [ ] 能讲清 **fused QKV 为什么比循环快**的三个层面（启动开销 / 内存带宽 / 利用率），并手算出输入读取量省 2/3（§4，**计划 line 1847 硬要求**）
- [ ] 能区分**横向融合（fused QKV）vs 纵向融合（Conv-BN-ReLU）**分别省什么（§4.6）
- [ ] 能复述 **causal mask 为什么是下三角**（§3.2，完成标准；承接 Day2）
- [ ] 自测题 §9 合上笔记能答，尤其 Q3/Q5 两条硬标准
- [ ] `W6_day3_log.md` 记录：实测加速比是多少 + 多头实现踩了 §8 哪个坑（大概率是 `.contiguous()` 或 split 维度）

> **今天的一句话总结**：多头注意力 = 把昨天的单头复制 N 份让它们分工（预算不变、视角变多），写法上是"切头→各自算→拼接→投影"四步。而真正值钱的是 **fused QKV**——把 Q/K/V 三个投影拼成一次大 GEMM，靠"少交开工费、少搬输入、喂大块给 GPU"换来 2-4 倍加速。它是你 AI Infra 主线**算子融合**的第一个真实案例，和 W5 的 Conv-BN-ReLU 一横一纵，共同指向那条贯穿始终的原理：**推理优化大量是在省"搬运"，不是省"计算"。**

---

## 12. 产出文件清单（本日交付）

```
week6_nanogpt/
├── src/
│   ├── attention_single.py        # Day2：单头（多头复用它的 Head）
│   └── multihead_attention.py     # 今天：教学版 + 工业版(fused QKV) + 加速 benchmark
├── tests/
│   └── test_attention.py          # 今天硬产出：5 条测试（含 loop==fused 等价性）
└── tech_notes/
    └── attention_complexity.md    # Day2 已写，今天不动
```

- `multihead_attention.py` 即 §5，可 `python src/multihead_attention.py` 跑，看到形状自检 + fused 加速比。
- `test_attention.py` 即 §6，在 `week6_nanogpt/` 下 `pytest tests/test_attention.py -v`，应 5 条全绿。

> **下一步（Day4，本周最重要的一天）**：今天的 `CausalSelfAttention` 明天会和 FeedForward、两个 LayerNorm 一起，装进一个 **Block**（`x + Attn(LN(x))` + `x + FFN(LN(x))` 两条残差通路），再堆 N 层组装成完整 GPT。多头是 Block 的左半边，明天把右半边和残差流补齐，nanoGPT 的骨架就立起来了。
