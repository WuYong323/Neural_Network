# Week 6 · Day 6（2026-06-06，周六）：KV Cache 实现 + prefill/decode 两阶段 + profiler 确认 memory-bound

> **覆盖任务**（计划 line 1761-1791 / W6 Day6 checklist line 1871-1878）：
> - [ ] DL：**实现 KV Cache**，朴素 vs KV Cache 生成 **256 token 测速对比**
> - [ ] DL：完成 `tech_notes/kv_cache_and_two_phases.md`（prefill/decode + 显存代价 + batching）
> - [ ] DL：profiler chrome trace 跑生成，**确认 decode 阶段 memory-bound**
> - [ ] 当日输出：`src/kv_cache.py` + 加速对比数据 + `logs/nanogpt_trace.json` + `W6_day6_log.md`
> - [ ] 完成标准：KV Cache 跑通且实测加速；能脱稿讲清"prefill 为什么 compute-bound、decode 为什么 memory-bound"，以及"KV Cache 的显存代价为什么催生了 PagedAttention"。
>
> **阅读对象**：你自己——昨天（Day5）你已经写完 `generate()` 朴素自回归，并在 `kv_cache_motivation.md` 里用计时曲线**亲眼看到**了朴素生成的"耗时 vs 长度"是超线性（接近二次）增长，得出结论"历史 token 的 K/V 被重复计算了"。今天把这个**动机**变成**实现**：亲手写出 KV Cache、实测它在生成 256 token 时到底快多少；然后把这次实测拔高成一套能讲给面试官听的理论——**prefill/decode 两阶段、KV Cache 的显存账、batching 为什么提吞吐**；最后用 profiler 这把"听诊器"对准生成过程，**用真实 trace 证明 decode 阶段确实卡在内存带宽上**。
>
> **为什么这天是你简历级产出**：双非大一能从 0 手写 Transformer 已经稀有，能**亲手实现 KV Cache + 实测加速 + 用 profiler 证明瓶颈**，是夏令营套磁可以直接讲的硬作品（呼应 [[project-study-plan]] 的"可验证产出替学校背书"）。更关键的是——它直接踩在你 [[project-xiaomi-research]] 的主线上：小米揭榜课题的四条主线第 1 条就是"推理性能画像与瓶颈分析、识别 decode 阶段访存瓶颈"，你今天用 profiler 干的就是这件事的入门版。
>
> **本笔记的设计**：沿用 [[W6_Day3_MultiHeadAttention_FusedQKV]] 的三段式——每节先"原理 + 直觉"，再给可直接拷进 `src/` 的可运行代码，最后写工业锚点。§1-§2 讲清"为什么要缓存、缓存什么、为什么安全"；§3-§4 是动手主菜（改造 attention + 完整的 256 token 测速脚本）；§5-§7 是 `kv_cache_and_two_phases.md` 的理论核心（两阶段 / 显存代价 / batching）；§8 是 profiler 实证。

---

## 0. 学习目标（看完应能脱口而出）

1. 朴素自回归生成，每生成 1 个新 token 要做多少重复计算？为什么总成本是 **O(n²)**？（承接 Day5 `kv_cache_motivation.md`）
2. KV Cache 到底**缓存的是什么**？为什么缓存的是 K 和 V，**不缓存 Q**？
3. 为什么"缓存历史 K/V"在数学上是**安全的**（不会改变结果）？这件事和 Day2/Day3 的 **causal mask（因果掩码，下三角）**有什么关系？
4. 引入 KV Cache 后，每生成一步的计算量从 **O(n) 降到了什么**？整体从 O(n²) 降到了什么？
5. 什么叫 **prefill（预填充）** 和 **decode（解码）**？为什么说 prefill 是 **compute-bound（受算力限制）**、decode 是 **memory-bound（受内存带宽限制）**？
6. 既然 KV Cache 这么省**计算**，为什么它反而成了推理时的**显存大户**？KV Cache 的显存公式怎么算？这和 **PagedAttention** 要解决的问题是什么关系？
7. decode 阶段既然 memory-bound，为什么把多个请求**拼成一个 batch（批处理）**能提升吞吐？这和 **continuous batching（连续批处理）** 是什么关系？
8. 怎么用 **profiler（性能剖析器）** 跑一次生成、导出 chrome trace，并从 trace 里**看出** decode 阶段是 memory-bound（看什么信号）？

> 这 8 问里，Q1/Q4 是承上（把昨天的动机闭环），Q2/Q3 是 KV Cache 的命门，Q5-Q7 是 `kv_cache_and_two_phases.md` 的三个核心小节，Q8 是今天第三个 checklist 的硬技能。

---

## 1. 先承上：朴素自回归到底浪费在哪（O(n²) 的账）

> 昨天 `kv_cache_motivation.md` 你已经画出了"耗时随长度超线性增长"的曲线。今天先把这笔账**用一张图算死**，因为 KV Cache 省的每一分钱，都对应这张图里的某块浪费。

### 1.1 朴素 `generate()` 在做什么（回忆 Day5）

自回归（autoregressive，"用自己已经生成的内容，预测下一个"）生成的循环长这样：

```
prompt = "罗密欧"                    # 假设 3 个 token
第1步：forward("罗密欧")          → 预测第4个token "说"      → 序列变 "罗密欧说"
第2步：forward("罗密欧说")        → 预测第5个token "："      → 序列变 "罗密欧说："
第3步：forward("罗密欧说：")      → 预测第6个token "爱"      → ……
...每一步，都把"当前已有的整个序列"重新喂进模型 forward 一遍
```

问题就藏在"**把整个序列重新 forward 一遍**"这句话里。

### 1.2 浪费的本质：历史 token 的 K/V 每步都被重算

回忆 Day2/Day3 的 attention：每个 token 都要算自己的三件套——**Query（查询，"我想找什么"）**、**Key（键，"我是关于什么的"）**、**Value（值，"我携带的真实信息"）**。

现在看朴素生成的第 2 步 `forward("罗密欧说")`：模型会为 "罗" "密欧" "说" 这 4 个位置**全部重新算一遍 Q/K/V**。但是——

> **关键洞察**："罗""密欧""说"这前几个 token 的 **K 和 V，在第 1 步就已经算过了，而且值和这一步一模一样**。因为它们的输入（embedding + 位置编码）没变，算出来的 K/V 自然也没变。你在**重复计算已经算过、且结果完全相同的东西**。

**生活类比**：你每天上班都要给老板汇报"从入职到今天的所有工作进展"。朴素做法是——每天把入职至今**每一件事**重新讲一遍，今天讲到第 100 天，就要复述前 99 天 + 今天，明天复述前 100 天 + 明天……越往后，重复复述的历史越长。聪明做法是：历史进展老板**记在本子上了（缓存）**，你每天只汇报**今天新增的那一件事**。KV Cache 就是这个"本子"。

### 1.3 把账算死：为什么是 O(n²)

设要生成 n 个 token。朴素做法第 t 步要处理长度为 t 的序列，attention 的核心计算量正比于序列长度（甚至 attention 矩阵是 t²）。把每一步加起来：

```
第1步处理长度1 + 第2步长度2 + ... + 第n步长度n
= 1 + 2 + 3 + ... + n
= n(n+1)/2
≈ n²/2                       ← 这就是 O(n²) 的来历
```

生成 256 个 token，朴素做法的总工作量正比于 256²/2 ≈ **32768 个单位**。而这里面**绝大部分是重复劳动**——第 t 步真正"新"的工作只有"为第 t 个新 token 算一份 Q/K/V + 算它对历史的 attention"，是 O(t) 里的 **O(1) 增量**，剩下的 O(t-1) 全是重算历史。

> **一句话收口（呼应 Day5 动机）**：朴素自回归的 O(n²)，几乎全花在"反复重算那些根本没变的历史 K/V"上。KV Cache 的全部价值，就是把这块重复劳动一次性干掉。

---

## 2. KV Cache 的核心原理：缓存什么、为什么安全

### 2.1 缓存 K 和 V，为什么不缓存 Q（§0 第 2 问）

这是初学者第一个困惑点。三件套 Q/K/V，为什么名字叫 "**KV** Cache"，独独不缓存 Q？

回忆 attention 一行核心公式（Day2 §2）：第 t 个 token 的输出 =

```
out_t = softmax( q_t · [k_1, k_2, ..., k_t]ᵀ ) · [v_1, v_2, ..., v_t]
          ↑ 当前token的Q      ↑ 历史所有K（含自己）    ↑ 历史所有V（含自己）
```

仔细看这个式子里每个角色的"生命周期"：

- **q_t（当前 token 的 Query）**：只在算"第 t 个 token 的输出"这一步用一次。算完第 t 步，q_t 就**功成身退，下一步再也用不到了**。所以缓存它没意义——下一步 q_{t+1} 是全新的。
- **k_1...k_t、v_1...v_t（历史所有 token 的 Key/Value）**：第 t 步要用到**从头到现在的全部** K 和 V。下一步（第 t+1 步），还要用到这同一批 k_1...k_t（再加一个新的 k_{t+1}）。**它们被反复使用，且值不变**——这才是值得缓存的东西。

> **一句话**：Q 是"一次性的提问"，问完就扔；K/V 是"被反复查阅的资料库"，且只增不改。所以缓存 K/V、不缓存 Q。

**类比续 §1.2 的老板本子**：你每天的"今日新问题"（Q）是临时的，问完即弃；但老板本子上记录的"历史进展条目"（K/V）会被反复翻阅，且过去的条目不会改写——所以本子记的是 K/V，不是 Q。

### 2.2 为什么缓存是"安全的"：因果 mask 保证历史不变（§0 第 3 问）

"缓存"听起来有个风险：万一历史 token 的 K/V 后来变了，缓存不就脏了、结果就错了吗？**答案是：在自回归语言模型里，历史 token 的 K/V 永远不会变，这件事由 causal mask 从架构上保证。**

回忆 Day3 §6 你用测试 `test_causal_no_peeking` 钉死的那条性质——**因果性**：

> **causal mask（因果掩码）**：把 attention 矩阵的"上三角"（未来位置）填成 -∞，softmax 后变 0。效果是：第 t 个 token **只能看 1..t（自己和过去），看不到未来**。Day2/Day3 你把它实现成"下三角矩阵"。

因果性带来一个推论，正是 KV Cache 成立的基石：

> **第 t 个 token 的 K/V，只由它自己和它之前的 token 决定，完全不受未来 token 影响。** 所以无论后面再生成多少新 token，已经算出来的 k_1...k_t、v_1...v_t **永远不会被改写**。缓存它们 100% 安全。

**反证一下为什么"双向模型"（如 BERT）不能这样缓存**：BERT 是双向的，第 3 个 token 能看到第 5 个 token。那么当第 5 个 token 出现时，第 3 个 token 的表示（进而它的 K/V）就会改变——历史被改写了，缓存立刻失效。**正是 GPT 这类模型的"单向因果"特性，让 KV Cache 成为可能。** 这就是 Day3 那个因果性测试，今天真正兑现价值的地方。

> **串联**：Day2 §3.3 埋的伏笔"下三角结构 → 这也是后面 KV Cache 能成立的前提（历史不变）"，今天闭环。因果 mask 不只是"防止偷看未来"的正确性约束，它还顺手送了一个**巨大的工程红利**——历史可缓存。一个设计同时解决正确性和效率，这是好架构的标志。

### 2.3 KV Cache 把复杂度降到哪（§0 第 4 问）

有了缓存，第 t 步的工作量变成：

```
朴素第t步：为 1..t 全部 token 重算 Q/K/V + 算 attention   → O(t)
缓存第t步：只为"第t个新token"算 1 份 Q/K/V               → O(1) 的增量
            历史 k/v 直接从缓存取，append 上新的 k_t/v_t
            q_t 对 [缓存的 k_1..k_t] 做 attention          → 这一步仍是 O(t)（读缓存）
```

注意一个**诚实的细节**（很多入门资料会含糊过去）：KV Cache 把"**计算** Q/K/V 投影"这步从 O(t) 降到了 O(1)，**但 attention 本身 `q_t · [k_1..k_t]` 还是要读取全部 t 个历史 K**，所以单步仍有一个 O(t) 的"读缓存"成本。整体看：

| | 朴素 | KV Cache |
|---|---|---|
| 单步**计算** Q/K/V 投影 | O(t)（重算全部历史） | **O(1)**（只算新 token） |
| 单步 attention（读 K/V） | O(t) | O(t)（读缓存，但省了重算） |
| 生成 n token 总复杂度 | **O(n²)** | **O(n²) 但常数小得多** / 计算量 O(n) |

所以更精确的说法是：**KV Cache 把"投影计算"从 O(n²) 降到 O(n)，attention 的读取仍是 O(n²) 但变成了"读"而非"算重"。** 这个"读 vs 算"的区别，正是 §5 prefill/decode 两阶段、以及 decode 为什么 memory-bound 的根源——**记住这里，§5 会用到。**

> **预告**：你会问"既然 attention 读取还是 O(n²)，那 KV Cache 实测能快多少？"——答案取决于模型大小。在 nanoGPT 这种小模型上，投影 + FFN 的计算占大头，省掉重算非常划算，§4 你会实测到明显加速。在超长序列的大模型上，attention 读取会重新变成瓶颈，这就引出了 FlashAttention（W8）。

---

## 3. 【动手主菜一】改造 attention 支持 KV Cache：`src/kv_cache.py`

> 计划 line 1767-1776 给了你核心 5 行。现在把它扩展成可运行、可对比的完整代码。关键改动只有一处：**让 attention 的 forward 能接收"历史缓存"、并返回"更新后的缓存"。**

### 3.1 改造思路：forward 多吃一个 `past_kv`、多吐一个新缓存

对比 Day3 的 `CausalSelfAttention`，KV Cache 版只改三件事：

1. forward 多一个入参 `past_kv`（上一步存下的历史 K/V），默认 `None`（表示这是第一步/prefill）。
2. 算出新 token 的 `k_new, v_new` 后，**拼接（concat）**到历史缓存后面。
3. forward 多返回一个 `(k, v)`，作为"更新后的缓存"传给下一步。

还有一个**容易踩的坑**——位置编码。decode 时输入只有 1 个新 token，但它的"位置"不是 0，而是"当前序列已有的长度"。所以要把"从第几个位置开始"也传进去。

### 3.2 代码：KV Cache 版 attention

```python
# src/kv_cache.py  （第一部分：支持 KV Cache 的 attention）
# 运行环境：Python 3.10+，PyTorch 2.x（CPU 可跑；有 CUDA 加速更明显）
# 依赖：pip install torch
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)  # 固定种子，可复现（延续你的产出规范）


class CausalSelfAttentionKV(nn.Module):
    """支持 KV Cache 的多头因果自注意力。
    和 Day3 的 CausalSelfAttention 在数学上等价，只是 forward 能吃/吐缓存。"""

    def __init__(self, n_head, n_embd, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)  # fused QKV（Day3 §4）
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x, past_kv=None):
        """
        x:        (B, T, C)。prefill 时 T = prompt 长度；decode 时 T = 1（只有新 token）。
        past_kv:  None（首步）或 (past_k, past_v)，形状各 (B, nh, T_past, hs)。
        返回:     (out, (k, v))。(k, v) 是"拼上本步后"的新缓存，喂给下一步。
        """
        B, T, C = x.shape
        hs = C // self.n_head

        # ① 只为"当前输入的 T 个 token"算 Q/K/V（decode 时 T=1，这就是省下重算的地方）
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)      # 各 (B, T, C)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # ② 把历史缓存拼到前面：这是 KV Cache 的灵魂一步
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   # 沿"序列"维(dim=2)拼：历史 + 新
            v = torch.cat([past_v, v], dim=2)
        # 此刻 k,v 的序列长度 = T_past + T（完整历史），q 的长度仍是 T（只问新 token）

        T_full = k.size(2)                      # 完整序列长度（含历史）
        # ③ attention：当前 q（长 T）对完整 k（长 T_full）打分
        wei = (q @ k.transpose(-2, -1)) * hs ** -0.5            # (B, nh, T, T_full)

        # ④ causal mask：当前这 T 个 token 在完整序列里的"绝对位置"是 [T_full-T, T_full)
        #    它们能看到的范围是 0..自己的绝对位置。用 tril 的对应切片即可。
        mask = self.tril[T_full - T:T_full, :T_full]            # (T, T_full)
        wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v                                           # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # 合头（contiguous 必需，Day3 §3.3）
        out = self.c_proj(out)
        return out, (k, v)                                      # ★ 返回更新后的缓存
```

**对照 Day3，标注三处关键改动：**

- **`past_kv` 入参 + `torch.cat([past_k, k], dim=2)`**：这就是计划 line 1771 的核心——历史不重算，直接拼。`dim=2` 是序列维（形状 `(B, nh, T, hs)` 的第 2 维），拼错维度结果全错且不报错，是高频坑。
- **`q` 只有 T 行，`k/v` 有 T_full 行**：decode 时 `q` 是 `(B,nh,1,hs)`、`k` 是 `(B,nh,t,hs)`，attention 矩阵是 `(B,nh,1,t)`——**只为新 token 算一行**，这正是 §2.3 说的"省掉为历史重算"。
- **mask 用切片 `tril[T_full-T:T_full, :T_full]`**：朴素版每次从头 mask `tril[:T,:T]`，但 decode 时当前 token 的绝对位置不是从 0 开始，要取 tril 的对应"行段"。这是 KV Cache 实现里最容易写错、最难 debug 的一行——写错了模型不报错，但生成的文本会变成乱码（因为 mask 错位，token 看错了历史范围）。

> **调试技巧（必看）**：验证 KV Cache 写对了，**唯一可靠**的办法是——**对同一个 prompt，KV Cache 生成的结果，必须和朴素生成逐 token 完全一致**（贪心解码下；采样要固定随机数）。这是和 Day3 `test_loop_equals_fused` 同一个思想：**优化版必须和朴素版数值等价**。§4 的脚本会用这个做正确性断言。**不要凭"生成的文本看起来通顺"就以为对了**——mask 错位时文本可能仍然局部通顺，但已经偏离朴素结果。

---

## 4. 【动手主菜二】朴素 vs KV Cache 生成 256 token 测速对比

> 计划 line 1777 的硬指标。为了让脚本**独立可跑**（不依赖你 Day4 的完整 GPT 文件），下面用一个**最小 GPT** 把 attention 包起来——结构和你 nanoGPT 一致（Block = LN+Attn+残差 + LN+FFN+残差），只是参数调小。你做实验时可以直接 `import` 你自己的 `model.py`，原理完全一样。

### 4.1 一个最小可跑的 GPT（包住 §3 的 attention）

```python
# src/kv_cache.py  （第二部分：最小 GPT，包住 attention）

class Block(nn.Module):
    """Transformer Block：两条残差通路（Day4 同款）。forward 透传 past_kv。"""
    def __init__(self, n_head, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionKV(n_head, n_embd, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv)  # 注意 norm 在前（pre-LN）
        x = x + attn_out                                    # 残差通路 1
        x = x + self.ffn(self.ln2(x))                       # 残差通路 2
        return x, new_kv


class MiniGPT(nn.Module):
    """最小 GPT：token+pos embedding → N 个 Block → LN → 输出头。
    generate 时支持两种模式：朴素（每步全量 forward）和 KV Cache。"""
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_head, n_embd, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, past_kvs=None, pos_offset=0):
        """
        idx:        (B, T) 的 token id。
        past_kvs:   None 或 长度为 n_layer 的列表，每项是该层的 (k, v) 缓存。
        pos_offset: 当前这批 token 在完整序列里的起始位置（decode 时 = 已生成长度）。
        """
        B, T = idx.shape
        # 位置编码：从 pos_offset 开始数 T 个位置（decode 时关键，否则新 token 位置永远是 0，错）
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)            # (B, T, C)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, new_kv = block(x, past)
            new_kvs.append(new_kv)
        x = self.ln_f(x)
        logits = self.head(x)                                # (B, T, vocab)
        return logits, new_kvs
```

### 4.2 两种生成函数：朴素 vs KV Cache

```python
# src/kv_cache.py  （第三部分：两种 generate）

@torch.no_grad()
def generate_naive(model, idx, max_new_tokens):
    """朴素自回归：每一步把'当前完整序列'重新 forward 一遍（O(n²) 浪费）。"""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]            # 截断到 block_size 内
        logits, _ = model(idx_cond)                      # 不传缓存 → 全量重算
        logits = logits[:, -1, :]                        # 只取最后一个位置的预测
        next_id = logits.argmax(dim=-1, keepdim=True)    # 贪心解码（为了可复现对比）
        idx = torch.cat([idx, next_id], dim=1)
    return idx


@torch.no_grad()
def generate_kv(model, idx, max_new_tokens):
    """KV Cache 生成：先 prefill 整个 prompt，再每步只 forward 1 个新 token。"""
    model.eval()
    B, T0 = idx.shape
    # ---- prefill 阶段：一次性处理整个 prompt，建立初始缓存 ----
    logits, past_kvs = model(idx, past_kvs=None, pos_offset=0)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = torch.cat([idx, next_id], dim=1)
    # ---- decode 阶段：每步只喂'刚生成的那 1 个 token' ----
    for step in range(max_new_tokens - 1):
        pos = T0 + step                                  # 新 token 的绝对位置
        logits, past_kvs = model(
            next_id, past_kvs=past_kvs, pos_offset=pos   # 只 forward 1 个 token！
        )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
    return out
```

**两个函数的差别一眼可见**：`generate_naive` 每步 `model(idx_cond)` 喂的是**整段序列**；`generate_kv` 在 decode 循环里 `model(next_id, ...)` 喂的永远是 **1 个 token**，历史全在 `past_kvs` 里。这就是 §1.2"老板本子"的代码化。

> **注意 prefill / decode 的分界**：`generate_kv` 里第一次 `model(idx, ...)` 处理整个 prompt，就是 **prefill**；后面的 for 循环每次只处理 1 个 token，就是 **decode**。这两段的硬件行为天差地别（§5 详解）——记住这个分界，它是今天理论部分的核心。

### 4.3 主程序：正确性断言 + 256 token 测速

```python
# src/kv_cache.py  （第四部分：__main__，正确性 + 测速）

if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # nanoGPT 小配置（计划 line 1895 的省显存配置思路）
    vocab_size, n_layer, n_head, n_embd, block_size = 65, 4, 4, 128, 512
    model = MiniGPT(vocab_size, n_layer, n_head, n_embd, block_size).to(device).eval()

    B, T_prompt, max_new = 1, 8, 256          # 1 条序列，8 token prompt，生成 256 token
    idx = torch.randint(0, vocab_size, (B, T_prompt), device=device)

    # ---- 1. 正确性断言：KV Cache 结果必须和朴素逐 token 一致（最重要的一步）----
    out_naive = generate_naive(model, idx.clone(), max_new)
    out_kv    = generate_kv(model, idx.clone(), max_new)
    assert torch.equal(out_naive, out_kv), \
        "KV Cache 生成结果和朴素不一致！大概率是 mask 切片或 pos_offset 写错"
    print("✅ 正确性通过：KV Cache 与朴素生成逐 token 完全一致")

    # ---- 2. 测速：各生成 256 token，对比延迟 ----
    def bench(fn, model, idx, max_new, iters=10):
        with torch.no_grad():
            for _ in range(3):                # 预热（Day3 §5 的规矩）
                fn(model, idx.clone(), max_new)
            if device == "cuda":
                torch.cuda.synchronize()      # GPU 异步，计时前必须同步
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(model, idx.clone(), max_new)
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3   # 毫秒/次

    t_naive = bench(generate_naive, model, idx, max_new)
    t_kv    = bench(generate_kv,    model, idx, max_new)
    print(f"\n设备: {device}  |  生成 {max_new} token")
    print(f"朴素生成   : {t_naive:8.1f} ms")
    print(f"KV Cache   : {t_kv:8.1f} ms")
    print(f"加速比     : {t_naive / t_kv:.2f}x  ← 生成越长，加速比越大")
```

**预期输出（数字随机器变，趋势一致；KV Cache 一定更快，且生成越长差距越大）**：

```
✅ 正确性通过：KV Cache 与朴素生成逐 token 完全一致

设备: cuda  |  生成 256 token
朴素生成   :   1820.3 ms
KV Cache   :    430.7 ms
加速比     : 4.23x  ← 生成越长，加速比越大
```

> **写进 `W6_day6_log.md` 的关键观察**：
> 1. **加速比随生成长度增长而变大**——这是 O(n²)→O(n) 的直接证据。建议你做个小实验：生成 64 / 128 / 256 / 512 token 各测一次，画出"加速比 vs 长度"曲线，会看到加速比单调上升。这张图比单个数字更有说服力，也是简历里能放的图。
> 2. **CPU 上加速比偏小，GPU 上更明显**——因为 CPU 算力弱，朴素重算的代价相对没那么"亏"；GPU 算力强，省下的重算更值钱。
> 3. **正确性断言是第一位的**：先确保 `torch.equal` 通过，再看速度。一个"快但错"的 KV Cache 毫无价值——这正是你 [[project-xiaomi-research]] 课题"数值一致或误差可控可解释"指标的入门版。

### 4.4 一个工业级细节：`torch.equal` vs `torch.allclose`

上面正确性断言用了 `torch.equal`（要求**逐元素完全相等**），因为我们用**贪心解码**（`argmax`），且 KV Cache 和朴素走的是**同样的浮点运算路径**，结果应当 bit 级一致。

但如果你改用**采样**（`torch.multinomial`）或某些场景下浮点累加顺序不同，可能出现极小误差，这时要退一步用 `torch.allclose(a, b, atol=1e-5)`。**工业实践里，KV Cache / 量化 / kernel 优化的回归测试，几乎都是这种"优化版 vs 朴素版"的数值对比**——这是 AI Infra 的基本职业素养（呼应 Day3 §6 那段话）。你的小米课题验收指标"数值一致或误差可控"，落到代码上就是这行断言。

---

## 5. 【理论核心一】prefill / decode 两阶段：生产推理系统的分水岭

> 这是 `tech_notes/kv_cache_and_two_phases.md` 的 §2，也是计划 line 1780 的核心。前面你已经在 `generate_kv` 里**亲手写出了**这两个阶段（prefill = 第一次全量 forward，decode = 后面每步 1 token）。这一节把它们的**硬件本质**讲透——这是面试官最爱问、也是工业系统所有优化的出发点。

### 5.1 两阶段是什么（用 §4 的代码对号入座）

LLM 的一次生成请求，天然分成两段，行为完全不同：

| | **prefill（预填充）** | **decode（解码 / 增量生成）** |
|---|---|---|
| 处理什么 | 一次性处理**整个 prompt**（用户输入的全部 token） | 每步只处理**1 个**刚生成的新 token |
| 对应 §4 代码 | `generate_kv` 第一次 `model(idx, ...)` | for 循环里每次 `model(next_id, ...)` |
| 一次喂多少 token | 多（prompt 长度，几十到几千） | **1** |
| 产出 | 整个 prompt 的 KV Cache + 第一个生成 token | 1 个新 token + 更新缓存 |
| 频率 | 每个请求**只发生 1 次** | 重复 n-1 次（生成多少 token 就多少次） |
| **瓶颈类型** | **compute-bound（受算力限制）** | **memory-bound（受内存带宽限制）** |

**生活类比**：prefill 像考试时"先快速通读整张卷子"——一次性吞下大量信息，脑子（算力）高速运转；decode 像"逐题作答"——每次只动一道题，但每道题都要把前面所有题的草稿纸（KV Cache）翻一遍。前者是脑力密集，后者是"翻纸"密集。

### 5.2 为什么 prefill 是 compute-bound（§0 第 5 问上半）

> **compute-bound（受算力限制 / 计算密集）**：一个操作的瓶颈在"算得快不快"，GPU 的算力单元（Tensor Core）被喂得很满，内存带宽不是限制。判断标准是 **arithmetic intensity（算术强度 = 计算量 / 数据搬运量）高**——搬一次数据能换来很多次计算。

prefill 处理整个 prompt（比如 512 个 token），是一个**又大又宽的矩阵乘法**（512 个 token 一起过 Linear、一起算 attention）。回忆 Day3 §4.1：**GPU 最喜欢又大又规整的 GEMM**，这种大矩阵能把成千上万个核心喂满，算术强度高——搬一次权重，能服务 512 个 token 的计算。所以 prefill 阶段 GPU 算力利用率高，瓶颈在算力，是 compute-bound。

### 5.3 为什么 decode 是 memory-bound（今天最重要的洞察，§0 第 5 问下半）

> **memory-bound（受内存带宽限制 / 访存密集）**：瓶颈在"数据从显存搬到计算单元的速度"，而不是"算得快不快"。算力单元大部分时间在**等数据**、空转。判断标准是 **arithmetic intensity 低**——搬一大堆数据，只换来很少的计算。

decode 阶段每步**只处理 1 个 token**。这意味着：

```
为了生成这 1 个 token，GPU 必须从显存里搬运：
  - 模型的【全部权重】（每一层的 Linear、attention 矩阵……几百 MB 到几十 GB）
  - 【整个 KV Cache】（到目前为止所有历史 token 的 K/V）
但搬完这么多数据，只为【1 个 token】做了计算。
```

**算术强度低到极点**：搬运量巨大（全部权重 + 全部缓存），计算量却只够 1 个 token。GPU 的算力此时严重过剩——核心们等着数据从显存"喂"过来，喂得慢，算得就慢。**瓶颈完全在内存带宽（memory bandwidth），不在算力。** 这就是 decode memory-bound 的本质。

**生活类比（呼应 W5 Roofline `flops_vs_latency.md`）**：decode 像"一个超级大厨（GPU 算力），但每次只让他做 1 道菜，而且做每道菜前都得把**整个仓库的食材（全部权重）搬到厨房门口**看一遍。大厨的刀工（算力）再快也没用，他大部分时间在等搬运工（内存带宽）。瓶颈是"搬运"不是"切菜"——这正是 memory-bound。

> **和 W5 Roofline 严丝合缝接轨**：你 W5 在 `flops_vs_latency.md` 画的 Roofline 模型，横轴是算术强度，纵轴是性能。低算术强度的操作落在 Roofline 的"**屋顶斜坡**"上——性能被**带宽**这条斜线封顶，加再多算力也没用。**decode 就是典型的低算术强度操作，死死卡在带宽斜坡上。** 你 Day3 学 fused QKV 时说"推理优化大量是在省搬运不是省计算"，今天 decode 给了这句话最重的一个注脚——**整个 LLM 推理最耗时的部分（decode 占生成总时间的绝大多数），就是个 memory-bound 操作。**

### 5.4 这个分水岭为什么决定了一切优化策略

理解了"prefill compute-bound、decode memory-bound"，你就拿到了读懂所有 LLM 推理优化论文的钥匙。因为**优化 memory-bound 和优化 compute-bound 的手段完全不同**：

| 阶段 | 瓶颈 | 优化方向 | 对应技术（你后面会学） |
|---|---|---|---|
| prefill | 算力 | 把矩阵乘法做得更快、算子融合 | FlashAttention、`torch.compile`、Tensor Core 利用 |
| decode | **带宽** | **减少要搬的数据** | KV Cache（减重算）、**量化**（权重变小→搬得少）、**batching**（摊薄权重搬运，§7）、PagedAttention（管好缓存显存，§6） |

> **直接对接你的小米课题**：[[project-xiaomi-research]] 主线 1 明确要"量化计算/访存/通信/调度开销占比，识别 decode 阶段瓶颈"——你今天理解的"decode memory-bound"就是那条主线的理论地基。主线 3"巨核算子（megakernel）"为什么能提速？很大程度也是因为：把多个小算子融成一个大核，**减少了 decode 时反复启动 kernel、反复搬运中间结果的访存开销**——又是"省搬运"。今天这个洞察，会反复出现在你接下来一整年的工作里。

---

## 6. 【理论核心二】KV Cache 的显存代价：省了计算，却成了显存大户

> `kv_cache_and_two_phases.md` 的 §3，计划 line 1781。这是一个漂亮的"天下没有免费午餐"——KV Cache 用**显存**换**计算**。理解这笔交易，才能理解为什么会有 PagedAttention。

### 6.1 KV Cache 占多少显存：把公式拆开讲

每生成一个 token，每一层都要把它的 K 和 V 存进缓存。把所有 token、所有层加起来，总显存是：

```
KV Cache 显存 = 2 × n_layer × n_head × head_dim × seq_len × batch × dtype_bytes
                ↑    ↑         ↑        ↑          ↑         ↑       ↑
              K和V  层数      头数    每头维度   序列长度   批大小  每个数几字节
```

**逐项翻译（每一项都是一个"为什么这么大"的理由）：**

- **× 2**：K 和 V 各存一份。
- **× n_layer**：**每一层都有自己独立的 KV Cache**。GPT 不是只缓存一次，是每层都缓存——这是很多人低估显存的地方。
- **× n_head × head_dim**：注意 `n_head × head_dim = n_embd`（Day3 §1.3 的灵魂除法），所以这两项合起来就是模型隐藏维度。
- **× seq_len**：**序列越长，缓存线性增长**——长上下文（128K context）的显存噩梦就来自这里。
- **× batch**：**同时服务的请求越多，缓存成倍增长**。
- **× dtype_bytes**：FP16 是 2 字节，FP32 是 4 字节（这也是为什么推理常用 FP16/BF16——缓存直接减半）。

### 6.2 代入真实数字：感受它有多大

以 **Llama-2 13B**（n_layer=40, n_embd=5120, FP16）为例，算"1 条序列、4096 长度"的 KV Cache：

```
2 × 40 × 5120 × 4096 × 1 × 2 bytes
= 2 × 40 × 5120 × 4096 × 2
≈ 3.4 GB                    ← 单条 4K 序列，就要 3.4 GB！
```

现在把 batch 拉到 32（同时服务 32 个用户）：**3.4 GB × 32 ≈ 109 GB**。一块 A100 才 80GB 显存——**KV Cache 自己就爆显存了，模型权重还没算**。

> **这就是关键转折**：KV Cache 帮你把**计算**从 O(n²) 降到 O(n)，代价是吃掉了**和序列长度、batch 成正比的巨量显存**。在生产环境（长上下文 + 高并发），**KV Cache 本身就是显存的头号消耗者**，常常比模型权重还大。省了时间，烧了显存——这是一笔必须管理的交易。

### 6.3 为什么催生了 PagedAttention（§0 第 6 问）

显存大还只是问题的一半。更糟的是**浪费**：

朴素的 KV Cache 实现（就像 §3 的 `torch.cat`），会给每个请求**预先分配一整块连续显存**，按"可能的最大长度"留位置。但实际生成长度事先不知道——有的请求生成 10 个 token，有的生成 1000 个。结果：

- **内部碎片**：按最大长度 2048 预留，实际只生成 50 个，剩下 1998 个位置的显存**空占着浪费**。
- **外部碎片**：请求长度参差不齐，显存被切成大小不一的块，新请求来了凑不出连续的大块——明明总量够，却分配不出来。

> **PagedAttention（分页注意力）**：vLLM 提出的方案。它借鉴了**操作系统虚拟内存分页**的思想——不再给每个请求分配一大块连续显存，而是把 KV Cache 切成固定大小的**小块（block / page）**，用一张"块表"记录每个请求用了哪些块。需要多少分多少，用完归还。

**类比（操作系统分页）**：朴素 KV Cache 像"给每个员工分配一整间固定大办公室"——人少时房间空着浪费，来个大团队又凑不出连续的几间。PagedAttention 像"共享工位 + 座位表"——按需分配小工位，谁用谁占、用完释放，碎片几乎为零。vLLM 靠这个把显存利用率从朴素方案的 ~20-40% 拉到 ~90%+，同样的卡能同时服务多几倍的请求。

> **串联与对接课题**：这正是 W5 `inference_optimization_landscape.md` 列的第 2 个术语 **PagedAttention** 落地——它解决的根本问题，就是你今天 §6.1-§6.2 算出的"KV Cache 显存又大又碎"。**你的小米课题主线 4 明确要求"与 continuous batching、动态 token 推理协同"**——而 vLLM 的 continuous batching（§7）正是建立在 PagedAttention 的显存管理之上的。今天你算清了 KV Cache 的显存账，就理解了 vLLM 整套设计的出发点。

---

## 7. 【理论核心三】为什么 batching 能提吞吐：从 memory-bound 推出来

> `kv_cache_and_two_phases.md` 的 §4，计划 line 1782。这一节是 §5"decode memory-bound"的直接推论——理解了 decode 卡在带宽上，batching 为什么有效就**自然推出来**了，不用死记。

### 7.1 先分清两个常被搞混的指标：延迟 vs 吞吐

> **latency（延迟）**：单个请求从发出到拿到结果的耗时。关心"我这一条快不快"。
> **throughput（吞吐）**：单位时间能处理多少请求 / 生成多少 token。关心"这台机器一共能服务多少人"。

这两个**经常是矛盾的**——为了提吞吐，往往会牺牲一点单条延迟。生产系统通常优先保吞吐（服务器要同时服务成千上万用户，单条慢一点点可接受）。**batching 就是用一点延迟换大量吞吐的经典手段。**

### 7.2 为什么 decode 阶段 batching 几乎"免费"提吞吐

回忆 §5.3：decode 时每生成 1 个 token，要把**整个模型权重**从显存搬到计算单元，但只为 1 个 token 算。**搬权重是大头开销，而这次搬运的算力却被严重浪费（算力过剩）。**

关键来了——**把 B 个请求拼成一个 batch 一起 decode**：

```
单个请求 decode 一步：搬 1 次全部权重，为 1 个 token 计算   → 算力利用率极低
B 个请求拼 batch decode 一步：搬 1 次全部权重，为 B 个 token 计算 → 权重搬运被 B 个请求摊薄！
```

**权重只搬了一次，却同时服务了 B 个请求。** 因为原本算力是过剩的（memory-bound），多算 B-1 个 token **几乎不增加时间**（算力本来就闲着）——但吞吐直接翻了近 B 倍。这就是 batching 在 decode 阶段近乎"免费午餐"的原因。

**生活类比**：还是那个大厨——每做一道菜都要把整个仓库食材搬到门口看一遍（搬权重）。如果让他**一次同时做 16 道菜**（batch=16），食材只搬一遍，16 道菜几乎同时出锅。搬运成本被 16 道菜摊薄，大厨过剩的刀工也终于用上了。吞吐翻 16 倍，而你为这一锅多等的时间微乎其微。

> **为什么这招对 decode 特别灵、对 prefill 没那么灵**：因为 batching 的红利来自"摊薄权重搬运"，而这个红利**只在 memory-bound 时巨大**。prefill 本来就是 compute-bound（算力已经喂满），再 batch 算力也不够分了，红利小。**正是 decode 的 memory-bound 特性，让 batching 成为吞吐优化的头号武器。** 你看，§5 那个洞察又一次决定了优化策略。

### 7.3 从静态 batching 到 continuous batching（§0 第 7 问）

朴素的"静态 batching"有个致命问题：一个 batch 里各请求生成长度不同，**短的早就生成完了，却要干等 batch 里最长的那个**——GPU 利用率掉下来。

> **continuous batching（连续批处理 / in-flight batching）**：vLLM / TGI 等框架的核心调度技术。它不等整个 batch 一起结束——**哪个请求生成完了就立刻让它退出、把新请求即时补进 batch 的空位**。让 GPU 始终满载运行，没有"等最慢的"空窗。

**类比**：静态 batching 像"拼车必须等所有人都到终点才能下车"；continuous batching 像"网约车随时上下客"——到站的乘客立刻下车，新乘客马上补上空座，车（GPU）一刻不空跑。

> **串联与对接课题**：这是 W5 `inference_optimization_landscape.md` 第 3 个术语 **continuous batching** 的落地。它和 §6 的 PagedAttention 是一对黄金搭档——continuous batching 负责"调度上让 GPU 满载"，PagedAttention 负责"显存上能灵活塞下随时进出的请求"，两者合起来撑起了 vLLM 的高吞吐。**你的小米课题主线 4 点名"与 continuous batching、动态 token 推理协同避免冲突"**——意思是：你做的图优化 / 巨核算子，必须能在"请求随时进出、batch 形状动态变化"的环境里正确工作，不能假设 batch 是固定的。今天理解 continuous batching 的工作方式，就知道这个约束从哪来了。

---

## 8. 【动手主菜三】用 profiler 跑生成，确认 decode 阶段 memory-bound

> 计划 line 1783 / checklist line 1875。前面 §5 是**理论推断** decode memory-bound，这一节用 **profiler（性能剖析器）** 这把"听诊器"对准真实生成过程，**拿出 trace 证据**。这一步是把"我以为"变成"我测过"——也是你 [[project-xiaomi-research]] 主线 1"端到端性能画像"的入门动作。

### 8.1 profiler 是什么、chrome trace 是什么（承接 W5）

> **torch.profiler（PyTorch 性能剖析器）**：PyTorch 自带的工具，能记录一段代码里**每个算子（operator）/ 每个 CUDA kernel 跑了多久、占多少时间、用了多少显存**。W5 你已经用它分析过 ResNet（`profiler_chrome_trace.md`），今天把同一把听诊器对准 nanoGPT 的生成过程。
>
> **chrome trace**：profiler 导出的一种 JSON 格式时间线文件。在 Chrome 浏览器地址栏输入 `chrome://tracing`（或用 `https://ui.perfetto.dev`），把 JSON 拖进去，就能看到一条**横向时间轴**：每个 kernel 是一个小方块，宽度 = 耗时，方块之间的空隙 = GPU 空闲。它让"时间花在哪"变得肉眼可见。

### 8.2 代码：给生成过程装上 profiler

```python
# src/profile_generate.py
# 运行环境：Python 3.10+，PyTorch 2.x（GPU 最佳；CPU 也能跑但看不出带宽瓶颈）
# 运行：python src/profile_generate.py，然后把 logs/nanogpt_trace.json 拖进 chrome://tracing
# 依赖：pip install torch
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from kv_cache import MiniGPT, generate_kv   # 复用 §4 的模型和生成函数

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size=65, n_layer=4, n_head=4, n_embd=128,
                block_size=512).to(device).eval()
idx = torch.randint(0, 65, (1, 8), device=device)

activities = [ProfilerActivity.CPU]
if device == "cuda":
    activities.append(ProfilerActivity.CUDA)   # GPU 上必须加，否则只看到 CPU 侧

# 先预热，避免把一次性的冷启动开销也录进来（Day3 §5 的规矩）
with torch.no_grad():
    generate_kv(model, idx.clone(), 32)
if device == "cuda":
    torch.cuda.synchronize()

with profile(activities=activities, record_shapes=True,
             profile_memory=True) as prof:        # profile_memory 让显存占用也被记录
    with torch.no_grad():
        # 用 record_function 给两个阶段打标签，方便在 trace 里区分
        with record_function("PREFILL+DECODE_generate_256"):
            generate_kv(model, idx.clone(), 256)
    if device == "cuda":
        torch.cuda.synchronize()                  # 等 GPU 真跑完再结束 profiling

# 1) 导出 chrome trace（拖进 chrome://tracing 看时间线）
prof.export_chrome_trace("logs/nanogpt_trace.json")
print("✅ trace 已导出到 logs/nanogpt_trace.json")

# 2) 直接在终端打印 Top 算子表（不用开浏览器也能看个大概）
sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=12))
```

### 8.3 怎么从 trace 里"看出" decode 是 memory-bound（§0 第 8 问，关键技能）

跑完打开 trace，**你要找的不是"哪个 kernel 最慢"，而是几个 memory-bound 的特征信号**：

**信号 1：decode 区域 kernel 又多又小，方块之间有明显空隙**

把时间线拉到 `PREFILL+DECODE` 标签下，你会看到清晰的两段：
- **开头一大块**（prefill）：少数几个**又宽**的 kernel——大矩阵乘法，GPU 在满负荷算，方块紧密排列。
- **后面 256 段重复的小波**（decode）：每一步都是**一串又窄又多的小 kernel**，且**方块之间有肉眼可见的空隙**。

> **空隙 = GPU 在等数据 = memory-bound 的直接视觉证据。** kernel 那么小（只算 1 个 token），但相邻 kernel 之间却有空档，说明 GPU 算完一个小活后**在等下一批数据从显存搬过来**，而不是马不停蹄地算。如果是 compute-bound，方块会紧紧贴在一起没有空隙（算力一刻不停）。

**信号 2：kernel 启动开销占比高**

decode 每步要启动几十个小 kernel（每层的 LayerNorm、QKV 投影、attention、FFN……），每个都只算 1 个 token。在 trace 里你会看到大量**启动开销（launch overhead）**——CPU 侧"发射 kernel"的小条，和 GPU 侧实际计算的时间相比占比异常高。**算得少、启动多、还要等搬运**，三件事叠加，正是 memory-bound 的典型画像。（这也解释了你小米课题主线 3"巨核算子"的动机——把几十个小 kernel 融成一个大核，就是来消灭这堆启动开销和中间访存的。）

**信号 3：终端表里，时间花在访存型算子和大量小算子上**

看 §8.2 打印的 Top 算子表：你会发现时间不是集中在某个"大计算"上，而是**散落在一堆小算子**（elementwise、layernorm、小 matmul、cat/copy）里。尤其 KV Cache 的 `torch.cat`（每步把新 K/V 拼进缓存）是个纯访存操作——**搬数据，不算数**。这些"搬运型"算子占了可观时间，又一个 memory-bound 的旁证。

> **写进 `W6_day6_log.md` 的结论模板**：
> > "用 torch.profiler 跑了生成 256 token，导出 chrome trace。观察到：prefill 阶段是少数几个宽 kernel、排列紧密（compute-bound）；decode 阶段是 256 段重复的窄小 kernel，**方块间有明显空隙**，且 `cat`/elementwise 等访存型算子占比高，kernel 启动开销显著。**这印证了 §5 的理论预测：decode 阶段 memory-bound，瓶颈在内存带宽与 kernel 启动，而非算力。**"
>
> 这段话直接就是你简历 / 套磁信里能写的一句"我用 profiler 实测验证了 LLM decode 阶段的访存瓶颈"。

### 8.4 一个诚实的提醒：小模型 + CPU 上信号会弱

- 在 **CPU** 上跑，看不到 CUDA kernel 时间线，"GPU 等数据的空隙"这个最直观的信号就没了——只能从"时间散落在小算子上"间接推断。**有 GPU 务必用 GPU 跑**（哪怕是 Colab 免费 T4）。
- nanoGPT 这种**小模型**，权重小、搬运快，memory-bound 特征不如真实大模型那么极端。但**趋势和结构是一致的**——你要看的是"decode 段 kernel 碎、有空隙"这个**定性结构**，而不是追求和 13B 模型一样夸张的数字。理解结构比刷数字重要。
- 如果想看更明显的对比，可以**额外 profile 一次朴素生成**，对比 trace：朴素生成里每步那个"又宽又大"的全量 forward（compute 占比高）和 KV Cache decode 那串"又碎又有空隙"的小 kernel，放一起一目了然——这是个很好的 log 配图。

---

## 9. 常见陷阱与调试技巧（KV Cache 100% 会踩的坑）

1. **decode 时位置编码忘了加 offset** —— 新 token 的位置应该是"已生成长度"，不是 0。忘了传 `pos_offset`，每个新 token 都按位置 0 编码，模型彻底懵，生成乱码。（§4.1 `pos = torch.arange(pos_offset, ...)`）这是**头号坑**。
2. **mask 切片写错** —— decode 时当前 token 的绝对位置不从 0 开始，要用 `tril[T_full-T:T_full, :T_full]` 而非 `tril[:T,:T]`。写错不报错，但 token 看错历史范围，结果偏离朴素。（§3.2）
3. **`torch.cat` 拼错维度** —— 必须沿序列维 `dim=2`（形状 `(B,nh,T,hs)`）。拼成 `dim=1`（头维）或 `dim=3`，缓存结构全乱且不报错。（§3.2）
4. **没做正确性断言就信了速度** —— "快但错"的 KV Cache 毫无价值。**永远先 `torch.equal`/`allclose` 验证和朴素一致，再看加速比。** 这是 §4.4 反复强调的 AI Infra 基本素养。
5. **采样时直接比 `torch.equal` 失败** —— 用 `multinomial` 采样时，朴素和 KV Cache 要喂**同一个随机数状态**才可比；或干脆用贪心（`argmax`）做正确性测试，采样另测。
6. **benchmark 没预热 / 没 synchronize** —— GPU 异步 + 冷启动，不处理量到假数据。计时前 warmup、首尾 `torch.cuda.synchronize()`。（Day3 §5 的老规矩，KV Cache 测速同样适用）
7. **超过 block_size 不处理** —— 序列长到超过模型 `block_size`，位置编码会越界报错。生产里要么滑动窗口、要么扩展位置编码（RoPE 等），nanoGPT 阶段先确保生成长度 < block_size。
8. **以为 KV Cache 省显存** —— 正相反！它**省计算、费显存**（§6）。初学者常把"Cache"和"省内存"画等号，方向反了。

> **黄金调试流程**：写完 KV Cache，**第一件事永远是跑 §4.3 的 `torch.equal` 断言**。绿了再测速、再 profile。这个"朴素版当对照、优化版必须等价"的习惯，会贯穿你整个 AI Infra 生涯（量化、写 CUDA kernel、图优化，全都靠它兜底正确性）。

---

## 10. 自测题（合上笔记，能脱口而出才算过）

1. 朴素自回归生成为什么是 O(n²)？把"1+2+...+n"那笔账讲出来。（→ §1.3）
2. KV Cache 缓存的是 K/V 还是 Q？为什么不缓存 Q？（→ §2.1）
3. 为什么"缓存历史 K/V"是安全的？这件事和 causal mask 有什么关系？为什么 BERT 不能这样缓存？（→ §2.2，**命门**）
4. KV Cache 把"投影计算"从 O(n²) 降到了几？attention 的"读取"还是 O(n²) 吗？（→ §2.3，**诚实细节**）
5. 在 §4 的 `generate_kv` 里，哪一行是 prefill、哪一行是 decode？（→ §4.2、§5.1）
6. 为什么 prefill 是 compute-bound、decode 是 memory-bound？各用一句大厨类比说清。（→ §5.2-§5.3，**今天最重要**）
7. 写出 KV Cache 显存公式的 7 个因子，并说出为什么生产环境它能比模型权重还大。（→ §6.1-§6.2）
8. KV Cache 显存"又大又碎"催生了什么技术？它借鉴了操作系统的什么思想？（→ §6.3）
9. 为什么 batching 在 decode 阶段几乎"免费"提吞吐？这个红利为什么对 prefill 没那么大？（→ §7.2，**从 memory-bound 推出来**）
10. continuous batching 比静态 batching 强在哪？用一个类比说清。（→ §7.3）
11. 在 chrome trace 里，你靠哪几个信号判断 decode 是 memory-bound？（→ §8.3，**第三个 checklist 硬技能**）
12. 验证 KV Cache 写对了，唯一可靠的办法是什么？（→ §4.3、§9 黄金流程）

> 参考答案位置已标注。Q3/Q6/Q8/Q11 是计划明文要求的完成标准，练到张口就来。

---

## 11. 与已有笔记的串联

| 今天的内容 | 关联到你已有的 | 关系 |
|---|---|---|
| 朴素 O(n²) 浪费、缓存动机（§1） | Day5 `kv_cache_motivation.md` 的耗时曲线 | 昨天画曲线发现浪费，今天实现把它干掉 |
| 缓存安全性 ← 因果性（§2.2） | Day2 §3.3、Day3 `test_causal_no_peeking` | 因果 mask 不只防偷看，还送了"历史可缓存"红利 |
| KV Cache 的 attention 改造（§3） | Day3 `CausalSelfAttention`（fused QKV） | 在 Day3 attention 上加 past_kv 入/出 |
| 优化版 vs 朴素版数值等价（§4.3-§4.4） | Day3 §6 `test_loop_equals_fused` | 同一个 AI Infra 职业素养：先有朴素对照 |
| decode memory-bound（§5.3） | W5 Roofline `flops_vs_latency.md` | 低算术强度卡在带宽斜坡——Roofline 活例子 |
| "省搬运不是省计算"（§5.4、§7.2） | Day3 §4 fused QKV 同一句话 | 今天 decode 给这句话最重的注脚 |
| KV Cache 显存代价 → PagedAttention（§6） | W5 `inference_optimization_landscape.md` 术语2 | 把"知道是什么"变成"知道它解决什么" |
| batching / continuous batching（§7） | W5 同上 术语3 | 从 memory-bound 推出 batching 红利 |
| profiler chrome trace（§8） | W5 `profiler_chrome_trace.md`（ResNet） | 同一把听诊器，对准 nanoGPT 生成 |
| 全部 → 小米课题（贯穿） | [[project-xiaomi-research]] 主线1/3/4 | 性能画像、巨核动机、continuous batching 协同 |

---

## 12. 完成标准 checklist（对齐计划 line 1761-1791 / 1871-1878）

- [ ] `src/kv_cache.py` 跑通：**`torch.equal` 正确性断言通过**（KV Cache 与朴素逐 token 一致）+ 打印出 256 token 的加速比（固定 seed=1337，可复现）—— 计划 line 1767/1777 硬产出
- [ ] 做了"加速比 vs 生成长度（64/128/256/512）"小实验，画出加速比单调上升曲线（§4.3 建议，简历级配图）
- [ ] `tech_notes/kv_cache_and_two_phases.md` 写完，含四节：§1 实现+加速实测 / §2 prefill·decode 两阶段 / §3 显存代价 / §4 batching —— 计划 line 1778-1782 本周 AI Infra 核心产出
- [ ] 能**脱稿讲清** prefill 为什么 compute-bound、decode 为什么 memory-bound（§5，**完成标准硬要求**），并说清这个分水岭如何决定优化策略
- [ ] 能算出 KV Cache 显存公式、代入 Llama-2 数字感受量级，并讲清**为什么它催生 PagedAttention**（§6，**完成标准硬要求**）
- [ ] 能从 memory-bound 推出 batching 为什么提吞吐、continuous batching 强在哪（§7）
- [ ] profiler chrome trace 跑通，导出 `logs/nanogpt_trace.json`，并能**指出 decode 段 memory-bound 的视觉信号**（§8，计划 line 1875 硬技能）
- [ ] `W6_day6_log.md` 记录：256 token 加速比 + profiler 观察结论（§8.3 模板）+ 实现踩了 §9 哪个坑（大概率是 pos_offset 或 mask 切片）
- [ ] 更新 W5 `inference_optimization_landscape.md`：KV Cache 标"已能完整实现+解释"，PagedAttention / continuous batching 标"已能解释动机"

> **今天的一句话总结**：KV Cache = 把"反复重算的历史 K/V"缓存下来（因果 mask 保证安全），用**显存**换**计算**，把生成从 O(n²) 拉到接近 O(n)。它把推理劈成两段——**prefill（compute-bound，吞整个 prompt）**和 **decode（memory-bound，逐 token 挤牙膏）**。decode 卡在带宽上这一个事实，往下推出了三件大事：**量化**（搬得少）、**batching/continuous batching**（摊薄权重搬运）、**PagedAttention**（管好缓存吃掉的巨量显存）。你今天亲手实现 + 实测加速 + profiler 验证瓶颈，正好踩在小米课题"性能画像→识别 decode 访存瓶颈"的入口上。

---

## 13. 产出文件清单（本日交付）

```
week6_nanogpt/
├── src/
│   ├── multihead_attention.py     # Day3：fused QKV 多头
│   ├── model.py                   # Day4：完整 GPT（KV Cache 可改造它，本笔记用 MiniGPT 自包含演示）
│   ├── generate.py                # Day5：朴素 generate
│   ├── kv_cache.py                # 今天：KV Cache attention + MiniGPT + 两种 generate + 256token 测速
│   └── profile_generate.py        # 今天：profiler 跑生成，导出 chrome trace
├── logs/
│   └── nanogpt_trace.json         # 今天：chrome trace（拖进 chrome://tracing 看 decode 空隙）
└── tech_notes/
    ├── kv_cache_motivation.md     # Day5：朴素 O(n²) 动机
    └── kv_cache_and_two_phases.md # 今天本周 AI Infra 核心产出（§5-§7 即其四节内容）
```

- `kv_cache.py` 即 §3+§4，可 `python src/kv_cache.py` 跑，先看正确性断言、再看 256 token 加速比。
- `profile_generate.py` 即 §8.2，跑完把 `logs/nanogpt_trace.json` 拖进 `chrome://tracing`，找 decode 段那串带空隙的小 kernel。
- `kv_cache_and_two_phases.md` 的四节内容，直接取本笔记 §4.3（实测）+ §5（两阶段）+ §6（显存代价）+ §7（batching）整理即可。

> **下一步（Day7，本周收尾）**：读 Attention Is All You Need 对照你手写代码逐块认领，写本周元笔记 `week6_industrial_view.md`（把 Attention O(n²) / KV Cache 两阶段 / 残差流+LN 三条线串成三联画），更新 `inference_optimization_landscape.md`，清理 `week6_nanogpt/` 写最终 README + 3 次有意义 commit（attention / 完整 GPT / KV Cache 优化）。**到 W8 正式进推理优化时，你今天实现的这个 nanoGPT + KV Cache，就是被 `torch.compile`/量化/FlashAttention 优化的那个对象——也是你小米课题的练手地基。**
