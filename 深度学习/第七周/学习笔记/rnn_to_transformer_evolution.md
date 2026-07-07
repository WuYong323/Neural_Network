# RNN → Transformer 演进:一条主线读懂"推理优化为什么死磕 Attention / KV Cache"

> **本笔记的唯一目标**:让你用**一条因果链**讲清楚——序列建模从 RNN 走到 Transformer,本质是"**串行→并行**"的胜利;而这场胜利的账单,落在了 **Attention 的 O(n²)** 和 **KV Cache 的显存**上。看完你要能脱口而出:**RNN 的 hidden state 是"有损压缩历史",KV Cache 是"无损保留全部历史"——更强,但更贵,所以推理优化的主战场必然在这里。**
>
> **串联**:这是 W7「Andrew Ng 深度学习专项 Course 5(Sequence Models)收尾」的演进总结,也是整个 DL 专项 5 门课的**完结里程碑**。它向上承接 W6「亲手实现 KV Cache / prefill / decode」,向下为 [W7 Day1 Roofline](../W7_Day1_Roofline_算术强度与H100脊点.md)(为什么 decode 是 memory-bound)和 [W7 Day2 Profiler 工具链](../W7_Day2_三级Profiler工具链_torchprofiler_nsys_ncu.md) 补上"**为什么瓶颈天然就在这**"的认知背景。Day1 里那句"KV-cache 的 `cat` 落在 Roofline 斜坡上",本笔记 §6 给它补上**底层实现解释**。

---

## 0. 开篇:你要能不看资料答出来的 6 个问题

1. 为什么说 RNN"**训练慢的锅不在算力,在串行**"?并行 vs 串行,差在哪一步?
2. LSTM / GRU 用门控解决了"长依赖遗忘",但**没解决**的那个更根本的问题是什么?
3. Transformer 的 self-attention 凭什么能并行?它把"串行的时间依赖"换成了什么?换来的**代价**是什么(两个,缺一不可)?
4. 一句话:RNN 的 **hidden state** 和 Transformer 的 **KV Cache**,在"如何携带历史"上,本质区别是什么?
5. 推理时 `k_cache = torch.cat([k_cache, k_new])` 这一步,**为什么会浪费显存**?为什么不直接"按需要的精确大小"申请?
6. 为什么说"推理优化的主战场在 attention / KV Cache",是这条演进线的**必然结论**而不是巧合?

> 第 4、5 题是这份笔记的灵魂。如果你能把第 5 题从"我知道会浪费"讲到"显存分配器为什么**故意**多给",这一天就值了。

---

## 1. 一页速览(AI Infra 视角的演进总表)

> 这一节就是任务要求的"**1 页演进总结**"。下面的表 + 一段话,是可以单独撕下来贴墙上的核心;§2 之后是为了让你**真懂**这张表而做的深挖。

| 阶段 | 怎么携带历史 | 能否并行(训练) | 时间/计算复杂度 | 显存代价 | AI Infra 一句话 |
|---|---|---|---|---|---|
| **RNN** | 单个固定大小的 hidden state `h`,逐步**覆盖式更新** | ❌ 串行(第 t 步等第 t−1 步) | O(n) 步,每步 O(d²) | 小(只存 1 个 `h`) | 串行 → GPU 算力大量闲置,训练慢的真因 |
| **LSTM / GRU** | hidden state + cell state,加**门控**选择性记忆/遗忘 | ❌ 仍串行 | O(n) 步 | 略大(多一个 cell state) | 缓解长依赖,但串行枷锁没解开 |
| **Transformer** | **不压缩**,保留每个 token 的 K、V 向量,用 attention 直接两两交互 | ✅ 完全并行(一个大矩阵乘) | 训练 O(n²),attention 矩阵 n×n | 大(注意力矩阵 + 推理期 KV Cache 随序列线性增长) | 并行换来吞吐,代价是 O(n²) 算力 + KV Cache 显存爆炸 |

**一段话核心论断(背下来)**:

> RNN 把"到目前为止的所有历史"**有损压缩**进一个固定大小的向量 `h`——像把一整本书逼着你用一句话复述,信息必然丢失,且远处的内容被反复覆盖、越传越糊(长依赖问题)。Transformer 干脆**不压缩**:它把每个 token 的 Key/Value 全部**无损保留**下来(这就是 KV Cache),让当前 token 用 attention 直接去"翻"任意一个历史 token。**表达能力的飞跃(无损 > 有损)直接兑换成了存储与带宽的飞跃账单(线性增长的 KV Cache + O(n²) 的注意力)**。于是,推理优化无论你从哪个方向切入(显存、延迟、吞吐),最后都会撞到同一堵墙:**attention 怎么算得快(FlashAttention)、KV Cache 怎么存得省(MQA/GQA、量化、PagedAttention)**。这不是巧合,是这条演进线的终点必然。

---

## 2. RNN:串行的代价(原理 → 代码 → 工业锚点)

### 2.1 原理 + 直觉:hidden state 是"一句话复述"

**是什么**:RNN(Recurrent Neural Network,循环神经网络)处理序列时,维护一个固定大小的向量 `h`(hidden state,隐藏状态)。每读一个新 token,就用"旧的 `h` + 新输入"算出"新的 `h`",然后**把旧的覆盖掉**。

**类比**:传话游戏。第 1 个人看完整句话,小声复述给第 2 个人(压缩成一句),第 2 个人没看过原文,只能基于"听到的 + 自己的理解"再传给第 3 个人……传到第 20 个人,最初的信息早被反复转述磨损得面目全非。`h` 就是"当前这个人脑子里记住的东西"——**大小固定**(脑容量有限),所以序列越长,早期信息被覆盖得越狠。这就是 RNN 臭名昭著的**长依赖(long-range dependency)问题**的直觉来源。

**核心**(别停在类比):`h` 是对"从开头到现在所有历史"的**有损压缩**。无论你喂 10 个还是 10000 个 token,`h` 的维度不变(比如 768 维)。信息论上,你不可能把任意长的历史无损塞进固定容量——必然丢信息。这一点,是后面对比 KV Cache 的关键锚点。

### 2.2 可运行代码:亲眼看见"串行依赖"

```python
# 环境: python>=3.9, torch>=2.0  (CPU 即可,无需 GPU)
# 运行: python rnn_serial_demo.py
import torch

torch.manual_seed(0)
T, d_in, d_h = 5, 4, 8          # 序列长 5,输入维度 4,隐藏维度 8

# 一个最朴素的 RNN cell: h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + b)
Wxh = torch.randn(d_in, d_h) * 0.1
Whh = torch.randn(d_h,  d_h) * 0.1
b   = torch.zeros(d_h)

def rnn_step(x_t, h_prev):
    # 为什么是这三项相加: 新输入的贡献 + 历史的贡献 + 偏置, tanh 把值压回 (-1,1) 防爆炸
    return torch.tanh(x_t @ Wxh + h_prev @ Whh + b)

xs = torch.randn(T, d_in)       # 5 个时间步的输入
h  = torch.zeros(d_h)           # h_0 全零

hs = []
for t in range(T):              # ★ 这个 for 循环就是 RNN 的"原罪"
    h = rnn_step(xs[t], h)      # 第 t 步必须拿到第 t-1 步的 h 才能开算
    hs.append(h)
hs = torch.stack(hs)
print("每步 hidden state 形状:", hs.shape)   # 预期输出: torch.Size([5, 8])
print("注意: 算 hs[3] 之前,hs[0..2] 必须已经算完——这是物理上的强制顺序")
```

**为什么这么写能说明问题**:`h = rnn_step(xs[t], h)` 里,等号右边的 `h` 是上一轮的输出。**第 t 步的输入,是第 t−1 步的输出**。这是一条无法打断的依赖链。哪怕你有 10000 个 GPU 核心,它们也只能干等着——因为算第 100 步必须先有第 99 步的结果。

### 2.3 工业锚点:串行 = 把 GPU 当单核 CPU 用

GPU 的本事是**同时**做几万个乘加(SIMT,单指令多线程)。RNN 的 for 循环却逼着它**一步一步来**,每一步只有一个小矩阵乘(`(1,d)@(d,d)`),算术强度极低,绝大多数计算单元在空转。这正是 [W7 Day1 Roofline](../W7_Day1_Roofline_算术强度与H100脊点.md) 讲的 **memory-bound + 利用率低**的双重暴击。

> **历史回响**:RNN 训练慢,慢的不是单步运算量,而是"**不能把时间维度铺开并行**"。Transformer 论文标题 *Attention Is All You Need* 真正的潜台词是 *Parallelism Is All You Need*——它最大的工程贡献,是把"沿序列方向的串行"干掉了。

---

## 3. LSTM / GRU:给压缩历史装上"阀门",但串行枷锁还在

### 3.1 原理 + 直觉:门控 = 选择性地记和忘

**问题背景**:朴素 RNN 的 `h` 每步被 `tanh(...)` 整个重写,早期信息很快被冲淡(数学上还会梯度消失/爆炸)。**LSTM(Long Short-Term Memory,长短期记忆网络)** 的解法:除了 `h`,再加一条**cell state `c`**(记忆主干),并用三个**门(gate)** 控制信息流——

- **遗忘门(forget gate)**:决定旧记忆 `c` 里哪些该丢。类比:整理笔记本时划掉过时的条目。
- **输入门(input gate)**:决定新信息里哪些值得写进 `c`。类比:只把重点抄进本子。
- **输出门(output gate)**:决定从 `c` 里读出多少给当前的 `h`。

**GRU(Gated Recurrent Unit,门控循环单元)** 是 LSTM 的精简版(2 个门、合并 `c` 和 `h`),参数更少、跑得更快,效果常常接近。

**核心**:门控的本质是**让"覆盖式更新"变成"可加性的、被调制的更新"**——`c_t = f_t * c_{t-1} + i_t * c̃_t`,这个"乘法门 + 加法"结构让梯度能沿 `c` 这条"高速公路"流得更远,这才是它缓解长依赖的真正原因(不是玄学,是 `c` 上没有反复的 `tanh` 挤压)。

### 3.2 关键代码:看清门控的"乘 + 加"主干

```python
# 环境: torch>=2.0; 这里只展示 LSTM 一步的核心算式(略去权重初始化细节)
import torch

def lstm_step(x_t, h_prev, c_prev, W):
    # W 把 [x_t, h_prev] 一次性映射出 4 组,分别给 i/f/g/o 四个门,工程上合并成一个大矩阵乘(更快)
    z = torch.cat([x_t, h_prev], dim=-1) @ W      # 一次矩阵乘搞定四个门 → 减少 kernel 启动
    i, f, g, o = z.chunk(4, dim=-1)
    i, f, o = i.sigmoid(), f.sigmoid(), o.sigmoid()   # 门: 0~1 的"开合度"
    g = g.tanh()                                       # 候选记忆
    c = f * c_prev + i * g          # ★ 记忆主干: 遗忘门*旧记忆 + 输入门*新记忆 (乘+加,梯度高速路)
    h = o * c.tanh()                # 输出门控制读出多少
    return h, c
```

> **工程细节(为什么合并成一个 `@ W`)**:四个门若分开算就是 4 次小矩阵乘 = 4 次 GPU kernel 启动,启动开销(launch overhead)在小算子上能占大头。合并成一个大矩阵乘再 `chunk`,是**算子融合(operator fusion)** 思想的雏形——这正是你在小米课题里要对"巨核 / AutoMegaKernel"做的事:把一堆小算子拼成一个大 kernel,少启动、少访存。

### 3.3 工业锚点:为什么 2017 年之后工业界基本抛弃了 LSTM 做大模型

LSTM/GRU 把"记不住远处"缓解了,但**那条 for 循环(串行)原封不动**。序列长度 n,就得跑 n 个串行步,GPU 利用率天花板被锁死。在"数据爆炸 + 显卡爆发"的时代,**能不能喂满 GPU**比"单个 cell 多聪明"重要得多。Transformer 赢在工程可扩展性(scalability),而不只是精度——**这是 AI Infra 视角最该记住的一课:能并行的次优结构,常常碾压不能并行的最优结构。**

---

## 4. Transformer:用 O(n²) 的并行,换掉 O(n) 的串行

### 4.1 原理 + 直觉:从"传话"到"开圆桌会"

**核心一句话**:Self-attention(自注意力)让序列里**每个 token 直接和所有 token 交互**,不再需要信息沿时间轴一棒一棒传。

**类比**:RNN 是传话游戏(信息逐人转述,失真且慢);Transformer 是**开圆桌会议**——每个人(token)同时看到桌上所有人的发言,自己决定"我要重点听谁的"(注意力权重)。没有先后顺序的强制依赖,所有人的"听谁"可以**一次性并行算出来**。

**机制**:每个 token 生成三个向量——
- **Query(查询,Q)**:我想找什么。类比:我举手要问的问题。
- **Key(键,K)**:我能提供什么。类比:每个人胸前的"我擅长 X"标签。
- **Value(值,V)**:我实际的内容。类比:我真正要说的话。

当前 token 用自己的 Q 去和**所有** token 的 K 做点积(算"相关度"),softmax 归一化成权重,再用这些权重去加权求和所有 V。`Q·Kᵀ` 是个 **n×n 的矩阵**——这就是 **O(n²)** 的来源,也是账单的第一项。

### 4.2 可运行代码:一个矩阵乘,吃掉整条 for 循环

```python
# 环境: torch>=2.0 (CPU 可跑). 运行: python attention_demo.py
import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, d = 6, 16                       # 序列长 6,每个 token 16 维

X = torch.randn(T, d)
Wq, Wk, Wv = (torch.randn(d, d) * 0.1 for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv   # 三个投影,(T, d)——注意:对所有 token 一次性算完,没有 for!

def attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d ** 0.5   # ★ (T,T) 相关度矩阵: 这一坨就是 O(n²)
    #          为什么除 sqrt(d): d 大时点积数值变大,softmax 会饱和(梯度趋零),缩放回稳定区间
    weights = F.softmax(scores, dim=-1)            # 每行归一化: 当前 token 对所有 token 的注意力分配
    return weights @ V                              # 用注意力加权求和所有 Value

out = attention(Q, K, V)
print("输出形状:", out.shape)        # 预期: torch.Size([6, 16])
print("注意力矩阵形状:", (Q @ K.transpose(-2,-1)).shape)   # torch.Size([6, 6]) ← n×n,n=6
print("关键: 上面没有任何沿时间步的 for 循环——6 个 token 的输出一个矩阵乘并行算出")
```

**为什么这是革命**:对比 §2.2 的 `for t in range(T)`,这里**整条序列的交互压成了两个矩阵乘**(`Q@Kᵀ` 和 `weights@V`)。矩阵乘正是 GPU 最爱、最能喂满算力的操作。训练时,n 个 token 的损失可以**同时**算、同时回传——这就是 Transformer 训练能堆到千卡集群的根本原因。

### 4.3 工业锚点:账单的两项

并行不是免费的,代价有两笔,**记牢**:

1. **计算 O(n²)**:`Q@Kᵀ` 是 n×n。序列翻倍,attention 计算量变 4 倍。长上下文(128k token)下,这是要命的。→ 业界解法:**FlashAttention**(不省 FLOPs,但靠分块+不落地 n×n 矩阵,把访存从 O(n²) 压到近似 O(n),大幅提速省显存)。
2. **显存(推理期):KV Cache 随序列线性增长**。这是下一节的主角,也是你科研方向的正中心。

---

## 5. 深层串联(本笔记灵魂):hidden state vs KV Cache

> 这一节回答开篇第 4、6 题。把它讲透,你就拿到了"为什么推理优化死磕 KV Cache"的**第一性原理**。

### 5.1 两种"携带历史"的哲学对立

回到最根本的问题:**模型怎么记住前文?**

- **RNN 派(有损压缩)**:历史 → 一个固定大小的 `h`。**优点**:省,无论多长序列,`h` 就那么大。**缺点**:有损,远处信息被覆盖,长依赖差。
- **Transformer 派(无损保留)**:不压缩!推理时,把**每个历史 token 的 K 向量和 V 向量原样存下来**——这堆存下来的 K、V,就是 **KV Cache(键值缓存)**。生成第 t 个 token 时,直接拿它的 Q 去和**缓存里全部历史 K** 算注意力。**优点**:无损,任意远的 token 都能精确"翻"到。**缺点**:贵,存的东西随序列**线性增长**。

| 维度 | RNN hidden state | Transformer KV Cache |
|---|---|---|
| 历史的表示 | 有损压缩成 1 个固定向量 | 无损保留每个 token 的 K、V |
| 大小随序列 n | **不变**(常数) | **线性增长** O(n) |
| 取用历史 | 间接(全糊在 `h` 里) | 直接(按 token 精确寻址) |
| 长依赖 | 差(被覆盖) | 强(精确保留) |
| 推理显存压力 | 几乎没有 | **巨大,主要矛盾** |

**一句话**:KV Cache 是"用空间换表达力"的极致——它把 RNN 那个被反复覆盖的 `h`,摊开成了"**永不覆盖、全程留底**"的一长条。**更强,但更贵。** 这八个字,就是推理优化这门学科存在的理由。

### 5.2 为什么推理要 Cache?——prefill / decode 与"重复计算"

为什么非缓存不可?因为自回归生成(autoregressive,一次吐一个 token,吐完拼回去再吐下一个)有大量重复:

- **Prefill(预填充)阶段**:把用户的整段 prompt 一次性喂进去,**并行**算出所有 token 的 K、V,存进 Cache。这一步是 compute-bound(大矩阵乘,GPU 吃得饱)。
- **Decode(解码)阶段**:逐个生成新 token。**关键洞察**:历史 token 的 K、V 在每一步里**完全相同**,若不缓存,每生成一个新词都要把前面几千个 token 的 K、V **从头重算一遍**——平方级浪费。缓存后,每步只需算**新 token 那一个** K、V,追加进 Cache 即可。

> 这就是 W6 你亲手实现的 prefill/decode 分离。而 decode 阶段"只算 1 个新 token、却要把整个模型权重 + 整个 KV Cache 读一遍"的特性,正是 [W7 Day1 Roofline](../W7_Day1_Roofline_算术强度与H100脊点.md) 里"decode 必然 memory-bound"的根因——算得少,搬得多。

---

## 6. 底层深挖:`torch.cat` 拼 KV Cache,到底为什么浪费显存?

> 这是你点名要的"深入内核":**为什么 `cat` 会请求多余的、浪费的空间?为什么不能按精确大小申请?** 下面从内存分配器讲到底。

### 6.1 朴素写法的两宗罪:O(n²) 拷贝 + 反复分配

最直觉的 decode KV Cache 更新,就是每步 `cat` 一下:

```python
# 朴素版 (能跑,但工业上是反面教材). 环境: torch>=2.0, 有 cuda 更明显
import torch
d = 128
k_cache = None
for t in range(1000):                       # 生成 1000 个 token
    k_new = torch.randn(1, d, device='cuda')           # 本步新 token 的 K 向量 (1, d)
    if k_cache is None:
        k_cache = k_new
    else:
        k_cache = torch.cat([k_cache, k_new], dim=0)   # ★ 罪魁: 看似"追加",实则重建
```

**`torch.cat` 到底干了什么(底层)**:它**不是**在原张量后面"接一段"。张量要求底层是一块**连续(contiguous)** 的内存,而原来的 `k_cache` 后面那块内存很可能已经被别人占了。所以 `cat` 的真实动作是:

1. 向显存分配器**申请一块全新的、能装下 (t+1, d) 的连续内存**;
2. 把旧的 `k_cache`(t 行)**整个拷贝**过去;
3. 把 `k_new`(1 行)拷在末尾;
4. 旧的 (t, d) 那块内存**作废**(等待回收)。

于是两宗罪:
- **罪一:O(n²) 的拷贝**。第 t 步要拷 t 行。1000 步累计拷贝 1+2+…+1000 ≈ 50 万行。序列越长,越平方级地浪费**显存带宽**——而 decode 本来就是 memory-bound,等于往伤口上撒盐。
- **罪二:反复分配 + 旧块作废**,把显存分配器搅得天翻地覆(见下)。

### 6.2 核心问题:为什么分配器"故意"多给、为什么不精确申请?

你的疑问非常对:既然每步只多 1 行,为什么不就申请那 1 行的精确大小?答案分两层。

**第一层:`cudaMalloc` 太慢,所以 PyTorch 自己管一个显存池(caching allocator,缓存分配器)。**

直接向显卡驱动要内存(`cudaMalloc`)是个**重操作**:它会**同步整个设备**(等所有流跑完)、走驱动、慢得像系统调用。如果每步 `cat` 都真去 `cudaMalloc` + `cudaFree`,decode 会被分配开销拖死。所以 PyTorch 在驱动之上自建了一个**缓存分配器**:一次性从驱动拿一大段显存,自己切块分给你;你 `free` 的块它不还给驱动,而是**留在池里复用**。`nvidia-smi` 看到 PyTorch 占了一大坨显存不降,就是这个池。

**第二层:为了让"还回来的块"能被"下次请求"复用,分配器必须把尺寸"向上取整",这就是浪费的来源。**

缓存分配器的复用逻辑是:你 free 一个块,它进 freelist;你下次申请,它找一个**够大的空闲块**切给你。这里有个致命细节——**如果块的尺寸是五花八门的精确值,几乎没有任何一个空闲块能匹配下一次的请求**,池子会碎成一地无法复用的小渣(碎片化,fragmentation)。

解决办法:**把请求尺寸向上取整到规整的档位**,让块尺寸标准化、可复用。PyTorch 的缓存分配器(`CUDACachingAllocator`)大致是这么干的(常量随版本微调):

- 申请尺寸先**向上取整到 512 字节的倍数**(`kMinBlockSize = 512`)——所以你要 1 行 `1×128` 的 fp16(256 字节),实际可能给你 512 字节;
- 小于 1 MB 的小块,从 **2 MB 的段**里切(`kSmallBuffer`);大于约 10 MB 的大块,**向上取整到 2 MB 的倍数**(`kRoundLarge`)再给。

**这就是"为什么请求会有多余浪费的空间"的根答案**:不是 bug,是**故意的设计权衡**——

> 用一点**内部碎片(internal fragmentation,块内多给的、用不上的那截)**,换来**块的标准化与可复用性**,从而避免昂贵的 `cudaMalloc` 和更可怕的**外部碎片**。"精确申请"看似省,实则会让池子彻底碎片化、复用率归零、被迫频繁找驱动要新内存——**整体反而更慢更费**。这是经典的"**空间换速度 + 空间换可复用性**"。

而 `cat` 的雪上加霜在于:它每步申请的尺寸**都在变大**(t 行 → t+1 行),刚 free 的"t 行块"装不下"t+1 行"的新请求,于是**旧块卡在池里、新块不断新切**,把碎片化和池膨胀拉满。

### 6.3 工业正解:一次性预分配 + 写切片(零增量分配、零拷贝)

```python
# 工业版: 开局按 max_len 预分配一整块,之后只"写切片",不再 cat. 环境: torch>=2.0
import torch
d, max_len = 128, 1024
k_cache = torch.empty(max_len, d, device='cuda')    # ★ 一次性分配,之后零增量分配
for t in range(1000):
    k_new = torch.randn(1, d, device='cuda')
    k_cache[t] = k_new            # 原地写入第 t 行: 不分配新内存、不拷贝旧数据
    k_valid = k_cache[:t + 1]     # 取"已生成部分": 这是 view(视图),不复制数据
```

**代价说明白**:预分配把"`cat` 的时间浪费(拷贝 + 分配)"换成了"**空间浪费**"——你按 `max_len=1024` 占着,哪怕实际只生成了 10 个 token,剩下 1014 行的显存也被你**预订占用、闲着**。这就是另一种"多余的空间",但它换来了 decode 路径上**完全没有分配和拷贝**,干净利落。

**所以你看到的两难是真实存在的**:`cat` 浪费**时间+带宽**(且churn 分配器),预分配浪费**容量**。二选一,看你缺什么。

### 6.4 业界最优解:PagedAttention(vLLM)——把"预订浪费"也干掉

预分配的痛点:**每个请求都得按最坏情况 `max_len` 预订**,并发 100 个请求时,绝大多数 token 没用上,显存被"预订"白白吃光(这叫 over-provisioning,过度预留)。

vLLM 的 **PagedAttention** 借了操作系统**虚拟内存分页**的思想:把 KV Cache 切成固定大小的**小块(block / page)**,需要时才从一个**全局共享的块池**里领一块,用逻辑→物理的**块表(block table)** 映射起来——逻辑上连续,物理上可以散落。好处:

- **几乎零内部碎片**(块很小),也不必为每个请求预订 `max_len`;
- 显存利用率从预分配的"经常 50%+ 浪费"提到**接近 96%+**,同样的卡能塞更多并发请求,吞吐大涨;
- 还能玩 **prefix 共享**(多个请求共用同一段系统 prompt 的 KV 块,copy-on-write)。

> **闭环**:从 RNN 的"省到极致(有损)"→ Transformer 的"全存(无损但贵)"→ `cat`/预分配的"时间 vs 空间二选一"→ PagedAttention 的"像操作系统管内存一样精细地管 KV Cache"。**整个演进的尽头,是把'怎么存历史'当成一个系统/内存管理问题来解**——这,就是 AI Infra。这也是 vLLM 源码值得你暑假精读的原因。

---

## 7. 为什么"主战场必在 attention / KV Cache"——收束成一条因果链

把全篇拧成一句话的因果链(能复述就是真懂了):

> 要并行(RNN 串行太慢)→ 选 Transformer → attention 让 token 两两交互(O(n²) 计算)→ 推理自回归要避免重算 → 必须缓存全部历史 K/V(KV Cache)→ KV Cache 随序列 + 并发**线性甚至成倍膨胀** → 它既吃**显存容量**、又吃 decode 的**访存带宽**(memory-bound)→ 于是显存怎么省(MQA/GQA、KV 量化、PagedAttention)、attention 怎么算得快(FlashAttention)就成了**绕不开的两条主线**。

| 痛点 | 根源(本笔记哪节) | 业界招式 |
|---|---|---|
| attention 算得慢 / n×n 落地显存 | §4.3 O(n²) | FlashAttention(分块,不落地中间矩阵) |
| KV Cache 太占显存 | §5.1 无损保留 | MQA / GQA(多 query 共享 KV 头)、KV Cache 量化(fp16→int8/fp8) |
| KV Cache 分配/碎片化 | §6.2 分配器取整 | 预分配、PagedAttention(分页) |
| decode 访存瓶颈 | §5.2 + W7 Day1 | 算子融合、连续批处理(continuous batching) |

---

## 8. 自测题(先合上笔记答,再翻对应节核对)

1. 用"传话游戏 vs 圆桌会议"解释 RNN 和 Transformer 的本质差异,并点明各自的代价。→ §2.1 / §4.1
2. LSTM 的门控为什么能缓解长依赖?写出 cell state 那行"乘+加"公式并解释它是"梯度高速路"。→ §3.1 / §3.2
3. Transformer 训练能并行,但代价是哪两项?各对应什么业界优化?→ §4.3
4. 一张表说清 hidden state 与 KV Cache 在"大小随序列""取用历史""长依赖"上的区别。→ §5.1
5. **(核心)** `k_cache = torch.cat([k_cache, k_new])` 为什么 O(n²) 拷贝?为什么分配器会多给空间、为什么不精确申请?预分配又浪费了什么?→ §6.1–6.3
6. 把"RNN→Transformer→KV Cache→推理优化主战场"用一条因果链讲出来。→ §7

> 参考答案不另附:每题后标了对应小节,**讲不出来就回去重读那节**,这比背答案有用。

---

## 9. 与已有笔记的串联 & 完成标准

### 9.1 串联表

| 关联笔记 | 关系 |
|---|---|
| W6 · 亲手实现 KV Cache / prefill / decode | 本笔记是它的**理论上游**:解释了你当时实现的 Cache 为什么非存不可 |
| [W7 Day1 · Roofline](../W7_Day1_Roofline_算术强度与H100脊点.md) | 本笔记 §5.2/§6 解释了"decode 为什么 memory-bound、`cat` 为什么是反模式";Day1 给的是**定量证据**,这里给**演进层面的因果** |
| [W7 Day2 · 三级 Profiler](../W7_Day2_三级Profiler工具链_torchprofiler_nsys_ncu.md) | 当你用 nsys/ncu 看到"`cat`/attention 是热点"时,本笔记告诉你**为什么它们天生是热点** |
| 小米课题 · AutoMegaKernel / 巨核 | §3.2 的"四门合并成一个矩阵乘"= 算子融合雏形,正是巨核要做的事 |
| 暑假 · vLLM 源码 | §6.4 PagedAttention 是精读入口 |

### 9.2 完成标准 checklist

- [x] 倍速刷完 Andrew Ng 深度学习专项 **Course 5(Sequence Models)**:RNN → LSTM/GRU → Attention → Transformer 的演进认知建立
- [x] 产出 1 页演进总表(§1)+ 深度解析(§2–§7),含可运行代码(RNN/LSTM/Attention/KV Cache)
- [x] 打通核心因果链:**有损压缩(hidden state)→ 无损保留(KV Cache)→ 推理优化主战场**
- [x] 啃下 `torch.cat` 浪费显存的**底层原因**(缓存分配器的尺寸取整 / 碎片化权衡),不再停留在"我知道会浪费"
- [ ] (下一步)对照本笔记 §6,回 W6 代码把朴素 `cat` 版 KV Cache 改成预分配版,用 W7 Day2 的 profiler 量出差距

---

## 🎯 里程碑:Andrew Ng 深度学习专项 5 门课全部完成

> **Deep Learning Specialization (deeplearning.ai) — DONE ✅**
>
> 1. Neural Networks and Deep Learning
> 2. Improving Deep Neural Networks (调参 / 正则化 / 优化)
> 3. Structuring Machine Learning Projects
> 4. Convolutional Neural Networks(对应 W5 CNN/ResNet）
> 5. **Sequence Models(本笔记主题:RNN → Attention → Transformer)** ← 收尾完成
>
> **意义**:DL 通识地基补齐。从此学习重心正式从"补课"切换到 **AI Infra 主线**(Roofline → Profiler → KV Cache 优化 → CUDA / vLLM)。这份笔记既是 Course 5 的演进总结,也是整个专项的**毕业证**和"从学知识到做系统"的**分水岭**。

*完结标记另存于 `../W7_里程碑_AndrewNg深度学习专项5门全完成.md`。*
