# nanoGPT 训练与 KV Cache 动机 · 自学笔记

> 主题:① Shakespeare 字符级训练(val loss ≈ 1.5)② 写 `generate()` + 保存 checkpoint ③ 写透 KV Cache 的动机(朴素生成 O(n²) 浪费)
> 定位:这三件事是一条线——**先把一个能用的小 GPT 训出来 → 让它会生成文本 → 发现生成很慢 → 引出 KV Cache 这个推理优化**。
> 对你的意义:第 3 点正是你小米「推理优化」项目的入门地基,vLLM 的核心优化 PagedAttention 就是把 KV Cache 做到极致。

---

## 0. 全景图:这三个任务为什么是一条线

很多人把这三件事当成三个孤立的 todo,其实它们是一个完整故事的三幕:

| 幕 | 任务 | 解决的问题 | 一句话 |
|---|---|---|---|
| 第一幕:训练 | Shakespeare 训到 val loss≈1.5 | 模型怎么从数据里学到语言规律 | 「教会它说话」 |
| 第二幕:推理 | 写 `generate()` + checkpoint | 怎么用训好的模型造出新文本、怎么把成果存下来 | 「让它开口 + 把它存档」 |
| 第三幕:优化 | KV Cache 动机 | 第二幕的生成方式慢得离谱,慢在哪、为什么 | 「发现它说话太慢,找病根」 |

**关键认知:训练(training)和推理(inference)是两个完全不同的阶段。**
- **训练**:喂给模型大量文本,让它调整内部参数(权重),目标是「学会」。这阶段算一次、改一次参数。
- **推理**:参数固定不动,拿它来「用」——输入开头,让它一个字一个字往下续写。

> 类比:训练像「学生备考刷题」(反复纠错、更新脑子里的知识);推理像「考场上答题」(知识固定,只管输出)。AI Infra 工程师(就是你的方向)**绝大部分精力花在推理优化上**——因为模型训好后要被亿万次调用,每次快 1 毫秒、省一点显存,乘以调用量就是巨大的成本节省。这就是为什么第三幕的 KV Cache 对你最重要。

---

## 第一部分:Shakespeare 字符级训练(目标 val loss ≈ 1.5)

### 1.1 是什么:我们到底在训一个什么东西

任务是:拿莎士比亚全集的文本(一个 1MB 左右的 `.txt`),训练一个小型 **GPT**,让它学会模仿莎翁的文风,能续写出像那么回事的古英文戏剧台词。

先拆几个第一次出现的名词:

**GPT(Generative Pre-trained Transformer,生成式预训练 Transformer)**
- 中文直译:「生成式 预训练 变换器」。
- 通俗解释:它本质是一个**「猜下一个字」的机器**。给它一段文字,它输出「下一个字最可能是什么」的概率。就这么简单——你以为它在思考,其实它在做一道永远做不完的填空题:已知前面所有字,猜下一个。
- 类比:像你手机输入法的「联想下一个词」,只不过 GPT 的「联想」强大到能写出整段连贯的文章。

**字符级(character-level)**
- 是什么:模型预测的最小单位是**单个字符**(`a`、`b`、空格、换行…),而不是单词或子词。
- 为什么用它入门:莎士比亚数据小、字符集就几十个,**词表(vocabulary)极小**,模型小、训得快,特别适合在一张卡甚至 CPU 上跑通整个流程。工业界大模型用的是 **子词级(subword,如 BPE)**,但原理完全一样,字符级是最干净的教学版。
- 类比:字符级像「用一个个字母拼单词」,子词级像「用偏旁部首拼汉字」——后者更高效,但前者更容易理解。

**词表 / tokenizer(分词器)**
- token 中文叫「词元」,是模型眼里的最小处理单位。tokenizer 负责把人类的文字**翻译成模型能懂的数字**(因为神经网络只会算数,不认识字母)。
- 字符级的 tokenizer 简单到一句话:给每个出现过的字符编个号。比如 `{'\n':0, ' ':1, '!':2, ..., 'a':39, ...}`。
- 编码(encode)= 文字→数字,解码(decode)= 数字→文字。

### 1.2 为什么:val loss ≈ 1.5 这个数字是怎么来的,凭什么算「好」

这是最容易被忽略、但最该想清楚的一点。先解释两个词:

**loss(损失)**
- 是什么:衡量「模型猜得有多差」的一个数。猜得越准,loss 越小。训练就是想尽办法把它压低。
- 这里用的是 **交叉熵损失(cross-entropy loss)**:简单说,它衡量「模型给正确答案分配的概率有多低」。模型如果对正确的下一个字给了 90% 概率,loss 就很小;只给了 1%,loss 就很大。

**val loss(validation loss,验证损失)**
- 我们把数据切成两份:**训练集(train)** 用来学,**验证集(val)** 模型从没见过,专门用来「考试」。
- val loss = 模型在没见过的数据上的 loss。**它才是真本事**——train loss 低可能只是「背题」,val loss 低才说明真学会了规律。
- 这两者拉开差距(train 很低、val 居高不下)就是**过拟合(overfitting)**:像学生死背answer,一换题就不会。

**那为什么 1.5 是个好目标?(深度部分,务必理解)**

交叉熵 loss 有个非常直观的物理意义:它约等于「模型平均每猜一个字,心里还剩多少 bit 的不确定性」(以 e 为底时单位是 nat,换算成 bit 要除以 ln2≈0.693)。

我们做几个对比,你就懂 1.5 的含金量:

- **完全瞎猜**:莎士比亚文本约 65 个不同字符。如果模型啥也没学、对 65 个字符均匀瞎猜,loss = ln(65) ≈ **4.17**。这是「文盲基线」。
- **训练初期**:loss 通常从 4.x 开始往下掉。
- **val loss ≈ 1.5**:意味着模型把不确定性从「65 选 1 的茫然」压缩到了「大概 e^1.5 ≈ 4.5 选 1 的纠结」。换算成 bit 约是 1.5/0.693 ≈ **2.16 bits/字符**。

> 含义:每写一个字符,模型心里基本只在 4~5 个候选里挑——这已经足够生成出**拼写正确、有词、有标点、句式像戏剧**的文本了。这就是任务描述里「生成像样文本」的量化标准。再往下压到 1.0 会更流畅,但 1.5 是「肉眼可见像模像样」的及格线。

**这是你该养成的习惯:任何 loss 数字,先找到它的「瞎猜基线」做参照,才知道模型到底学到了没。** 工业界看 loss 从不看绝对值,看的是「离理论下界还有多远」。

### 1.3 怎么做:最小可运行的训练骨架

下面是一份**精简但贴近 nanoGPT 真实结构**的训练代码。环境依赖写在注释里。

```python
# ===================================================================
# 环境依赖:python>=3.9, torch>=2.0  (pip install torch)
# 数据:input.txt = 莎士比亚全集纯文本(nanoGPT 仓库 data/shakespeare_char/)
# 能在单张 GPU 上几分钟跑到 val loss≈1.5;纯 CPU 也能跑,只是慢
# 这份代码是「教学骨架」:结构对标 nanoGPT,去掉了分布式/混合精度等工程细节,
# 让你先看懂主干。工业级加速见后文「工业接轨」。
# ===================================================================
import torch, torch.nn as nn
from torch.nn import functional as F

# ---------- 超参数(hyperparameters):训练前由人设定、训练中不变的旋钮 ----------
batch_size  = 64      # 一次喂多少段文本并行训练。越大越稳但越吃显存
block_size   = 256    # 上下文长度(context length):模型一次最多看多少个字符来预测下一个
                      # 这就是后面 KV Cache 故事里的「n」,记住它
n_embd       = 384    # 每个字符被表示成多少维的向量(embedding 维度)
n_head       = 6      # 多头注意力的「头」数(后面讲)
n_layer      = 6      # 堆叠多少个 Transformer Block
dropout      = 0.2    # 随机「丢弃」比例,防过拟合的手段
learning_rate = 3e-4  # 学习率:每次纠错时参数迈多大步。太大震荡,太小学得慢
max_iters    = 5000   # 总共训练多少步
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 有卡用卡,没卡用 CPU

# ---------- 数据准备:文字 → 数字 ----------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))   # 所有出现过的字符,去重排序 = 词表
vocab_size = len(chars)           # 词表大小,莎翁约 65
stoi = {ch: i for i, ch in enumerate(chars)}   # string→int 编码表
itos = {i: ch for i, ch in enumerate(chars)}   # int→string 解码表
encode = lambda s: [stoi[c] for c in s]            # 文字→数字列表
decode = lambda l: ''.join([itos[i] for i in l])   # 数字列表→文字

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))          # 前 90% 训练,后 10% 验证——这就是 train/val 划分
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    """随机取一批 (输入x, 目标y)。y 是 x 整体右移一位——
    因为任务是『已知前面的字,猜下一个字』,所以目标就是把输入错开一位。"""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))   # 随机起点
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])     # 关键:右移一位
    return x.to(device), y.to(device)
```

> **为什么 y 是 x 右移一位?** 这是理解 GPT 训练的题眼。模型在每个位置都要预测「我后面那个字」,所以位置 i 的输入对应的正确答案,就是位置 i+1 的真实字符。一次 forward,所有位置**同时**算 loss——这叫 teacher forcing(强制教学):训练时永远喂真实答案当上文,不用模型自己的预测,这样训得又快又稳。记住这点,它和第三幕「推理时为什么慢」形成鲜明对比。


接下来是模型本体。**注意力机制(attention)是重点**,因为它既是 GPT 的核心,也是第三幕 KV Cache 要优化的对象。

```python
class Head(nn.Module):
    """单个注意力头(attention head)。
    注意力一句话:每个字在预测下一个字时,会『回头看』前面所有字,
    并决定『该重点参考谁』——这就是它能理解长距离上下文的原因。"""
    def __init__(self, head_size):
        super().__init__()
        # 每个字生成三个向量:Query(我想找什么)、Key(我是什么)、Value(我能提供什么信息)
        self.key   = nn.Linear(n_embd, head_size, bias=False)   # K
        self.query = nn.Linear(n_embd, head_size, bias=False)   # Q
        self.value = nn.Linear(n_embd, head_size, bias=False)   # V
        # tril = 下三角矩阵,用来做「因果遮罩」:只能看前面,不能偷看后面(未来)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape          # Batch, Time(序列长度), Channel(维度)
        k = self.key(x)            # (B,T,head_size)
        q = self.query(x)
        # 注意力分数 = Q 和 K 做点积:衡量「每个字该多关注其它每个字」
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5   # 缩放,防数值过大
        # 因果遮罩:把「未来」的位置设成 -inf,softmax 后变 0,即看不到未来
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)        # 归一化成概率,每行加起来=1
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v             # 用注意力权重加权求和 Value = 输出
```

> **Q/K/V 用类比理解(关键):** 想象一场学术研讨会。每个人(每个字)手里有:
> - **Query(查询)**= 我现在想问的问题(「我想了解关于 X 的事」)
> - **Key(键)**= 我胸前的名牌,标明我擅长什么
> - **Value(值)**= 我肚子里真正的干货内容
>
> 一个字要预测下一个字时,它拿自己的 Query 去和在场所有人的 Key 比对(点积),谁的 Key 跟我的 Query 越匹配,我就越关注谁,然后按这个关注度去汇总大家的 Value。**这个「Q 找 K、再取 V」的过程,就是注意力。** 记住 K 和 V——它们就是 KV Cache 里要缓存的那个「KV」。

```python
class MultiHeadAttention(nn.Module):
    """多头:并行跑多个 Head,每个头关注不同角度(有的看语法、有的看语义),再拼起来。"""
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # 拼接所有头
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """前馈网络:注意力负责『看上下文』,FFN 负责『消化加工』这些信息。"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.ReLU(),   # 先升维加工
            nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))  # 再降回原维度
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    """一个 Transformer Block = 注意力 + 前馈,各带残差连接和 LayerNorm。"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        # x + ... 是残差连接(residual):让梯度好传,深层网络才训得动
        x = x + self.sa(self.ln1(x))      # 先归一化再进注意力(Pre-LN,现代标配)
        x = x + self.ffwd(self.ln2(x))
        return x
```

> **为什么要残差连接(residual connection)和 LayerNorm?** 这俩是「深层网络能训起来」的两大功臣。残差(`x + f(x)`)给梯度开了条高速公路,避免层数一深梯度就消失;LayerNorm 把每层数据拉回稳定的分布,防止数值乱飞。**没有它们,6 层还行,几十层上百层的大模型根本训不动。** 这是从「能跑」到「能 scale 到大模型」的关键工程设计。


```python
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)    # 字符→向量
        self.position_embedding = nn.Embedding(block_size, n_embd) # 位置→向量(让模型知道字的先后)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)   # 最后映射回「每个字符的分数」

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                            # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb        # 字的含义 + 字的位置 = 完整输入
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)     # (B,T,vocab_size):每个位置对每个字符打的分

        if targets is None:
            loss = None              # 推理时没有标准答案,不算 loss
        else:
            # 把 (B,T,vocab) 拉平成 (B*T, vocab) 才能算交叉熵
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
```

```python
# ---------- 训练循环 ----------
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW 是当前标配优化器

@torch.no_grad()                     # 评估时关掉梯度计算,省显存、提速
def estimate_loss():
    """分别在 train/val 上各跑几批,取平均——单批波动大,平均才靠谱。"""
    out = {}
    model.eval()                     # 切到评估模式(关掉 dropout)
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()                    # 切回训练模式
    return out

for it in range(max_iters):
    if it % 500 == 0:                # 每 500 步「考一次试」,盯 val loss
        l = estimate_loss()
        print(f"step {it}: train loss {l['train']:.4f}, val loss {l['val']:.4f}")
        # 你要的目标:看着 val loss 一路降到 ≈1.5 就达标了

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # 清空上一步的梯度
    loss.backward()                        # 反向传播:算出每个参数该往哪调
    optimizer.step()                       # 真正更新参数——这就是「学习」发生的瞬间
```

> **训练循环的四步心法(背下来,所有 PyTorch 训练都是这套):**
> 1. `forward` 算预测和 loss(猜 + 看错多少)
> 2. `zero_grad` 清空旧梯度(擦黑板)
> 3. `backward` 反向传播算梯度(找出每个参数该改的方向)
> 4. `step` 更新参数(真正改一点点)
>
> 五千步循环跑完,val loss 从 4.x 降到 1.5,模型就「学会说莎士比亚话」了。**每 500 步打印 val loss 是工业界的基本素养**——不盯着它,你根本不知道模型是在学还是在原地打转,或者已经过拟合了。

---

## 第二部分:写 `generate()` + 保存 checkpoint

### 2.1 是什么:让训好的模型「开口说话」

训练只是让模型「学会」了,但它还没「说」过一个字。`generate()` 就是那个让它开口的函数:**给它一个开头(哪怕只是一个换行符),让它一个字一个字往下续写。**

这里要引出一个最重要的概念:

**自回归生成(autoregressive generation)**
- 中文:「自己回过头来参考自己」的生成方式。
- 是什么:模型每生成一个新字,就把这个新字**接回输入末尾**,再用「包含新字的完整序列」去预测下一个字。如此循环。
- 类比:像接龙写句子。你写「今天」,看着「今天」想出「天气」,再看着「今天天气」想出「真好」……**每一步都要把前面已经写出来的全部重读一遍**,再决定下一个词。
- ⚠️ 记住这个「每步都把前面全部重读一遍」——它就是第三幕 KV Cache 要解决的浪费的根源。


### 2.2 怎么做:generate() 的实现

```python
@torch.no_grad()                     # 生成是推理,不需要梯度
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    idx: 起始上下文,形状 (B, T),装的是已有字符的编号
    max_new_tokens: 要续写多少个新字符
    temperature: 温度,控制随机性(后面解释)
    top_k: 只从概率最高的 k 个候选里采样(后面解释)
    """
    for _ in range(max_new_tokens):
        # ★关键且低效★:每一步都把整个序列裁到 block_size 再喂进去重算
        idx_cond = idx[:, -block_size:]        # 只保留最后 block_size 个(超长了截断)
        logits, _ = self(idx_cond)             # forward 整个序列!(浪费就在这,见第三幕)
        logits = logits[:, -1, :]              # 只要最后一个位置的预测(其它都白算了)
        logits = logits / temperature          # 温度缩放
        if top_k is not None:                  # top-k 过滤:只留最可能的 k 个
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)      # 转成概率分布
        idx_next = torch.multinomial(probs, num_samples=1)  # 按概率抽一个字(不是取最大)
        idx = torch.cat((idx, idx_next), dim=1)  # ★把新字接回末尾★ → 下一轮重算
    return idx

# 把它绑到 GPT 类上(实际写代码时直接写进 class 里):
GPT.generate = generate

# ---------- 用法 ----------
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 从一个换行符(编号0)起头
out = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=200)
print(decode(out[0].tolist()))   # 解码成文字打印出来
```

> **temperature(温度)和 top_k 是两个控制「生成风格」的旋钮,工业界天天调:**
> - **temperature**:想象给概率分布「加热」。温度高(>1)→ 分布变平,模型更敢冒险、更有创意但容易胡言乱语;温度低(<1)→ 分布变尖,模型更保守、更确定但可能重复无聊。0.8 是常用的「略保守又不死板」值。
> - **top_k**:只在概率最高的 k 个字里抽,把长尾的怪字直接排除。防止模型偶尔抽到一个离谱字符把整句带跑偏。
>
> 你调 ChatGPT API 时见到的 `temperature`、`top_p` 参数,就是这同一套东西。**这俩参数不影响模型本身,只影响「怎么从模型的预测里挑字」——这是推理阶段的调味,不是训练。**

### 2.3 怎么做:保存与加载 checkpoint(检查点)

**checkpoint(检查点)**
- 是什么:把模型训练到某一刻的「全部状态」存成文件,以后能原样恢复。
- 为什么必须做:训练动辄几小时几天,中途断电/掉线/被抢卡(你在共享集群上太常见了!)就前功尽弃。checkpoint 让你能「存档读档」。
- 类比:打游戏的存档点。打到一半存一下,挂了从存档点继续,不用从头再来。

```python
# ---------- 保存 ----------
checkpoint = {
    'model': model.state_dict(),        # 模型参数(最核心)
    'optimizer': optimizer.state_dict(),# 优化器状态(AdamW 的动量等,续训必需)
    'iter': it,                         # 训到第几步了
    'val_loss': best_val_loss,          # 当时的 val loss
    'config': {                         # 超参数!不存的话加载时不知道怎么搭模型
        'vocab_size': vocab_size, 'n_embd': n_embd, 'n_head': n_head,
        'n_layer': n_layer, 'block_size': block_size,
    },
    'stoi': stoi, 'itos': itos,         # 词表!不存的话没法 encode/decode
}
torch.save(checkpoint, 'ckpt.pt')

# ---------- 加载 ----------
ckpt = torch.load('ckpt.pt', map_location=device)
model = GPT()                           # 先按 config 搭出一样结构的空壳
model.load_state_dict(ckpt['model'])    # 再把参数灌进去
model.to(device); model.eval()          # 推理前务必 .eval()(关 dropout)
```

> **新手最常踩的 checkpoint 坑(工业界血泪):**
> 1. **只存了 `model.state_dict()`,没存 config 和词表**——下次加载时不知道模型多大、字符怎么编号,直接抓瞎。**checkpoint 要存「重建现场所需的一切」,不只是参数。**
> 2. **存了整个 model 对象**(`torch.save(model)`)——看似省事,但换了代码结构、换了机器就加载失败。工业界铁律:**只存 `state_dict`(纯参数字典),不存整个对象。**
> 3. **加载后忘了 `.eval()`**——dropout 还开着,生成结果每次都飘,还以为模型坏了。
> 4. **续训不存 optimizer 状态**——AdamW 有动量,丢了等于学习节奏从头乱起,loss 会抖一下。


---

## 第三部分:KV Cache 动机 —— 朴素生成为什么是 O(n²) 浪费

> 这是你小米推理优化项目的地基。请务必把这一部分嚼烂。
> 产出物对应任务:`tech_notes/kv_cache_motivation.md`(朴素生成 O(n²) 浪费 + 耗时曲线)

### 3.1 问题背景:回看 generate(),浪费藏在哪

回到第二部分的 `generate()`,盯住这两行:

```python
idx_cond = idx[:, -block_size:]    # 拿整个序列
logits, _ = self(idx_cond)         # 整个序列重新 forward 一遍
logits = logits[:, -1, :]          # 但只用最后一个位置的结果,其余全扔
```

发现矛盾没有?**我们每生成 1 个字,都把前面所有字重新完整计算了一遍,但只用了最后一个位置的输出,前面所有位置的计算结果全部扔掉。**

举个具体例子,生成 "ROMEO" 这 5 个字符:

| 第几步 | 喂进模型的内容 | 模型实际算了几个位置 | 真正需要的 |
|---|---|---|---|
| 1 | `R` | 1 | 第1个位置 |
| 2 | `R O` | 2 | 只要第2个位置,但第1个又算了一遍 |
| 3 | `R O M` | 3 | 只要第3个,前2个又算了一遍 |
| 4 | `R O M E` | 4 | 只要第4个,前3个又重算 |
| 5 | `R O M E O` | 5 | 只要第5个,前4个又重算 |

总计算量 = 1+2+3+4+5 = 15 次「位置计算」,但**真正有用的只有 5 次**(每步的最后一个)。剩下 10 次全是重复劳动。

### 3.2 核心原理:为什么是 O(n²)

**O(n²)(读作「大 O n 平方」)——复杂度记号**
- 是什么:描述「工作量随规模 n 增长得多快」的记号。O(n) 是线性(n 翻倍,工作翻倍);O(n²) 是平方(n 翻倍,工作翻 4 倍)。
- 类比:O(n) 像 n 个人排队挨个握一次手;O(n²) 像 n 个人开会两两都要握手,人一多握手次数暴涨。

生成 n 个字符,第 k 步要重算 k 个位置,总位置计算量:

```
1 + 2 + 3 + ... + n = n(n+1)/2 ≈ n²/2   →   O(n²)
```

**这就是「朴素生成是 O(n²)」的来历。** 序列越长,浪费越夸张:
- 生成 100 字 → 约 5,000 次位置计算(理想只需 100)
- 生成 1000 字 → 约 500,000 次(理想只需 1000)
- 生成 10000 字 → 约 5 千万次(理想只需 10000)

**理想情况应该是 O(n):每生成一个字只算这一个字。** 差距随长度平方级拉大——这正是长文本生成慢到无法忍受的根因。

### 3.3 病根诊断:到底哪部分被重复算了,能不能省

回到注意力机制(第一部分的 Q/K/V)。生成第 k 个字时,新字的 Query 要和**前面所有字的 Key、Value** 做运算。关键洞察:

> **前面那些字的 K 和 V,在它们被生成出来的那一刻就固定了,永远不会变。**

第 3 步算过的 `R`、`O` 的 K/V,到第 4 步、第 5 步还是一模一样。我们却在每一步重新计算它们——**这就是纯粹的浪费,而且是可以完全避免的浪费。**

**KV Cache(KV 缓存)的思想就一句话:把每个字算出来的 K 和 V 存起来(缓存),下一步直接取用,不再重算。**

- 类比:你每天上班走同一条路。朴素生成像「每天重新用地图从头规划整条路线」;KV Cache 像「第一次记下路线,以后直接照着走,只规划新增的那一段」。
- 加上缓存后:生成第 k 个字,只需算**新字这一个位置**的 K/V(其余从缓存取),复杂度从 O(n²) 降到 **O(n)**。

```python
# 伪代码:加了 KV Cache 的生成(对比第二部分的朴素版)
cache_k, cache_v = [], []          # 缓存:存每一层、每个历史字的 K 和 V
for _ in range(max_new_tokens):
    # 只把【最新的那一个字】喂进去,不是整个序列!
    logits = model.forward_with_cache(idx_next, cache_k, cache_v)
    #   ↑ 内部:新字算自己的 q/k/v;k/v 追加进 cache;
    #          注意力用【新q】对【全部缓存的k/v】做——历史的 k/v 直接取,不重算
    idx_next = sample(logits)       # 采样下一个字
# 工作量:每步只算 1 个位置 → 总共 O(n),不再是 O(n²)
```


### 3.4 怎么做:亲手测出「耗时曲线」(任务的核心产出)

光说 O(n²) 是纸上谈兵。**真正让你理解的是:亲手画出耗时随长度增长的曲线,看它怎么弯上去。** 这也是 `kv_cache_motivation.md` 要求的「耗时曲线」。

```python
# ===================================================================
# 实验:测「朴素生成」耗时随生成长度的增长曲线
# 依赖:torch, matplotlib  (pip install torch matplotlib)
# 目的:亲眼看到耗时是怎么从「线性」弯成「平方」的
# ===================================================================
import time, torch
import matplotlib.pyplot as plt

model.eval()
lengths = [50, 100, 200, 400, 800, 1600]   # 测不同生成长度
naive_times = []

for n in lengths:
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    torch.cuda.synchronize() if device == 'cuda' else None  # GPU 异步,计时前必须同步!
    t0 = time.time()
    _ = model.generate(ctx, max_new_tokens=n)               # 朴素生成
    torch.cuda.synchronize() if device == 'cuda' else None
    naive_times.append(time.time() - t0)
    print(f"生成 {n} 字符,耗时 {naive_times[-1]:.3f}s")

# 画图:如果是 O(n²),曲线会明显上弯(凸形),而不是直线
plt.plot(lengths, naive_times, 'o-', label='naive (no cache)')
plt.xlabel('生成长度 n'); plt.ylabel('耗时 (秒)')
plt.title('朴素自回归生成:耗时随长度的增长'); plt.legend(); plt.grid(True)
plt.savefig('kv_cache_timing.png', dpi=120)
print("曲线已保存 kv_cache_timing.png")
```

> **怎么看这条曲线(实验的灵魂):**
> - 如果耗时是 O(n),点会连成一条**直线**。
> - 实际你会看到点连成一条**向上弯的曲线**(凸的)——长度翻倍,耗时不止翻倍。这就是 O(n²) 的视觉证据。
> - **进阶做法**:再画一条「每个字符的平均耗时」(总耗时 / n)。朴素版这条线会**随 n 上升**(越往后每个字越慢,因为要重读的历史越长);理想的 O(n) 版本这条线应该是**水平的**。这个对比最能说明问题。
>
> ⚠️ **计时陷阱(工业界必踩):** GPU 计算是异步的——`time.time()` 可能在 GPU 还没算完时就记了时间,测出来的数全错。必须在计时前后加 `torch.cuda.synchronize()` 强制等 GPU 算完。这是性能测量的头号坑,记死它。

### 3.5 与工业界接轨:从这个玩具到 vLLM(你的项目落点)

你现在测出来的这条 O(n²) 曲线,正是整个大模型推理优化领域的**起点问题**。产业界顺着它做出了一整条技术线:

| 优化 | 解决什么 | 和你的关系 |
|---|---|---|
| **KV Cache** | 消除重复计算,O(n²)→O(n) | 你现在理解的这个,是一切的基础 |
| **KV Cache 的新问题:吃显存** | 缓存所有历史 K/V,长序列+多用户时显存爆炸 | 引出下面的 PagedAttention |
| **PagedAttention(vLLM 核心)** | 像操作系统管内存一样分页管理 KV Cache,大幅减少显存碎片浪费 | **这就是你要钻研的 vLLM 的看家本领** |
| **Continuous Batching** | 多个请求的生成动态拼批,提高 GPU 利用率 | vLLM 吞吐高的另一半原因 |

> **一句话点透你的项目定位:** KV Cache 把推理从 O(n²) 救到 O(n),但代价是「用显存换计算」——缓存要占大量显存。于是「**如何高效管理这块 KV Cache 显存**」成了新战场,vLLM 的 PagedAttention 就是这个战场上的明星方案。你小米项目的「巨核大模型推理优化」,本质就是在这条线上继续往前推。**今天你手动测出的这条 O(n²) 曲线,就是你理解 vLLM 为什么存在的第一课。**

### 3.6 常见误区与自检清单

**容易搞错的点:**
- ❌「KV Cache 让模型更聪明/loss 更低」——错。它**不改变任何输出结果**,只是算得快,纯工程优化。
- ❌「训练时也用 KV Cache」——错。训练用 teacher forcing,一次性并行算所有位置(第一部分讲的),不存在「一个个往下生成」,所以没有重复、不需要缓存。**KV Cache 只在推理(生成)阶段用。**
- ❌「缓存了就没成本」——错。它是**拿显存换速度**:序列越长、并发用户越多,KV Cache 占的显存越恐怖,这才有了后续一堆显存优化。

**自检:你真懂了吗?**
1. 为什么训练不需要 KV Cache,但推理需要?(答:训练并行算全部位置一次过;推理自回归逐字生成,才有重复)
2. 朴素生成为什么是 O(n²)?省掉的是哪部分计算?(答:每步重算历史的 K/V;历史 K/V 其实固定不变)
3. KV Cache 把复杂度降到多少?代价是什么?(答:O(n);代价是吃显存)
4. 计时实验里为什么要 `torch.cuda.synchronize()`?(答:GPU 异步,不同步会测到错误的时间)

> 这四题能用自己的话答出来,这三个任务就算真正打通了。答不上来的,回去重读对应小节。

---

## 附:三个任务的交付清单

- [ ] **训练**:跑出 val loss ≈ 1.5,保存训练日志(每 500 步的 train/val loss)
- [ ] **generate()**:能从 checkpoint 加载模型并生成一段像样的莎士比亚文本,贴几句到笔记里
- [ ] **checkpoint**:`ckpt.pt` 能存能加载,确认存了 config + 词表
- [ ] **kv_cache_motivation.md**:含 O(n²) 推导 + 亲手跑出的耗时曲线图 `kv_cache_timing.png` + 一段你自己话写的「为什么浪费、怎么省」

> 写 `kv_cache_motivation.md` 时,别复制我的话——**用你自己的语言重讲一遍 O(n²) 的故事**,能讲清楚才是真懂。这份笔记是你的「输入」,那份 md 是你的「输出」,两者不能一样。
