# Embedding 学习笔记

> 目标：从直觉 → 数学 → 训练 → 应用，建立一套能迁移到 Transformer / RAG / 向量数据库的底层认知。

---

## 0. 一句话定义

**Embedding（嵌入）：把离散的、高维的、无语义的对象（词、用户、商品、图片、节点……），映射成一个低维、稠密、带语义的实数向量，使得"对象之间的语义关系"对应"向量之间的几何关系"。**

记作一个映射：

$$
f: \text{Object} \rightarrow \mathbb{R}^d
$$

其中 $d$ 通常是 64、128、300、768、1536 等远小于原始词表大小的数。

---

## 1. 为什么需要 Embedding（动机）

### 1.1 计算机不认识"猫"，只认识数字

最朴素的方案是 **One-Hot 编码**：词表有 $V$ 个词，每个词用一个长度 $V$ 的向量表示，只有自己那一位是 1。

```
词表 = [我, 你, 猫, 狗, 国王, 女王]   (V = 6)
猫   = [0, 0, 1, 0, 0, 0]
狗   = [0, 0, 0, 1, 0, 0]
```

### 1.2 One-Hot 的三大致命缺陷

| 缺陷 | 说明 |
|------|------|
| **维度灾难** | 中文词典轻松 10 万词 → 每个词 10 万维向量 |
| **极度稀疏** | 99.999% 是 0，存储和计算都浪费 |
| **语义全失** | 任意两个不同词的余弦相似度恒为 0；"猫 vs 狗" 和 "猫 vs 国王" 距离一样 |

### 1.3 Embedding 解决了什么

- **稠密**：每一维都是有意义的实数
- **低维**：从 $V$ 维压到 $d$ 维（$d \ll V$）
- **有语义**：相似对象 → 相近向量

---

## 2. Embedding 的核心性质（最重要的直觉）

### 2.1 语义相似 ⇔ 向量相近

度量两个向量相似度，常用：

- **余弦相似度**（最常用，方向决定语义）：

$$
\cos(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}
$$

- **欧氏距离**：

$$
d(\vec{a}, \vec{b}) = \sqrt{\sum_i (a_i - b_i)^2}
$$

### 2.2 向量空间中的"语义算术"

经典例子（Word2Vec 论文）：

$$
\vec{King} - \vec{Man} + \vec{Woman} \approx \vec{Queen}
$$

$$
\vec{Beijing} - \vec{China} + \vec{France} \approx \vec{Paris}
$$

**为什么会有这种性质？** 因为训练目标让"性别"、"国家-首都"这类语义关系，在向量空间里变成了**近似平行的方向向量**。

### 2.3 一个关键直觉：维度 = 隐含的语义因子

虽然单个维度通常不可解释，但你可以想象 300 维里隐式编码了：
- 第 17 维 ≈ "是否是动物"
- 第 42 维 ≈ "积极/消极情绪"
- 第 88 维 ≈ "正式/口语化"
- ……

这些**潜在因子**是模型从数据里**自动发现**的，不是人工设计的。

---

## 3. 原理：Embedding 是怎么"学"出来的

### 3.1 哲学基础：分布假说（Distributional Hypothesis）

> "You shall know a word by the company it keeps." —— J.R. Firth, 1957

**一个词的含义，由它经常出现的上下文决定。**

- "我家的 **猫** 喜欢吃鱼"
- "我家的 **狗** 喜欢吃骨头"

→ "猫"和"狗"上下文相似 → 向量应当相似。

### 3.2 Word2Vec：两种训练范式

#### (a) CBOW (Continuous Bag-of-Words)：用上下文预测中心词

```
输入: [我家, 的, ___, 喜欢, 吃]
预测: 猫
```

#### (b) Skip-gram：用中心词预测上下文（更常用，对低频词更好）

```
输入: 猫
预测: 我家, 的, 喜欢, 吃
```

### 3.3 训练流程（以 Skip-gram 为例）

1. **初始化**：每个词随机给一个 $d$ 维向量，组成嵌入矩阵 $E \in \mathbb{R}^{V \times d}$
2. **前向**：取出中心词向量 $\vec{v}_c$，对每个候选上下文词 $w_o$ 计算：

$$
P(w_o \mid w_c) = \frac{\exp(\vec{u}_o \cdot \vec{v}_c)}{\sum_{w \in V} \exp(\vec{u}_w \cdot \vec{v}_c)}
$$

3. **损失**：负对数似然（cross entropy）

$$
\mathcal{L} = -\sum \log P(w_o \mid w_c)
$$

4. **反向传播**：调整 $\vec{v}_c$、$\vec{u}_o$，让真实上下文概率变大
5. **训练完毕**：嵌入矩阵 $E$ 的每一行就是对应词的 embedding

> ⚡ **关键洞察**：embedding 不是直接的训练目标，而是**模型为了完成"预测上下文"任务而顺便学到的副产品**。这一点对理解后面所有 embedding 模型都至关重要。

### 3.4 工程加速技巧

朴素 softmax 分母要遍历整个词表，太慢。实际用：

- **负采样 (Negative Sampling)**：每次只更新真实上下文词 + 几个随机负样本
- **Hierarchical Softmax**：用霍夫曼树把 $O(V)$ 降到 $O(\log V)$

### 3.5 后续演进路线

| 模型 | 关键改进 |
|------|----------|
| Word2Vec (2013) | 静态词向量，分布假说 |
| GloVe (2014) | 基于全局共现矩阵的矩阵分解视角 |
| FastText (2016) | 用子词 (subword) 解决 OOV 和形态学 |
| ELMo (2018) | **上下文相关**词向量（同一个词在不同句子里向量不同） |
| BERT / GPT (2018+) | Transformer 产出深度上下文 embedding |
| Sentence-BERT, OpenAI text-embedding-3 | 句/段落级 embedding，专为检索优化 |

**重要分水岭**：
- **静态 embedding**（Word2Vec / GloVe）：一个词一个向量，"苹果"在"吃苹果"和"苹果手机"里向量一样
- **上下文 embedding**（BERT 之后）：同一个词在不同句子里向量**不同**，由整句话决定

---

## 4. Embedding 不止于词

| 对象 | 模型示例 | 应用 |
|------|----------|------|
| 词 | Word2Vec, GloVe | NLP 输入层 |
| 句/段落 | Sentence-BERT, OpenAI Embedding | 语义搜索、RAG |
| 图片 | CNN 倒数第二层、CLIP | 以图搜图、多模态 |
| 用户 / 商品 | 矩阵分解、双塔模型 | 推荐系统 |
| 图节点 | DeepWalk, Node2Vec, GNN | 社交网络、知识图谱 |
| 代码 | CodeBERT | 代码搜索、补全 |
| 跨模态 | CLIP（图+文）、ImageBind（多模态） | 文搜图、零样本分类 |

---

## 5. 应用全景

### 5.1 语义搜索 / 向量数据库

```
query  ──embedding──► 向量 q
docs   ──embedding──► 向量 d1, d2, ..., dN  (预先存入向量数据库)

检索 = 在向量空间里找 cos(q, di) 最大的 top-k
```

向量数据库：**Pinecone, Milvus, Faiss, Qdrant, Chroma**

### 5.2 RAG（Retrieval-Augmented Generation）

让 LLM"开卷考试"：

```
1. 把知识库切片 → 每片做 embedding → 存向量库
2. 用户问问题 → 问题 embedding → 检索相关切片
3. 把切片 + 问题 拼成 prompt 喂给 LLM → 生成答案
```

### 5.3 推荐系统（双塔模型）

```
用户特征 ──塔A──► 用户向量 u
物品特征 ──塔B──► 物品向量 v
评分 = u · v  → 召回 / 排序
```

### 5.4 大模型内部

Transformer 第一步就是 **Token Embedding + Position Embedding**：

```
"你好" → [tok_id_1, tok_id_2] → [vec_1, vec_2] (查表) → 进入 attention 层
```

LLM 学到的 token embedding 矩阵，本身就是几十亿参数里的关键一部分。

### 5.5 多模态对齐（CLIP）

把图片和文字**映射到同一个向量空间**：

```
"一只猫" ──text encoder──► 向量 t
[猫的图片] ──image encoder──► 向量 i
训练目标：让真实(图,文)对的 cos(t,i) 大，错配对的小
```

→ 实现了"用文字搜图"、"零样本图像分类"。

---

## 6. 实战注意事项（容易踩的坑）

1. **embedding 维度选择**：太小欠拟合，太大过拟合 + 算力浪费。常见 64~1536，看任务和数据量。
2. **是否归一化**：用余弦相似度时，先 L2 归一化向量再做点积更稳。
3. **不同模型的 embedding 不能混用**：OpenAI 的 1536 维和 BGE 的 1024 维生活在完全不同的空间。
4. **静态 vs 上下文**：做关键词匹配可以用 Word2Vec；做问答检索一定要用上下文 embedding（BERT 系）。
5. **冷启动问题**：新词 / 新用户没向量怎么办？→ 用子词（FastText）、内容特征、或预训练模型零样本编码。
6. **向量漂移**：模型升级后整个库要重新 embedding，否则新旧向量空间不对齐。

---

## 7. 与 ICPC / 算法的连接（给你的迁移直觉）

| 算法概念 | Embedding 中的对应 |
|----------|-------------------|
| 哈希 | embedding 是"带语义的哈希"——相似的输入给出相近的输出 |
| KD-Tree / 最近邻 | 向量检索的底层结构（高维下要用 HNSW、IVF 等近似算法） |
| 字符串相似度 (编辑距离) | embedding 把它升级成"语义距离" |
| 矩阵分解 | GloVe / 推荐系统 embedding 的数学本质 |
| 图论（最短路、连通） | Node Embedding（DeepWalk 用随机游走） |

---

## 8. 自检清单（学完应当能回答）

- [x] One-Hot 有哪三个缺陷？Embedding 怎么解决？
- [x] 为什么"国王 - 男人 + 女人 ≈ 女王"？这反映了什么？
- [x] 分布假说是什么？它如何指导 Word2Vec 训练？
- [x] CBOW 和 Skip-gram 的区别？哪个对低频词更友好？
- [x] 为什么需要负采样？
- [x] 静态 embedding 和上下文 embedding 的核心区别？
- [x] RAG 流程里 embedding 出现在哪几步？
- [x] CLIP 是怎么把图片和文字放进同一个向量空间的？
- [x] 为什么不同模型的 embedding 不能直接混用？

---

## 9. 推荐学习路径

1. **直觉**：本笔记 + 3Blue1Brown 关于 word embedding 的视频
2. **动手**：用 `gensim` 训练一个 Word2Vec，可视化（t-SNE / UMAP 降到 2D）观察聚类
3. **进阶**：读 Word2Vec 原论文 (Mikolov 2013) → GloVe → BERT 论文中的 embedding 部分
4. **工程**：搭一个最小 RAG（langchain + chroma + 任一 embedding 模型）
5. **理论**：研究"为什么神经网络能学到这种向量空间"——表示学习 (Representation Learning) 是后续 ML 的核心主题

---

## 10. 一句话收束

> **Embedding 是机器理解世界的"通用货币"——一旦把任何东西变成向量，相似度、检索、分类、生成、跨模态对齐就全都统一成了向量运算。这是现代 AI 最重要的底层范式之一。**
