# Attention 复杂度：O(T²) 手算 + 缩放因子 1/√d 的作用

> **归属**：W6 Day2（2026-06-02）AI Infra 锚点。配套学习笔记见桌面 `W6_Day2_EP6_nanoGPT_单头SelfAttention.md`。
> **一句话**：单头 self-attention 的两个工程命脉——(T,T) 注意力矩阵带来的 **O(T²) 算力+显存爆炸**，和 **1/√d 缩放**对 softmax 稳定性的保护。这两点是 KV Cache / FlashAttention / 长上下文的根。

---

## 1. 单头 self-attention 的 5 行（认领对象）

```python
q = self.query(x)                                  # (B, T, hs)
k = self.key(x)                                    # (B, T, hs)
v = self.value(x)                                  # (B, T, hs)
wei = q @ k.transpose(-2, -1) * head_size ** -0.5  # (B, T, T) 匹配度 + 缩放
wei = wei.masked_fill(tril[:T,:T] == 0, float('-inf'))  # causal mask 下三角
wei = F.softmax(wei, dim=-1)                       # (B, T, T) 注意力比例
out = wei @ v                                      # (B, T, hs) 加权汇总
```

复杂度全部来自第 4 行那个 **(B, T, T)** 的 `wei`。

---

## 2. O(T²) 手算

### 2.1 平方的来源

序列有 T 个 token，每个 token 要和**所有 T 个** token 算一次点积匹配度 → T×T = **T²** 次。`wei` 形状 (T, T)。

- **算力**：`Q @ Kᵀ` = T² 个点积，每个 d 维 → **O(T²·d)** 次乘加。
- **显存**：(T, T) 的 `wei` 矩阵要实际存进显存（前向 softmax 用、反向算梯度用）→ 大小本身 **O(T²)**。

> 两个都是 O(T²)。长上下文真正的杀手是**显存**这一半。

### 2.2 手算具体数字（单头，FP32 = 4 字节/数）

```
单个注意力矩阵 = T × T × 4 字节

T=1024   (GPT-2):       1024²   × 4 = 4 MB
T=8192   (8K):          8192²   × 4 ≈ 256 MB      (T×8 → 显存×64)
T=131072 (GPT-4 128K):  131072² × 4 ≈ 64 GB       (单张 A100-80G 装不下一个头)
```

**规律**：上下文每翻倍，注意力矩阵显存 ×4。

### 2.3 放进真实模型规模

```
总注意力显存 ≈ batch × layers × heads × (T² × 4)

batch=8, 12 层, 12 头, T=1024:
   8 × 12 × 12 × 4MB ≈ 4.6 GB    (仅注意力矩阵，未含权重/KV/优化器状态)
```

---

## 3. 缩放因子 1/√d 的作用

### 3.1 是什么

第 4 行 `* head_size**-0.5` = 除以 √d（d=head_size）：`wei = (Q @ Kᵀ) / √d`。

### 3.2 为什么（防 softmax 饱和 → 防梯度消失）

- q、k 分量近似均值0方差1时，点积 = d 项之和，**方差≈d，标准差≈√d**。d 越大点积摆得越开。
- 大数喂 softmax 会饱和：`softmax([11,5,3]) ≈ [0.998, 0.002, 0]`，退化成 one-hot 硬选择。
- softmax 在饱和区**梯度≈0** → 梯度消失，这部分学不动。

### 3.3 怎么解决

除以 √d 把方差从 d 拉回 1，softmax 工作在健康区间，与 d 无关。

```python
import torch
from torch.nn import functional as F
d = 64
q, k = torch.randn(8, d), torch.randn(8, d)
print((q @ k.T).std().item())            # ≈ 8  = √64
print((q @ k.T * d**-0.5).std().item())  # ≈ 1
print(F.softmax(q @ k.T, dim=-1)[0].max().item())        # 往往 >0.9 饱和
print(F.softmax(q @ k.T * d**-0.5, dim=-1)[0].max().item())  # 温和
```

### 3.4 同源思想（控方差≈1 母题）

| 位置 | 控谁的方差 | 手段 |
|---|---|---|
| Kaiming 初始化（W4 `init_and_stability.md`） | 每层激活 | × 1/√fan_in |
| BatchNorm / LayerNorm（W5/W4） | 每层输出 | 减均值除标准差 |
| **Attention 缩放** | 点积 | ÷ √d |

---

## 4. AI Infra 地图（一切从 (T,T) 分叉）

| 问题 | 解法 | 学习节点 |
|---|---|---|
| (T,T) 显存 O(T²) 存不下 | FlashAttention（分块 SRAM，不落显存） | W8/暑假 |
| 生成时重算历史 K/V | KV Cache（缓存历史，每步 O(T)） | W6 Day6 + W8 |
| KV 显存碎片 | PagedAttention（vLLM 分页） | 暑假 |
| decode 受 KV 带宽限 | memory-bound（W5 Roofline） | W6 Day6 |

> **causal mask 下三角 → 历史 K/V 永不变 → KV Cache 正确性的根基**。

---

## 5. 速查结论

1. Attention O(T²)：算力 O(T²·d) + 显存 O(T²)，两个都是。
2. T 翻倍 → 注意力矩阵显存 ×4。T=1024/FP32 单头 = 4MB；128K = 64GB。
3. 1/√d 把点积方差从 d 拉回 1，防 softmax 饱和导致的梯度消失。
4. 与 Kaiming 初始化、LayerNorm 同属"控方差≈1"母题。
5. 下三角因果性是 KV Cache 成立的前提。
