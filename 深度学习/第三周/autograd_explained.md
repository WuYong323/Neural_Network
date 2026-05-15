# Autograd 是怎么自动算梯度的

> 第3周技术笔记 · 2026-05-17
> 关键词：计算图、反向传播、micrograd、PyTorch、推理优化伏笔

---

## 0. 一句话总结

**Autograd = 在正向计算时偷偷把每一步运算记下来（构建计算图），到了 `.backward()` 时按图反向走一遍，用链式法则把梯度一节一节传回去。**

第2周我们是**人**充当 autograd——拿纸推完 6 个公式再写死在 `nn.py` 里。这周 micrograd 让**机器**充当 autograd——你只管写 `loss = ...`，它自己把所有梯度算好。

---

## 1. 为什么需要"计算图"这个概念

### 1.1 第2周的痛点

我的 `backward()` 长这样：

```python
def backward(X, Z1, A1, Z2, A2, Y, W2):
    dZ2 = A2 - Y
    dW2 = A1.T @ dZ2 / N
    db2 = dZ2.sum(axis=0) / N
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1 / N
    db1 = dZ1.sum(axis=0) / N
    return ...
```

**这段代码绑死了具体网络结构。** 一旦我想：
- 多加一层 → 整个 backward 重写
- 把 ReLU 换成 tanh → ReLU 那一行的导数得手动改成 `(1 - tanh²)`
- 中间多一个 skip connection → 链式法则要重新推一遍

人脑推导 + 手写代码这条路，只能撑到三四层。Transformer 那种几十层、分支无数的结构，**靠手写绝对不可能不出错**。

### 1.2 计算图给了我们什么

计算图把每次运算都建模成一个**节点**，节点里存：

| 字段 | 含义 |
|---|---|
| `data` | 这个节点的值（前向算出来的数字） |
| `grad` | 反向传播会往这里写的梯度，初始为 0 |
| `_prev` | 这个节点是由哪几个节点运算出来的（父节点集合） |
| `_op` | 运算类型（`+`、`*`、`tanh`、…） |
| `_backward` | 一个闭包函数：知道怎么把自己的 grad 分发给 `_prev` |

**关键设计：每个节点只关心一件事——"我对我的输入贡献多少梯度"**。这就把"全局推一长串链式法则"拆成了"每个节点本地一小步"，自动可扩展到任意深度。

---

## 2. micrograd 的 `_backward` 闭包：核心机制

```python
def __add__(self, other):
    out = Value(self.data + other.data, _prev=(self, other), _op='+')

    def _backward():
        self.grad  += out.grad        # ∂out/∂self  = 1
        other.grad += out.grad        # ∂out/∂other = 1
    out._backward = _backward
    return out

def __mul__(self, other):
    out = Value(self.data * other.data, _prev=(self, other), _op='*')

    def _backward():
        self.grad  += other.data * out.grad   # ∂out/∂self  = other
        other.grad += self.data  * out.grad   # ∂out/∂other = self
    out._backward = _backward
    return out
```

### 2.1 三个必须想清楚的点

**(1) 为什么用闭包而不是普通函数？**
因为 `_backward` 要访问 `self`、`other`、`out` 这几个特定的 Value 对象——它们是**前向计算时**才确定的。闭包刚好能"捕获"这些变量，每次 `__add__` 调用都会生成一个独属于那次运算的 `_backward`。

**(2) 为什么是 `+=` 而不是 `=`？**
看这个最小例子：
```python
x = Value(3.0)
y = x + x          # 同一个 x 在图里出现了两次
y.backward()
```
正确答案是 `dy/dx = 2`。如果 `_backward` 写成 `self.grad = out.grad`，第二次会**覆盖**第一次的贡献，结果错成 1。`+=` 保证多路径贡献能正确累加。

**(3) 为什么要拓扑排序？**
反向传播必须**保证一个节点的 `grad` 累加完毕，才能用它去更新它的 `_prev`**。如果乱序，可能我用 `dA1` 去更新 `dZ1`，但 `dA1` 自己还没被某条路径累加完，这就漏算了。拓扑排序（按"产生关系"逆序）保证了这一点。

```python
def backward(self):
    topo = []
    visited = set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

---

## 3. micrograd vs PyTorch：同一个思想，两套实现

PyTorch 的 autograd 内核思路**完全一致**，只是工程化得多：

| 维度 | micrograd | PyTorch |
|---|---|---|
| 图节点 | `Value`（标量） | `Tensor`（任意 shape） |
| 反向钩子 | `_backward` 闭包 | `grad_fn`（C++ 实现的 `Function.backward`） |
| 拓扑排序 | Python DFS | C++ 优先队列 |
| 内存策略 | 全部保留 | 自动释放中间张量（除非 `retain_graph=True`） |
| 自定义算子 | 自己写一个返回新 Value 的方法 | 继承 `torch.autograd.Function`，实现 `forward`/`backward` 静态方法 |

**对应关系最直接的：**
```python
# micrograd
def __mul__(self, other):
    out = Value(self.data * other.data, _prev=(self, other))
    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad
    out._backward = _backward
    return out
```
```python
# PyTorch (简化版)
class Mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        return b * grad_out, a * grad_out   # 对应 self.grad, other.grad
```

**`ctx.save_for_backward` 等价于闭包捕获变量；`backward` 静态方法等价于 `_backward` 闭包。**

理解了 micrograd，PyTorch autograd 90% 的疑惑都消失了。剩下 10% 是工程优化（fused kernel、内存复用、CUDA stream），那是后面推理优化要学的事。

---

## 4. 与第2周手写 numpy 网络的对比

| 维度 | 第2周 numpy 手写 | 第3周 micrograd | PyTorch |
|---|---|---|---|
| **代码量** | ~200 行（含梯度检验） | ~100 行 Value 类 + 几十行 nn | 调用即用 |
| **梯度来源** | 人推 6 个公式 → 写死在代码里 | 每个运算自带 `_backward` → 自动组合 | 同 micrograd，C++ 实现 |
| **修改网络结构成本** | 高（每次都要重推公式） | 低（写新表达式即可） | 极低 |
| **运行速度** | 快（numpy 向量化） | **极慢**（每个标量都是 Python 对象 + 建图开销） | 快（GPU + 融合 kernel） |
| **支持的运算粒度** | 矩阵级 | 标量级 | 张量级 |
| **支持反向的算子** | 我手写的几个 | `__add__/__mul__/__pow__/tanh/exp/relu` | 几千个 |
| **能调试到中间梯度吗** | 能（自己变量） | 能（每个 Value.grad） | 能（`.retain_grad()` 后） |
| **能不能跑 GPU** | 不能 | 不能 | 能 |

### 4.1 关键洞察

**micrograd 是"教学版" PyTorch**：它演示了"自动微分"这个想法的核心机制，但**牺牲了一切性能**——每个标量都是独立的 Python 对象，建图 + 拓扑 + 闭包调用全部走 Python 解释器，比 numpy 慢 100~1000 倍。

**numpy 网络是"高性能但脆弱"**：底层调用 BLAS，速度很快，但梯度逻辑全部写死。

**PyTorch 把两者优点结合**：autograd 自动化（来自 micrograd 的思想）+ 张量化 BLAS/cuBLAS（来自 numpy 的性能哲学）+ 显存/kernel 工程化。

> 我第2周那个 `dZ2 = A2 - Y` 是怎么自动得到的？
> ——softmax 的 `_backward` + cross_entropy 的 `_backward` 在拓扑排序中相邻执行，链式法则自动把 `-Y/A2`（cross-entropy 对 A2 的导数）和 `A2 * (I - A2)`（softmax 雅可比）相乘，化简后正好是 `A2 - Y`。**人推一次和机器推一次结果完全相同，但机器推可以推任意复杂的表达式。**

---

## 5. 推理优化伏笔：为什么推理时不需要梯度

> 这部分是第8周（推理优化入门）的入口。先种下意识，后面遇到具体技术不会陌生。

### 5.1 `torch.no_grad()` 到底关掉了什么

```python
with torch.no_grad():
    output = model(x)
```

这一行做了**三件大事**：

1. **不构建计算图**：每次运算不再创建 `grad_fn` 节点、不再把父节点存到 `_prev` 里。
2. **不保留中间激活**：第2周我们要在 backward 里用到 `Z1`、`A1`、`Z2`，所以前向时不能释放它们；推理不需要 backward，**所有中间张量算完就丢**。
3. **关闭某些层的训练模式**：通常配合 `model.eval()` 一起用——BN 切到 running mean/var，Dropout 直接关掉。

### 5.2 这件事在工程上意味着什么

对一个 7B 参数的 Transformer 推理 1 个 token：

| 资源 | 训练（含梯度） | 推理（`no_grad`） |
|---|---|---|
| 参数显存 | 1× 参数 | 1× 参数 |
| 梯度显存 | 1× 参数 | **0** |
| 优化器状态（Adam） | 2× 参数 | **0** |
| 激活值显存 | 全部保留（O(层数×batch×seqlen)） | **只保留当前层** |
| **总显存放大倍数** | **~4×** | **~1×** |

> 关掉梯度 = **省 4 倍显存 + 省 4 倍内存带宽 + 跳过 backward 的所有计算量**。
> 这是为什么同一张 GPU 能推理 70B 模型却装不下 13B 的训练。

### 5.3 这条线一直通到哪里（第8周预告）

第8周会接触到的所有概念，本质都建立在"推理不需要梯度，所以可以做训练时不敢做的优化"这个前提上：

- **量化（INT8/FP16）**：训练时梯度需要 FP32 精度才能稳定累加；推理不算梯度，敢直接降精度，速度 ×2、显存 ÷2。
- **KV Cache**：自回归生成时，前面 token 的 K、V 算过一次就不变，缓存复用。**前提是不需要对它们反向求导**——训练时是不允许这么干的。
- **算子融合（`torch.compile`）**：把多个小算子融成一个 kernel。训练时为了构建反向图，每个算子必须单独存在；推理时可以激进地融合，因为不需要再"拆回去"做 backward。
- **FlashAttention 推理变种**：训练版需要存 softmax 中间结果给 backward，推理版完全不用，I/O 量直接减半。

**核心一句话**：autograd 是训练的"福利"，但对推理是"负担"。第8周做的每件事，本质都是"把训练时为 autograd 付的代价拿回来"。

---

## 6. 检验自己是否真的懂了

闭卷回答这几个问题（参考答案对应回到上文章节）：

1. micrograd 的 `_backward` 为什么必须用闭包？普通函数行不行？（§2.1）
2. `x = Value(3); y = x + x; y.backward()` 的 `x.grad` 是多少？为什么必须 `+=`？（§2.1）
3. 不做拓扑排序直接反向调用 `_backward` 会出什么问题？（§2.1）
4. 我第2周写死的 `dZ2 = A2 - Y`，在 micrograd 里是由哪两个算子的 `_backward` 自动组合出来的？（§4.1）
5. `torch.no_grad()` 节省的不只是显存，还有什么？说出至少 3 项。（§5.1）
6. KV Cache 为什么是推理专有优化，训练时为什么不能用？（§5.3）

如果第 4、5、6 题答不出来，回到对应章节再读一遍——这是后面三周的地基。

---

## 附：与课程的对应

- Karpathy EP1（micrograd）：本笔记 §2 全部、§3 前半。
- Karpathy EP2（bigram NN 版）：本笔记没展开，见 `bigram/comparison.md`。
- 第2周 `tech_notes/backprop_derivation.md`：本笔记 §1.1、§4 的对照基准。
- 第4周 EP4（BatchNorm 训练 vs 推理）：会重新激活 §5.1 中"`model.eval()` 关掉了什么"的讨论。
- 第8周（推理优化入门）：§5.2、§5.3 全部成为入口。
