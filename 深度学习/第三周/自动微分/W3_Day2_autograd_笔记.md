# W3 Day2 学习笔记：自动微分——让梯度沿计算图反向流动（micrograd 第二步）

> 日期：2026-05-12（周二）
> 前置：你已完成 Day1，`Value` 类能正向构图，5 条断言全部通过
> 视频范围：Karpathy "Neural Networks: Zero to Hero" EP1 第 1:00 – 2:25
> 今日核心：**把"手推梯度公式"从你脑子里搬进代码里**——这就是自动微分（Automatic Differentiation, AD）的全部魔法

---

## 0. 先看清楚今天要拿到什么

完成这份笔记 + 代码后，你应该能：

1. 用一句话回答：**自动微分为什么只写正向传播就能拿到所有梯度？它和数值微分、符号微分有什么区别？**
2. 默写出链式法则的核心一行：`dL/dx = dL/dy × dy/dx`，并解释为什么反向传播**只需要局部导数**。
3. 手写出 `+` 和 `*` 两种运算的 `_backward` 闭包，**不看视频**。
4. 解释"为什么梯度要用 `+=` 累加而不是 `=` 赋值"，并举一个具体反例。
5. 手算一个 3 节点的小图，再用代码跑出来，两边梯度完全一致。
6. 写出拓扑排序的递归版本，解释为什么反向传播必须先拓扑排序。

> ⚠️ 名词澄清：你问的"自动积分"，准确名字是 **自动微分（Automatic Differentiation, AD）**——求的是导数（微分），不是原函数（积分）。深度学习从头到尾都是求导数，从不求积分。英文里 differentiation（微分/求导）和 integration（积分）严格区分。本笔记此后一律用"自动微分"。

如果第 3、4、6 条任意一条做不到，**不要急着进 Day 3**——这三条是自动微分的灵魂。

---

## 1. 知识点串讲：从"手推公式"到"自动求导"

### 1.1 三种求导方式的对比

你已经会"手推梯度公式"了（第 2 周），也隐约知道"PyTorch 能自动求"。其实在工程上求导只有三条路：

| 方法 | 做法 | 缺点 | 代表 |
|---|---|---|---|
| **数值微分** | 用极限定义：`(f(x+h) − f(x)) / h`，取 h 很小 | 精度差（浮点丢位数）、慢（每个参数都要 forward 一遍） | 只用于测试时校验梯度 |
| **符号微分** | 像数学课那样代数化简，得到导数的解析表达式 | 表达式爆炸；有 if/for 就没法写 | Mathematica |
| **自动微分（AD）** | 把每种运算的导数规则**编码在运算本身上**，按计算图反向走一遍，用链式法则把局部导数串起来 | 需要先有计算图（Day1 你就干这个了） | **PyTorch / TF / micrograd** |

**关键洞察：** 自动微分不是"算出导数的表达式"，而是"对当前这个具体输入，直接给出导数的数值"。所以它和 `if / for / 动态形状` 完全兼容——每次正向都重新建图就行。

### 1.2 链式法则：一张图就懂

高数课本里的链式法则：如果 `y = f(x)`，`z = g(y)`，那么 `dz/dx = dz/dy × dy/dx`。

换成计算图的语言：

```
      x  ──[ f ]──▶  y  ──[ g ]──▶  z
```

如果**下游已经把 `dz/dy` 算好递给我**（`y.grad`），而**我自己知道 `dy/dx`**（`f` 的局部导数），那么：

```
dz/dx  =  dz/dy  ×  dy/dx
          ─────┬────     ─────┬────
          下游递给我的        我自己算的
```

**这就是反向传播的全部秘密。** 每个运算节点只需要知道两件事：

1. **局部导数**（`dy/dx`）：数学课教过你怎么算
2. **下游梯度**（`dz/dy`）：下游节点会递给它

两者相乘再累加到前驱上，梯度就往后传了一层。一层一层传下去，从 loss 到每个叶子都能拿到梯度。

### 1.3 每种运算的"局部导数"长什么样

下表请**背下来**（至少 5 秒内能推出来）：

| 正向运算 | 对 a 的局部导数 | 对 b 的局部导数 |
|---|---|---|
| `out = a + b` | **1** | **1** |
| `out = a * b` | **b** | **a** |
| `out = tanh(a)` | **1 − out²** | — |
| `out = max(0, a)`（ReLU） | **1 if a>0 else 0** | — |
| `out = exp(a)` | **out**（因为 `(eᵃ)' = eᵃ`） | — |
| `out = a / c`（c 为常数） | **1/c** | — |

**怎么记？**
- **加法**：`a + b` 对 a 求导，b 不动，a 变 1 结果就变 1，所以是 1。
- **乘法**：`a * b` 对 a 求导，b 当常数看，导数就是 b。

这张表就是你在 `_backward` 里会看到的所有数学——真没有别的。

### 1.4 "局部"和"全局"的解耦是自动微分最关键的设计

再看 §1.2 的图：**加法节点根本不需要知道它的下游是什么**。它只需要：
1. 从 `out.grad` 里拿到"下游递给我的梯度"
2. 乘上自己的局部导数（加法是 1）
3. 累加到 `self.grad` 和 `other.grad` 上

**正是这种解耦让自动微分可以支持任意结构的计算图**——不管下游是 tanh 还是乘法，加法都不用管它。

这也回答了你第 2 周的困惑："为什么 PyTorch 换一个激活函数不用重写 backward？"——因为每个激活函数节点自己负责自己的那一层链式法则，**其它节点看都不看它一眼**。

### 1.5 为什么梯度必须用 `+=` 而不是 `=`

考虑一个小陷阱：`y = x + x`。

计算图里 `y._prev = {x}`（set 去重后只有一个）。但数学上 `dy/dx = 2`——x 以两条路径影响 y。

如果 `__add__` 的 `_backward` 写成：

```python
self.grad  = 1.0 * out.grad   # 赋值（错）
other.grad = 1.0 * out.grad   # 赋值（错）
```

当 `self is other`（都是同一个 x）时，后一行会**覆盖**前一行，最终 `x.grad = 1` 而不是 2。

正确写法：

```python
self.grad  += 1.0 * out.grad   # 累加（对）
other.grad += 1.0 * out.grad   # 累加（对）
```

**更一般的原因：** 一个节点如果被多个下游用到（神经网络里极其常见——skip connection、参数共享），会从多条路径收到梯度。数学上这些梯度必须**全部加起来**。所以 `+=` 是唯一正确的更新方式。

**副作用：** 每次新的 backward 之前必须把所有 grad 清零——这就是 PyTorch `optimizer.zero_grad()` 的由来。

### 1.6 为什么要拓扑排序？

反向传播有一条硬约束：**调用一个节点的 `_backward` 时，它的 `grad` 必须已经是"所有下游梯度的累加结果"**，否则就把不完整的梯度往前传了。

所以必须按**逆拓扑序**走：先处理最下游（loss 那个节点），再处理它的前驱、前驱的前驱……直到叶子。经典做法：DFS 后序遍历 + visited 集合。§2 的代码里就 8 行。

---

## 2. 手算一遍：把反向传播在纸上走通

在写代码之前，**强烈建议你拿笔和纸走一遍**。只有手算过一次，代码里每一行你才能"看出意义"。

### 2.1 目标计算图

```
   a(2.0)     b(-3.0)
      \        /
       \      /
       [ * ]  ← c = a*b = -6
          |
          v                d(10.0)
        c(-6)               /
          \                /
           \              /
            [ + ]  ← L = c + d = 4
               |
               v
             L(4.0)   ← 假装这是最终的 loss
```

### 2.2 反向传播的 4 个步骤

**Step 0：初始化。** 所有节点 `grad = 0`。

**Step 1：给最顶端的 L 一个种子梯度。**
`L.grad = dL/dL = 1`（对自己求导恒为 1，这是反向传播的起点）。

**Step 2：反向经过加法节点 `L = c + d`。** 用链式法则：
- `dL/dc = dL/dL × dL/dc(局部) = 1 × 1 = 1` → `c.grad += 1` → `c.grad = 1`
- `dL/dd = dL/dL × dL/dd(局部) = 1 × 1 = 1` → `d.grad += 1` → `d.grad = 1`

**Step 3：反向经过乘法节点 `c = a*b`。** 局部导数：`dc/da = b = -3`，`dc/db = a = 2`。
- `dL/da = dL/dc × dc/da = 1 × (-3) = -3` → `a.grad += -3` → `a.grad = -3`
- `dL/db = dL/dc × dc/db = 1 × 2 = 2` → `b.grad += 2` → `b.grad = 2`

**Step 4：所有节点都访问过，反向传播结束。**

**最终结果：** `a.grad = -3, b.grad = 2, c.grad = 1, d.grad = 1, L.grad = 1`。

### 2.3 代码里你要做的事

对照手算的四步，代码只需要：

1. 给每种运算定义"局部怎么分发梯度"——**就是 `_backward` 闭包**
2. 从终点 L 出发按逆拓扑序走一遍，每个节点调用自己的 `_backward`——**就是 `backward()` 方法**

---

## 3. `Value` 类完整实现（正向 + 反向）

下面是 `week3_karpathy/micrograd_follow/value.py` 的完整版。直接替换 Day1 的版本，然后 `python value.py` 运行。

```python
"""
W3 Day2: 在 Day1 构图的基础上加反向传播。
每个运算节点挂一个 _backward 闭包，backward() 按逆拓扑序调用它们。
"""

import math


class Value:
    """能记住出身、并能反向传递梯度的标量。"""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # _backward 是"把我的 grad 按局部导数分发给前驱"的函数。
        # 叶子节点没有前驱，所以默认是 no-op。
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        if self.label:
            return f"Value(label='{self.label}', data={self.data}, grad={self.grad})"
        return f"Value(data={self.data}, grad={self.grad})"

    # --------- 运算 ---------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 加法的局部导数都是 1，所以 out.grad 直接往两边传
            # 必须用 += 累加，不能 =
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 局部导数：dout/dself = other.data, dout/dother = self.data
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # tanh 的局部导数：1 - tanh(x)^2 = 1 - out.data^2
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # --------- 便捷运算（让 3 + a、a - b、-a 等也能用） ---------

    def __radd__(self, other):      # 10 + a 时被调用（左边不是 Value）
        return self + other

    def __neg__(self):              # -a
        return self * -1

    def __sub__(self, other):       # a - b
        return self + (-other)

    def __rmul__(self, other):      # 10 * a
        return self * other

    # --------- 反向传播主入口 ---------

    def backward(self):
        """从 self 出发，反向传播梯度到所有前驱（把 self 当作 loss）。"""
        # 1. 拓扑排序：后序 DFS，结果是"父节点一定在子节点之后"
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2. 种子梯度：dL/dL = 1
        self.grad = 1.0

        # 3. 逆拓扑序调用每个节点的 _backward
        for v in reversed(topo):
            v._backward()


# ----------------------------------------------------------------------
# 验证 1：和手算对齐（§2.2 的小图）
# ----------------------------------------------------------------------
def demo_small_graph():
    print("=" * 50)
    print("验证 1：小图手算对照")
    print("=" * 50)
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b;        c.label = 'c'
    d = Value(10.0, label='d')
    L = c + d;        L.label = 'L'

    L.backward()

    for v in [a, b, c, d, L]:
        print(f"  {v.label}.grad = {v.grad}")

    # 和 §2.2 手算结果逐项断言
    assert a.grad == -3.0, "a.grad 应当是 -3"
    assert b.grad == 2.0,  "b.grad 应当是 2"
    assert c.grad == 1.0,  "c.grad 应当是 1"
    assert d.grad == 1.0,  "d.grad 应当是 1"
    assert L.grad == 1.0,  "L.grad 应当是 1（种子）"
    print("  OK 小图梯度与手算完全一致")


# ----------------------------------------------------------------------
# 验证 2：y = x + x 的累加陷阱
# ----------------------------------------------------------------------
def demo_plus_trap():
    print("=" * 50)
    print("验证 2：y = x + x，正确答案是 dy/dx = 2")
    print("=" * 50)
    x = Value(3.0, label='x')
    y = x + x;  y.label = 'y'
    y.backward()
    print(f"  x.grad = {x.grad}  （若为 1 说明 += 写成了 =）")
    assert x.grad == 2.0, "x.grad 应当是 2；若是 1，说明梯度更新用了 = 而不是 +="
    print("  OK += 累加正确")


# ----------------------------------------------------------------------
# 验证 3：用数值微分校验（黄金标准）
# ----------------------------------------------------------------------
def demo_numerical_check():
    print("=" * 50)
    print("验证 3：与数值微分对比（f(x+h)-f(x-h))/(2h)")
    print("=" * 50)

    def f(a_val, b_val):
        # 一个稍微复杂点的表达式
        a = Value(a_val); b = Value(b_val)
        out = (a * b + b).tanh() + a * a
        return out

    a_val, b_val = 1.5, -2.0
    h = 1e-5

    # 解析梯度（我们自己的 AD）
    a = Value(a_val, label='a')
    b = Value(b_val, label='b')
    L = (a * b + b).tanh() + a * a
    L.backward()

    # 数值梯度（中心差分）
    grad_a_num = (f(a_val + h, b_val).data - f(a_val - h, b_val).data) / (2 * h)
    grad_b_num = (f(a_val, b_val + h).data - f(a_val, b_val - h).data) / (2 * h)

    print(f"  a.grad  (AD)        = {a.grad:.6f}")
    print(f"  a.grad  (numerical) = {grad_a_num:.6f}")
    print(f"  b.grad  (AD)        = {b.grad:.6f}")
    print(f"  b.grad  (numerical) = {grad_b_num:.6f}")

    assert abs(a.grad - grad_a_num) < 1e-4, "a 的 AD 梯度与数值梯度偏差过大"
    assert abs(b.grad - grad_b_num) < 1e-4, "b 的 AD 梯度与数值梯度偏差过大"
    print("  OK 解析梯度与数值梯度吻合（误差 < 1e-4）")


if __name__ == "__main__":
    demo_small_graph()
    print()
    demo_plus_trap()
    print()
    demo_numerical_check()
```

### 3.1 预期运行输出

```
==================================================
验证 1：小图手算对照
==================================================
  a.grad = -3.0
  b.grad = 2.0
  c.grad = 1.0
  d.grad = 1.0
  L.grad = 1.0
  OK 小图梯度与手算完全一致

==================================================
验证 2：y = x + x，正确答案是 dy/dx = 2
==================================================
  x.grad = 2.0
  OK += 累加正确

==================================================
验证 3：与数值微分对比（f(x+h)-f(x-h))/(2h)
==================================================
  a.grad  (AD)        = 3.xxxxxx
  a.grad  (numerical) = 3.xxxxxx
  b.grad  (AD)        = 0.xxxxxx
  b.grad  (numerical) = 0.xxxxxx
  OK 解析梯度与数值梯度吻合（误差 < 1e-4）
```

具体数值以机器跑出来为准。**只要三段的断言全部通过，你的自动微分就写对了。**

---

## 4. 逐段讲解：把每一行看穿

### 4.1 闭包（closure）是怎么"挂"在节点上的

看 `__mul__` 里的关键三行：

```python
out = Value(self.data * other.data, (self, other), '*')   # ①

def _backward():                                           # ②
    self.grad  += other.data * out.grad
    other.grad += self.data  * out.grad

out._backward = _backward                                  # ③
```

- **①** 正向算完，构造新节点 `out`。此时它的 `_backward` 还是默认的 `lambda: None`。
- **②** 定义一个**内部函数** `_backward`。注意它引用了外层的 `self`、`other`、`out`——这三个变量**不是函数参数**，是从外层函数"捕获"进来的。这种现象叫**闭包**。即使 `__mul__` 返回后、作用域消失了，这三个引用依然存活，因为闭包替它们"保活"。
- **③** 把闭包挂到 `out._backward` 属性上。将来调用 `out._backward()` 时，它还能访问到 `self / other / out`——这就是"每个节点自己知道该怎么分发梯度"的技术实现。

**如果你把这三行理解成 "out 节点带着一把私有的钥匙，钥匙能打开 self 和 other 的 grad 字段来累加"——你就懂了。**

### 4.2 为什么 `out.grad` 要写在闭包里，而不是函数参数

对比两种写法：

```python
# 版本 A（micrograd 采用）
def _backward():
    self.grad += other.data * out.grad   # 闭包里读 out.grad
```

```python
# 版本 B（不行）
def _backward(upstream):
    self.grad += other.data * upstream   # 参数传入
```

B 看起来更干净，为什么不用？

因为 `backward()` 在调用的时候**一次只调用一个节点的 `_backward`**，它不知道"这个节点的下游是谁、下游把多少梯度递过来"。梯度是**累加**在节点本身的 `.grad` 字段上的（想想 §1.5，一个节点可能被多个下游用到）。等所有下游都把梯度累加进来了，轮到自己时，直接读 `out.grad` 就拿到了完整的下游梯度——不用也不能从参数传。

**这个设计的本质：把"节点有多少下游"这件事藏起来了——每个节点只看自己 `.grad` 里的累计结果。**

### 4.3 拓扑排序的递归写法

```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)   # 注意：先递归 children，再 append 自己——后序
build_topo(self)
```

- **为什么递归后再 append？** 后序遍历保证子节点**先**进 `topo`，父节点**后**进 `topo`。
- **正向 `topo` 是叶子在前、根在后。** 所以 `reversed(topo)` 就是"根在前、叶子在后"——正是反向传播需要的顺序。
- **`visited` 的作用：** 防止同一个节点被处理多次。考虑 skip connection：`z = x + f(x)`，`x` 会从两条路径被遍历到，但 `topo` 里只能出现一次，否则 `_backward` 会被调两次，梯度翻倍。

### 4.4 `self.grad = 1.0` 这一行为什么放在拓扑排序之后

调用 `loss.backward()` 时，我们要让 loss 自己的 grad 从 0 变成 1（种子），这样 loss 节点的 `_backward`（如果有）从 `out.grad` 读出来的才是正确的种子 1。如果种子放在拓扑之前赋值，后续多次 backward 需要清 grad 也会更麻烦。习惯上**种子放在拓扑排序之后、遍历之前**。

### 4.5 几个容易踩的坑

1. **忘了 `out._backward = _backward`** → 定义了闭包但没挂上，反向传播等于没写。
2. **把 `_backward` 写在 `__init__` 里** → `__init__` 时还不知道这个节点是怎么来的（是加还是乘），必须在 `__add__` / `__mul__` 里具体化之后再挂。
3. **闭包里写成 `self.grad = ...`（没有 +=）** → §1.5 讲过的 `y = x + x` 陷阱。
4. **`tanh` 的局部导数写成 `1 - self.data**2`** → 错。应当是 `1 - out.data**2`（对 **输出** 的平方，不是输入）。数学推导：`tanh'(x) = 1 - tanh²(x)`，而 `tanh(x)` 就是 `out.data`。
5. **拓扑排序忘了 visited** → 钻石图会把节点加多次，导致 `_backward` 被多次调用，梯度翻倍。
6. **多次 backward 忘了清 grad** → 第二次 backward 时梯度会在上一次的基础上继续累加。修复：加一个 `zero_grad()` 方法遍历整张图把 grad 清 0。

---

## 5. 一个完整的端到端例子：单个神经元的反向传播

前面的小图过于"玩具"。下面这个例子就是**一个真正的神经元**——两输入一输出，tanh 激活。这也是 Karpathy 视频 EP1 最后的经典例子。

### 5.1 神经元的数学定义

```
n = x1*w1 + x2*w2 + b          # 线性组合（pre-activation）
o = tanh(n)                    # 激活
```

给定：`x1 = 2, x2 = 0, w1 = -3, w2 = 1, b = 6.8813735870195432`（这个古怪的 b 是 Karpathy 刻意挑的，让 `n` 正好等于 0.8813…，以便 `tanh(n) ≈ 0.7071`，输出刚好是一个好看的数）。

我们想要的东西：**`do/dw1`、`do/dw2`、`do/db`** ——也就是"这个神经元的三个参数各自对输出贡献多少梯度"，这正是训练时更新参数需要的东西。

### 5.2 代码

把这段加到 `value.py` 的 `__main__` 里，或者单独存成 `neuron_demo.py`：

```python
def demo_neuron():
    print("=" * 50)
    print("验证 4：单个神经元的反向传播")
    print("=" * 50)

    # 输入
    x1 = Value(2.0,  label='x1')
    x2 = Value(0.0,  label='x2')
    # 权重 + 偏置
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0,  label='w2')
    b  = Value(6.8813735870195432, label='b')

    # 正向
    x1w1 = x1 * w1;        x1w1.label = 'x1*w1'
    x2w2 = x2 * w2;        x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b;      n.label = 'n'
    o = n.tanh();          o.label = 'o'

    print(f"  正向输出 o.data = {o.data:.6f}   （应约为 0.7071）")

    # 反向
    o.backward()

    print(f"  do/dw1 = {w1.grad:.6f}   （期望 ≈ 1.0）")
    print(f"  do/dw2 = {w2.grad:.6f}   （期望  = 0.0，因为 x2=0）")
    print(f"  do/db  = {b.grad:.6f}    （期望 ≈ 0.5）")
    print(f"  do/dx1 = {x1.grad:.6f}   （期望 ≈ -1.5）")
    print(f"  do/dx2 = {x2.grad:.6f}   （期望 ≈ 0.5）")


# 在 __main__ 中加一行：
# demo_neuron()
```

### 5.3 这些"期望值"是怎么来的

你不用信代码，亲手推一遍——这正是 §1.2 链式法则的练习：

- **种子：** `o.grad = 1`
- **tanh 节点：** 局部导数 `1 - o² = 1 - 0.7071² ≈ 0.5`。所以 `n.grad = 1 × 0.5 = 0.5`。
- **加法 `n = x1w1x2w2 + b`：** 梯度 1:1 传递。所以 `x1w1x2w2.grad = 0.5`，`b.grad = 0.5`。
- **加法 `x1w1x2w2 = x1w1 + x2w2`：** 同样 1:1。`x1w1.grad = 0.5`，`x2w2.grad = 0.5`。
- **乘法 `x1w1 = x1 * w1`：** 局部导数 `dw1 = x1 = 2`，`dx1 = w1 = -3`。所以 `w1.grad = 0.5 × 2 = 1`，`x1.grad = 0.5 × (-3) = -1.5`。
- **乘法 `x2w2 = x2 * w2`：** 局部导数 `dw2 = x2 = 0`，`dx2 = w2 = 1`。所以 `w2.grad = 0.5 × 0 = 0`，`x2.grad = 0.5 × 1 = 0.5`。

**手推结果 = 代码结果 = ✅。** 你已经掌握反向传播了。

### 5.4 这例子告诉了你什么

- **`w2.grad = 0` 不是 bug，是数学。** 因为 `x2 = 0`，所以 `w2` 在这次前向里**完全没起作用**，自然没有梯度。训练时这个权重会"停在原地"，要等 `x2 ≠ 0` 的样本进来才更新。这是为什么**神经网络需要多样化的训练数据**。
- **`b.grad = 0.5`，只是 tanh 的局部导数。** 偏置项总是直接吃到 `1 × upstream`，所以它的梯度就是 "上游梯度"。这是为什么**偏置几乎总在更新**——它被所有样本"用到"。
- **`x1.grad` 也有值。** 在最顶层的神经元里输入的梯度没啥用（输入是数据不是参数），但下层神经元的"输入"其实是上层神经元的"输出"，所以**输入的梯度就是"上一层的输出梯度"**，反向传播正是靠它一层一层往回传。

---

## 6. 自测 5 题（做完代码必答）

每题 2-3 句话写在 `W3_day2_log.md` 里：

1. **自动微分、数值微分、符号微分有什么区别？** 为什么深度学习必须选自动微分？（提示：动态图、性能、精度）
2. **为什么反向传播只需要"局部导数"？** 如果我改了 loss 函数，加法节点的 `_backward` 要改吗？为什么不用改？
3. **如果把 `_backward` 里的 `+=` 改成 `=`，写一个最短的计算图能复现 bug。** 手推 + 代码各一遍。
4. **为什么必须先拓扑排序再反向传播？** 举一个不拓扑排序就会算错的例子。
5. **`o.backward()` 这一行背后发生了什么？** 用 3 步描述（种子 / 拓扑 / 逆序调用闭包）。

---

## 7. 常见错误速查表

| 现象 | 可能原因 | 修复 |
|---|---|---|
| 所有 grad 都是 0 | 没有调 `self.grad = 1.0` 种子；或 `_backward` 没挂上 | 检查 `backward()` 里的种子；检查每个运算函数末尾有 `out._backward = _backward` |
| `y = x + x` 的 `x.grad = 1`（应为 2） | `+=` 写成 `=` | 把所有梯度更新改成 `+=` |
| 第二次 `backward()` 结果翻倍 | 没清零 | 每次 backward 前遍历图把 grad 清 0（或加 `zero_grad()`） |
| `tanh` 的梯度对不上 | 局部导数写成 `1 - self.data**2` | 改成 `1 - out.data**2`（对 **输出** 平方） |
| 拓扑排序 stack overflow | 图太深递归爆栈 | 改迭代版（用显式 stack），或 `sys.setrecursionlimit` |
| `Value(10) + Value(20)` 报 TypeError | `__add__` 没有 `other = Value(other) if ...` 包装 | 补上包装（已在代码里） |
| 同一个节点被重复 append 到 topo | 忘了 `visited` 检查 | 在 `build_topo` 开头加 `if v not in visited` |

---

## 8. 与 PyTorch 的一一对应

你今天写的每一行，PyTorch 里都有对应实现：

| micrograd（你今天写的） | PyTorch |
|---|---|
| `Value.data` | `Tensor.data` / `Tensor.detach()` |
| `Value.grad` | `Tensor.grad` |
| `Value._prev` | `Tensor.grad_fn.next_functions` |
| `Value._op` | `Tensor.grad_fn`（类型即运算） |
| `out._backward` 闭包 | `torch.autograd.Function.backward()` |
| `build_topo` 拓扑排序 | `torch.autograd.engine`（C++ 实现，并行化） |
| `self.grad = 1.0` 种子 | `loss.backward(gradient=torch.ones_like(loss))` |
| `+= 累加` | `Tensor.grad.add_(...)` |
| 需要手动 zero_grad | `optimizer.zero_grad()` 的由来 |

**区别只有两点：**
1. PyTorch 的 tensor 是多维数组，micrograd 是标量。标量版适合教学，把自动微分最核心的逻辑暴露得一清二楚。
2. PyTorch 的 grad_fn 是**类**不是闭包（C++ 里闭包不方便），但原理完全一样。

---

## 9. 给 Day3 的伏笔

Day3 的任务是"**不看源码独立复现**"。建议你：

1. 关掉 Day1、Day2 的代码，打开一个新的 `micrograd_redo/value.py`，**盲写一遍**。写不出来的地方不要翻回去看，而是先对着 §1.2–1.5 的原理回忆，推不出来再翻。
2. 独立复现时重点自测：`y = x + x`、`z = x * x`、以及 §5 的神经元例子。
3. **额外挑战：** 加一个 `exp()`、`__pow__`、`__truediv__`。这三个运算有了之后，你就能实现 `sigmoid`、`softmax`、交叉熵等真实神经网络的全部组件。

Day4 的 make_moons 训练会用到这三个运算中的至少一个，所以 Day3 顺手做了的话 Day4 就没障碍。

---

## 10. 完成检查清单（睡前对照）

- [x] `value.py` 实现 `__add__ / __mul__ / tanh` 的 `_backward`，且全部用 `+=`
- [x] `backward()` 方法写好，包含拓扑排序 + 种子 + 逆序调用
- [x] §3 的三段验证全部通过（小图、累加陷阱、数值梯度对比）
- [x] §5 的神经元例子跑通，5 个梯度值与手推完全一致
- [x] `W3_day2_log.md` 回答了 §6 的 5 个自测题
- [x] README.md 的进度条勾选了 Day 2

完成以上 → 今天合格。
能口述"为什么 `+=`、为什么拓扑排序、为什么闭包" → 今天优秀。
Day 3 可以直接开始独立复现。

---

## 11. 一句话总结这一整天

> **自动微分 = 每个运算自带局部导数 + 链式法则在拓扑图上反向传播。**
> 你今天写的不是一段"仿 PyTorch 的玩具代码"——你写的就是 PyTorch 的核心原理，只是用 Python 标量版呈现。把 `_backward` 闭包、`+=` 累加、拓扑排序这三件事理解到能默写，你就打通了所有深度学习框架的任督二脉。

---

*笔记生成时间：2026-05-12*
*前置依赖：W3_Day1_micrograd_笔记.md*
*下一篇：W3_Day3_micrograd_独立复现笔记.md*

