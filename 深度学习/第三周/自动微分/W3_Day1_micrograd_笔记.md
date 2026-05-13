# W3 Day1 学习笔记：从计算图说起（micrograd 第一步）

> 日期：2026-05-11（周一）  
> 来源任务：第3周 Day1 的 DL 部分（3h）  
> 视频范围：Karpathy "Neural Networks: Zero to Hero" EP1 第 0:00 – 1:00  
> 今日原则：**只搭计算图，不写 backward**。先把"图"建对，明天再让梯度沿图反向流动。

---

## 0. 先看清楚今天要拿到什么

完成这份笔记 + 代码后，你应该能做到：

1. 用一句话回答：**为什么不能用 numpy 的 ndarray 直接做自动微分？micrograd 多出来的那点东西到底是什么？**
2. 默写出 `Value` 类的 5 个核心属性（`data / grad / _prev / _op / label`），并解释每个的用途。
3. 用自己的 `Value` 类构造表达式 `c = a*b + Value(10.0)`，打印每个节点的 `_prev`，且关系正确。
4. 把这张计算图画出来（手画或 graphviz 都行），说明哪些是叶子节点、哪些是中间节点。

如果以上有任何一条做不到，**不要进入 Day 2**——Day 2 会在这套结构上加 `_backward`，结构没立稳，反向传播一定踩坑。

---

## 1. 知识点串讲：从你第 2 周的经验出发

### 1.1 为什么你第 2 周写得很爽，今天却要重新开始？

第 2 周你在 `nn.py` 里写的是这样的反向传播：

```python
dZ2 = A2 - Y
dW2 = A1.T @ dZ2 / N
db2 = dZ2.sum(axis=0) / N
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)
dW1 = X.T @ dZ1 / N
db1 = dZ1.sum(axis=0) / N
```

这 7 行能跑，但它有两个**致命限制**：

| 限制 | 含义 | 现实代价 |
|---|---|---|
| **梯度公式写死** | 网络结构一改（比如多加一层、把 ReLU 换成 tanh），你必须**重新手推**所有公式 | 改个激活函数都要重推半天 |
| **只能描述固定形状的网络** | `dZ1 = dA1 * (Z1 > 0)` 这种代码假定了"上一层一定是 ReLU"。如果是动态计算图（比如 RNN 的循环展开），这个写法根本没法用 | RNN/Transformer 全部跑不了 |

PyTorch 不会让你写 `dZ2 = A2 - Y`。你只写 **正向传播 + `loss.backward()`**，所有梯度它自动算出来。它怎么做到的？

**核心答案就两步：**

1. 正向传播时，每做一次运算（加、乘、tanh…），系统**偷偷记录一笔账**：这个新值是由哪两个旧值、用什么运算产生的。
2. 反向传播时，从最终的 loss 出发，按照这本账**回溯**整张网络，每经过一个节点就调一下它对应运算的"局部导数公式"，把梯度累加上去。

第 1 步记录的那本账，就是**计算图（Computation Graph）**。  
今天要做的，就是把这本账的数据结构（`Value` 类）写出来。

> 🔑 **一句话总结：** numpy 的 ndarray 不知道自己是怎么来的，所以无法自动求导；`Value` 类的核心创新就是——**每个数都自己记住自己的"出身"**。

### 1.2 计算图到底是个什么图？

举个最小例子：`c = a * b + 10`，其中 `a=2, b=-3`。

人类怎么算：先算 `a*b = -6`，再算 `-6 + 10 = 4`。

计算机要画成图的话：

```
        a(2.0)         b(-3.0)
            \           /
             \         /
              [ * ]  ←  一个"乘法节点"，输出值 = -6
                |
                v
              e(-6.0)        Value(10.0)
                  \             /
                   \           /
                    [ + ]  ←  一个"加法节点"，输出值 = 4
                       |
                       v
                     c(4.0)
```

**关键约定（务必想清楚）：**

- 图里的**节点**是 `Value` 对象，每个节点存一个**数值**（标量）。
- 图里的**边**表达"谁是谁的输入"。一条从 `a` 指向 `e` 的边，意思是"`a` 是 `e` 的一个输入"。
- 每个节点除了自己的数值，还要记录：
  - **`_prev`**：我是由哪些 `Value` 算出来的（就是我入边连出去的那些前驱）
  - **`_op`**：用什么运算算出来的（`'+'` / `'*'` / `'tanh'` …）
  - **`grad`**：最终 loss 对我求偏导是多少（**今天先初始化为 0，不算**）
  - **`label`**：可选的人类可读名字，比如 `'a'`、`'e'`，纯粹是为了画图好看

**叶子节点（leaf）vs 中间节点：**

- 叶子节点：`_prev = ()` 空集，`_op = ''` 空字符串。它们是用户直接 `Value(2.0)` 创建的。在真正的网络里，**叶子节点就是网络的权重、偏置、输入数据**。
- 中间节点：`_prev` 非空，`_op` 是某种运算。它们是运算的产物。

第 2 周你网络里的 `W1, b1, W2, b2` 就是 4 个叶子，`Z1, A1, Z2, A2, loss` 都是中间节点。只不过那时候你用 numpy，**它们之间没有连接关系**，所以你必须手写 `dZ2 = ...` 来"补"这层连接。今天，你要让连接关系自己活在数据结构里。

### 1.3 为什么今天故意不写 backward？

Karpathy 在 EP1 里也是分开讲的：先 0:00–1:00 讲 `Value` 类和构图，再 1:00–2:25 加 `_backward`。**这个顺序极其重要**，因为：

1. **解耦学习**：构图是"数据结构题"，反向传播是"算法题"。先把数据结构定下来，算法才好写。
2. **避免一上来被链式法则绕晕**：很多人第一次学是从"链式法则"开始的，结果把数学和代码混在一起，两边都没搞清。今天你**只做数据结构**，明天再上链式法则。
3. **强化你对 `_prev` 的理解**：今天你只能用 `_prev` 来打印图、画图、做断言，所以会反复盯着它看。等明天写拓扑排序时，你会发现拓扑排序操作的对象就是 `_prev` 构成的 DAG（有向无环图）。

### 1.4 `_prev` 为什么用 set/tuple，不用 list？

Karpathy 视频里用的是 `set`（集合）。原因：

- **去重**：考虑 `y = x + x` 这种情况，`y._prev` 应该只记一个 `x`，不是 `[x, x]`。明天写反向传播时，遍历 `_prev` 是不希望同一个节点被走两次的（每个节点的 `_backward` 闭包里**自己**已经处理了"梯度对两个 x 都要累加"的事）。
- **快速判重**：拓扑排序的 `visited` 集合用 set 也最自然。

不过用 `tuple` 也行（Karpathy 后来在 micrograd 仓库的 README 里说 set 其实有点小坑，因为 set 的元素必须 hashable，Python 默认对象是 id 哈希所以也 OK）。**今天先跟视频用 `set`**，统一风格。

---

## 2. 三个任务的完整实现

### 任务 A：搭建项目目录

#### 目标结构

```
week3_karpathy/
├── micrograd_follow/       ← 跟视频敲的代码放这里（Day 1-2）
│   └── value.py
├── micrograd_redo/         ← Day 3 不看视频独立复现用
│   └── (暂时空)
├── logs/
│   └── (训练日志、截图)
└── README.md
```

#### 一次性建好的命令（在 Git Bash / PowerShell / cmd 里都能跑）

**Bash（推荐，因为你 Claude Code 默认 bash）：**

```bash
cd ~/Desktop   # 或者你想放项目的任何位置
mkdir -p week3_karpathy/micrograd_follow
mkdir -p week3_karpathy/micrograd_redo
mkdir -p week3_karpathy/logs
touch week3_karpathy/README.md
touch week3_karpathy/micrograd_follow/value.py
```

#### `README.md` 的初版内容（直接复制进去）

```markdown
# week3_karpathy

跟随 Andrej Karpathy 的 "Neural Networks: Zero to Hero" EP1-EP2 的复现项目。

## 目录说明

- `micrograd_follow/`：Day 1-2 跟视频敲的版本
- `micrograd_redo/`：Day 3 缓冲日不看源码独立复现版本
- `logs/`：训练日志、计算图截图

## 运行

```bash
cd micrograd_follow
python value.py   # 运行 Day 1 的构图 demo
```

## 进度

- [x] Day 1：Value 类 + 加法/乘法，构图打印
- [ ] Day 2：_backward 闭包 + 拓扑排序 + backward()
- [ ] Day 3：独立复现（micrograd_redo/）
- [ ] Day 4：用 micrograd 训练 make_moons 二分类
- [ ] Day 5-6：bigram 语言模型（EP2）
- [ ] Day 7：技术笔记 + autograd_explained.md
```

---

### 任务 B：实现 `Value` 类（只做正向 + 构图）

下面这份代码可以**直接保存为 `week3_karpathy/micrograd_follow/value.py` 并 `python value.py` 运行**。代码后面附了逐段讲解。

```python
"""
W3 Day1: micrograd 第一步——只构图，不反向传播。
跟随 Karpathy EP1 第 0:00-1:00。
"""


class Value:
    """一个能记住自己 '出身' 的标量。

    属性
    ----
    data : float
        节点存储的数值本身。
    grad : float
        最终 loss 对这个节点的偏导数。今天先初始化为 0，不计算。
    _prev : set[Value]
        产生这个节点的所有前驱 Value（叶子节点为空集）。
        命名以下划线开头，表示 '这是内部状态，外人少碰'。
    _op : str
        产生这个节点用的运算符（'+', '*' 等）。叶子节点为空字符串。
    label : str
        可选的人类可读标签，仅用于打印和画图，不参与计算。
    """

    def __init__(self, data, _children=(), _op='', label=''):
        # 把传入的 _children（任何可迭代对象）统一存成 set，方便去重和判重
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        # 控制 print(v) 看到的样子。带 label 时显示更友好。
        if self.label:
            return f"Value(label='{self.label}', data={self.data})"
        return f"Value(data={self.data})"

    def __add__(self, other):
        # self + other 时被调用
        # 把对方也包装成 Value（兼容 a + 10 这种写法，10 会被自动包成 Value(10)）
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data + other.data,
            _children=(self, other),   # 记录前驱：我是由 self 和 other 加出来的
            _op='+',
        )
        return out

    def __mul__(self, other):
        # self * other 时被调用
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data * other.data,
            _children=(self, other),
            _op='*',
        )
        return out


# ----------------------------------------------------------------------
# 验证：构造一个简单计算图，检查 _prev 关系是否正确
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 用 label 让节点有人类可读名字
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b                # c = -6
    c.label = 'c'
    d = Value(10.0, label='d')
    e = c + d                # e = 4
    e.label = 'e'

    print("===== 节点信息 =====")
    for node in [a, b, c, d, e]:
        prev_labels = [p.label or f"<no-label data={p.data}>" for p in node._prev]
        print(f"{node!r}  | _op='{node._op}'  | _prev={prev_labels}")

    print("\n===== 验证断言 =====")
    # 1. 叶子节点的 _prev 必须为空
    assert a._prev == set() and b._prev == set() and d._prev == set(), \
        "叶子节点的 _prev 应该是空集"
    # 2. c 的前驱必须正好是 {a, b}
    assert c._prev == {a, b}, "c 的前驱应当是 {a, b}"
    # 3. e 的前驱必须正好是 {c, d}
    assert e._prev == {c, d}, "e 的前驱应当是 {c, d}"
    # 4. 数值正确
    assert c.data == -6.0
    assert e.data == 4.0
    # 5. 今天梯度还都是 0
    assert all(v.grad == 0.0 for v in [a, b, c, d, e])
    print("✅ 所有断言通过。计算图构建正确。")
```

#### 预期运行输出

```
===== 节点信息 =====
Value(label='a', data=2.0)  | _op=''  | _prev=[]
Value(label='b', data=-3.0)  | _op=''  | _prev=[]
Value(label='c', data=-6.0)  | _op='*'  | _prev=['b', 'a']  ← 顺序可能不同（set 无序）
Value(label='d', data=10.0)  | _op=''  | _prev=[]
Value(label='e', data=4.0)  | _op='+'  | _prev=['c', 'd']

===== 验证断言 =====
✅ 所有断言通过。计算图构建正确。
```

#### 逐段讲解（盯着代码看）

| 代码 | 在做什么 | 为什么这样 |
|---|---|---|
| `self.grad = 0.0` | 初始化梯度为 0 | 今天不算，但留好坑位。明天反向传播时往里累加。 |
| `set(_children)` | 把前驱存成集合 | 自动去重，且后面 `in` 判断 O(1)。 |
| `other if isinstance(other, Value) else Value(other)` | 把数字自动包装成 Value | 让 `a + 10` 这种写法能跑（否则 10 没有 `_prev` 等属性，下游会炸） |
| `_children=(self, other)` | 关键一行：记账 | 这就是"每个节点记住自己出身"的具体实现 |
| `return out` | 返回新 `Value` | **正向传播会一路构造新节点**——这就是计算图的生长过程 |

#### 三个**最容易踩的坑**（提前预警）

1. **写成 `self._prev = _children`（少了 `set()`）**：如果传进来的是 tuple，明天写 `_prev.add(...)` 时会报 `AttributeError`。即便不写 add，集合的去重特性也很重要（想想 `y = x + x` 的情况）。
2. **`__add__` 里忘了把 `other` 转成 Value**：写 `c = a + 10`，10 是 int 没有 `.data`，整段会崩。
3. **把 `_op` 写在 `_prev` 里**：`_prev` 只放前驱 Value 对象，`_op` 是个字符串，单独的字段，不要混。

---

### 任务 C：可视化计算图

视频里 Karpathy 用了 graphviz，给了现成函数。下面给两种方案，**任选其一**：

#### 方案 1（推荐）：用 graphviz 自动画

**前置安装：**

```bash
pip install graphviz
```

外加 graphviz 本体（Windows 上去 https://graphviz.org/download/ 下安装包，安装时勾选"添加到 PATH"，重启终端）。如果懒得装系统 graphviz，可以跳到方案 2。

把下面这段加到 `value.py` 末尾（或者另存为 `draw.py`），运行后会生成 `logs/graph_demo.png`。

```python
# ---- 把这段加在 value.py 末尾（或单独 draw.py） ----
from graphviz import Digraph


def trace(root):
    """从 root 出发，遍历整个计算图，返回所有节点 + 所有边。"""
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))   # 注意方向：child 流向 v
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, filename='graph_demo'):
    """画出从 root 回溯到所有叶子的整张计算图。"""
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})  # 从左到右

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # 用矩形画 Value 节点：label | data | grad
        dot.node(
            name=uid,
            label=f"{{ {n.label or '?'} | data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape='record',
        )
        if n._op:
            # 给每个运算节点单独画一个圆圈
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.render(filename, directory='../logs', cleanup=True)
    print(f"✅ 计算图已生成：../logs/{filename}.png")


# 在 if __name__ == "__main__" 里追加一行：
# draw_dot(e, filename='graph_demo')
```

#### 方案 2（零依赖）：纯文本打印

实在装不了 graphviz，下面这个函数用 ASCII 输出树状结构，足够你验证 `_prev` 关系：

```python
def print_graph(node, indent=0):
    """从 node 出发反向打印计算图，越深越偏右。"""
    prefix = "  " * indent + ("└─ " if indent > 0 else "")
    name = node.label or f"v_{id(node) % 1000}"
    print(f"{prefix}{name}  (data={node.data:.4f}, op='{node._op or 'leaf'}')")
    for child in node._prev:
        print_graph(child, indent + 1)


# 调用
# print_graph(e)
```

预期输出：

```
e  (data=4.0000, op='+')
  └─ c  (data=-6.0000, op='*')
    └─ a  (data=2.0000, op='leaf')
    └─ b  (data=-3.0000, op='leaf')
  └─ d  (data=10.0000, op='leaf')
```

#### 方案 3（手画）：当然可以——画一张拍照存到 `logs/graph_demo.jpg`。手画反而能加深印象，第一次推荐这样做。

---

## 3. 今天的"思考闭环"——做完代码必答这 5 题

写完代码不算完，要在 `W3_day1_log.md` 里回答这 5 个问题（每题 2-3 句话即可）：

1. **`_prev` 用 set 而不用 list，到底是为了什么？** 给一个具体的例子说明 list 会带来什么问题。
2. **如果我把 `__add__` 里的 `_children=(self, other)` 改成 `_children=()`，正向算出来的 `data` 还对吗？画图会怎样？** （提示：data 对，画图断成两半）
3. **`Value(10.0)` 是叶子还是中间节点？为什么？** （提示：看 `_op` 和 `_prev`）
4. **`a + b + c` 这个表达式会产生几个新节点？请画出来。** （提示：Python 会拆成 `(a+b)+c`，所以产生 2 个新节点）
5. **第 2 周里你写的 `Z1 = X @ W1 + b1`，如果这一句要进入 micrograd 风格的计算图，会产生几个中间 Value？** （提示：矩阵乘法是一个节点，加法是一个节点；但今天的 micrograd 是**标量**版的，矩阵化要等到后面 PyTorch tensor 才支持）

---

## 4. 常见错误与排查（出问题时回来看）

| 现象 | 可能的原因 | 怎么修 |
|---|---|---|
| `TypeError: unsupported operand type(s) for +: 'Value' and 'int'` | `__add__` 里没有把 other 自动包成 Value | 加 `other = other if isinstance(other, Value) else Value(other)` |
| `c._prev` 是空集 | `__mul__` 里忘了传 `_children=(self, other)` | 补上 |
| `print(v)` 显示一长串内存地址 | 没写 `__repr__`，或者写成了 `__str__` | 用 `__repr__`，因为列表打印用的是 repr |
| `c._prev == {a, b}` 断言失败 | 多半是 `_children=(self, other)` 写成了 `_children=[self, other]` 然后 set 化时漏了 | 仔细看 `set(_children)` 是否生效 |
| graphviz 报 `ExecutableNotFound` | 装了 Python 包但没装系统 graphviz | Windows 下到官网下安装包，重启终端 |

---

## 5. 给明天的伏笔（Day 2 预告，今天**不要**做）

- 明天加 `_backward` **闭包**：每个运算节点上挂一个函数，这个函数知道"我自己的局部导数是什么、要把多少梯度往前驱传"。
- 加 `backward()` 方法：先**拓扑排序**所有节点，从最后一个节点（loss）开始，按逆拓扑序依次调用 `_backward`。
- 关键 bug 预警：所有梯度更新必须用 `+=` 而不是 `=`，否则 `y = x + x` 这种式子会算错。

**你今天写好的 `_prev` 就是明天拓扑排序操作的对象**——这就是为什么今天必须把图建对。

---

## 6. 与你已有知识的对照表

| 第 2 周（numpy 手写网络） | 今天（micrograd Day1） |
|---|---|
| `Z1 = X @ W1 + b1`：一行就完成正向 | `out = Value(self.data + other.data, _children=(self, other), _op='+')`：每一次运算都要构造新节点 |
| `dZ2 = A2 - Y`：硬编码的反向公式 | 暂时**没有反向**——明天把"反向公式"挂在节点上作为闭包 |
| `W1, b1` 等参数是 `np.ndarray` | 网络参数会变成 `Value` 对象，每个标量一个 |
| 改激活函数 = 重推公式 | 改激活函数 = 加一个新运算节点（明天会加 `tanh`） |

**核心交易：** 用"运行慢、对象多"换"灵活到任意结构都能自动求导"。所以 Day 4 用 micrograd 训练 make_moons 时会发现它**比第 2 周的 numpy 网络慢得多**——这是必然的，明天你就明白为什么。

---

## 7. 完成检查清单（睡前对照）

- [ ] `week3_karpathy/` 目录建好，结构如本笔记 §2-A
- [ ] `value.py` 实现 `__init__ / __repr__ / __add__ / __mul__`，**没有写 backward**
- [ ] `python value.py` 跑通，5 条断言全部通过
- [ ] 计算图画出来了（graphviz / ASCII / 手画 三选一），存到 `logs/`
- [ ] `W3_day1_log.md` 写完，至少回答了 §3 的 5 个问题
- [ ] README 的进度条勾选了 Day 1

完成全部以上 → 今天合格。  
完成 §3 的全部 5 题并能口述 → 今天优秀，可以放心进 Day 2。

---

*笔记生成时间：2026-05-11*  
*作者：你的学习助手*
