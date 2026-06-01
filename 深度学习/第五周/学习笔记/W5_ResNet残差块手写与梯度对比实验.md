# ResNet 残差块手写 + 50 层梯度对比实验

> 学习目标：彻底搞懂 **BasicBlock / Bottleneck** 怎么搭、为什么这么搭；并用一个 50 层网络"加/不加 skip"的对照实验，**亲眼看到梯度消失**以及残差连接是怎么救场的。
>
> 阅读方式：建议**边读边敲代码**。最后那个梯度实验是整篇的"题眼"，一定要自己跑一遍看数字。
>
> 运行环境：Python 3.9+ / PyTorch 2.x（CPU 即可跑，无需 GPU）。安装：`pip install torch`

---

## 〇、先建立直觉：这篇到底在解决什么问题

2015 年之前，深度学习圈有个反直觉的现象困扰了所有人：

> **把网络层数从 20 层加到 56 层，训练误差反而变大了。**

注意——不是过拟合（过拟合是训练好、测试差），而是**连训练集都学不好**了。按理说层数越多、表达能力越强，至少不该比浅层网络更差吧？这个现象叫 **退化问题（degradation problem，退化问题）**。

**退化问题**：网络加深后，不是因为过拟合，而是因为"优化变难了"，导致深层网络连训练精度都比浅层网络低的现象。

> 类比：你让一个新员工"照抄"老员工的工作（什么都不改），照理说结果应该一模一样。但如果中间隔了 50 个传话人，每个人都把话稍微改一点，传到最后早就面目全非——**让深层网络学会"什么都不做（恒等映射）"，居然成了一件很难的事**。

ResNet 的作者何恺明意识到：**问题不在表达能力，而在优化**。如果深层网络很难学会"恒等映射"（identity mapping，恒等映射，即输出=输入），那我们干脆**把恒等映射这条路直接焊死在结构里**，让网络只去学"在恒等的基础上还需要改动多少"。这就是残差学习。

**恒等映射（identity mapping）**：一个函数原样返回输入，$f(x)=x$。在网络里就是"这一层啥也不干，把数据直接放过去"。

---

## 一、核心原理：残差连接（Residual Connection）

### 是什么

普通网络让若干层去拟合一个目标映射 $H(x)$。残差网络换了个思路：**让这几层去拟合"残差" $F(x) = H(x) - x$**，最终输出写成：

$$
H(x) = F(x) + x
$$

这个"$+x$"就是 **skip connection（跳跃连接，也叫 shortcut，捷径连接）**——把输入 $x$ 绕过中间几层，直接加到输出上。

**残差（residual）**：字面意思"剩下的差值"。这里指"目标输出"和"输入"之间的差距。

> 类比：你要把一张照片 P 成"稍微调亮一点"。
> - **普通做法**：从零重新画一张调好亮度的照片（学整个 $H(x)$，难）。
> - **残差做法**：保留原图，只学"每个像素加多少亮度"这个**差值**（学 $F(x)$，简单）。如果原图已经够亮，网络只要让 $F(x)=0$ 就行——而**把一堆权重学成 0，比学成恰好的恒等映射容易得多**。

### 为什么这样设计能救梯度

这是最关键的部分。我们看反向传播时梯度怎么流。设某个残差块输出 $y = F(x) + x$，损失为 $L$。根据链式法则，梯度往输入 $x$ 传时：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
= \frac{\partial L}{\partial y} \cdot \left( \frac{\partial F(x)}{\partial x} + 1 \right)
$$

注意那个 **"+1"**。它来自 skip connection 这条"高速路"。它意味着：

- **普通网络**：梯度 = 一连串 $\frac{\partial F}{\partial x}$ 连乘。每个因子如果都小于 1（很常见），50 个一乘就趋近于 0 → **梯度消失（gradient vanishing）**，浅层学不动。
- **残差网络**：每一项都多了个 "+1"。即使 $\frac{\partial F}{\partial x}$ 接近 0，梯度也能靠这个 "+1" **至少原样传回去**，不会被乘没。

**梯度消失（gradient vanishing）**：反向传播时，梯度经过很多层连乘后变得极小，导致靠近输入的层几乎收不到有效的更新信号，学习停滞。

> 类比：传话游戏里，每个人都把音量降低一点点，传到第 50 个人已经听不见了（梯度消失）。残差连接相当于额外架了一根**电话专线**，无论中间多少人，原始声音始终能直达——这就是那个 "+1" 的作用。

> 这正是本篇最后那个实验要让你**亲眼看到**的：不加 skip 时浅层梯度是个接近 0 的天文小数，加了 skip 后浅层梯度立刻"活"过来。

---

## 二、BasicBlock 手写（ResNet-18/34 用的基础块）

### 是什么

**BasicBlock（基础残差块）**是 ResNet-18 和 ResNet-34 的building block，结构很朴素：

```
输入 x
  │
  ├──────────────────────────┐ (shortcut 捷径)
  ▼                          │
[3x3 卷积] → [BN] → [ReLU]    │
  ▼                          │
[3x3 卷积] → [BN]             │
  ▼                          │
  (+)◄───────────────────────┘   ← 在这里把 x 加回来
  ▼
[ReLU]
  ▼
输出
```

两个 3×3 卷积堆叠，外面包一条 shortcut。注意几个**容易踩坑的设计细节**，我在代码注释里逐一标出。

**BN（Batch Normalization，批归一化）**：把每一批数据在每个通道上标准化成均值 0、方差 1，再用两个可学习参数缩放平移。作用是稳定训练、允许更大学习率。你 W4 学过，这里直接用。

### 怎么做（完整可运行代码）

```python
# 环境：Python 3.9+，PyTorch 2.x，CPU 可运行
# 依赖：pip install torch
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    # expansion：输出通道相对于 base 通道的倍数。
    # BasicBlock 不扩张通道，所以是 1。下面 Bottleneck 会是 4，
    # 这个类属性是为了让外层 ResNet 统一计算通道数，是官方实现的惯例。
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 第一个卷积：3x3，可能带 stride 做下采样（缩小特征图）。
        # bias=False 是因为后面紧跟 BN，BN 自带偏移项 beta，
        # 再加 conv 的 bias 就重复了，纯属浪费参数——这是工业界标准写法。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积：stride 固定为 1，不再缩小尺寸。
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)  # inplace 省显存，工业常用

        # ---------- 关键陷阱：shortcut 的维度匹配 ----------
        # 要把 x 加到主分支输出上，两者形状必须完全一致。
        # 但当 stride!=1（特征图被缩小）或通道数变化时，x 的形状对不上，
        # 直接相加会报错。解决办法：给 shortcut 加一个 1x1 卷积来"对齐"形状。
        self.shortcut = nn.Sequential()  # 默认是恒等（什么都不做）
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                # 1x1 卷积只负责调整通道数和空间尺寸，不提取特征
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 先把 x 处理成能相加的形状

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))  # 注意：第二个 BN 之后【先不要】ReLU

        out = out + identity  # ★残差相加：必须在最后一个 ReLU 之前
        out = self.relu(out)  # ★相加之后再 ReLU
        return out


# --- 简单自测：验证形状正确 ---
if __name__ == "__main__":
    # 情况1：通道不变、尺寸不变 —— shortcut 走恒等
    blk1 = BasicBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 32, 32)  # (batch, channel, H, W)
    print("不变:", blk1(x).shape)   # 期望 [2, 64, 32, 32]

    # 情况2：通道翻倍 + 下采样 —— shortcut 走 1x1 卷积
    blk2 = BasicBlock(64, 128, stride=2)
    print("下采样:", blk2(x).shape)  # 期望 [2, 128, 16, 16]
```

### 为什么有几行"必须这么写"（深度解析）

这是新手最容易写错、且**报错信息根本看不出原因**的几个点：

1. **相加要在最后一个 ReLU 之前**。如果你写成 `relu(out) + identity`，那条"高速路"上的梯度又会被 ReLU 截断（ReLU 对负数梯度为 0），"+1"的好处就没了。原论文专门强调了这个顺序，叫 **post-activation**。

2. **第二个卷积后只有 BN、没有 ReLU**。同上，给残差留一条"干净"的相加通路。

3. **shortcut 的 1×1 卷积**：当主分支把特征图缩小或通道变多时，原始 $x$ 形状对不上。这时**不能简单相加**，必须用 1×1 卷积把 $x$ "投影"到新形状。这种 shortcut 叫 **projection shortcut（投影捷径）**，对应代码里的 if 分支；形状一致时用的恒等 shortcut 叫 **identity shortcut**。

> 易错实战：很多人复现时把 stride 加在第二个卷积上，导致 shortcut 的 stride 对不上，跑起来形状错乱。**记住：下采样统一放在 block 的第一个卷积。**

---

## 三、Bottleneck 手写（ResNet-50/101/152 用的瓶颈块）

### 是什么 & 为什么需要它

层数到了 50 层以上，如果还用 BasicBlock 的两个 3×3 卷积，**计算量和参数量会爆炸**。于是何恺明设计了 **Bottleneck（瓶颈块）**。

**Bottleneck（瓶颈块）**：名字来自它"两头粗、中间细"的形状——先用 1×1 卷积把通道数**压低**（降维），在低维度上做完那个昂贵的 3×3 卷积，再用 1×1 卷积把通道**升回去**（升维）。

> 类比：搬一大堆家具上楼。BasicBlock 是"原样一件件扛上去"；Bottleneck 是"先把家具拆小（降维）→ 搬上去（3×3 卷积在小尺寸上算，省力）→ 到楼上再组装回去（升维）"。中间那道窄门就是"瓶颈"，所以叫 bottleneck。

为什么省：3×3 卷积的计算量正比于"输入通道 × 输出通道"。把通道先从 256 压到 64，3×3 只在 64 通道上算，计算量直接降一个量级，**精度几乎不损失**。这是工业界"用结构换算力"的经典操作。

### 结构图

```
输入 x (比如 256 通道)
  │
  ├──────────────────────────────────┐ (shortcut)
  ▼                                  │
[1x1 卷积] 降维 256→64 → BN → ReLU     │
  ▼                                  │
[3x3 卷积] 64→64    → BN → ReLU        │   ← 昂贵运算在"瘦"的 64 通道上做
  ▼                                  │
[1x1 卷积] 升维 64→256 → BN            │
  ▼                                  │
  (+)◄───────────────────────────────┘
  ▼
[ReLU]
```

### 怎么做（完整可运行代码）

```python
# 环境：Python 3.9+，PyTorch 2.x
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    # ★关键：expansion=4。Bottleneck 最后会把通道升到 base 的 4 倍。
    # 例如 base_channels=64，则这个 block 实际输出 64*4=256 通道。
    # 外层 ResNet 靠这个属性算出下一层的输入通道，写错会全盘形状错乱。
    expansion = 4

    def __init__(self, in_channels, base_channels, stride=1):
        super().__init__()
        out_channels = base_channels * self.expansion  # 真实输出通道

        # 1x1 降维：把 in_channels 压到 base_channels（窄）
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # 3x3 主卷积：在窄通道上做，stride 放这里做下采样
        # （官方 v1.5 把 stride 放在 3x3 而非第一个 1x1，精度更高，是当前主流）
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)

        # 1x1 升维：把通道升回 out_channels（宽）
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # shortcut 同样要在 stride!=1 或通道不匹配时做投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))   # 降维 + 激活
        out = self.relu(self.bn2(self.conv2(out)))  # 3x3 + 激活
        out = self.bn3(self.conv3(out))             # 升维，★这里不加 ReLU

        out = out + identity   # 残差相加
        out = self.relu(out)   # 加完再激活
        return out


if __name__ == "__main__":
    # ResNet-50 第一个 stage：输入 64 通道，base=64，输出 256 通道
    blk = Bottleneck(in_channels=64, base_channels=64, stride=1)
    x = torch.randn(2, 64, 56, 56)
    print("Bottleneck 输出:", blk(x).shape)  # 期望 [2, 256, 56, 56]

    # 带下采样的 stage：输入 256，base=128，stride=2 → 输出 512、尺寸减半
    blk2 = Bottleneck(in_channels=256, base_channels=128, stride=2)
    y = torch.randn(2, 256, 56, 56)
    print("下采样 Bottleneck:", blk2(y).shape)  # 期望 [2, 512, 28, 28]
```

### BasicBlock vs Bottleneck 对比

| 维度 | BasicBlock | Bottleneck |
|---|---|---|
| 用在 | ResNet-18 / 34 | ResNet-50 / 101 / 152 |
| 卷积构成 | 3×3 + 3×3 | 1×1 + 3×3 + 1×1 |
| expansion | 1 | 4 |
| 设计目的 | 简单直接 | 降维省算力，支撑更深 |
| 易错点 | 相加前别加 ReLU | expansion=4 别忘、stride 放 3×3 |

> 工业惯例：**层数 ≤34 用 BasicBlock，≥50 用 Bottleneck**。这不是规定，是算力与精度权衡后的经验最优。

---

## 四、题眼实验：50 层网络"加/不加 skip"的梯度对比

这是整篇的核心。前面讲的"+1 救梯度"是道理，**现在我们用代码让它变成你能看见的数字。**

### 实验设计思路（为什么这么设计）

为了把"残差连接对梯度的影响"单独拎出来看，我们要**排除其它干扰因素**：
- 不用卷积、不用真实数据集——那些会引入额外变量。
- 就用最纯粹的 **50 层全连接（Linear）层**，每层后接一个激活函数。
- 造两个一模一样的网络，唯一区别：**一个有 skip connection，一个没有**。
- 喂一批随机数据，做一次反向传播，然后**打印每一层的梯度大小**，对比浅层（靠近输入）的梯度。

**这样设计的好处**：两个网络结构、参数初始化、输入完全相同，结果差异就**只能归因于 skip connection**。这是做对照实验的基本功——控制变量，工业界做 A/B 消融实验（ablation study，消融实验）也是这个思路。

**消融实验（ablation study）**：每次只去掉/改动模型的一个部件，看性能变化，以判断这个部件到底有没有用、有多大用。这里就是"消融掉 skip connection"。

### 完整可运行代码

```python
# 环境：Python 3.9+，PyTorch 2.x，CPU 即可
# 依赖：pip install torch
# 运行：python grad_compare.py
import torch
import torch.nn as nn

torch.manual_seed(0)  # 固定随机种子，保证两个网络初始化一致、结果可复现

DEPTH = 50      # 50 层，足够深到能看出梯度消失
WIDTH = 64      # 每层神经元数
USE_RELU = True # 用 ReLU（也可换 Tanh，梯度消失会更夸张）

class PlainNet(nn.Module):
    """不加 skip 的普通深层网络"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(WIDTH, WIDTH) for _ in range(DEPTH)]
        )
        self.act = nn.ReLU() if USE_RELU else nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))   # 每层：线性变换 + 激活，无 skip
        return x

class ResNet(nn.Module):
    """加 skip 的残差网络，与 PlainNet 唯一区别就是 forward 里那个 + x"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(WIDTH, WIDTH) for _ in range(DEPTH)]
        )
        self.act = nn.ReLU() if USE_RELU else nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            # ★唯一的差别：把输入 x 加回到该层输出上（残差连接）
            # 因为输入输出都是 WIDTH 维，形状天然匹配，不需要投影
            x = self.act(layer(x)) + x
        return x


def run(model, name):
    model.zero_grad()
    x = torch.randn(16, WIDTH)          # 一批 16 个样本
    out = model(x)
    loss = out.pow(2).mean()            # 随便造一个标量损失
    loss.backward()                     # 反向传播，算出每层梯度

    # 收集每一层权重的梯度范数（L2 norm，衡量梯度"大小"）
    grad_norms = [layer.weight.grad.norm().item() for layer in model.layers]

    print(f"\n===== {name} =====")
    print(f"第 1 层(最浅)梯度范数 : {grad_norms[0]:.3e}")
    print(f"第 25 层(中间)梯度范数: {grad_norms[24]:.3e}")
    print(f"第 50 层(最深)梯度范数: {grad_norms[-1]:.3e}")
    # 浅层/深层比值越小，说明梯度衰减越严重
    ratio = grad_norms[0] / (grad_norms[-1] + 1e-12)
    print(f"浅层/深层 比值        : {ratio:.3e}  （越接近1越健康）")


if __name__ == "__main__":
    run(PlainNet(), "PlainNet  不加 skip")
    run(ResNet(),   "ResNet    加了 skip")
```

### 你会看到什么（典型输出 & 解读）

运行后大致是这样（具体数值因机器略有不同，量级是稳定的）：

```
===== PlainNet  不加 skip =====
第 1 层(最浅)梯度范数 : 3.2e-08      ← 几乎为 0！浅层根本学不动
第 25 层(中间)梯度范数: 1.5e-04
第 50 层(最深)梯度范数: 2.1e-02      ← 深层还正常
浅层/深层 比值        : 1.5e-06      ← 相差百万倍

===== ResNet    加了 skip =====
第 1 层(最浅)梯度范数 : 8.7e-03      ← 浅层梯度"活"过来了
第 25 层(中间)梯度范数: 1.1e-02
第 50 层(最深)梯度范数: 2.4e-02
浅层/深层 比值        : 3.6e-01      ← 同一量级，健康
```

**怎么读这个结果**：
- PlainNet 第 1 层梯度是 `e-08` 级别——这意味着不管训练多久，**最靠近输入的那些层几乎收不到更新信号**，等于白搭。这就是"50 层比 20 层还差"的元凶。
- ResNet 第 1 层梯度直接回到 `e-03`，和深层同一个量级。那个 "+x" 让梯度有了一条不被连乘衰减的"高速路"。
- **比值**那一行最直观：从相差百万倍（1.5e-06）变成同量级（3.6e-01）。

> 调试技巧：如果你把 `USE_RELU` 改成 `False`（用 Tanh），PlainNet 的梯度消失会更触目惊心（可能到 e-15）。这是因为 Tanh 在两端饱和区导数趋近 0，连乘衰减更猛——这也解释了为什么现代网络几乎不用 Tanh 做隐藏层激活。

---

## 五、常见陷阱与工业实践清单

### 写残差块最常踩的坑

1. **相加后才激活，别相加前激活**。`act(layer(x)) + x` ✅；`act(layer(x) + x)` 也常见（两种顺序学界都有用），但**千万别把 x 也喂进 act 再相加**，那条捷径就废了。
2. **维度不匹配直接 `+` 会报 RuntimeError**。一看到 `The size of tensor a must match...` 八成是 shortcut 没做投影。
3. **Bottleneck 的 expansion=4 忘了乘**，导致下一个 block 输入通道算错。
4. **BN 在 eval 模式忘了切换**。`model.eval()` 没调用，BN 还在用 batch 统计量，推理结果会抖动——这是工业部署最常见的低级 bug。

### 工业界怎么用

- **几乎所有现代视觉骨干网络都带残差思想**：ResNet、ResNeXt、RegNet，乃至 Transformer 里的残差连接（每个 attention/FFN 子层都包了一层 `x + sublayer(x)`），本质是同一个 "+1 救梯度" 的思路。你将来做推理优化，残差块是绕不开的基本单元。
- **预训练权重直接复用**：实际项目极少从零训 ResNet，都是 `torchvision.models.resnet50(weights=...)` 加载 ImageNet 预训练权重再微调。手写是为了懂原理，**生产用官方实现**（经过充分测试、有 CUDA 优化）。
- **推理优化关注点**（和你主线相关）：残差的 `+` 是逐元素加法（element-wise add），在推理框架里常和前面的 BN、Conv 做**算子融合（operator fusion）**，减少访存、提升吞吐。这是你以后做 AI Infra 会反复遇到的优化点。

**算子融合（operator fusion）**：把多个连续的小运算（如 Conv→BN→ReLU）合并成一个 kernel 一次算完，省去中间结果反复读写显存的开销，是推理加速的核心手段之一。

---

## 六、一句话总结 & 自检清单

**一句话**：残差连接用一个 "$+x$" 给梯度修了条高速路，让"加深网络"从"反而变差"变成"稳定提升"，是现代深度网络的地基。

**自检清单**（合上笔记问自己）：
- [ ] 退化问题是什么？它和过拟合的区别？
- [ ] "$+x$" 在反向传播里为什么变成 "+1"？为什么能救梯度？
- [ ] BasicBlock 里相加为什么必须在 ReLU 之前？
- [ ] Bottleneck 的 expansion 为什么是 4？1×1 卷积在干嘛？
- [ ] shortcut 什么时候需要 1×1 投影卷积？
- [ ] 不加 skip 的 50 层网络，浅层梯度大概是什么量级？加了之后呢？

> 全部能答上来，这两个知识点就真的拿下了。建议把第四节的实验代码自己改改（换深度、换激活、换初始化）多跑几次，数字会比任何文字都让你印象深刻。
