# 神经网络学习 · 从 NumPy 手写到 Triton 手写 Kernel

这是我从零自学深度学习与 AI infra 的完整记录仓库。

今天2026年8月18日正式结束。

接下来开启暑假篇章（推理引擎项目）。

> 时间跨度：2026-05 至2026-08 · 已推进 8 周 · 从"手推公式 + NumPy 造轮子"一路走到"手写 Triton kernel 并对标 torch.compile"。

---

## 关于这个仓库

这里记录我学习深度学习的全过程——从最基础的数学推导，到用 NumPy 从零实现一个能跑的网络，再到用 PyTorch 复现经典模型，最后转向 **AI Infra 主线**：性能画像、算子融合、量化、手写 GPU kernel。

不追求"看起来很专业"，只追求**真的把每个细节搞懂了**。所以仓库里有大量手写推导扫描件、踩坑记录、反复修改的实现，以及每天的学习笔记——这是学习真实的样子。

每一周都有一份配套的 `学习笔记/`，把当天学的东西讲清楚、串成线；不是复制结论，是**自己重讲一遍确认真的懂了**。

## 关于我

- 双非院校大一在读，正在备战 ICPC，C++ 算法基础持续打磨中
- 学习重心已从"补课式刷课"切换到 **AI Infra 主线**：Roofline → Profiler → 算子融合 → 手写 kernel
- 目标方向：大模型推理优化 / AI Infra

## 仓库结构

按学习周组织，每周一个目录，内含代码、实验、图表和 `学习笔记/`。

```
.
├── README.md                        # 你现在看到的这个文件
├── 概率论系统入门.md                 # 数学补充：概率论系统梳理
│
└── 深度学习/
    ├── 神经网络学习笔记.md            # 主学习笔记（贯穿全程，最长）
    ├── 工业级神经网络学习.md          # 工业/工程视角的补充
    ├── 深度学习核心公式参考手册.md      # 核心公式速查
    │
    ├── 第一周/   PyTorch 上手 + MNIST 分类基线
    ├── 第二周/   NumPy 从零手写全连接网络 + 反向传播推导
    ├── 第三周/   自动微分引擎（micrograd 风格）+ bigram 语言模型
    ├── 第四周/   makemore MLP：Embedding / BatchNorm / 优化器 / 调参
    ├── 第五周/   CNN：im2col / LeNet / ResNet / CIFAR-10 + profiler
    ├── 第六周/   Transformer：WaveNet → nanoGPT + KV Cache
    ├── 第七周/   AI Infra 转向：Roofline / Profiler / torch.compile / 量化
    └── 第八周/   Triton 手写 Kernel + FlashAttention + 端到端对标
```

## 逐周进度

| 周 | 主题 | 核心内容 | 代表产出 |
|----|------|---------|---------|
| **第一周** | PyTorch 上手 | MNIST 分类基线、数据增强 + Otsu、BatchNorm/Dropout 对比 | `MNIST学习/`、`torch学习.ipynb` |
| **第二周** | NumPy 造轮子 | Xavier 初始化推导、反向传播 6 公式手推、梯度检查、完整训练循环 | `numpy手写MNIST（无CNN）/`、`手写图片（公式推导）/` |
| **第三周** | 自动微分 | micrograd 风格 `Value` 类（前向/反向/可视化）、bigram 语言模型（计数法 vs 神经网络法） | `自动微分/`、`bigram语言模型/`、`手写神经网络（完整）/` |
| **第四周** | makemore MLP | 复现 Bengio 神经概率语言模型、Embedding、BatchNorm 训练/推理双行为、优化器从零实现、lr range test / 调度、首次 profiler | `makemore_MLP/`（含参考论文） |
| **第五周** | 卷积网络 | 朴素卷积 → im2col、LeNet-5 复现、ResNet 残差块（BasicBlock/Bottleneck + 消融）、CIFAR-10 训 ResNet18、显存四份账 + 梯度检查点、Roofline、chrome trace | `吴恩达Course 4 (CNN)/` |
| **第六周** | Transformer | WaveNet（Module 容器抽象）、nanoGPT：单头/多头 self-attention、fused QKV、残差流 + LayerNorm、attention O(N²) 复杂度、KV Cache 两阶段与显存代价 | `EP5/`、`EP6：nanoGPT/` |
| **第七周** | AI Infra 转向 | Roofline / 算术强度 / H100 脊点、三级 profiler 工具链（torch profiler / nsys / ncu）、kernel launch 开销与算子融合、torch.compile 图优化、FP32/FP16/BF16/INT8 量化地基 · **里程碑：Andrew Ng 专项 5 门全完成** | `profiler/`、`compile/`、`量化/`、`小模型实测/` |
| **第八周** | Triton 手写 Kernel | vector-add → fused pointwise → RMSNorm(+autotune) → fused attention（online softmax / FlashAttention 思想）→ 手写 kernel 集成进 nanoGPT，全程以 torch.compile 为分母对标 + 三尺子验误差；多 GPU（RTX3080 / A100 / H100）benchmark | `Triton/`、`FlashAttention(纯python底层实现)/`、`端到端集成/` |

## 学习路线

按时间推进，已完成的打勾。

### 第一阶段：基础打底 
- [x] 线性回归 / 逻辑回归手写实现（Andrew Ng Course 1）
- [x] 反向传播完整推导（6 个核心公式，手写扫描件在 `第二周/手写图片（公式推导）/`）
- [x] NumPy 从零实现全连接网络
- [x] MNIST 分类（含数据增强、BatchNorm/Dropout）

### 第二阶段：框架与卷积 
- [x] PyTorch 入门，重写第一阶段的网络
- [x] CNN 基础（朴素卷积 → im2col、感受野、FLOPs）
- [x] LeNet / ResNet 复现，在 CIFAR-10 上训 ResNet18

### 第三阶段：现代深度学习 
- [x] RNN / LSTM / GRU（Andrew Ng Course 5）
- [x] Attention 机制、多头注意力、fused QKV
- [x] Transformer 从零实现（nanoGPT）+ KV Cache
- [x] 端到端小项目：手写字符级 GPT

### 第四阶段：AI Infra 主线
- [x] Roofline 模型、三级 profiler 工具链、算子融合动机
- [x] torch.compile 图优化、数值精度与量化地基
- [x] Triton 手写融合 kernel + FlashAttention 思想（online softmax）
- [x] 手写 kernel 集成进真实模型并对标 torch.compile
- [ ] 下沉 CUDA（tiled matmul / shared memory / bank conflict）
- [ ] 精读 FlashAttention 源码
- [ ] 精读 vLLM 源码（PagedAttention 入口）

> 路线会根据进度和兴趣调整，不强求完全按顺序。

## 数据与复现

为了保持仓库轻量，**数据集、模型权重、profiler 追踪、训练日志都不入库**（已在 `.gitignore` 排除）：

- **数据集**（MNIST / CIFAR-10 / 手势数字 h5）——首次运行脚本时由 `torchvision` 等自动下载到本地 `data/` 目录
- **模型权重**（`*.pth` / `*.pt` / `*.npz`）、**日志**（`logs/`、`*.csv`）、**profiler 追踪**（`*trace.json`、`gpu_trace.txt`）——运行对应脚本即可重新生成

依赖以每周实际用到为准，核心为 `torch`、`triton`、`numpy`；第八周的 Triton kernel 需要 NVIDIA GPU + CUDA。

## 学习资源

主要参考：

- [Coursera — Deep Learning Specialization (Andrew Ng)](https://www.coursera.org/specializations/deep-learning)
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [Dive into Deep Learning](https://d2l.ai/)
- Andrej Karpathy 的 micrograd / makemore / nanoGPT 系列
- [OpenAI Triton 官方文档与教程](https://triton-lang.org/)

辅助：学习过程中用 Claude / Deepseek 等帮助理解推导和 Debug，但所有理解都会**自己手写一遍**确保真的懂了，不只是复制结论。

## 写在最后

学习深度学习是个长期工程。从第一周手推反向传播，到第八周手写 kernel 对标 torch.compile，最珍贵的不是那些代码，而是清楚知道自己会什么、边界在哪、下一步该攻什么。

我的一位导师说“be prepared, be open。不要一味的追求一个设定的任务和目标，过程有无限可能，结果超乎想象。如果真的喜欢并心怀热爱，论文与升学只是水到渠成。” 而我们需要做的就是做自己想做的事并享受其间过程，仅此而已。至于结果如何我想是不会差。

