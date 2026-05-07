# Neural Networks Learning

这是用来学习神经网络而建立的一个仓库，只要我还在学习神经网络，就会一直更新。

---

## 关于这个仓库

这里记录我从零开始学习神经网络的全过程——从最基础的数学推导，到亲手用 Numpy 实现一个能跑的网络，再到使用 PyTorch 等框架做更复杂的模型。代码、推导、笔记、实验结果都会放在这里。

不追求"看起来很专业"，只追求**真的把每个细节搞懂了**。所以会有大量推导草稿、踩坑记录、和反复修改的实现——这是学习的真实样子。

## 关于我

- 双非院校大一在读
- 正在备战 ICPC，C++ 算法基础持续打磨中
- 机器学习路线：跟随 Andrew Ng 系列课程学习
- 数学背景：线代和微积分够用，统计在补
- 目标方向：人工智能，关注就业和技术深度

## 仓库结构（持续更新）

```
.
├── notes/                # 学习笔记（Markdown）
│   ├── xavier_init.md           # Xavier 权重初始化
│   ├── forward_components.md    # ReLU / Softmax / CrossEntropy / Forward
│   └── ...
│
├── week2_numpy_nn/       # 用 Numpy 从零实现一个全连接网络
│   ├── src/
│   │   └── nn.py                # 初始化 / 前向 / 反向 / 训练循环
│   ├── experiments/             # 训练实验和图表
│   └── README.md
│
├── derivations/          # 手写推导扫描件 / 拍照存档
│
└── README.md             # 你现在看到的这个文件
```

## 学习路线

按时间推进，已完成的会打勾。

### 第一阶段：基础打底
- [x] 线性回归 / 逻辑回归手写实现（Andrew Ng Course 1 Week 1-2）
- [ ] 反向传播完整推导（6 个核心公式）
- [ ] Numpy 从零实现一个 2 层全连接网络
- [ ] MNIST 分类，准确率 > 95%

### 第二阶段：框架与卷积
- [ ] PyTorch 入门，重写第一阶段的网络
- [ ] CNN 基础（卷积、池化、感受野）
- [ ] 在 CIFAR-10 上训练一个能用的 CNN

### 第三阶段：现代深度学习
- [ ] RNN / LSTM 基础
- [ ] Attention 机制
- [ ] Transformer 从零实现
- [ ] 至少一个端到端的小项目

> 路线会根据学习进度和兴趣调整，不强求完全按这个顺序。

## 当前进度

正在做：**Week 2 - Numpy 从零实现全连接网络**

- 已完成：Xavier 初始化、ReLU / Softmax / Cross Entropy、前向传播
- 进行中：反向传播、训练循环
- 下一步：在 MNIST 上跑通完整流程

## 学习资源

主要参考：

- [Coursera - Deep Learning Specialization (Andrew Ng)](https://www.coursera.org/specializations/deep-learning)
- [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [Dive into Deep Learning](https://d2l.ai/)

辅助：在学习过程中使用 Claude / ChatGPT 帮助理解推导和 Debug，所有理解都会**自己手写一遍**确保真的懂了，不只是复制结论。

## 更新频率

每天有进展就更新。学习日志放在 `notes/` 里，文件名按 `Wn_dayX_log.md` 命名，对应每天的任务清单。

## 写在最后

学习神经网络是个长期工程，做这个仓库的目的不是给别人看，是逼自己**把每一步都写清楚**。如果将来有同样在自学的人路过，看到一两个推导或实现觉得有用，就更好了。
