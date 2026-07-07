# 🎯 里程碑 · Andrew Ng 深度学习专项 5 门课全部完成

> **达成日期**:2026-06-27(W7)
> **状态**:✅ Deep Learning Specialization (deeplearning.ai) — 全部完成

---

## 完成清单

| # | 课程 | 对应学习周 | 状态 |
|---|---|---|---|
| 1 | Neural Networks and Deep Learning | W1–W3 地基 | ✅ |
| 2 | Improving Deep Neural Networks(调参/正则化/优化) | W4(含 BatchNorm、optimizer) | ✅ |
| 3 | Structuring Machine Learning Projects | 穿插 | ✅ |
| 4 | Convolutional Neural Networks | W5(CNN / ResNet / profiler) | ✅ |
| 5 | **Sequence Models**(RNN → LSTM/GRU → Attention → Transformer) | **W7 收尾** | ✅ |

---

## 收尾产出

- **演进总结笔记**:[`tech_notes/rnn_to_transformer_evolution.md`](./tech_notes/rnn_to_transformer_evolution.md)
  - 1 页演进总表(AI Infra 视角)+ 深度解析 + 可运行代码(RNN / LSTM / Attention / KV Cache)
  - 核心因果链:**hidden state(有损压缩历史)→ KV Cache(无损保留历史)→ 推理优化主战场在 attention/KV Cache**
  - 底层深挖:`torch.cat` 拼 KV Cache 为何浪费显存(缓存分配器尺寸取整 / 碎片化权衡 / PagedAttention)

## 这块里程碑的意义

DL 通识地基补齐。从此学习重心正式从"补课式刷课"切换到 **AI Infra 主线**:
Roofline(W7 Day1)→ Profiler 工具链(W7 Day2)→ KV Cache 优化 → CUDA / vLLM 源码(暑假)。

Course 5 的 Sequence Models 不是终点,而是**分水岭**——它把"模型怎么记住历史"这个问题,
直接交到了"怎么把历史存得又快又省"的 AI Infra 手上。下一阶段开始**做系统**,不再只是学知识。

---

## 下一步(承接)

- [ ] 对照演进笔记 §6,把 W6 的朴素 `cat` 版 KV Cache 改成**预分配版**,用 W7 Day2 profiler 量出差距
- [ ] 衔接 W7 AI Infra 主线 / 小米课题(AutoMegaKernel、巨核大模型推理优化)
- [ ] 暑假精读 vLLM 源码(PagedAttention 为入口)

> 相关记忆:学习计划见 `大一下每周学习计划.md`;方向锚点见 user-profile / project-study-plan。
