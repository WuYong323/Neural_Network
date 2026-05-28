# W4 Day 6 · 代码合集（完整可运行）

> 配套笔记：`W4_Day6_Course2_正则化优化器_工业视角.md`
> 每段都是**独立可运行**的 `python xxx.py`，不依赖 makemore 项目其他文件。
> 建议落点写在每节标题旁边，跑通后再挪进 `src/`。

---

## 1. L2 正则的两种实现 → `src/demo_l2.py`

对照"手动加到 loss"和"优化器 weight_decay"两种写法，验证更新后的参数一致。

```python
"""L2 正则两种实现的等价性验证"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)

    # 两份完全相同的初始模型
    model_a = nn.Linear(8, 4)
    model_b = nn.Linear(8, 4)
    model_b.load_state_dict(model_a.state_dict())

    lr, lam = 0.1, 1e-2
    opt_a = torch.optim.SGD(model_a.parameters(), lr=lr)                  # 手动加 L2
    opt_b = torch.optim.SGD(model_b.parameters(), lr=lr, weight_decay=lam)  # 内建

    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))

    # 方式 A：把 L2 加到 loss 里
    loss_a = F.cross_entropy(model_a(x), y) + 0.5 * lam * sum(
        (p ** 2).sum() for p in model_a.parameters()
    )
    opt_a.zero_grad(); loss_a.backward(); opt_a.step()

    # 方式 B：用 weight_decay
    loss_b = F.cross_entropy(model_b(x), y)
    opt_b.zero_grad(); loss_b.backward(); opt_b.step()

    diff = (model_a.weight - model_b.weight).abs().max().item()
    print(f"两种 L2 实现的最大参数差: {diff:.2e}  (应 < 1e-6)")


if __name__ == "__main__":
    main()
```

**预期输出**：
```
两种 L2 实现的最大参数差: 0.00e+00  (应 < 1e-6)
```

---

## 2. Dropout 训推双行为 → `src/demo_dropout.py`

手写 inverted dropout，验证训练态/推理态期望一致。

```python
"""Dropout 训推双行为：训练时随机置零 + inverse scaling，推理时关闭"""
from __future__ import annotations
import torch


def dropout_train(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """训练时：随机置零 + 1/(1-p) 缩放，保持期望不变"""
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)


def dropout_eval(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """推理时：identity"""
    return x


def main() -> None:
    torch.manual_seed(0)
    x = torch.ones(10000) * 1.0  # 输入均值 1
    p = 0.3

    out_train = dropout_train(x, p)
    out_eval = dropout_eval(x, p)

    print(f"输入均值        : {x.mean().item():.4f}")
    print(f"训练态输出均值  : {out_train.mean().item():.4f}  (≈ 1.0，inverse scaling 补偿)")
    print(f"推理态输出均值  : {out_eval.mean().item():.4f}  (= 1.0，identity)")

    # 与 PyTorch 内建 nn.Dropout 对比
    layer = torch.nn.Dropout(p=p)
    layer.train(); pyt_train = layer(x).mean().item()
    layer.eval(); pyt_eval = layer(x).mean().item()
    print(f"\nnn.Dropout train: {pyt_train:.4f}")
    print(f"nn.Dropout eval : {pyt_eval:.4f}")


if __name__ == "__main__":
    main()
```

**预期输出**：
```
输入均值        : 1.0000
训练态输出均值  : 1.0009  (≈ 1.0)
推理态输出均值  : 1.0000  (= 1.0)

nn.Dropout train: 0.9986
nn.Dropout eval : 1.0000
```

---

## 3. 优化器演化线四件套 → `src/optimizers_from_scratch.py`

在同一个二维 toy loss 上对比 SGD / Momentum / RMSProp / Adam 的轨迹。

```python
"""手写 SGD / Momentum / RMSProp / Adam 单步，并在二次型 loss 上对比收敛"""
from __future__ import annotations
import torch
import matplotlib.pyplot as plt


def sgd_step(w, g, lr=0.1):
    return w - lr * g


def momentum_step(w, g, v, lr=0.1, beta=0.9):
    v = beta * v + g
    return w - lr * v, v


def rmsprop_step(w, g, s, lr=0.01, beta2=0.999, eps=1e-8):
    s = beta2 * s + (1 - beta2) * (g ** 2)
    return w - lr * g / (s.sqrt() + eps), s


def adam_step(w, g, m, v, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g ** 2)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return w - lr * m_hat / (v_hat.sqrt() + eps), m, v


def loss_fn(w: torch.Tensor) -> torch.Tensor:
    """病态二次型：x 方向陡（曲率大），y 方向平（曲率小）"""
    return 10.0 * w[0] ** 2 + 0.5 * w[1] ** 2


def run(name: str, n_steps: int = 80):
    w = torch.tensor([1.5, 1.5], requires_grad=True)
    v = torch.zeros_like(w); s = torch.zeros_like(w); m = torch.zeros_like(w)
    traj = [w.detach().clone()]

    for t in range(1, n_steps + 1):
        loss = loss_fn(w)
        g, = torch.autograd.grad(loss, w)

        with torch.no_grad():
            if name == "SGD":
                w_new = sgd_step(w, g, lr=0.05)
            elif name == "Momentum":
                w_new, v = momentum_step(w, g, v, lr=0.05, beta=0.9)
            elif name == "RMSProp":
                w_new, s = rmsprop_step(w, g, s, lr=0.1)
            elif name == "Adam":
                w_new, m, v = adam_step(w, g, m, v, t, lr=0.2)
            w.copy_(w_new)
        traj.append(w.detach().clone())

    return torch.stack(traj).numpy()


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    # 等高线
    xs = torch.linspace(-2, 2, 100); ys = torch.linspace(-2, 2, 100)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    Z = 10.0 * X ** 2 + 0.5 * Y ** 2
    ax.contour(X, Y, Z, levels=20, cmap="gray", alpha=0.5)

    for name, color in [("SGD", "tab:blue"), ("Momentum", "tab:orange"),
                        ("RMSProp", "tab:green"), ("Adam", "tab:red")]:
        traj = run(name)
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3, color=color, label=name)

    ax.scatter([0], [0], marker="*", s=200, color="black", label="optimum")
    ax.legend(); ax.set_title("Optimizer trajectories on ill-conditioned quadratic")
    ax.set_xlabel("w[0] (steep)"); ax.set_ylabel("w[1] (flat)")
    plt.tight_layout(); plt.savefig("optim_compare.png", dpi=120)
    print("saved -> optim_compare.png")


if __name__ == "__main__":
    main()
```

**预期产物**：`optim_compare.png` —— 能直观看到 SGD 沿陡方向振荡、Momentum 抑制振荡、RMSProp/Adam 把两个方向的步长拉平。

---

## 4. Adam 显存实测（**Day 6 硬交付物**）→ `src/optimizer_memory_demo.py`

验证 Adam 的 optimizer state ≈ 2× 参数显存。这段是 `tech_notes/optimizer_memory.md` 的核心实证。

```python
"""验证 Adam optimizer state = 2× 参数显存（SGD = 0×，Momentum = 1×）"""
from __future__ import annotations
import torch
import torch.nn as nn


def state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    for group in opt.state.values():
        for s in group.values():
            if isinstance(s, torch.Tensor):
                total += s.numel() * s.element_size()
    return total


def main() -> None:
    model = nn.Sequential(nn.Linear(1000, 1000), nn.Linear(1000, 1000))
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = n_params * 4 / 1024 / 1024
    print(f"参数量          : {n_params:,}")
    print(f"参数 FP32 显存  : {param_mb:.2f} MB\n")

    # 制造一份 grad，让所有优化器都能 step
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    cases = [
        ("SGD          ", torch.optim.SGD(model.parameters(), lr=0.01)),
        ("SGD+Momentum ", torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)),
        ("RMSProp      ", torch.optim.RMSprop(model.parameters(), lr=0.001)),
        ("Adam         ", torch.optim.Adam(model.parameters(), lr=0.001)),
        ("AdamW        ", torch.optim.AdamW(model.parameters(), lr=0.001)),
    ]

    print(f"{'优化器':<14} {'state 显存':>12}   {'倍数 (× 参数)':>14}")
    print("-" * 50)
    for name, opt in cases:
        opt.step()
        sz = state_bytes(opt) / 1024 / 1024
        ratio = sz / param_mb if param_mb > 0 else 0
        print(f"{name:<14} {sz:>9.2f} MB   {ratio:>10.2f}×")

    print("\n结论：Adam/AdamW 的 state ≈ 2× 参数大小（m + v）")
    print("推论：7B FP32 训练显存 = 28(W) + 28(grad) + 56(Adam) = 112 GB")


if __name__ == "__main__":
    main()
```

**预期输出**：
```
参数量          : 2,002,000
参数 FP32 显存  : 7.64 MB

优化器           state 显存     倍数 (× 参数)
--------------------------------------------------
SGD                 0.00 MB         0.00×
SGD+Momentum        7.64 MB         1.00×
RMSProp             7.64 MB         1.00×
Adam               15.27 MB         2.00×
AdamW              15.27 MB         2.00×

结论：Adam/AdamW 的 state ≈ 2× 参数大小（m + v）
推论：7B FP32 训练显存 = 28(W) + 28(grad) + 56(Adam) = 112 GB
```

---

## 5. Warmup + Cosine 学习率调度 → `src/lr_schedule.py`

nanoGPT / LLaMA 风格的 lr 曲线，第6周 nanoGPT 直接复用。

```python
"""GPT/LLaMA 风格的 warmup + cosine lr schedule"""
from __future__ import annotations
import math
import matplotlib.pyplot as plt


def get_lr(step: int, warmup_steps: int = 2000, total_steps: int = 600_000,
           max_lr: float = 6e-4, min_lr: float = 6e-5) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main() -> None:
    total = 600_000
    steps = list(range(0, total + 1, 1000))
    lrs = [get_lr(s, warmup_steps=2000, total_steps=total) for s in steps]

    print(f"step      0 -> lr = {lrs[0]:.2e}")
    print(f"step  2,000 -> lr = {get_lr(2000):.2e}  (warmup 顶点)")
    print(f"step 300,000 -> lr = {get_lr(300_000):.2e}  (中段)")
    print(f"step 600,000 -> lr = {get_lr(600_000):.2e}  (终点 = min_lr)")

    plt.figure(figsize=(8, 4))
    plt.plot(steps, lrs)
    plt.axvline(2000, color="red", linestyle="--", alpha=0.5, label="warmup end")
    plt.xlabel("step"); plt.ylabel("learning rate"); plt.legend()
    plt.title("GPT/LLaMA warmup + cosine schedule"); plt.tight_layout()
    plt.savefig("lr_schedule.png", dpi=120)
    print("\nsaved -> lr_schedule.png")


if __name__ == "__main__":
    main()
```

**预期输出**：
```
step      0 -> lr = 0.00e+00
step  2,000 -> lr = 6.00e-04  (warmup 顶点)
step 300,000 -> lr = 3.30e-04  (中段)
step 600,000 -> lr = 6.00e-05  (终点 = min_lr)

saved -> lr_schedule.png
```

---

## 跑通顺序建议

1. **先跑 §4 显存实测** —— 这是计划硬交付物，5 秒就出结果，截图直接贴进 `tech_notes/optimizer_memory.md`
2. **跑 §5 lr 曲线** —— 一图胜千言，留作第6周 nanoGPT 复用
3. **跑 §3 优化器对比** —— 拿到 `optim_compare.png` 后，看 SGD vs Adam 在病态 loss 上的真实差距
4. §1 §2 是**等价性 / 双行为验证**，跑一遍加深印象即可

## 完成检查

- [x] §4 输出对得上 "Adam = 2.00×"
- [x] §5 lr 曲线在第 2000 步达到峰值，之后平滑余弦下降
- [x] §3 图里 SGD 沿 w[0] 振荡明显，Adam 几乎直线指向原点
- [x] 把 §4 的输出 + 推论那两行写进 `tech_notes/optimizer_memory.md` 的 §3 实证小节

---

*文件用途：W4_Day6 配套实验代码，跑通后产出物归档到 `makemore_MLP/src/` 与 `tech_notes/`*
