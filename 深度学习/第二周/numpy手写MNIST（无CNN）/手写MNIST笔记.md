# 纯 NumPy 手写两层神经网络训练 MNIST（第2周实战教程）

> 目标读者：已经手推过反向传播、写完 forward/backward 的初学者（也就是当前的你）。
> 周目标（2026-05-04 ~ 2026-05-10）：**不依赖任何深度学习框架自动求导**，训出 test acc > 95% 的两层 MLP，并把训练 pipeline 打磨到工程可复现。

---

## 0. 学习目标与本周里程碑映射

对照你第 2 周任务清单：

| 第2周任务 | 本文位置 | 预计时长 |
|---|---|---|
| 手推反向传播（已完成） | §2 | 已完成 |
| relu/softmax/CE/forward（已完成） | §3 | 已完成 |
| backward/update_params（已完成） | §3 | 已完成 |
| shape_check（已完成） | §3 | 已完成 |
| 初始 loss ≈ 2.3 验证（已完成） | §3 末 | 已完成 |
| **gradient_check 数值梯度** | §4 | 1–2 h |
| **mini-batch + shuffle 训练循环** | §5 | 2–3 h |
| **训练日志 CSV** | §5 | 0.5 h |
| **调参实验 exp_compare.csv** | §6 | 3–4 h |
| **保存最优模型 best.npz** | §7 | 0.5 h |
| **test acc > 95%** | §6.3 给出配方 | 贯穿 |
| **本周复盘** | §9 | 1 h |

**顺序一定是：§4 梯度检验 → §5 训练 → §6 调参。** 不先通过数值梯度检验就直接训，后面调参出问题你会分不清是 bug 还是超参。

---

## 1. 环境与项目结构

### 1.1 环境
- Python 3.10+
- `numpy`、`matplotlib`、`scikit-learn`（只用它加载数据）

Windows PowerShell：

```powershell
cd C:\Users\donk\Desktop
mkdir mnist_nn
cd mnist_nn

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install numpy matplotlib scikit-learn
```

如果激活脚本被策略禁用：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，然后重开 PowerShell。

### 1.2 目录结构

```
mnist_nn/
├── nn.py                  # 模型：前向/反向/更新/shape_check
├── gradient_check.py      # 数值梯度检验
├── data_loader.py         # MNIST 加载 & one-hot
├── train.py               # mini-batch 训练主脚本
├── run_experiments.py     # 批量调参脚本（生成 exp_compare.csv）
├── eval.py                # 加载 checkpoint 并复现 test_acc
├── logs/                  # 每次训练的日志 CSV
├── checkpoints/           # best.npz
└── experiments/           # exp_compare.csv
```

一键创建：

```powershell
mkdir logs, checkpoints, experiments
ni nn.py, gradient_check.py, data_loader.py, train.py, run_experiments.py, eval.py
```

---

## 2. 数学回顾：为什么 `dZ2 = (A2 - Y) / N`

理解这一步，backward 就能一行写对。

### 2.1 记号
- `z`：输出层 logits，长度 10
- `a = softmax(z)`，`a_k = exp(z_k) / Σⱼ exp(z_j)`
- `y`：one-hot 标签
- `L = −Σ_k y_k log a_k`

### 2.2 一步到位的推导

利用 softmax 的导数 `∂a_j/∂z_k = a_j(δ_{jk} − a_k)`：

```
∂L/∂z_k = −Σⱼ y_j · (1/a_j) · ∂a_j/∂z_k
        = −Σⱼ y_j · (δ_{jk} − a_k)
        = a_k · Σⱼ y_j  − y_k
        = a_k − y_k    （因为 Σⱼ y_j = 1）
```

批量 N 条样本，loss 取平均，于是：

```
dZ2 = (A2 − Y) / N       # 形状 (N, 10)
```

**这就是 backward 第一行的由来。写成 `(A2 − Y)` 后再在别处除 N 也可以，但一定要除。**

### 2.3 剩下的链式（都带形状）

```
dW2 = A1ᵀ @ dZ2           (H, 10)
db2 = Σ dZ2 over axis=0   (10,)
dA1 = dZ2 @ W2ᵀ           (N, H)
dZ1 = dA1 ⊙ 1[Z1 > 0]     (N, H)   # ReLU 导数
dW1 = Xᵀ @ dZ1            (784, H)
db1 = Σ dZ1 over axis=0   (H,)
```

**形状是你最好的武器**：每写一行先脑补 shape，不合就是错的。

---

## 3. `nn.py`：前向 / 反向 / 更新（含 shape_check）

### 3.1 这一节解决什么
把你已经写过的函数组织成一个可调用的单文件模型，顺带加上 **L2 正则**（调参阶段会用到）和一个自检入口。

### 3.2 关键原理
- **Softmax 数值稳定**：`exp(z)` 很大会溢出 → 先减每行最大值。结果不变。
- **交叉熵数值稳定**：`log(a)` 趋近 0 会炸 → 加 `eps=1e-12`。
- **He 初始化**：ReLU 专用，`W ~ N(0, sqrt(2/fan_in))`。否则前几步梯度幅度会失控。
- **L2 正则**：loss 加 `λ/2 · (‖W1‖² + ‖W2‖²)`；backward 里 `dW += λ·W`。

### 3.3 完整代码

```python
# nn.py
import numpy as np


def init_params(input_dim=784, hidden_dim=256, output_dim=10, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))
    b2 = np.zeros(output_dim)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    eZ = np.exp(Z)
    return eZ / np.sum(eZ, axis=1, keepdims=True)


def cross_entropy(A2, Y, eps=1e-12):
    N = Y.shape[0]
    return -np.sum(Y * np.log(A2 + eps)) / N


def forward(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_loss(A2, Y, params, l2=0.0):
    loss = cross_entropy(A2, Y)
    if l2 > 0:
        loss += 0.5 * l2 * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    return loss


def backward(Y, cache, params, l2=0.0):
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]
    W2 = params["W2"]
    N = Y.shape[0]

    dZ2 = (A2 - Y) / N                  # (N, 10)
    dW2 = A1.T @ dZ2                    # (H, 10)
    db2 = np.sum(dZ2, axis=0)           # (10,)
    dA1 = dZ2 @ W2.T                    # (N, H)
    dZ1 = dA1 * (Z1 > 0)                # ReLU 导
    dW1 = X.T @ dZ1                     # (784, H)
    db1 = np.sum(dZ1, axis=0)           # (H,)

    if l2 > 0:
        dW1 += l2 * params["W1"]
        dW2 += l2 * params["W2"]

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def update_params(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]
    return params


def shape_check(params, X, Y):
    A2, cache = forward(X, params)
    assert A2.shape == Y.shape, f"A2 {A2.shape} vs Y {Y.shape}"
    grads = backward(Y, cache, params)
    for k in params:
        assert params[k].shape == grads[k].shape, \
            f"{k} shape mismatch: {params[k].shape} vs {grads[k].shape}"
    print("[shape_check] OK")


if __name__ == "__main__":
    # 冒烟测试：随机输入下初始 loss ≈ log(10) ≈ 2.30
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 784))
    y_idx = rng.integers(0, 10, size=64)
    Y = np.eye(10)[y_idx]
    params = init_params()
    shape_check(params, X, Y)
    A2, _ = forward(X, params)
    print(f"init loss = {cross_entropy(A2, Y):.4f}  (expected ~2.30)")
```

```powershell
python nn.py
```
期望：
```
[shape_check] OK
init loss = 2.30xx  (expected ~2.30)
```

### 3.4 常见错误
- 初始 loss 不是 2.3：init 方差错，或 softmax 忘记减最大值。
- loss NaN：softmax 没稳定化 / log 没加 eps。
- 形状不对：90% 是把 `@`（矩阵乘）写成了 `*`（逐元素乘）。
- update 无效：忘了 `-=`，或者新字典没接住。

### 3.5 自测
- [x] `python nn.py` 输出 shape_check OK 且 loss ≈ 2.30
- [ ] 不看教程也能独立写出 backward 的 6 行
- [x] 能解释 dZ2 为什么要除以 N

---

## 4. `gradient_check.py`：数值梯度检验（**必做**）

### 4.1 为什么非做不可
解析梯度一个符号错、一个 axis 写反，loss 一样会下降，只是方向不对。**唯一能证明 backward 正确的，就是和数值梯度对比。** 这是新手代码和工业代码的分水岭。

### 4.2 原理
中心差分：

```
num_grad_i = ( J(θ + ε·e_i) − J(θ − ε·e_i) ) / (2ε)
```

相对误差：

```
rel_err = |g_num − g_ana| / max(|g_num|, |g_ana|, 1e-12)
```

判定：
- `< 1e-7`：完美
- `< 1e-5`：可接受
- `> 1e-3`：几乎肯定有 bug

### 4.3 完整代码

```python
# gradient_check.py
import numpy as np
from nn import init_params, forward, backward, compute_loss


def numerical_gradient(params, X, Y, param_name, eps=1e-5,
                       n_samples=20, l2=0.0):
    rng = np.random.default_rng(0)
    W = params[param_name]
    idxs = rng.choice(W.size, size=min(n_samples, W.size), replace=False)

    num_grad = np.zeros(len(idxs))
    for i, idx in enumerate(idxs):
        coords = np.unravel_index(idx, W.shape)

        orig = W[coords]
        W[coords] = orig + eps
        A2_p, _ = forward(X, params)
        loss_p = compute_loss(A2_p, Y, params, l2=l2)

        W[coords] = orig - eps
        A2_m, _ = forward(X, params)
        loss_m = compute_loss(A2_m, Y, params, l2=l2)

        W[coords] = orig
        num_grad[i] = (loss_p - loss_m) / (2 * eps)
    return idxs, num_grad


def check(param_name, l2=0.0):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 784))
    y_idx = rng.integers(0, 10, size=8)
    Y = np.eye(10)[y_idx]

    params = init_params(hidden_dim=32)        # 小模型，跑得快
    _, cache = forward(X, params)
    grads = backward(Y, cache, params, l2=l2)

    idxs, num_g = numerical_gradient(params, X, Y, param_name, l2=l2)
    ana_g = grads[param_name].ravel()[idxs]

    rel_err = np.abs(num_g - ana_g) / np.maximum(
        np.maximum(np.abs(num_g), np.abs(ana_g)), 1e-12
    )
    ok = "OK" if rel_err.max() < 1e-5 else "FAIL"
    print(f"[{param_name}] max rel_err = {rel_err.max():.3e}, "
          f"mean = {rel_err.mean():.3e}  {ok}")
    return rel_err.max()


if __name__ == "__main__":
    print("== without L2 ==")
    for name in ["W1", "b1", "W2", "b2"]:
        check(name, l2=0.0)
    print("\n== with L2 = 1e-3 ==")
    for name in ["W1", "b1", "W2", "b2"]:
        check(name, l2=1e-3)
```

```powershell
python gradient_check.py
```

目标：**每一行都 OK（rel_err < 1e-5）**。

### 4.4 常见错误
- **整体 rel_err 量级 1e-3**：backward 某一处漏除 N 或多除 N。
- **只有 `W1` 失败**：ReLU 导数写错（`Z1 > 0` 或 `A1 > 0` 都可，但不能写成 `X > 0`）。
- **只有 `b` 失败**：`sum(axis=0)` 写成 `axis=1` 或漏了。
- **加 L2 才失败**：`dW += l2 * W` 忘写，或 loss 里漏了 `0.5 *`。
- **没有 L2 时 rel_err 就是 0（太完美）**：`eps` 太小被浮点吃掉。取 1e-4 再试一次验证非巧合。

### 4.5 自测
- [ ] 4 参数 × (有/无 L2) 共 8 项全部 < 1e-5
- [ ] 能解释为何用中心差分而不是单侧
- [ ] 能解释分母里的 `max(..., 1e-12)` 起什么作用

---

## 5. `train.py`：mini-batch 训练 + 日志 + 评估

### 5.1 这一节解决什么
从"能前向能反向"到"能训出模型"。关键三件事：**mini-batch + shuffle**、**epoch 循环**、**指标日志**。

### 5.2 关键原理
- **为什么 mini-batch**：全量 GD 一步代价大，纯 SGD 方差大；batch 64~256 在实践上最佳。
- **为什么 shuffle**：MNIST 原始大致按类排序，不打乱会让每个 batch 类别分布失衡，梯度剧烈震荡。**每 epoch 开头打乱一次**。
- **evaluate 不走 backward**：只算 loss/acc，且最好分批过，避免 70000×784 的大矩阵占内存。
- **accuracy**：`argmax(A2, axis=1) == argmax(Y, axis=1)` 的均值。

### 5.3 `data_loader.py`

```python
# data_loader.py
import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(normalize=True):
    print("Loading MNIST (首次运行会联网下载) ...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32)        # (70000, 784)
    y = mnist.target.astype(np.int64)        # (70000,)

    if normalize:
        X = X / 255.0

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    def one_hot(y, C=10):
        Y = np.zeros((len(y), C), dtype=np.float32)
        Y[np.arange(len(y)), y] = 1.0
        return Y

    Y_train, Y_test = one_hot(y_train), one_hot(y_test)
    print(f"train: X {X_train.shape}, Y {Y_train.shape}")
    print(f"test:  X {X_test.shape},  Y {Y_test.shape}")
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    load_mnist()
```

### 5.4 `train.py`

```python
# train.py
import argparse, csv, os, time
import numpy as np

from nn import init_params, forward, backward, update_params, compute_loss
from data_loader import load_mnist


def evaluate(params, X, Y, batch=1024):
    total_loss, total_correct, n = 0.0, 0, 0
    for i in range(0, len(X), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        A2, _ = forward(xb, params)
        total_loss += compute_loss(A2, yb, params, l2=0.0) * len(xb)
        total_correct += (np.argmax(A2, 1) == np.argmax(yb, 1)).sum()
        n += len(xb)
    return total_loss / n, total_correct / n


def train(args):
    X_train, Y_train, X_test, Y_test = load_mnist()
    params = init_params(hidden_dim=args.hidden, seed=args.seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"logs/{args.tag}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc",
             "test_loss", "test_acc", "lr", "time_s"]
        )

    rng = np.random.default_rng(args.seed)
    N = len(X_train)
    best_test_acc, lr = 0.0, args.lr
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        idx = rng.permutation(N)
        Xs, Ys = X_train[idx], Y_train[idx]

        for i in range(0, N, args.batch):
            xb, yb = Xs[i:i + args.batch], Ys[i:i + args.batch]
            _, cache = forward(xb, params)
            grads = backward(yb, cache, params, l2=args.l2)
            update_params(params, grads, lr)

        train_loss, train_acc = evaluate(params, X_train, Y_train)
        test_loss, test_acc = evaluate(params, X_test, Y_test)

        if args.lr_decay > 0 and epoch % args.lr_decay == 0:
            lr *= 0.5

        dt = time.time() - t0
        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_acc={test_acc:.4f} lr={lr:.4f} time={dt:.1f}s")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, test_loss, test_acc, lr, dt]
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            np.savez(f"checkpoints/{args.tag}_best.npz",
                     W1=params["W1"], b1=params["b1"],
                     W2=params["W2"], b2=params["b2"],
                     epoch=epoch, test_acc=test_acc)

    print(f"best test_acc = {best_test_acc:.4f}")
    return best_test_acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--lr_decay", type=int, default=10,
                   help="每 N 个 epoch lr 减半；0 表示不衰减")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
```

试跑：

```powershell
python train.py --tag baseline --epochs 5
```

参考表现（CPU 大约 30s~1min/epoch）：
- epoch 1 结束：test_acc ≈ 0.93
- epoch 5 结束：test_acc ≈ 0.96

### 5.5 常见错误
- **epoch 1 test_acc < 0.5**：lr 太大或 init 错；先试 `--lr 0.01` 确定是哪种。
- **test_acc 纹丝不动**：shuffle 漏了 / batch 索引错 / 评估时用了训练集。
- **train_acc=1.0，test_acc=0.93**：过拟合，加大 L2 或减小 hidden。
- **loss 爆 NaN**：lr 过大导致权重爆炸；lr 除以 10 重跑。
- **每 epoch 末 test_acc 回落**：lr 太大，开 `--lr_decay 5` 或降 lr。

### 5.6 自测
- [ ] `python train.py --tag baseline --epochs 5` 能跑完
- [ ] `logs/baseline.csv` 有 5 行数据
- [ ] `checkpoints/baseline_best.npz` 被生成
- [ ] 能用一句话解释为什么 shuffle 必须每个 epoch 做

---

## 6. 调参实验：系统化做 `lr × hidden × batch × L2`

### 6.1 原则（比数字重要）
- **一次只动一个超参**（控制变量法）。
- **固定 seed**，保证可复现。
- **所有组跑相同 epochs**，公平比较。
- **看 best test_acc**，不看最后一个 epoch。
- **调参顺序：`lr → hidden → batch → L2`**。理由：lr 不对连训都训不起来；hidden 决定容量；batch 影响稳定性和速度；L2 是确定过拟合后的最后一招。

### 6.2 批量脚本 `run_experiments.py`

```python
# run_experiments.py
import csv, os
from argparse import Namespace
from train import train

BASELINE = {"lr": 0.1, "hidden": 256, "batch": 128, "l2": 1e-4,
            "epochs": 15, "lr_decay": 8, "seed": 42}

GRID = {
    "lr":     [0.3, 0.1, 0.05, 0.01],
    "hidden": [128, 256, 512],
    "batch":  [64, 128, 256],
    "l2":     [0.0, 1e-4, 1e-3],
}


def main():
    os.makedirs("experiments", exist_ok=True)
    out = "experiments/exp_compare.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "lr", "hidden", "batch", "l2", "best_test_acc"])

        for axis, values in GRID.items():
            for v in values:
                cfg = dict(BASELINE)
                cfg[axis] = v
                tag = f"{axis}_{v}"
                print(f"\n===== {tag} =====")
                best = train(Namespace(**cfg, tag=tag))
                w.writerow([axis, cfg["lr"], cfg["hidden"],
                            cfg["batch"], cfg["l2"], f"{best:.4f}"])


if __name__ == "__main__":
    main()
```

```powershell
python run_experiments.py
```

采用"绕着 baseline 扫一维"的方式，避免组合爆炸（13 组 vs 完整笛卡尔积的 108 组），但每一维都能看清边际影响。

### 6.3 冲 >95% 的推荐配方

| 超参 | 建议值 | 说明 |
|---|---|---|
| hidden | 256 | 够大但不过拟合 |
| lr | 0.1 | 纯 SGD + 小 batch 的经典起点 |
| batch | 128 | 速度与稳定性平衡 |
| l2 | 1e-4 | 轻微正则 |
| epochs | 25 | 收敛充分 |
| lr_decay | 每 10 epoch 减半 | 尾段精调 |
| init | He | 能过 95% 的关键之一 |

一条命令：

```powershell
python train.py --hidden 256 --lr 0.1 --batch 128 --l2 1e-4 --epochs 25 --lr_decay 10 --tag final
```

最后几个 epoch 的 test_acc 通常在 **97.5%~98%**。

### 6.4 常见误判
- **同一配置跑两次差 ±0.5%**：正常的随机性，看**趋势**别盯单次。
- **batch 开到 1024 反而差**：batch 越大单 epoch 更新次数越少，需要同步提高 epochs 或 lr。
- **L2 加到 1e-2 反而差**：L2 压过信号；先确认是否真过拟合（train_acc 远高于 test_acc）再加。

### 6.5 自测
- [ ] `experiments/exp_compare.csv` 有 13 行（4+3+3+3）
- [ ] 能指出哪组最差并解释原因
- [ ] 能用一句话总结 lr 对 test_acc 的影响曲线

---

## 7. 保存最优模型 & 复现

### 7.1 保存（train.py 已做）
每当 test_acc 刷新，写 `checkpoints/{tag}_best.npz`。

### 7.2 加载并复现

```python
# eval.py
import sys
import numpy as np
from nn import forward
from data_loader import load_mnist


def load_checkpoint(path):
    d = np.load(path)
    return {"W1": d["W1"], "b1": d["b1"], "W2": d["W2"], "b2": d["b2"]}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final_best.npz"
    params = load_checkpoint(path)
    _, _, X_test, Y_test = load_mnist()
    A2, _ = forward(X_test, params)
    acc = (np.argmax(A2, 1) == np.argmax(Y_test, 1)).mean()
    print(f"{path}  test_acc = {acc:.4f}")
```

```powershell
python eval.py checkpoints/final_best.npz
```

**复现的 acc 必须和训练时打印的 `best test_acc` 一致**（差 ≤ 1e-4 是浮点正常）。不一致说明保存/加载漏了 key 或 shape 错。

---

## 8. 常见 bug 速查表（按现象定位）

| 现象 | 最可能原因 | 第一步验证 |
|---|---|---|
| 初始 loss ≠ 2.30 | 初始化方差错 / softmax 没减最大值 | 打印 `W1.std()` 应 ≈ 0.05 |
| 全部 rel_err > 1e-3 | backward 某行公式错 | 单独 check 4 参数，看哪个 FAIL |
| 只有 W1 FAIL | ReLU 导写错 | 打印 `(Z1>0).mean()` 应在 0.3~0.7 |
| 加 L2 才 FAIL | `dW += l2*W` 漏；loss 里 `0.5*` 漏 | 设 l2=0 先确认原版对 |
| 第一个 batch loss=NaN | lr 过大 / softmax 未稳定化 | lr ÷ 10 再跑 |
| train_loss ↓ 但 test_acc 不动 | shuffle 漏 / 评估数据用错 | 打印 X_test[0] 和 X_train[0] 对比 |
| train_acc=1.00 test_acc=0.93 | 过拟合 | 加 L2 / 减 hidden |
| 每 epoch 后 test_acc 反复跳 | lr 过大 | 开 lr_decay 或降 lr |
| checkpoint 复现 acc 对不上 | savez key 漏 / shape 错 | `print(np.load(path).files)` 核对 |

---

## 9. 本周复盘模板（直接填）

```markdown
# Week 2 复盘（2026-05-04 ~ 2026-05-10）

## 做完了什么
- [x] 手推 BP
- [x] nn.py 全部函数
- [x] gradient_check 8 项全 < 1e-5
- [x] train.py 跑通，baseline test_acc = ___
- [x] 调参实验 13 组（见 experiments/exp_compare.csv）
- [x] 最优 test_acc = ___（是否超过 95% 目标：是 / 否）
- [x] checkpoint 复现 acc 与训练时一致

## 最难的点（具体写，不要空话）
1. ……（例：dZ2 = A2−Y 的推导一开始没理解为什么能消项）
2. ……
3. ……

## 学到但没完全吃透的
- ……（例：He 初始化 `sqrt(2/fan_in)` 的方差守恒是怎么推的）

## 踩过的坑（留给未来的自己）
1. bug：__ | 症状：__ | 根因：__ | 怎么发现：__
2. ……

## 下周要继续
- [ ] 读 PyTorch 版两层 MLP，对照 NumPy 实现
- [ ] 加 momentum / Adam，看同配置上限
- [ ] 写 2 分钟口述版 BP 推导（能背的程度）

## 一句话总结
本周我最大的收获是：____。
```

---

## 10. 学习自测题（含答案要点）

### Q1. Softmax 为什么要先减掉每行最大值？不减会怎样？
**答案要点**：避免 `exp(z)` 溢出。分子分母同乘常数不改变 softmax 输出。不减时若 z 中有大值（比如 1000），`exp(1000)=inf`，整行变 NaN。

### Q2. 推导 `∂L/∂z_k = a_k − y_k`（softmax+交叉熵）。
**答案要点**：用 `∂a_j/∂z_k = a_j(δ_{jk} − a_k)`，代入 `∂L/∂z_k = −Σⱼ (y_j/a_j)·∂a_j/∂z_k`，化为 `a_k·Σⱼ y_j − y_k = a_k − y_k`（因为 `Σⱼ y_j = 1`）。

### Q3. 数值梯度为何用中心差分而不是单侧？`ε=1e-5` 而不是 1e-10 的原因？
**答案要点**：中心差分截断误差 O(ε²)，单侧 O(ε)。ε 太小会放大浮点减法的消位误差，反而更差；太大又偏离真导数。1e-5 是权衡后的经验值。

### Q4. 为什么每个 epoch 都要 shuffle？不 shuffle 会怎样？
**答案要点**：每 batch 组成固定会让梯度路径固定、loss 出现周期性震荡；MNIST 原始大致按类排列，不 shuffle 会让每 batch 只含少数几类，梯度偏差极大。

### Q5. 怎么区分"过拟合"与"lr 太大导致震荡"？
**答案要点**：过拟合 = train_acc 非常高（接近 1），test_acc 明显低且趋稳；lr 过大 = train_loss 本身不下降、来回跳甚至 NaN。看 train 集表现就能分开。

### Q6. He 初始化的方差为什么是 `2/fan_in` 而不是 `1/fan_in`？
**答案要点**：ReLU 会把约一半输入置零，输出方差砍半，要补 2 倍才能让前向信号在层间方差守恒。tanh/sigmoid 用的是 Xavier（系数 1）。

### Q7. 解析梯度与数值梯度的 rel_err 分别是 1e-8 和 1e-3，哪个可能实现正确？为什么？
**答案要点**：1e-8 表明几乎完全吻合，几乎肯定正确；1e-3 通常说明解析梯度有 bug——因为数值梯度在 ε=1e-5 时本身精度在 1e-6~1e-8，能差到 1e-3 只能是公式错。

### Q8. test_acc 卡在 94% 怎么办？按什么顺序尝试？
**答案要点**：先排查是否 bug（gradient_check 是否仍过、初始 loss 是否 2.30）→ 增 hidden（容量）→ 调 lr 与 lr_decay → 增 epochs 观察 train/test gap → 加轻微 L2 → 最后才考虑 momentum 等优化器技巧。

---

祝本周通关。跑完这套 pipeline 以后，你就真的"从 0 手搓过一个神经网络"了，面试时讲 BP 不会再心虚。

