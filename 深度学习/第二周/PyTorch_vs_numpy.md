# PyTorch vs 手写 NumPy —— 深度学习实现对比总结

> 目标：理解两种方式的本质差异，不是"哪个更好"，而是"为什么 PyTorch 能做到 NumPy 做不到的事"。

---

## 一、核心问题：为什么需要对比？

手写 NumPy 实现神经网络，是理解反向传播原理的最佳路径。
PyTorch 则是工业/研究级框架，解决了手写实现的三大痛点：

| 痛点 | NumPy 手写 | PyTorch 解决方案 |
|------|-----------|----------------|
| 反向传播 | 每层手动推导梯度公式 | 自动微分（Autograd） |
| GPU 加速 | 不支持 | `.cuda()` 一行迁移 |
| 可扩展性 | 增加层结构需重写大量代码 | 模块化 `nn.Module` |

---

## 二、最核心差异：自动微分 vs 手动反向传播

### 2.1 手写 NumPy：每层都要推导梯度

以单层全连接 + Sigmoid + MSE Loss 为例：

```python
import numpy as np

# 前向传播
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Z = X @ W + b          # 线性变换
A = sigmoid(Z)         # 激活
loss = np.mean((A - Y) ** 2)  # MSE Loss

# 反向传播 —— 必须手动推导每一步链式法则
dL_dA = 2 * (A - Y) / Y.size                    # ∂L/∂A
dA_dZ = A * (1 - A)                              # sigmoid 导数
dL_dZ = dL_dA * dA_dZ                           # 链式法则
dL_dW = X.T @ dL_dZ                             # ∂L/∂W
dL_db = np.sum(dL_dZ, axis=0, keepdims=True)    # ∂L/∂b

# 更新参数
W -= lr * dL_dW
b -= lr * dL_db
```

**关键点**：每换一个激活函数、损失函数，就要重新推导所有梯度公式。

---

### 2.2 PyTorch：自动微分，只需写前向

```python
import torch
import torch.nn as nn

# 定义模型（只写前向）
model = nn.Linear(in_features, out_features)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
optimizer.zero_grad()       # 清空上一轮梯度
output = torch.sigmoid(model(X))  # 前向传播
loss = criterion(output, Y)
loss.backward()             # 自动计算所有梯度（计算图反向传播）
optimizer.step()            # 更新参数
```

**PyTorch 如何做到自动微分？**

PyTorch 在前向传播时，会构建一张**计算图（Computational Graph）**，记录每个操作及其输入输出关系。`loss.backward()` 沿此图反向，利用链式法则自动计算每个叶节点（参数）的梯度。

```
X → [Linear] → Z → [sigmoid] → A → [MSELoss] → loss
                                                   ↓
                                             .backward()
                                      自动沿图反向传播梯度
```

---

## 三、数据结构对比：ndarray vs Tensor

| 特性 | NumPy `ndarray` | PyTorch `Tensor` |
|------|----------------|-----------------|
| 设备 | 仅 CPU | CPU / GPU（`.cuda()`） |
| 梯度追踪 | 无 | `requires_grad=True` 时追踪 |
| 广播规则 | 支持 | 支持（规则相同） |
| 互转 | — | `.numpy()` / `torch.from_numpy()` |

```python
import numpy as np
import torch

arr = np.array([1.0, 2.0, 3.0])

# NumPy → Tensor（共享内存，修改一个另一个也变）
t = torch.from_numpy(arr)

# Tensor → NumPy（需先 .detach() 断开计算图）
arr2 = t.detach().numpy()

# 启用梯度追踪
t_grad = torch.tensor([1.0, 2.0], requires_grad=True)
```

---

## 四、模型构建方式对比

### NumPy：一切手动，状态管理靠自己

```python
class TwoLayerNet:
    def __init__(self, d_in, d_hidden, d_out):
        self.W1 = np.random.randn(d_in, d_hidden) * 0.01
        self.b1 = np.zeros((1, d_hidden))
        self.W2 = np.random.randn(d_hidden, d_out) * 0.01
        self.b2 = np.zeros((1, d_out))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)   # ReLU
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2

    def backward(self, X, dL_dZ2):
        dL_dW2 = self.A1.T @ dL_dZ2
        dL_db2 = np.sum(dL_dZ2, axis=0)
        dL_dA1 = dL_dZ2 @ self.W2.T
        dL_dZ1 = dL_dA1 * (self.Z1 > 0)   # ReLU 导数
        dL_dW1 = X.T @ dL_dZ1
        dL_db1 = np.sum(dL_dZ1, axis=0)
        return dL_dW1, dL_db1, dL_dW2, dL_db2
```

### PyTorch：继承 `nn.Module`，只写前向

```python
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 反向传播由 loss.backward() 自动完成
```

---

## 五、训练循环对比

```
                 NumPy 手写                    |              PyTorch
--------------------------------------------|--------------------------------------------
1. 前向传播（手写每层）                        | 1. model(X)
2. 计算 Loss（手写公式）                       | 2. criterion(output, Y)
3. 反向传播（手写每层梯度）                     | 3. loss.backward()
4. 手动更新 W -= lr * dW                      | 4. optimizer.step()
5. 手动清零无梯度概念（直接覆盖）                | 5. optimizer.zero_grad()（需显式调用！）
```

> **易错点**：PyTorch 的梯度是**累加**的，每次迭代前必须调用 `optimizer.zero_grad()`，否则梯度会叠加。NumPy 手写无此问题，因为每次直接覆盖梯度变量。

---

## 六、什么时候用哪个？

```
用 NumPy 手写：
  ✓ 学习反向传播原理（强烈推荐！）
  ✓ 理解链式法则、梯度消失/爆炸的本质
  ✗ 实际项目（代码量大、易出错、无法用 GPU）

用 PyTorch：
  ✓ 实际训练模型（CV、NLP、强化学习...）
  ✓ 快速实验不同架构
  ✓ 需要 GPU 加速
  ✗ 学习原理时（太"黑盒"，不适合入门理解）
```

---

## 七、学习路径建议

```
阶段 1：NumPy 手写 单层感知机 → 理解前向/反向传播
       ↓
阶段 2：NumPy 手写 两层网络 → 理解多层链式法则、激活函数梯度
       ↓
阶段 3：用 PyTorch 复现同样网络 → 验证结果一致，感受 Autograd 的魔法
       ↓
阶段 4：用 nn.Module 构建更复杂网络（CNN/RNN）→ 工程能力
```

---

## 八、一句话总结

> **NumPy 手写让你理解"为什么"，PyTorch 让你专注"做什么"。**
> 掌握手写反向传播后再用 PyTorch，才能真正理解框架在替你做什么，出问题时才能 debug。

---

*生成时间：2026-05-06 | 适用：ICPC+DL备赛期 W1 总结*
