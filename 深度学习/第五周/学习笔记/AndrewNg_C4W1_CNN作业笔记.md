# 吴恩达 C4W1 作业笔记：从零搭一个卷积神经网络

> 配套作业：`Convolutional Model: step by step` + `Convolutional Model: Application`
> 写作日期：2026-05-29（W5 Day1 起步，呼应 [[W4_Day5_BatchNorm]] 的训推差异，为 [[W6_nanoGPT]] 之前夯实视觉模型基础）
> 风格沿用 W4 Day 2 三段式：**原理+直觉 → 可运行代码 → 工业锚点**

---

## 0. 学习目标问题（看完笔记后你要能脱口而出）

1. 为什么图像任务不直接用全连接（Fully Connected, FC）层？卷积到底在省什么？
2. `padding`、`stride`、`filter` 三个超参数对输出尺寸的影响公式是什么？为什么记不住的人在工业代码里会踩坑？
3. 一个形状 `(m, n_H, n_W, n_C)` 的 batch 走过 Conv → Pool → Conv → Flatten → Dense，每一步张量形状怎么算？
4. 为什么 cuDNN 实际并不"按定义"做卷积？`im2col` 是个什么把戏？
5. 推理时为什么要把 BatchNorm 折叠（fold）进 Conv？省了什么？
6. `'SAME'` 和 `'VALID'` padding 在不同框架里行为差异在哪儿？哪里会咬人？

> 末尾会回到这 6 题对答案。

---

## 1. 背景：为什么要从 FC 走到 CNN

### 1.1 一个反常识的数字

假设输入是一张 224×224 的彩色图（这是 ImageNet 的标准尺寸），你直接接一个 1000 单元的全连接层：

```
参数量 = 224 × 224 × 3 × 1000 = 150,528,000 ≈ 1.5 亿
```

**只是第一层** 就 1.5 亿参数。这意味着：

- 显存爆炸
- 数据需求爆炸（参数越多越容易过拟合）
- 计算量爆炸

而一个标准的 ResNet-50 全网络才 2500 万参数。**差了 6 倍，还是把整个网络 vs FC 单层。**

### 1.2 FC 的两个根本毛病

类比：你在一张图里找一只猫。

- **毛病一：位置不变性丢失**。FC 把图像拉成一维向量，左上角的猫和右下角的猫，对它来说是完全不同的两个像素组合。它得"分别学一遍"。
- **毛病二：参数没有共享**。识别"猫耳朵"这个特征，左上角学一套权重，右上角学一套，浪费到离谱。

CNN 的两个核心设计就是直接对应这两个毛病：

| 毛病 | CNN 的对策 | 中文名 | 直觉 |
|---|---|---|---|
| 位置敏感 | **Translation Invariance** | 平移不变性 | 同一个滤波器在整张图上滑动，哪儿出现猫耳朵都能检测到 |
| 参数浪费 | **Parameter Sharing** | 参数共享 | 一个 3×3 滤波器只有 9 个参数，整张图复用 |

> **类比**：FC 像是雇 1.5 亿个保安，每人只盯一个像素；CNN 像是给 64 个保安每人一个望远镜（filter），让他们在整张图上来回巡逻。

---

## 2. 核心概念逐个击破

### 2.1 Filter / Kernel（滤波器 / 卷积核）

**是什么**：一个小矩阵（典型尺寸 3×3、5×5、7×7），在输入图像上"滑动"，每个位置做一次"对应位置相乘再求和"的操作（element-wise product + sum）。

**例子**：一个 3×3 的边缘检测滤波器：

```
[[-1, -1, -1],
 [-1,  8, -1],
 [-1, -1, -1]]
```

把它放到一张全 100 的灰图上，输出全是 0（没边缘）；放到一个边界处，输出会很大（检测到了边缘）。

**工业锚点**：现代 CNN 里你 **不会手写** 这种 filter，filter 的数值是训练学出来的。但 ResNet 第一层学出来的 filter 可视化后，长得就像"边缘检测器"和"色块检测器"——这是网络自动从数据里发现了 80 年代图像处理的经典手工特征。

### 2.2 Padding（填充）

**是什么**：在输入边缘补一圈 0（zero padding），让卷积之后输出尺寸不缩水。

**为什么需要**：不 pad 的话，每过一层 conv 图就缩小一圈。10 层之后 224×224 缩成 204×204，**深度网络会越来越瘦**。

**两种常见模式**：

- `'VALID'`：不 pad，输出会变小。
- `'SAME'`：pad 到让输出尺寸等于输入（步长为 1 时）。

**踩坑预警**：**`'SAME'` 在 TensorFlow 和 PyTorch 里行为不一样。**

- TF 的 `'SAME'` 会自动算 pad，且左右可能不对称（左 1 右 2）。
- PyTorch 的 `padding='same'` 是后加的（1.9+），且要求 stride=1。

工业项目从 TF 迁移到 PyTorch 时，这是导致 **数值精度对不齐** 的常见元凶。我之前看 [issue 跟踪](https://github.com/pytorch/pytorch/issues/3867) 里就有大量讨论。

### 2.3 Stride（步长）

**是什么**：filter 每次滑动的距离。stride=1 一次走一格，stride=2 一次走两格。

**直觉**：stride 向下采样（downsample）的"代偿"——stride=2 直接让输出尺寸减半，比 pooling 更激进。

**输出尺寸公式**（背下来）：

```
n_H_out = floor( (n_H + 2*pad - f) / stride ) + 1
n_W_out = floor( (n_W + 2*pad - f) / stride ) + 1
n_C_out = filter 个数
```

其中 `f` 是 filter 边长。

> **记忆口诀**："输入加 pad 减 f，除以步长加 1，向下取整。"

### 2.4 Channel（通道）

**是什么**：彩色图有 RGB 3 个通道；中间层的 feature map 通道数等于上一层 filter 的个数。

**关键性质**：filter 的通道数 **必须等于** 输入的通道数。比如输入是 `(h, w, 64)`，那么 filter 形状是 `(f, f, 64)`，每个 filter 输出一个二维 feature map，多个 filter 堆叠就形成新的通道维度。

**类比**：64 个通道像 64 个不同视角的"灰图"。filter 在所有视角上同时做卷积，再把结果加起来——相当于"综合 64 个视角投一票"。

---

## 3. Assignment 1：用 numpy 手写卷积前向传播

这一节对应作业 `Convolutional Model: step by step`。**不要跳过手写**——cuDNN 是怎么对它做加速的，你必须先知道 naive 版长什么样。

### 3.1 Zero Padding 工具函数

```python
import numpy as np

def zero_pad(X, pad):
    """
    给一个 batch 的图像四周补 0。
    
    X: shape (m, n_H, n_W, n_C) — m 张图，HWC 顺序
    pad: 单边补 0 的层数（int）
    
    返回 X_pad: shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    # 为什么 pad_width 这样写：
    # 第 0 维 (m, batch) 不补；第 3 维 (channel) 不补；
    # 中间两维 (H, W) 才是图像空间维度，要补。
    X_pad = np.pad(
        X,
        pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode='constant',
        constant_values=0
    )
    return X_pad

# 测试
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print(x.shape, '->', x_pad.shape)
# (4, 3, 3, 2) -> (4, 7, 7, 2)
```

> **工业锚点**：实际部署里 zero pad 经常被融合（fuse）进 conv kernel 内，避免显式分配那块多出来的内存。TensorRT 和 ONNX Runtime 都做这种 padding fusion。

### 3.2 Single Step of Convolution（卷积单步）

**是什么**：filter 已经对齐到输入的某个位置上，做"对应相乘再求和"。

```python
def conv_single_step(a_slice_prev, W, b):
    """
    a_slice_prev: 输入的一个切片，shape (f, f, n_C_prev)
    W: 一个 filter 的权重，shape (f, f, n_C_prev)
    b: 偏置，shape (1, 1, 1)
    
    返回一个标量 Z（这个位置的卷积输出）
    """
    # element-wise 乘法 —— 注意不是矩阵乘
    s = np.multiply(a_slice_prev, W)
    
    # 把所有元素加起来
    Z = np.sum(s)
    
    # 加偏置；float() 是为了把 (1,1,1) 的 numpy 数组转成 Python float
    Z = float(Z + b)
    return Z
```

> **常见误区**：很多人第一次写会用 `np.dot` 或 `@`——错。卷积是 **逐元素相乘求和**，不是矩阵乘。但下面 §6.1 的 im2col 会告诉你工业实现里它 **实际上** 又被改写成了矩阵乘。这不矛盾——是数学上等价的两种实现。

### 3.3 Conv Forward 完整版

```python
def conv_forward(A_prev, W, b, hparameters):
    """
    A_prev: 上一层激活值, shape (m, n_H_prev, n_W_prev, n_C_prev)
    W:      filter 权重,    shape (f, f, n_C_prev, n_C)  
    b:      偏置,           shape (1, 1, 1, n_C)
    hparameters: dict, 含 'stride' 和 'pad'
    """
    # 1. 拆出维度（这步就是大家口中的 "shape gymnastics"）
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # 2. 算输出尺寸（这就是 §2.3 的公式）
    n_H = int((n_H_prev - f + 2*pad) / stride) + 1
    n_W = int((n_W_prev - f + 2*pad) / stride) + 1
    
    # 3. 初始化输出
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # 4. pad 输入
    A_prev_pad = zero_pad(A_prev, pad)
    
    # 5. 四重循环 —— naive 实现，慢得令人发指，但好懂
    for i in range(m):                       # 遍历 batch
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):                 # 遍历输出高度
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):             # 遍历输出宽度
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):         # 遍历输出通道
                    # 切出 filter 要看的那块输入
                    a_slice_prev = a_prev_pad[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :  # 所有输入通道一起进
                    ]
                    # 调用单步卷积
                    Z[i, h, w, c] = conv_single_step(
                        a_slice_prev, W[:, :, :, c], b[:, :, :, c]
                    )
    
    cache = (A_prev, W, b, hparameters)  # 反向传播要用
    return Z, cache
```

**这段代码有 4 层 for 循环**，对一个 224×224 输入跑一次大约要几秒——你能直观感受到为什么 GPU + cuDNN 是必须的。

### 3.4 Pooling 池化前向

**是什么**：取一个窗口里的最大值（max pool）或平均值（avg pool）。**没有可学习参数**。

**为什么用**：
- 降采样，减少后续计算
- 提供一定的局部平移不变性（窗口里挪一格，max 还是同一个值）

```python
def pool_forward(A_prev, hparameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']
    
    # 注意 pool 不改变通道数
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vs, ve = h*stride, h*stride+f
                    hs, he = w*stride, w*stride+f
                    a_slice = A_prev[i, vs:ve, hs:he, c]
                    
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_slice)
    
    cache = (A_prev, hparameters)
    return A, cache
```

> **工业锚点**：现代 CNN（如 ConvNeXt、Vision Transformer 之后的设计）越来越少用 max pool，**用 stride=2 的 conv 代替**。原因：pool 信息丢失太死板，stride conv 是"可学习的下采样"。这也是 W5 学 ResNet 时你会注意到的——ResNet 早期用 maxpool，后期 block 内全是 stride conv。

---

## 4. Assignment 2：用 TF/Keras 训练一个真 CNN

作业 2 用 TensorFlow 在 SIGNS 数据集（手势数字 0-5）上训练。这一节我会用更现代的写法（TF 2.x + Keras），并补上工业实践会做、但作业不会教的细节。

### 4.1 Sequential API（最简单）

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_sequential(input_shape=(64, 64, 3), num_classes=6):
    model = models.Sequential([
        # 第 1 个 conv 块
        layers.Conv2D(8, kernel_size=4, strides=1, padding='same',
                      input_shape=input_shape),
        layers.ReLU(),
        layers.MaxPool2D(pool_size=8, strides=8, padding='same'),
        
        # 第 2 个 conv 块
        layers.Conv2D(16, kernel_size=2, strides=1, padding='same'),
        layers.ReLU(),
        layers.MaxPool2D(pool_size=4, strides=4, padding='same'),
        
        # 分类头
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model_sequential()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  
    # ↑ 注意：sparse 版的 loss 直接吃整数标签 (0,1,2..)，
    #   不需要 one-hot。工业项目 90% 用这个，省内存。
    metrics=['accuracy']
)
model.summary()
```

### 4.2 Functional API（更工业，处理多输入/分支）

```python
def build_model_functional(input_shape=(64, 64, 3), num_classes=6):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.Conv2D(8, 4, padding='same')(inputs)
    x = layers.BatchNormalization()(x)        # ← BN 实战出场，呼应 W4 Day5
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(8, 8, padding='same')(x)
    
    x = layers.Conv2D(16, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(4, 4, padding='same')(x)
    
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

**为什么作业里没用 BN，但工业代码必须加？**

- 训练快得多（learning rate 可以调大）
- 一定的正则化效果
- ResNet/EfficientNet 等所有现代结构都默认带 BN（或它的变体 LayerNorm/GroupNorm）

详见 [[batchnorm_inference]]——你 W4 Day5 写过的笔记，那里讲了 BN 训推双行为。

### 4.3 训练 + 验证（含工业最佳实践）

```python
# 1. 数据 pipeline —— 用 tf.data 而不是 numpy 喂
train_ds = (tf.data.Dataset
            .from_tensor_slices((X_train, Y_train))
            .shuffle(buffer_size=1024)
            .batch(64)
            .prefetch(tf.data.AUTOTUNE))   # ← 关键：让 CPU 准备数据时 GPU 不闲着

val_ds = (tf.data.Dataset
          .from_tensor_slices((X_test, Y_test))
          .batch(64)
          .prefetch(tf.data.AUTOTUNE))

# 2. 回调（callback）—— 工业项目必加
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True
    ),  # ← 5 个 epoch val_loss 没降就停，回滚到最好的权重
    
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3
    ),  # ← val_loss 卡了 3 个 epoch 就把 lr 砍半
    
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', save_best_only=True
    ),
]

# 3. 训练
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2  # ← 一个 epoch 一行，不要进度条（写日志友好）
)
```

> **常见错误**：直接用 `model.fit(X, Y, batch_size=64)` 也能跑，但你失去了 prefetch、shuffle、autotune 的好处。在 GPU 上这能差 2~3 倍速度——`nvidia-smi` 看到 GPU 利用率只有 30% 就是这种事。

### 4.4 推理时怎么用（工业部署视角）

```python
# 训练完保存
model.save('signs_cnn.keras')

# 部署端加载并推理
loaded = tf.keras.models.load_model('signs_cnn.keras')

# 单张图推理 —— 注意要扩 batch 维度
import numpy as np
single_img = np.random.rand(64, 64, 3).astype(np.float32)
batch = single_img[np.newaxis, ...]   # (1, 64, 64, 3)
probs = loaded.predict(batch, verbose=0)
pred = np.argmax(probs, axis=-1)
```

**踩坑**：忘记加 batch 维度是初学者排第一的报错——`Input 0 of layer "conv2d" is incompatible with the layer: expected min_ndim=4, found ndim=3`。看到这个直接想"我是不是少了一维"。

---

## 5. 反向传播的直觉（选做但建议读）

作业里反向传播是选做，但搞 AI Infra 你必须懂大概在干嘛。

### 5.1 Conv 反向传播的核心一句话

> **卷积的反向传播，本身也是一种卷积**（对 filter 做 180° 翻转后的卷积）。

这意味着前向算子的高效实现，反向也能复用——cuDNN 的 conv backward 本质上调的还是 conv forward 的内核。

### 5.2 Max Pool 反向传播

只有"当时被选中"的那个位置接收梯度，其它位置梯度为 0。

```python
# 简化示意
mask = (a_slice == np.max(a_slice))  
# 那一格梯度等于上游梯度，其它为 0
dA_prev_slice = mask * dA[..., this_pos]
```

### 5.3 Avg Pool 反向传播

梯度被均匀分摊回窗口里每个位置（每个位置拿 `dA / (f*f)`）。

---

## 6. 工业锚点（重点中的重点，呼应你 AI Infra 方向）

### 6.1 im2col：你以为 cuDNN 在做卷积，它其实在做矩阵乘

**问题**：§3.3 那个四层 for 循环慢得没法用。GPU 适合做的是 **大矩阵乘** （GEMM），不是带空间索引的滑动窗口。

**im2col 的把戏**：把每个 filter 要看的小窗口"拉直"成一列，整张图变成一个大矩阵；把所有 filter 也"拉直"。这样卷积 = 矩阵乘。

```
原始：filter 3×3×64,  在 56×56×64 上滑动
变形：把每个 3×3×64 的窗口拉成一个 576 长的列
     共 56×56 = 3136 列 → 矩阵 (576, 3136)
     filter 也拉成 (n_filters, 576)
     卷积 = filter_mat @ im2col_mat
```

**代价**：内存膨胀 `f²` 倍（窗口重叠的部分被重复存了）。
**收益**：能直接调 cuBLAS 的 GEMM kernel，速度提升 10-100 倍。

> **W5 Day1 必看**：PyTorch 源码 `aten/src/ATen/native/Convolution.cpp` 里就有 `slow_conv2d_forward` 和走 cuDNN 的两条路径，cuDNN 路径背后正是 im2col + GEMM（或 Winograd）。

### 6.2 Winograd 卷积

**是什么**：对 3×3 卷积有专门的代数变换，可以把乘法次数从 9 次降到 4 次（F(2,3) 算法），代价是几次廉价的加法。

**适用场景**：3×3 stride=1 的 conv，正好是 ResNet 等模型的主力。

**工业意义**：TensorRT 默认对 3×3 conv 启用 Winograd，能把推理 latency 砍 30% 左右。

### 6.3 NCHW vs NHWC：内存布局之争

| 布局 | 全称 | 谁用 |
|---|---|---|
| **NHWC** | (batch, Height, Width, Channel) | TensorFlow 默认、CPU 友好 |
| **NCHW** | (batch, Channel, Height, Width) | PyTorch / cuDNN 默认 |

**为什么这事重要**：

- **NHWC** 在做"对每个像素的所有 channel 做某操作"（如 BN、激活）时连续访存友好。
- **NCHW** 在做"对每个 channel 的整张图做卷积"时连续访存友好。
- **跨框架转模型** 时如果 layout 没对齐，模型会 silently 输出全是噪声。这在 ONNX 转换里非常常见。

**调试技巧**：模型从 PyTorch 转 TensorFlow 后输出全错？先看 layout：用 `np.transpose(x, (0, 2, 3, 1))` 把 NCHW 转 NHWC。

### 6.4 推理优化：BN 折叠（BN Folding）

**问题**：训练时 Conv → BN → ReLU 是三个独立算子。推理时 BN 用的是固定的 running_mean/var，本质是个 affine 变换 `y = γ*x + β`。这个 affine 可以 **吸收进** 前面 Conv 的 weight 和 bias 里——融合后只剩 Conv → ReLU。

```python
# 折叠公式（PyTorch 风格）
W_fold = W * (gamma / sqrt(var + eps)).reshape(C, 1, 1, 1)
b_fold = (b - running_mean) * gamma / sqrt(var + eps) + beta
```

**收益**：
- 算子数减少 → kernel 启动开销减少
- 显存读写减少 → 带宽友好
- 推理 latency 通常降 10~20%

**踩坑**：BN 折叠后再做 INT8 量化，分布会变——必须重新校准（calibration）。

> 这条直接接你 W4 Day5 的 [[batchnorm_inference]] 笔记，是工业落地里最高频的优化之一。

### 6.5 INT8 量化卷积

把 Conv 的 weight 从 FP32 转成 INT8，推理时整个 conv 在 INT8 上算，最后 dequant 回 FP32（或 FP16）。

- **存储**：4 倍压缩
- **算力**：现代 GPU（Ampere+）的 INT8 Tensor Core 比 FP32 快 4-8 倍
- **精度损失**：通常 <1%，但 **量化感知训练（QAT）** 比训后量化（PTQ）效果更好

W8 推理优化入门时你会重点学这块，这里先有个印象就行。

---

## 7. 常见陷阱清单

| 陷阱 | 症状 | 解决 |
|---|---|---|
| 输入图像没归一化（0-255 直接喂） | loss NaN，或者 acc 卡在随机水平 | 训练前 `x = x / 255.0` 或减均值除标准差 |
| 忘记加 batch 维度 | `expected ndim=4, found ndim=3` | `x[np.newaxis, ...]` |
| label 形状不对 | `loss=sparse_categorical_crossentropy` 报错 | 用整数 label，不要 one-hot |
| filter 数太大，第一层就 OOM | `RESOURCE_EXHAUSTED` / CUDA OOM | 减 batch size，或减第一层 filter 数 |
| `'SAME'` padding 在 PyTorch 报错 | `padding='same' is not supported with strides>1` | 改成手动 `nn.ZeroPad2d` |
| BN 用在 batch_size=1 推理时崩溃 | inference 输出全是怪值 | 切到 `model.eval()`，PyTorch 会用 running_stats |

---

## 8. 自测题（合上笔记答）

1. 输入 `(32, 64, 64, 3)`，经过 `Conv2D(filters=16, kernel=5, stride=2, pad=2)`，输出形状？
2. 同样输入，经过 `MaxPool2D(pool=2, stride=2)`，输出形状？通道数变没变？
3. 用一句话解释：为什么卷积比全连接更适合图像？（要点：参数共享、平移不变性）
4. cuDNN 怎么把卷积变成矩阵乘？这么做的代价是什么？
5. BN 在训练和推理时行为有什么不同？为什么推理时可以折叠进 Conv？

> 参考答案在 §2.3、§3.4、§1.2、§6.1、§6.4。如果有 1 道答不上来，回到对应章节再读一遍。

---

## 9. 与已有笔记的串联

| 这次学到的 | 串联回 | 串联到 |
|---|---|---|
| Conv 前向手写 | [[W3_micrograd]] 的 forward 思路 | [[W6_nanoGPT]] 里的 attention forward |
| BN 折叠 | [[W4_Day5_batchnorm_inference]] | [[W8_推理优化_BN_fusion]] |
| profiler 在 conv 上的输出 | [[W4_Day3_first_profiler]] | W5 Day3 计划做的 ResNet profile |
| 参数共享 → 显存节省 | [[W4_Day6_optimizer_memory]] | W5 ResNet 实战 |

---

## 10. 完成 Checklist（学完打勾）

- [ ] 能默写 conv 输出尺寸公式
- [ ] 不看作业代码，能独立写出 `conv_single_step` 和 `conv_forward`（哪怕慢）
- [ ] 在 SIGNS 上跑通了 Sequential 版，test acc > 80%
- [ ] 加了 BN 后跑 Functional 版，对比有没有更稳/更快
- [ ] 看过一次 `tf.profiler` 或 `torch.profiler` 的输出，知道 conv 占了多少时间
- [ ] 能用一句话解释 im2col 是什么
- [ ] 能用一句话解释 BN folding 省了什么

---

## 11. 下一步建议（衔接 W5 后续）

- **Day 2-3**：把 SIGNS 替换成 CIFAR-10，搭一个 5-block 的 plain CNN，故意先不加 residual connection——下周接 ResNet 时你会切肤体会"为什么需要 skip connection"。
- **Day 4-5**：用 `torch.profiler` profile 你写的 plain CNN，找出最耗时的算子。**这条是 AI Infra 视角的关键动作**。
- **Day 6**：起一个 `tech_notes/conv_im2col.md`，把 §6.1 的内容用代码 demo 一下（自己写一个 numpy 版的 im2col 和等价的 GEMM）。

---

> **元笔记**：这份笔记按"原理 → 手写实现 → 框架实战 → 工业优化"四段式扩写自原 Day 2 三段式。AI Infra 锚点（§6 整节）是为你目标方向特意补的，作业本身不会教这些。

---

## 附录 A：PyTorch 等价实现（推荐主用版本）

> 如果你目标是 AI Infra，**这一节才是你真正要写的代码**。§4 的 TF 版只用于读懂 Andrew Ng 原作业模板。

### A.1 关键差异速查表（先看这个再写代码）

| 维度 | TensorFlow / Keras | PyTorch |
|---|---|---|
| 默认 layout | NHWC `(N, H, W, C)` | NCHW `(N, C, H, W)` |
| 卷积层 | `Conv2D(filters, kernel_size)` | `nn.Conv2d(in_ch, out_ch, kernel_size)` ← **必须显式写 in_ch** |
| padding `'same'` | 任意 stride 都支持 | 仅 stride=1 支持 `padding='same'`，否则手写 `nn.ZeroPad2d` |
| 训练 API | `model.fit(...)` 全自动 | 必须自己写训练循环 ← **好事，能看清每步** |
| 模式切换 | 自动 | `model.train()` / `model.eval()` ← **忘了切 BN 会出诡异 bug** |
| 标签格式 | `sparse_categorical_crossentropy` 吃 int | `nn.CrossEntropyLoss` 直接吃 int（且内置 softmax）|

### A.2 模型定义（两种写法）

```python
import torch
import torch.nn as nn

# 写法 1：nn.Sequential —— 等价于 Keras Sequential
model_seq = nn.Sequential(
    # PyTorch 是 NCHW，输入 (N, 3, 64, 64)
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, padding='same'),
    nn.BatchNorm2d(8),       # ← 注意是 BatchNorm2d，不是 BatchNormalization
    nn.ReLU(inplace=True),   # inplace=True 省一次显存分配，工业里默认开
    nn.MaxPool2d(kernel_size=8, stride=8, padding=0),
    
    nn.Conv2d(8, 16, kernel_size=2, padding='same'),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    nn.Linear(16 * 2 * 2, 6)  # ← 这个 16*2*2 要算对，否则 shape mismatch
    # 不加 softmax —— CrossEntropyLoss 内置了
)

# 写法 2：nn.Module 子类 —— 工业项目主流，能塞复杂控制流
class SignsCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(8, 8)
        
        self.conv2 = nn.Conv2d(8, 16, 2, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(4, 4)
        
        self.fc = nn.Linear(16 * 2 * 2, num_classes)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)  # 保留 batch 维
        return self.fc(x)
```

> **算 `Linear` 输入维度的工业技巧**：不想手算？建一个假 batch 跑一遍 forward 到 flatten 之前 `print(x.shape)`，比纸笔推导靠谱。或者用 `nn.LazyLinear(num_classes)` —— 第一次 forward 时自动推断输入维度。

### A.3 数据 pipeline（对标 TF 的 `tf.data`）

```python
from torch.utils.data import TensorDataset, DataLoader

# 假设 X_train shape (m, 64, 64, 3)，要转成 NCHW
X_train_t = torch.from_numpy(X_train).permute(0, 3, 1, 2).float() / 255.0
Y_train_t = torch.from_numpy(Y_train).long()  # ← CrossEntropyLoss 要 long

train_ds = TensorDataset(X_train_t, Y_train_t)
train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # ← 多进程加载，CPU 准备数据时 GPU 不闲（对应 TF 的 prefetch）
    pin_memory=True,      # ← 锁页内存，CPU→GPU 拷贝快一档
    persistent_workers=True,  # ← worker 不在每个 epoch 重启，省启动开销
)
```

**踩坑**：Windows 上 `num_workers > 0` 经常会因为 spawn 模式炸——必须把训练代码包在 `if __name__ == '__main__':` 里。这条你 Windows 11 环境会遇到。

### A.4 训练循环（手写，看清每一步）

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignsCNN(num_classes=6).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3
)

best_val_acc = 0.0
patience_counter = 0
EARLY_STOP_PATIENCE = 5

for epoch in range(50):
    # ============ 训练阶段 ============
    model.train()  # ← 关键：BN/Dropout 切到训练模式
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # non_blocking=True 配合 pin_memory，让拷贝异步进行
        
        optimizer.zero_grad(set_to_none=True)
        # set_to_none=True 比默认 zero_ 略快，工业默认开
        
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * x.size(0)
        train_correct += (logits.argmax(1) == y).sum().item()
        train_total += x.size(0)
    
    # ============ 验证阶段 ============
    model.eval()  # ← 关键：BN 用 running_stats，Dropout 关闭
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():  # ← 不建反向图，省一半显存
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits, y).item() * x.size(0)
            val_correct += (logits.argmax(1) == y).sum().item()
            val_total += x.size(0)
    
    val_acc = val_correct / val_total
    print(f"Epoch {epoch:02d} | "
          f"train_loss={train_loss/train_total:.4f} acc={train_correct/train_total:.4f} | "
          f"val_loss={val_loss/val_total:.4f} acc={val_acc:.4f}")
    
    scheduler.step(val_loss / val_total)
    
    # 手写 EarlyStopping + best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_signs.pt')  # 只存权重，工业唯一姿势
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break
```

### A.5 推理与保存（工业姿势）

```python
# ===== 保存：只存 state_dict，绝不存整个 model 对象 =====
torch.save(model.state_dict(), 'signs.pt')

# ===== 加载 =====
model = SignsCNN(num_classes=6)         # 先建结构
model.load_state_dict(torch.load('signs.pt', map_location='cpu'))
model.eval()                            # 切到推理模式
model.to(device)

# ===== 推理（注意 NCHW 和 batch 维） =====
import numpy as np
single_img = np.random.rand(64, 64, 3).astype(np.float32)
x = torch.from_numpy(single_img).permute(2, 0, 1)  # HWC -> CHW
x = x.unsqueeze(0).to(device)                       # (1, 3, 64, 64)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(-1).item()
```

> **为什么不存整个模型对象？**：`torch.save(model, ...)` 会 pickle 类定义，部署端如果代码路径变了就加载失败；CI/CD 里很容易踩。`state_dict` 只是权重字典，跨版本/跨代码兼容性强得多。这是工业唯一可接受的做法。

### A.6 接 `torch.profiler`（呼应 W4 Day3）

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
) as prof:
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        prof.step()
        if step >= 10:
            break

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=15))
```

输出里你会看到 `aten::cudnn_convolution` 占大头——**这就是 §6.1 说的 cuDNN 卷积**。把它的执行时间记下来，到 W5 Day 4 加 BN folding 之后再 profile 一次，对比时间差，那条曲线就是你 AI Infra 路上第一个量化收益。

### A.7 PyTorch 视角的 §7 陷阱补丁

| 新增陷阱 | 症状 | 解决 |
|---|---|---|
| 忘了 `model.eval()` 直接推理 | val_acc 比 train 还低很多，且每次跑结果不同 | 推理前必 `model.eval()` |
| 忘了 `optimizer.zero_grad()` | loss 不降反升、显存爆掉 | 每个 step 第一行就 zero_grad |
| 标签是 float 喂给 CrossEntropyLoss | `expected scalar type Long but found Float` | `.long()` 转一下 |
| `num_workers>0` 在 Windows 卡住 | DataLoader 启动后无响应 | 主代码包到 `if __name__ == '__main__':` |
| `tensor.to(device)` 之后不重新赋值 | 模型还在 CPU | `x = x.to(device)`，不是 `x.to(device)` |

---

## 附录 B：TF / PyTorch 数值对齐 Mini-Lab（最值钱的练习）

**目标**：同样的输入、同样的权重，TF 模型和 PyTorch 模型输出对齐到小数点后 4 位。

```python
# 1. 先训一个 TF 模型，导出权重为 numpy
tf_weights = {layer.name: layer.get_weights() for layer in tf_model.layers}

# 2. 在 PyTorch 模型里手动赋值
#    ★ 关键：Conv 的权重 layout 不一样
#    TF:      (kH, kW, in_C, out_C)
#    PyTorch: (out_C, in_C, kH, kW)
W_tf = tf_weights['conv2d'][0]              # (4, 4, 3, 8)
W_pt = W_tf.transpose(3, 2, 0, 1)           # (8, 3, 4, 4)
pt_model.conv1.weight.data = torch.from_numpy(W_pt)
pt_model.conv1.bias.data   = torch.from_numpy(tf_weights['conv2d'][1])

# 3. 同一张图，两边推理，diff 应该 < 1e-5
x_np = np.random.rand(1, 64, 64, 3).astype(np.float32)
y_tf = tf_model(x_np).numpy()                                # NHWC
y_pt = pt_model(torch.from_numpy(x_np).permute(0,3,1,2)).detach().numpy()
print('max diff:', np.abs(y_tf - y_pt).max())
```

**做完这个练习你会一辈子记得**：Conv 权重的 layout 转置是 `(3, 2, 0, 1)`。这是 ONNX/TFLite 转换器内部干的事，自己亲手做一次比读 100 篇博客都有用。

---

> **元元笔记**：附录 A 是你的主用代码，§4 的 TF 版仅用于读懂作业原文。附录 B 那个 mini-lab 强烈建议挤出 1 小时做——AI Infra 工程师的高频日常就是这种"两个框架对齐"。
