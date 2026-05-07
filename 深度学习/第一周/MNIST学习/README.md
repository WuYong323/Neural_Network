# 深度学习学习记录

基于 PyTorch 的深度学习入门项目，记录从零开始学习神经网络的过程。

## 项目结构

```
JupyterProject1/
├── 深度学习/
│   └── 第一周/
│       ├── torch学习.ipynb                          # PyTorch 基础语法学习
│       └── MNIST学习/
│           ├── MNIST baseline.ipynb                 # 基础 CNN 模型训练
│           ├── MNIST baseline（调试）.ipynb          # 调试版本
│           ├── MNIST识别优化(数据增强与otsu）.ipynb  # 数据增强 + Otsu 阈值优化
│           ├── BatchNorm与Dropout.ipynb             # 正则化技术实验
│           ├── 手写MINIST.ipynb                     # 手写数字识别实现
│           ├── 推理脚本.ipynb                       # 推理流程演示
│           ├── config(配置文件）.ipynb              # 配置项说明
│           ├── config.py                            # 训练超参数配置
│           ├── infer.py                             # 推理脚本（命令行）
│           └── mnist_cnn.pth                        # 训练好的模型权重
```

## 环境配置

**要求：** Python 3.8+，CUDA（可选，有 GPU 时自动启用）

```bash
pip install torch torchvision numpy pillow matplotlib
```

## 模型结构

两层卷积 + 全连接的 CNN：

```
Conv2d(1→32) → BatchNorm → ReLU → MaxPool
Conv2d(32→64) → BatchNorm → ReLU → MaxPool
Flatten
Linear(3136→128) → BatchNorm → ReLU → Dropout(0.5)
Linear(128→10)
```

## 训练超参数

| 参数 | 值 |
|------|----|
| seed | 42 |
| batch_size | 64 |
| learning_rate | 1e-3 |
| epochs | 10 |
| optimizer | Adam |

## 使用方法

**训练：** 直接运行 Notebook 中的训练单元格即可。

**推理（命令行）：**

```bash
cd 深度学习/第一周/MNIST学习
python infer.py <图片路径>

# 指定权重文件
python infer.py my_digit.jpg --weights mnist_cnn.pth

# 保存结果图
python infer.py my_digit.jpg --output result.png
```

推理脚本会输出：
- 预测数字及置信度
- 原图 / 预处理后图 / 概率分布的可视化图

## 实验结果

| 实验 | 测试集准确率 |
|------|------------|
| Baseline CNN | ~99% |
| + 数据增强 + Otsu | ~99%+ |

**注意：** 在自己拍摄的手写图片上准确率会有所下降，主要原因是图片预处理（背景噪点、笔画粗细）与 MNIST 数据集分布存在差异，后续可继续改进预处理流程。

## 学习内容

- **第一周**
  - PyTorch 张量操作、自动微分基础
  - CNN 搭建与训练流程
  - BatchNorm、Dropout 正则化
  - 数据增强（随机旋转、仿射变换）
  - Otsu 自适应阈值图像预处理
  - 模型保存与加载推理
