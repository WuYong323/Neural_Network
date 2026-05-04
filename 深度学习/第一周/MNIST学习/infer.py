"""
infer.py —— MNIST 手写数字推理脚本
用法：python infer.py <图片路径>
例如：python infer.py my_digit.jpg
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# ── 1. 模型定义（必须和训练时完全一致）──────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── 2. Otsu 自适应阈值（纯 numpy 实现）──────────────────────────────────────
def _otsu_threshold(arr):
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_total = np.dot(np.arange(256), hist)

    best_thresh, best_var = 0, 0
    sum_bg, weight_bg = 0.0, 0.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0 or weight_bg == total:
            continue
        weight_fg = total - weight_bg
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var, best_thresh = var, t

    return best_thresh


# ── 3. 图像预处理（照片 → 28×28 的 MNIST 格式）──────────────────────────────
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')       # 转灰度
    arr = np.array(img, dtype=np.float32)

    # MNIST 是黑底白字；如果你的图是白底黑字，自动翻转
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Otsu 自适应二值化：去除背景噪点，保留笔画
    thresh = _otsu_threshold(arr)
    arr = np.where(arr > thresh, arr, 0.0)

    # 找出数字的包围盒（有像素的行和列）
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)

    if not rows.any():
        return Image.fromarray(np.zeros((28, 28), dtype=np.uint8))

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # 四周加 20% 的留白，防止数字贴边
    h = r_max - r_min + 1
    w = c_max - c_min + 1
    pad = int(max(h, w) * 0.2)
    r_min = max(0, r_min - pad)
    r_max = min(arr.shape[0] - 1, r_max + pad)
    c_min = max(0, c_min - pad)
    c_max = min(arr.shape[1] - 1, c_max + pad)
    cropped = arr[r_min:r_max + 1, c_min:c_max + 1]

    # 补零边让裁剪区域变成正方形，再缩放到 20×20
    h, w = cropped.shape
    if h > w:
        left = (h - w) // 2
        right = (h - w) - left
        cropped = np.pad(cropped, ((0, 0), (left, right)))
    elif w > h:
        top = (w - h) // 2
        bottom = (w - h) - top
        cropped = np.pad(cropped, ((top, bottom), (0, 0)))

    pil_20 = Image.fromarray(cropped.astype(np.uint8)).resize((20, 20), Image.LANCZOS)

    # 放进 28×28 画布正中间（每边留 4 像素），和 MNIST 格式对齐
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = np.array(pil_20)

    return Image.fromarray(canvas)


# ── 4. 推理主函数 ─────────────────────────────────────────────────────────────
def predict_image(image_path, model, device, save_path='prediction_result.png'):
    # 预处理
    img_processed = preprocess_image(image_path)

    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform_infer(img_processed).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs_np = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted = int(probs_np.argmax())

    # 可视化
    original = Image.open(image_path).convert('L')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(img_processed, cmap='gray')
    axes[1].set_title(
        f'Model Input (28x28)\nPredicted: {predicted}  ({probs_np[predicted] * 100:.1f}%)',
        fontsize=12
    )
    axes[1].axis('off')

    colors = ['#4C72B0'] * 10
    colors[predicted] = '#DD4444'
    bars = axes[2].bar(range(10), probs_np * 100, color=colors,
                       edgecolor='white', linewidth=0.5)
    axes[2].set_xticks(range(10))
    axes[2].set_xlabel('Digit Class', fontsize=11)
    axes[2].set_ylabel('Probability (%)', fontsize=11)
    axes[2].set_title('Probability Distribution', fontsize=13)
    axes[2].set_ylim(0, 115)
    axes[2].yaxis.grid(True, linestyle='--', alpha=0.5)
    axes[2].set_axisbelow(True)

    for bar, p in zip(bars, probs_np):
        if p > 0.005:
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{p * 100:.1f}%',
                ha='center', va='bottom', fontsize=9
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\nPredicted: {predicted}   Confidence: {probs_np[predicted] * 100:.2f}%')
    print('All probabilities:')
    for i, p in enumerate(probs_np):
        bar_str = '█' * int(p * 40)
        print(f'  {i}: {bar_str} {p * 100:.2f}%')

    return predicted, probs_np


# ── 5. 程序入口 ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MNIST 手写数字推理')
    parser.add_argument('image', help='输入图片路径')
    parser.add_argument('--weights', default='mnist_cnn.pth', help='模型权重文件路径（默认 mnist_cnn.pth）')
    parser.add_argument('--output', default='prediction_result.png', help='结果图保存路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = CNN().to(device)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f'Loaded weights from: {args.weights}')
    except FileNotFoundError:
        print(f'[ERROR] 找不到权重文件: {args.weights}')
        print('请先运行训练代码，或用 --weights 指定正确路径')
        sys.exit(1)

    predict_image(args.image, model, device, save_path=args.output)


if __name__ == '__main__':
    main()
