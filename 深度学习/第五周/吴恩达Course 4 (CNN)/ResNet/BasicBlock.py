import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)

        self.conv2=nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels*self.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )

    def forward(self,x):
        identity=self.shortcut(x)

        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))

        out=out+identity
        out=self.relu(out)
        return out



if __name__=="__main__":
    # 情况1：通道不变、尺寸不变 —— shortcut 走恒等
    blk1 = BasicBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 32, 32)  # (batch, channel, H, W)
    print("不变:", blk1(x).shape)  # 期望 [2, 64, 32, 32]

    # 情况2：通道翻倍 + 下采样 —— shortcut 走 1x1 卷积
    blk2 = BasicBlock(64, 128, stride=2)
    print("下采样:", blk2(x).shape)  # 期望 [2, 128, 16, 16]




































