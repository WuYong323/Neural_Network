import torch
import torch.nn as nn



class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,in_channels:int,base_channels:int,stride=1):
        super().__init__()
        out_channels=base_channels*self.expansion

        self.conv1=nn.Conv2d(in_channels,base_channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(base_channels)

        self.conv2=nn.Conv2d(base_channels,base_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(base_channels)

        self.conv3=nn.Conv2d(base_channels,out_channels,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)

        self.relu=nn.ReLU(inplace=True)

        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        identity=self.shortcut(x)

        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))

        out=out+identity
        out=self.relu(out)
        return out



if __name__=="__main__":
    # ResNet-50 第一个 stage：输入 64 通道，base=64，输出 256 通道
    blk = Bottleneck(in_channels=64, base_channels=64, stride=1)
    x = torch.randn(2, 64, 56, 56)
    print("Bottleneck 输出:", blk(x).shape)  # 期望 [2, 256, 56, 56]

    # 带下采样的 stage：输入 256，base=128，stride=2 → 输出 512、尺寸减半
    blk2 = Bottleneck(in_channels=256, base_channels=128, stride=2)
    y = torch.randn(2, 256, 56, 56)
    print("下采样 Bottleneck:", blk2(y).shape)  # 期望 [2, 512, 28, 28]








































