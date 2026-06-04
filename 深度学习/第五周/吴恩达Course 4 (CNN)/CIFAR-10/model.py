import torch.nn as nn
from torchvision.models import resnet18



def build_cifar_resnet18(num_classes:int=10)->nn.Module:
    model=resnet18(weights=None,num_classes=num_classes)
    model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()

    for m in model.modules():
        if isinstance(m,nn.ReLU):
            m.inplace=False
    return model







































