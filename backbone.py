import torch
import torch.nn as nn
from torchvision.models import resnet50

class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)

        self.stage1 = nn.Sequential(*list(resnet.children())[:4])
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c3, c4, c5]
