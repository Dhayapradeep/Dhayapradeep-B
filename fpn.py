import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, channels=[512, 1024, 2048], out_channels=256):
        super().__init__()

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in channels
        ])

        self.output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in channels
        ])

    def forward(self, features):
        p5 = self.lateral[2](features[2])
        p4 = self.lateral[1](features[1]) + F.interpolate(p5, scale_factor=2)
        p3 = self.lateral[0](features[0]) + F.interpolate(p4, scale_factor=2)

        return [
            self.output[0](p3),
            self.output[1](p4),
            self.output[2](p5)
        ]
