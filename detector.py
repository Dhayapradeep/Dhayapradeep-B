import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, num_anchors=2):
        super().__init__()

        self.cls_head = nn.Conv2d(256, num_anchors, 1)
        self.box_head = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, fpn_feats):
        cls_logits, bbox_regs = [], []

        for f in fpn_feats:
            cls_logits.append(self.cls_head(f))
            bbox_regs.append(self.box_head(f))

        return cls_logits, bbox_regs
