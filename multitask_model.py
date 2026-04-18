import torch.nn as nn
from models.backbone import Backbone
from models.fpn import FPN
from models.detector import DetectionHead
from models.roi_heads import AttributeHeads

class MultiTaskFaceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.fpn = FPN()
        self.detector = DetectionHead()
        self.attributes = AttributeHeads()

    def forward(self, images, proposals=None):
        features = self.backbone(images)
        fpn_feats = self.fpn(features)

        cls_logits, bbox_regs = self.detector(fpn_feats)

        outputs = {
            "detection_cls": cls_logits,
            "detection_bbox": bbox_regs
        }

        if proposals is not None:
            age_logits, emotion_logits = self.attributes(fpn_feats[0], proposals)
            outputs["age"] = age_logits
            outputs["emotion"] = emotion_logits

        return outputs
