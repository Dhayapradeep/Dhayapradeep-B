import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_detection_loss(outputs, targets):
    """
    Compute detection loss for face detection
    
    Args:
        outputs: Model outputs containing detection predictions
        targets: Ground truth targets
    
    Returns:
        Detection loss value
    """
    if 'detection_cls' not in outputs or 'detection_bbox' not in outputs:
        return torch.tensor(0.0, requires_grad=True)
    
    # Classification loss (binary cross entropy for face/no-face)
    cls_loss = F.binary_cross_entropy_with_logits(
        outputs['detection_cls'], 
        targets.get('cls_targets', torch.zeros_like(outputs['detection_cls']))
    )
    
    # Bounding box regression loss (smooth L1 loss)
    bbox_loss = F.smooth_l1_loss(
        outputs['detection_bbox'], 
        targets.get('bbox_targets', torch.zeros_like(outputs['detection_bbox']))
    )
    
    # Combine losses
    total_loss = cls_loss + bbox_loss
    
    return total_loss