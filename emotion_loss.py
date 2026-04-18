import torch
import torch.nn as nn
import torch.nn.functional as F

def emotion_loss_fn(emotion_pred, emotion_target):
    """
    Emotion classification loss using cross entropy
    
    Args:
        emotion_pred: Predicted emotion logits (batch_size, num_emotions)
        emotion_target: Target emotion labels (batch_size,) or one-hot (batch_size, num_emotions)
    
    Returns:
        Emotion classification loss
    """
    if emotion_pred is None:
        return torch.tensor(0.0, requires_grad=True)
    
    # Handle different target formats
    if emotion_target.dim() == 1:
        # Target is class indices
        loss = F.cross_entropy(emotion_pred, emotion_target.long())
    else:
        # Target is one-hot or soft labels
        loss = F.cross_entropy(emotion_pred, emotion_target)
    
    return loss

def focal_emotion_loss(emotion_pred, emotion_target, alpha=1.0, gamma=2.0):
    """
    Focal loss for emotion classification to handle class imbalance
    
    Args:
        emotion_pred: Predicted emotion logits (batch_size, num_emotions)
        emotion_target: Target emotion labels (batch_size,)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    if emotion_pred is None:
        return torch.tensor(0.0, requires_grad=True)
    
    ce_loss = F.cross_entropy(emotion_pred, emotion_target.long(), reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()

# Emotion class names for reference
EMOTION_CLASSES = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
]