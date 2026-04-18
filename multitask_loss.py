import torch
import torch.nn as nn


class AdaptiveMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, det_loss, age_loss, emo_loss):
        loss = (
            torch.exp(-self.log_vars[0]) * det_loss + self.log_vars[0] +
            torch.exp(-self.log_vars[1]) * age_loss + self.log_vars[1] +
            torch.exp(-self.log_vars[2]) * emo_loss + self.log_vars[2]
        )
        return loss


def compute_multitask_loss(pred_age, true_age, pred_emotion, true_emotion, 
                           lambda_age=1.0, lambda_emotion=1.0):
    """
    Compute multi-task loss for age and emotion prediction
    
    Args:
        pred_age: Predicted age logits [batch_size, num_age_bins]
        true_age: Ground truth age labels [batch_size]
        pred_emotion: Predicted emotion logits [batch_size, num_emotions]
        true_emotion: Ground truth emotion labels [batch_size]
        lambda_age: Weight for age loss
        lambda_emotion: Weight for emotion loss
    
    Returns:
        total_loss: Combined weighted loss
        age_loss: Age classification loss
        emotion_loss: Emotion classification loss
    """
    age_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()
    
    age_loss = age_criterion(pred_age, true_age)
    emotion_loss = emotion_criterion(pred_emotion, true_emotion)
    
    total_loss = lambda_age * age_loss + lambda_emotion * emotion_loss
    
    return total_loss, age_loss, emotion_loss
