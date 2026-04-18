import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class AgeHead(nn.Module):
    def __init__(self, in_channels, num_age_bins=8):
        super(AgeHead, self).__init__()
        
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.age_classifier = nn.Linear(256, num_age_bins)
    
    def forward(self, roi_features):
        """
        roi_features: [N, C]
        """
        x = F.relu(self.bn1(self.fc1(roi_features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        
        age_logits = self.age_classifier(x)
        return age_logits


class EmotionHead(nn.Module):
    def __init__(self, in_channels, num_emotions=7):
        super(EmotionHead, self).__init__()
        
        self.fc1 = nn.Linear(in_channels, 256)
        self.dropout = nn.Dropout(0.3)
        self.emotion_classifier = nn.Linear(256, num_emotions)
    
    def forward(self, roi_features):
        x = F.relu(self.fc1(roi_features))
        x = self.dropout(x)
        emotion_logits = self.emotion_classifier(x)
        return emotion_logits


class AttributeHeads(nn.Module):
    def __init__(self, num_emotions=7, num_age_bins=8):
        super().__init__()
        
        # ROI feature dimension after pooling
        in_channels = 256 * 7 * 7
        
        self.age_head = AgeHead(in_channels, num_age_bins)
        self.emotion_head = EmotionHead(in_channels, num_emotions)
    
    def forward(self, feature_map, proposals):
        roi_feats = roi_align(
            feature_map,
            proposals,
            output_size=(7, 7),
            spatial_scale=1/16
        )
        
        roi_feats = roi_feats.view(roi_feats.size(0), -1)
        
        age_logits = self.age_head(roi_feats)
        emotion_logits = self.emotion_head(roi_feats)
        
        return age_logits, emotion_logits


# Age and Emotion Labels for decoding predictions
AGE_LABELS = [
    "0-10",
    "11-20",
    "21-30",
    "31-40",
    "41-50",
    "51-60",
    "61-70",
    "71+"
]

EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", 
    "Happy", "Sad", "Surprise", "Neutral"
]


def decode_predictions(age_logits, emotion_logits):
    """
    Convert model predictions to human-readable labels
    
    Args:
        age_logits: Tensor of shape [batch_size, num_age_bins]
        emotion_logits: Tensor of shape [batch_size, num_emotions]
    
    Returns:
        age_label: String representing age range
        emotion_label: String representing emotion
    """
    age_pred = torch.argmax(age_logits, dim=1)
    emotion_pred = torch.argmax(emotion_logits, dim=1)
    
    return AGE_LABELS[age_pred.item()], EMOTION_LABELS[emotion_pred.item()]
