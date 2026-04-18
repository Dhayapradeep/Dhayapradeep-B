import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def age_distribution_loss(age_pred, age_target):
    """
    Age distribution loss using KL divergence
    
    Args:
        age_pred: Predicted age distribution (batch_size, num_age_bins)
        age_target: Target age values or distributions
    
    Returns:
        Age loss value
    """
    if age_pred is None:
        return torch.tensor(0.0, requires_grad=True)
    
    # If target is a single age value, convert to distribution
    if age_target.dim() == 1:
        # Convert age to bin index (assuming 8 age bins: 0-10, 11-20, ..., 71+)
        age_bins = torch.tensor([10, 20, 30, 40, 50, 60, 70, 100], device=age_target.device)
        bin_indices = torch.searchsorted(age_bins, age_target.float())
        bin_indices = torch.clamp(bin_indices, 0, len(age_bins) - 1)
        
        # Create one-hot distribution
        age_target_dist = F.one_hot(bin_indices, num_classes=8).float()
    else:
        age_target_dist = age_target
    
    # Apply softmax to predictions to get probability distribution
    age_pred_dist = F.softmax(age_pred, dim=1)
    
    # KL divergence loss
    kl_loss = F.kl_div(
        F.log_softmax(age_pred, dim=1), 
        age_target_dist, 
        reduction='batchmean'
    )
    
    return kl_loss

def age_regression_loss(age_pred, age_target):
    """
    Simple regression loss for age prediction
    
    Args:
        age_pred: Predicted age values (batch_size, 1)
        age_target: Target age values (batch_size,)
    
    Returns:
        Age regression loss
    """
    if age_pred is None:
        return torch.tensor(0.0, requires_grad=True)
    
    return F.mse_loss(age_pred.squeeze(), age_target.float())