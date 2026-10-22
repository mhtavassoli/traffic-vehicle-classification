"""models_phase13.py — Entropy-Regularized MIMO Combiner (Fixed Collapse)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_phase12 import Phase12Net

class Phase13Net(Phase12Net):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)

    def forward(self, x, return_all=False):
        out = super().forward(x, return_all=True)
        return out if return_all else out["memberships"]


class Phase13Loss(nn.Module):
    def __init__(self, beta_shortcut=0.1, gamma_entropy=0.001): # <--- 0.01 به جای 0.5
        super().__init__()
        self.beta_shortcut = beta_shortcut
        self.gamma_entropy = gamma_entropy

    def forward(self, memberships, attention, expert_weights, y_true):
        # 1. Classification Loss (Safe Implementation)
        # If 'memberships' is the output of a Softmax layer, this line is correct. 
        # If the output is from a Sigmoid layer, it is better to use F.binary_cross_entropy. 
        # Here, we use 'clamp' to avoid log(0) and log(1).        
        probs = memberships.clamp(min=1e-6, max=1.0 - 1e-6)
        log_probs = torch.log(probs)
        loss_cls = F.nll_loss(log_probs, y_true)

        # 2. Shortcut Penalty (Revised)
        top_w = attention[:, 0]
        bot_w = attention[:, 2]
        # We apply a penalty if Bot is greater than Top (with a margin of 0.1)
        loss_shortcut = F.relu(bot_w - top_w + 0.1).mean()

        # 3. Entropy Regularization (with a very low weight)
        log_expert = torch.log(expert_weights.clamp(min=1e-6))
        entropy = -(expert_weights * log_expert).sum(dim=1).mean()

        # Total Loss
        loss = loss_cls + self.beta_shortcut * loss_shortcut + self.gamma_entropy * entropy
        
        return loss, loss_cls.item(), loss_shortcut.item(), entropy.item()