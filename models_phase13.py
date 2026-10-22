"""models_phase13.py — Ultimate Entropy-Regularized MIMO Combiner (Anti-Collapse)."""
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
    def __init__(self, beta_shortcut=0.1, gamma_entropy=0.1, delta_balance=0.5):
        super().__init__()
        self.beta_shortcut = beta_shortcut
        self.gamma_entropy = gamma_entropy
        self.delta_balance = delta_balance

    def forward(self, memberships, attention, expert_weights, y_true):
        # 1. Classification Loss (Stabilized via Softmax Normalization)
        # Prevents the model from outputting 1.0 for all classes (fixes a bug resulting in 22% accuracy)
        probs = memberships / (memberships.sum(dim=1, keepdim=True) + 1e-8)
        log_probs = torch.log(probs.clamp(min=1e-8, max=1.0))
        loss_cls = F.nll_loss(log_probs, y_true)

        # 2. Shortcut Penalty
        top_w = attention[:, 0]
        bot_w = attention[:, 2]
        loss_shortcut = F.relu(bot_w - top_w + 0.1).mean()

        # 3. Target Entropy (Preventing Winner-Takes-All Collapse)
        # We want the entropy to be around 0.5 (i.e., a peaked distribution, but not 1.0 or 0.0)
        log_expert = torch.log(expert_weights.clamp(min=1e-8))
        entropy = -(expert_weights * log_expert).sum(dim=1).mean()
        target_entropy = 0.5 
        loss_entropy = (entropy - target_entropy) ** 2

        # 4. Load Balancing Loss (Key MoE discovery)
        # If the average selection rate of an expert in a batch exceeds 0.5, a penalty is applied.
        avg_weights = expert_weights.mean(dim=0) # Average over the entire batch
        loss_balance = torch.sum(avg_weights ** 2) # Its minimum is 0.5 (0.5² + 0.5²)

        # Total Loss
        loss = loss_cls + \
               (self.beta_shortcut * loss_shortcut) + \
               (self.gamma_entropy * loss_entropy) + \
               (self.delta_balance * loss_balance)
        
        return loss, loss_cls.item(), loss_shortcut.item(), entropy.item(), loss_balance.item()