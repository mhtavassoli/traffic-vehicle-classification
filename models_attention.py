"""models_attention.py — Multi-Region Attention + Fuzzy Head for anti-shortcut.

Motivation (Phase 11):
    Grad-CAM analysis showed the model attends to the LOWER part of images
    (license plate, bumper) rather than ambulance-specific features (blue
    flashers, AMBULANCE text). This is a classic "shortcut learning" failure.

Solution:
    1. MultiRegionAttention: extract features from 4 regions (top, middle,
       bottom, full) and combine with learned attention weights.
    2. FuzzyClassifier: replace softmax with Gaussian membership functions
       so predictions can be uncertain (sum of memberships ≠ 1).

Reference:
    - Geirhos et al., "Shortcut Learning in Deep Neural Networks", Nature MI 2020.
    - Zadeh, "Fuzzy Sets", Information and Control 1965.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

from config import NUM_CLASSES, IMG_SIZE


# ============================================================
# 1) Multi-Region Attention
# ============================================================
class MultiRegionAttention(nn.Module):
    """
    Split feature map into 4 regions and combine via attention.

    Regions:
        - top 1/3    → expects flashers, AMBULANCE text (upper body)
        - middle 1/3 → expects body, windows
        - bottom 1/3 → expects license plate, bumper (shortcut zone!)
        - full       → global context

    Attention weights are learned — the model must decide how much to
    weight each region. An entropy penalty encourages diffuse attention
    (preventing collapse to a single region).
    """

    def __init__(self, in_features=512, n_regions=4):
        super().__init__()
        self.n_regions = n_regions
        self.in_features = in_features

        # Attention scorer: takes pooled region features → scalar logit
        self.attention = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Normalization after combination
        self.norm = nn.LayerNorm(in_features)

    def _pool_regions(self, feature_map):
        """
        feature_map: (B, C, H, W)
        returns: (B, n_regions, C) — one feature vector per region
        """
        B, C, H, W = feature_map.shape
        # Adaptive split into thirds (H must be divisible by 3)
        # ResNet18 at 128×128 → 4×4 feature map → H=4. We'll split as 1/1/2.
        top_h = max(1, H // 3)
        mid_h = max(1, H // 3)
        bot_h = H - top_h - mid_h

        top = feature_map[:, :, :top_h, :].mean(dim=(2, 3))         # (B, C)
        mid = feature_map[:, :, top_h:top_h + mid_h, :].mean(dim=(2, 3))
        bot = feature_map[:, :, top_h + mid_h:, :].mean(dim=(2, 3))
        full = feature_map.mean(dim=(2, 3))

        return torch.stack([top, mid, bot, full], dim=1)  # (B, 4, C)

    def forward(self, feature_map, return_attention=False):
        regions = self._pool_regions(feature_map)          # (B, 4, C)

        # Compute attention logits per region
        logits = self.attention(regions).squeeze(-1)       # (B, 4)
        weights = F.softmax(logits, dim=1)                 # (B, 4)

        # Weighted sum over regions
        combined = (weights.unsqueeze(-1) * regions).sum(dim=1)  # (B, C)
        combined = self.norm(combined)

        if return_attention:
            return combined, weights
        return combined


# ============================================================
# 2) Fuzzy Classification Head
# ============================================================
class FuzzyClassifier(nn.Module):
    """
    Replace softmax with Gaussian membership functions.

    For each class c, learn:
        - center_c ∈ R^D
        - sigma_c  > 0  (learnable bandwidth)

    Membership: mu_c(x) = exp(-||x - center_c||^2 / (2 * sigma_c^2))

    Output is NOT normalized — it expresses "how much this sample looks
    like class c" independent of other classes. This allows uncertain
    predictions (all memberships low) and multi-class ambiguity
    (two memberships high).
    """

    def __init__(self, in_features=512, num_classes=NUM_CLASSES):
        super().__init__()
        # Make the centers larger (use 1.0 instead of 0.01)
        # NO Initialize centers to small random values
        # Very small initial value to prevent a large gap at the beginning
        self.centers = nn.Parameter(torch.randn(num_classes, in_features) * 0.01)
        # nn.init.kaiming_normal_(self.centers, mode='fan_out', nonlinearity='relu')
        # Log-sigma for numerical stability (sigma = exp(log_sigma))
        # Initialize log_sigma with a reasonable value instead of zero
        # Since the distances are normalized, sigma=2 is a reasonable choice
        self.log_sigma = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, return_distance=False):
        """
        x: (B, D)
        returns: memberships (B, C) — values in (0, 1], not normalized
        """
        
        # 1. L2 Normalization of the Feature Vector
        x_norm = F.normalize(x, p=2, dim=1)
        # 2. L2 Normalization of Centers
        centers_norm = F.normalize(self.centers, p=2, dim=1)
        
        # Now the distances are between 0 and 2 (not 1000)
        # Pairwise distances: (B, C)
        dist_sq = torch.cdist(x_norm, self.centers, p=2).pow(2)
        
        sigma = torch.exp(self.log_sigma).unsqueeze(0)      # (1, C)
        mu = torch.exp(-dist_sq / (2 * sigma.pow(2) + 1e-8))
        if return_distance:
            return mu, dist_sq
        return mu

    def predict(self, x):
        """Return hard prediction (argmax) and confidence (max membership)."""
        mu = self.forward(x)
        conf, pred = mu.max(dim=1)
        return pred, conf


# ============================================================
# 3) Full model: ResNet18 + MultiRegionAttention + FuzzyClassifier
# ============================================================
class AntiShortcutNet(nn.Module):
    """
    End-to-end model that avoids the bottom-region shortcut.
    """

    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super().__init__()
        # Backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Attention over regions
        self.multi_region = MultiRegionAttention(in_features=512, n_regions=4)

        # Fuzzy classification head
        self.classifier = FuzzyClassifier(in_features=512, num_classes=num_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            for name, p in self.named_parameters():
                if name.startswith(("conv1", "bn1", "layer1", "layer2", "layer3")):
                    p.requires_grad = False

    def features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x  # (B, 512, H', W')

    def forward(self, x, return_attention=False):
        fmap = self.features(x)
        if return_attention:
            feats, attn = self.multi_region(fmap, return_attention=True)
            memberships = self.classifier(feats)
            return {
                "features": feats,
                "attention": attn,
                "memberships": memberships,
            }
        feats = self.multi_region(fmap)
        memberships = self.classifier(feats)
        return memberships


# ============================================================
# 4) Loss functions
# ============================================================
class FuzzyLoss(nn.Module):
    """
    Loss for fuzzy classifier:
        L = CE_loss + alpha * attention_entropy + beta * sigma_regularization

    CE_loss: standard cross-entropy on log(mu) — works because mu ∈ (0, 1].
    attention_entropy: encourages diffuse attention (not collapse to 1 region).
    sigma_regularization: prevents sigma from becoming too small or too large.
    """

    def __init__(self, alpha=0.0, beta=0.1, margin=0.5):
        super().__init__()
        self.alpha = alpha # For entropy (default set to zero)
        self.beta = beta # For the lower-region penalty (Shortcut Penalty)
        self.margin = margin # margin of safety

    def forward(self, memberships, attention, targets):
        # Cross-entropy on log-membership (mu > 0)
        # 1. Classification Loss
        log_probs = torch.log(memberships + 1e-8)
        loss_ce = F.nll_loss(log_probs, targets)

        # Attention entropy — encourage uniform-ish attention
        # 2. Attention Entropy (Optional – has no effect if alpha=0)
        attn_entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=1).mean()

        # 3. Anti-Shortcut Penalty (The most important part)
        # Indices: 0=TOP, 1=MIDDLE, 2=BOTTOM, 3=FULL
        # We want to place more emphasis on TOP than on BOTTOM. 
        # If BOTTOM > TOP, this term becomes positive and increases the Loss. 
        # Consequently, to minimize the Loss, the optimizer is compelled to increase TOP and decrease BOTTOM.
        # Goal: TOP - BOTTOM >= margin
        # If TOP - BOTTOM >= 0.5, the penalty becomes zero, and the model is unconstrained. 
        # If TOP - BOTTOM < 0.5, the penalty becomes positive, increasing the loss.
        top_weight = attention[:, 0]
        bottom_weight = attention[:, 2]
        shortcut_penalty = F.relu(bottom_weight - top_weight + self.margin).mean()

        # Sigma regularization — keep bandwidth reasonable
        # The final composition
        loss = loss_ce + self.alpha * attn_entropy + self.beta * shortcut_penalty
        return loss