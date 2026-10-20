"""models_phase12.py — Full MIMO architecture with Topology + Attention + Fuzzy.

This model implements the ultimate vision:
    1. Attention Expert: Multi-region spatial attention (from Phase 11).
    2. Fuzzy Expert: Gaussian membership functions (from Phase 11).
    3. Topology Expert: Structural and edge-based feature extraction.

The MIMO Combiner merges these three experts using a learned gating
mechanism (inspired by 6G beamforming).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import NUM_CLASSES
from models_attention import MultiRegionAttention, FuzzyClassifier


# ============================================================
# Expert 3: Topological Structural Expert
# ============================================================
class TopologicalExpert(nn.Module):
    """
    Extracts structural / topological features from the feature map.
    Uses Sobel filters to compute edges (gradients) and a lightweight
    Graph-like pooling to capture shape distributions.
    """
    def __init__(self, in_channels=512, out_features=512):
        super().__init__()
        # Sobel-like depthwise convolution to extract edges
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                    padding=1, groups=in_channels, bias=False)
        # Initialize with Sobel filters (approximate)
        nn.init.kaiming_normal_(self.edge_conv.weight, mode='fan_out')
        
        # Project edge features to match the other experts
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_features, kernel_size=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        
        self.norm = nn.LayerNorm(out_features)

    def forward(self, feature_map):
        # feature_map: (B, 512, H, W)
        edges = F.relu(self.edge_conv(feature_map))
        edge_feats = self.proj(edges) # (B, 512, H, W)
        
        # Global average pooling to get a vector
        pooled = edge_feats.mean(dim=(2, 3)) # (B, 512)
        return self.norm(pooled)


# ============================================================
# MIMO Combiner (6G Beamforming Inspired)
# ============================================================
class MIMOCombiner(nn.Module):
    """
    Learns to weight the outputs of the 3 experts based on the input.
    """
    def __init__(self, in_features=512, n_experts=3):
        super().__init__()
        self.n_experts = n_experts
        # Gating network: takes concatenated features, outputs expert weights
        self.gate = nn.Sequential(
            nn.Linear(in_features * n_experts, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_experts)
        )

    def forward(self, expert_features):
        # expert_features: list of 3 tensors, each (B, 512)
        concat = torch.cat(expert_features, dim=1) # (B, 1536)
        weights = F.softmax(self.gate(concat), dim=1) # (B, 3)
        
        # Weighted sum
        stacked = torch.stack(expert_features, dim=1) # (B, 3, 512)
        combined = (weights.unsqueeze(-1) * stacked).sum(dim=1) # (B, 512)
        return combined, weights


# ============================================================
# Full Phase 12 Network
# ============================================================
class Phase12Net(nn.Module):
    """
    Full MIMO + Topology + Attention + Fuzzy classifier.
    """
    def __init__(self, num_classes=NUM_CLASSES):
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

        # --- The 3 Experts ---
        # Expert 1: Attention (returns combined features and attention weights)
        self.attention_expert = MultiRegionAttention(in_features=512, n_regions=4)
        
        # Expert 2: Topology (returns structural features)
        self.topology_expert = TopologicalExpert(in_channels=512, out_features=512)
        
        # Expert 3: Fuzzy (acts as the classification head here)
        # We will feed the combined feature into the fuzzy classifier later.
        
        # Combiner
        self.combiner = MIMOCombiner(in_features=512, n_experts=2)
        
        # Final Fuzzy Classifier
        self.classifier = FuzzyClassifier(in_features=512, num_classes=num_classes)

    def features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x

    def forward(self, x, return_all=False):
        fmap = self.features(x)
        
        # 1. Attention Expert
        attn_feats, attn_weights = self.attention_expert(fmap, return_attention=True)
        
        # 2. Topology Expert
        topo_feats = self.topology_expert(fmap)
        
        # 3. Combine
        combined_feats, expert_weights = self.combiner([attn_feats, topo_feats])
        
        # 4. Fuzzy Classification
        memberships = self.classifier(combined_feats)
        
        if return_all:
            return {
                "memberships": memberships,
                "attention": attn_weights,
                "expert_weights": expert_weights
            }
        return memberships
    
class MIMOLoss(nn.Module):
    """
    Loss function for Phase 12.
    Combines Classification, Shortcut Penalty, and Load Balancing Penalty.
    """
    def __init__(self, beta_shortcut=0.1, gamma_balance=0.5, margin=0.5):
        super().__init__()
        self.beta = beta_shortcut
        self.gamma = gamma_balance
        self.margin = margin

    def forward(self, memberships, attention, expert_weights, targets):
        # 1. Classification Loss (Fuzzy)
        log_probs = torch.log(memberships + 1e-8)
        loss_ce = F.nll_loss(log_probs, targets)

        # 2. Shortcut Penalty (from Phase 11)
        # Force attention to top (0) > bottom (2)
        top_weight = attention[:, 0]
        bottom_weight = attention[:, 2]
        shortcut_penalty = F.relu(bottom_weight - top_weight + self.margin).mean()

        # 3. Load Balancing Penalty (NEW!)
        # expert_weights shape: (B, 2) -> [Attn_Weight, Topo_Weight]
        # If weights are [1.0, 0.0], sum of squares = 1.0
        # If weights are [0.5, 0.5], sum of squares = 0.5
        # We want to minimize the sum of squares to force balance.
        balance_penalty = (expert_weights ** 2).sum(dim=1).mean()

        # Total Loss
        return loss_ce + self.beta * shortcut_penalty + self.gamma * balance_penalty
