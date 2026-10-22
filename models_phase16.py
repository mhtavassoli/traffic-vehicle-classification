"""models_phase16.py — Quantum-Inspired Neural Network (QINN).

Replaces the Fuzzy Classifier with a Complex-Valued Quantum Head.
Uses the Born Rule (Probability = |amplitude|^2) to compute class probabilities.
This naturally simulates quantum superposition and prevents the probability collapse.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_phase14 import Phase14Net

class QuantumHead(nn.Module):
    def __init__(self, in_features=512, num_classes=8):
        super().__init__()
        # In quantum mechanics, the state is a complex vector (a + bi).
        # We simulate this with two real linear layers: one for the real part, one for the imaginary part.
        self.linear_real = nn.Linear(in_features, num_classes)
        self.linear_imag = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 1. Compute the complex amplitude (superposition state)
        z_real = self.linear_real(x)
        z_imag = self.linear_imag(x)
        
        # 2. Apply the Born Rule: P = |z|^2 = a^2 + b^2
        # This is the quantum measurement step.
        probs = z_real**2 + z_imag**2
        
        # 3. Normalize to get a valid probability distribution (sum to 1)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

class Phase16Net(Phase14Net):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)
        # Replace the Fuzzy Classifier with the Quantum Head
        self.classifier = QuantumHead(in_features=512, num_classes=num_classes)

    def forward(self, x, return_all=False):
        # We need to override forward to handle the new classifier
        fmap = self.features(x)
        
        # Attention Expert
        attn_feats, attn_weights = self.attention_expert(fmap, return_attention=True)
        
        # Topology Expert (TDA)
        topo_feats = self.topology_expert(fmap)
        
        # Combine via MIMO
        combined_feats, expert_weights = self.combiner([attn_feats, topo_feats])
        
        # Quantum Classification
        probs = self.classifier(combined_feats)
        
        if return_all:
            return {
                "memberships": probs, # In Phase 16, this is now a valid probability distribution
                "attention": attn_weights,
                "expert_weights": expert_weights
            }
        return probs