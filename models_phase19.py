"""models_phase19.py — Calibrated Quantum Head (Temperature Scaling).

Fixes the overconfidence issue of the Born Rule by introducing a learnable
temperature parameter.
"""
import torch
import torch.nn as nn
from models_phase16 import Phase16Net

class CalibratedQuantumHead(nn.Module):
    def __init__(self, in_features=512, num_classes=8):
        super().__init__()
        self.linear_real = nn.Linear(in_features, num_classes)
        self.linear_imag = nn.Linear(in_features, num_classes)
        # Temperature parameter (learnable). Initialized to 1.0 (standard Born rule).
        # If T > 1, the distribution becomes softer (less confident).
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z_real = self.linear_real(x)
        z_imag = self.linear_imag(x)
        
        # Amplitude magnitude
        amplitude_sq = z_real**2 + z_imag**2
        
        # Apply Temperature Scaling
        # Clamp temperature to avoid division by zero or negative values
        T = torch.clamp(self.temperature, min=0.1)
        
        # Calibrated Born Rule: P = |z|^(2/T) / sum(...)
        probs = torch.pow(amplitude_sq, 1.0 / T)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

class Phase19Net(Phase16Net):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)
        # Replace the standard QuantumHead with the Calibrated one
        self.classifier = CalibratedQuantumHead(in_features=512, num_classes=num_classes)