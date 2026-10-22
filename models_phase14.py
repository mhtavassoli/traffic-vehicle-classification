"""models_phase14.py — True Topological Expert using Persistent Homology.

Replaces the Sobel-based Topology Expert with a real TDA-based one.
Extracts Persistence Images from the feature map, which capture the
"shape" of the vehicle (e.g., number of loops, structural longevity).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceImage
from models_phase13 import Phase13Net, Phase13Loss

class TopologicalExpertTDA(nn.Module):
    def __init__(self, in_channels=512, out_features=512):
        super().__init__()
        # We apply TDA on the spatial feature map (H x W)
        self.cubical = CubicalPersistence(homology_dimensions=[0, 1])
        self.pi = PersistenceImage(n_bins=10)
        
        # The Persistence Image produces a flat vector. We project it.
        self.proj = nn.Sequential(
            nn.Linear(200, out_features), 
            # FIXED: Replaced BatchNorm1d with LayerNorm to support batch sizes of 1 (prevents crashes on last batch)
            nn.LayerNorm(out_features), 
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(out_features)

    def forward(self, feature_map):
        # feature_map: (B, 512, H, W)
        B, C, H, W = feature_map.shape
        
        # Reduce channels to 1 by taking the mean (makes TDA computationally feasible)
        spatial_map = feature_map.mean(dim=1) # (B, H, W)
        
        # Convert to numpy for gtda processing
        spatial_np = spatial_map.detach().cpu().numpy()
        
        # Compute Persistence Diagrams and Images
        diagrams = self.cubical.fit_transform(spatial_np)
        pis = self.pi.fit_transform(diagrams) 
        
        # Flatten PI and project to 512-D
        pis_flat = torch.tensor(pis.reshape(B, -1), dtype=torch.float32).to(feature_map.device)
        
        # Handle variable size (pad or truncate to 200)
        if pis_flat.size(1) < 200:
            pis_flat = F.pad(pis_flat, (0, 200 - pis_flat.size(1)))
        else:
            pis_flat = pis_flat[:, :200]
            
        return self.norm(self.proj(pis_flat))

# Phase14Net simply swaps the expert
class Phase14Net(Phase13Net):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)
        # Swap the Sobel expert with the TDA expert
        self.topology_expert = TopologicalExpertTDA(in_channels=512, out_features=512)
