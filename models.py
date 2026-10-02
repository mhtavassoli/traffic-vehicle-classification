"""CNN baseline + ResNet18 transfer-learning wrappers."""
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

class TrafficCNN(nn.Module):
    """Simple 4-block CNN. Easy to ablate (pool type, dropout)."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, pool_type="max"):
        super().__init__()
        Pool = nn.MaxPool2d if pool_type == "max" else nn.AvgPool2d
        self.features = nn.Sequential(
            nn.Conv2d(3,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), Pool(2),
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), Pool(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), Pool(2),
            nn.Conv2d(128,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), Pool(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

def build_resnet18(mode="feature_extract", num_classes=NUM_CLASSES):
    """
    mode='feature_extract' -> freeze backbone, train head only.
    mode='fine_tune'       -> unfreeze layer4 + head; lower LR for pretrained params.
    """
    assert mode in {"feature_extract", "fine_tune"}
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the classifier head for our 8 classes.
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    if mode == "feature_extract":
        for p in net.parameters(): p.requires_grad = False
        for p in net.fc.parameters(): p.requires_grad = True
    else:  # fine_tune
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith("layer4") or name.startswith("fc")
    return net

def resnet_param_groups(net, lr_head=1e-3, lr_backbone=1e-4, wd=0.0):
    """Different LR for pretrained vs new parameters (a standard fine-tune trick)."""
    head, backbone = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad: continue
        (head if name.startswith("fc") else backbone).append(p)
    groups = [{"params": head, "lr": lr_head}]
    if backbone: groups.append({"params": backbone, "lr": lr_backbone})
    return groups