"""models_mimo.py — MIMO-Inspired Multi-Expert Architecture."""
import torch, torch.nn as nn, torch.nn.functional as F
from config import NUM_CLASSES


class MIMOClassifier(nn.Module):
    """
    MIMO-Inspired Multi-Expert Classifier.

    Instead of a single linear head, we use FOUR experts that each
    capture a different aspect of the input:

    1. Classification Expert: standard linear classifier
    2. Prototype Expert: distance to class prototypes
    3. Reconstruction Expert: autoencoder reconstruction error
    4. Confidence Expert: predicts its own reliability
    """
    def __init__(self, backbone, in_features=512, num_classes=NUM_CLASSES,
                 n_prototypes_per_class=1):
        super().__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.num_classes = num_classes

        # ---- Expert 1: Standard classification ----
        self.classifier = nn.Linear(in_features, num_classes)

        # ---- Expert 2: Prototype-based ----
        # Learnable prototypes: (C, K, D) where K = prototypes per class
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, n_prototypes_per_class, in_features) * 0.01
        )

        # ---- Expert 3: Reconstruction (compact autoencoder) ----
        self.reconstructor = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, in_features),
        )

        # ---- Expert 4: Confidence predictor ----
        # Predicts whether the classifier's prediction is reliable
        self.confidence_head = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x, return_all=False):
        # Extract features
        feats = self._extract_features(x)

        # Expert 1: logits
        logits = self.classifier(feats)

        if not return_all:
            return logits

        # Expert 2: prototype distances
        # feats: (B, D), prototypes: (C, K, D)
        # dist: (B, C, K) → min over K → (B, C)
        diffs = feats.unsqueeze(1).unsqueeze(1) - self.prototypes.unsqueeze(0)
        # (B, C, K, D)
        proto_dists = diffs.norm(dim=-1)  # (B, C, K)
        proto_dists = proto_dists.min(dim=-1).values  # (B, C)

        # Expert 3: reconstruction error
        recon = self.reconstructor(feats)
        recon_err = (feats - recon).norm(dim=1, keepdim=True)  # (B, 1)

        # Expert 4: confidence
        confidence = self.confidence_head(feats)  # (B, 1)

        return {
            "logits": logits,
            "prototype_dists": proto_dists,
            "recon_error": recon_err,
            "confidence": confidence,
            "features": feats,
        }

    def _extract_features(self, x):
        """Extract penultimate features from the backbone."""
        net = self.backbone
        x = net.conv1(x); x = net.bn1(x); x = net.relu(x); x = net.maxpool(x)
        x = net.layer1(x); x = net.layer2(x); x = net.layer3(x); x = net.layer4(x)
        x = net.avgpool(x)
        return x.flatten(1)


class MIMOLoss(nn.Module):
    """
    Multi-task loss combining all four experts.

    L_total = α·L_classification + β·L_prototype + γ·L_reconstruction + δ·L_confidence
    """
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.3, delta=0.3,
                 confidence_threshold=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.conf_thresh = confidence_threshold

    def forward(self, outputs, targets):
        # 1. Classification loss (standard CE)
        l_cls = F.cross_entropy(outputs["logits"], targets)

        # 2. Prototype loss: features should be close to their class prototype
        # For each sample, find min distance to prototype of its true class
        true_proto_dists = outputs["prototype_dists"][
            torch.arange(len(targets)), targets
        ]  # (B,)
        l_proto = true_proto_dists.mean()

        # 3. Reconstruction loss: features should be reconstructible
        l_recon = outputs["recon_error"].mean()

        # 4. Confidence loss: predict whether classification is correct
        with torch.no_grad():
            preds = outputs["logits"].argmax(1)
            is_correct = (preds == targets).float().unsqueeze(1)  # (B, 1)
        l_conf = F.binary_cross_entropy(outputs["confidence"], is_correct)

        return (self.alpha * l_cls
                + self.beta * l_proto
                + self.gamma * l_recon
                + self.delta * l_conf)


def build_mimo_classifier(num_classes=NUM_CLASSES):
    """Build a MIMO classifier with a pretrained ResNet18 backbone."""
    from torchvision import models
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Keep only the feature extractor part
    backbone.fc = nn.Identity()
    return MIMOClassifier(backbone, in_features=512, num_classes=num_classes)