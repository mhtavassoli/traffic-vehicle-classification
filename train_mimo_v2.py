"""train_mimo_v2.py — Fixed MIMO training with warm-up + normalization + unfreezing."""
import os, json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import DataLoader
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, VAL_FRACTION, SEED)
from data import IndexedImageDataset, train_transform, eval_transform, stratified_val_split
from audit import list_split
from models_mimo import build_mimo_classifier


class NormalizedMIMOLoss(nn.Module):
    """Loss with running-normalized components to avoid domination."""
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.3, delta=0.3):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.register_buffer("cls_scale",   torch.tensor(1.0))
        self.register_buffer("proto_scale", torch.tensor(1.0))
        self.register_buffer("recon_scale", torch.tensor(1.0))
        self.register_buffer("conf_scale",  torch.tensor(1.0))
        # Warm-up flags
        self.use_proto = False
        self.use_recon = False
        self.use_conf  = False

    def set_phase(self, epoch):
        """Curriculum: activate experts gradually."""
        self.use_proto = epoch >= 4
        self.use_recon = epoch >= 8
        self.use_conf  = epoch >= 11

    def forward(self, outputs, targets):
        l_cls = F.cross_entropy(outputs["logits"], targets)

        # Update running scales
        if self.training:
            with torch.no_grad():
                self.cls_scale = 0.99 * self.cls_scale + 0.01 * l_cls.detach()
                self.proto_scale = (0.99 * self.proto_scale
                                    + 0.01 * outputs["prototype_dists"].mean().detach())
                self.recon_scale = (0.99 * self.recon_scale
                                    + 0.01 * outputs["recon_error"].mean().detach())
                self.conf_scale  = (0.99 * self.conf_scale
                                    + 0.01 * F.binary_cross_entropy(
                                        outputs["confidence"],
                                        torch.zeros_like(outputs["confidence"])).detach())

        total = self.alpha * l_cls / (self.cls_scale + 1e-8)

        if self.use_proto:
            true_proto = outputs["prototype_dists"][torch.arange(len(targets)), targets]
            l_proto = true_proto.mean()
            total = total + self.beta * l_proto / (self.proto_scale + 1e-8)

        if self.use_recon:
            l_recon = outputs["recon_error"].mean()
            total = total + self.gamma * l_recon / (self.recon_scale + 1e-8)

        if self.use_conf:
            with torch.no_grad():
                preds = outputs["logits"].argmax(1)
                is_correct = (preds == targets).float().unsqueeze(1)
            l_conf = F.binary_cross_entropy(outputs["confidence"], is_correct)
            total = total + self.delta * l_conf / (self.conf_scale + 1e-8)

        return total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, return_all=True)
        loss = criterion(outputs, y)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        preds = outputs["logits"].argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, return_all=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def unfreeze_layer4(model):
    """Progressive Unfreezing: allow layer4 + fc to train."""
    for name, p in model.backbone.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True
    print("  [unfreeze] layer4 + fc enabled")


def main():
    print("=" * 70)
    print("MIMO v2: Warm-up + Normalized Loss + Progressive Unfreezing")
    print("=" * 70)

    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    tr_items, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                                    VAL_FRACTION, SEED)

    ds_tr = IndexedImageDataset(tr_items, class_to_idx, train_transform(True))
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model = build_mimo_classifier(NUM_CLASSES).to(DEVICE)

    # Freeze backbone initially
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Optimizer with param groups for different LRs
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("backbone")],
         "lr": 1e-3},
        {"params": model.backbone.parameters(),
         "lr": 1e-4},
    ], weight_decay=1e-4)

    criterion = NormalizedMIMOLoss(alpha=1.0, beta=0.3, gamma=0.3, delta=0.3)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=25, eta_min=1e-5)

    best_acc = 0.0
    EPOCHS = 25

    for epoch in range(1, EPOCHS + 1):
        # Curriculum: set active experts
        criterion.set_phase(epoch)

        # Progressive unfreezing: at epoch 15, unfreeze layer4
        if epoch == 15:
            unfreeze_layer4(model)

        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        va_acc = evaluate(model, va_loader, DEVICE)
        scheduler.step()

        active = []
        if criterion.use_proto: active.append("P")
        if criterion.use_recon: active.append("R")
        if criterion.use_conf:  active.append("C")
        active_str = "".join(active) if active else "-"

        print(f"ep {epoch:02d} | experts[{active_str}] | "
              f"tr_loss {tr_loss:6.3f} | tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "arch": "mimo_resnet18_v2",
                "config": {"epoch": epoch, "val_acc": va_acc},
            }, os.path.join(ARTIFACTS, "mimo_resnet18_v2.pt"))

    print(f"\n✓ Best val acc: {best_acc:.4f}")
    print(f"  (Previous MIMO: 0.7750, resnet_ft_v2: 0.9091)")


if __name__ == "__main__":
    main()