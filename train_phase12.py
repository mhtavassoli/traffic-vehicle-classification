"""train_phase12.py — Train the full Phase12Net (Topology + MIMO)."""
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TRAIN_DIR,
    VAL_FRACTION, SEED, BATCH_SIZE,
)
from data import (
    IndexedImageDataset, train_transform, eval_transform,
    stratified_val_split,
)
from audit import list_split
from models_phase12 import Phase12Net
# from models_attention import FuzzyLoss
from models_phase12 import Phase12Net, MIMOLoss # add MIMOLoss


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    attn_w_sum, topo_w_sum = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        out = model(x, return_all=True)
        loss = criterion(out["memberships"], out["attention"], out["expert_weights"], y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        preds = out["memberships"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)
        
        # Recording weights for display
        attn_w_sum += out["expert_weights"][:, 0].sum().item()
        topo_w_sum += out["expert_weights"][:, 1].sum().item()
        
    avg_attn = attn_w_sum / total
    avg_topo = topo_w_sum / total

    return total_loss / total, correct / total, avg_attn, avg_topo


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, return_all=True)
        preds = out["memberships"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    print("=" * 70)
    print("PHASE 12: Full MIMO (Topology + Attention + Fuzzy)")
    print("=" * 70)

    # Data
    items = list_split(TRAIN_DIR)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    items = [(p, c) for p, c in items if c in c2i]
    tr_items, va_items, _, _ = stratified_val_split(items, c2i, VAL_FRACTION, SEED)

    ds_tr = IndexedImageDataset(tr_items, c2i, train_transform(True))
    ds_va = IndexedImageDataset(va_items, c2i, eval_transform())
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=64, shuffle=False,
                           num_workers=2, pin_memory=True)

    # Model
    model = Phase12Net(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Optionally Warm-start backbone from Phase 11
    ckpt_path = os.path.join(ARTIFACTS, "antishortcut.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        # Filter backbone + attention_expert keys
        phase12_state = model.state_dict()
        for k, v in ckpt["state_dict"].items():
            if k in phase12_state and phase12_state[k].shape == v.shape:
                phase12_state[k] = v
        model.load_state_dict(phase12_state)
        print("  ✓ Warm-started from Phase 11 checkpoint.")

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    EPOCHS = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5,
    )
    # criterion = FuzzyLoss(alpha=0.0, beta=0.1) # Keep shortcut penalty
    criterion = MIMOLoss(beta_shortcut=0.1, gamma_balance=0.5)

    # Training
    best_acc = 0.0
    history = []
    t0 = time.time()

    print("\n  Training ...")
    for epoch in range(1, EPOCHS + 1):
        # tr_loss, tr_acc = train_epoch(model, loader_tr, optimizer, criterion, DEVICE)
        tr_loss, tr_acc, w_attn, w_topo = train_epoch(model, loader_tr, optimizer, criterion, DEVICE)
        va_acc = evaluate(model, loader_va, DEVICE)
        scheduler.step()

        print(f"ep {epoch:02d} | loss {tr_loss:.4f} | "
              f"tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f} | "
              f"weights: Attn={w_attn:.2f}, Topo={w_topo:.2f}")

        history.append({"epoch": epoch, "tr_loss": tr_loss, "va_acc": va_acc})

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": c2i,
                "arch": "phase12_mimo_topology",
                "config": {"epoch": epoch, "val_acc": va_acc},
            }, os.path.join(ARTIFACTS, "phase12.pt"))

    elapsed = time.time() - t0
    print(f"\n✓ Best val acc: {best_acc:.4f} ({elapsed/60:.1f} min)")

    with open(os.path.join(ARTIFACTS, "phase12_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()