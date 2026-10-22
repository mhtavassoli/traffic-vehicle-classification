"""train_phase16.py — Train the Quantum-Inspired Network.

Warm-starts from Phase 14 and fine-tunes using standard Cross-Entropy.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TRAIN_DIR, VAL_FRACTION, SEED, BATCH_SIZE
from data import IndexedImageDataset, train_transform, eval_transform, stratified_val_split
from audit import list_split
from models_phase16 import Phase16Net

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        out = model(x, return_all=True)
        probs = out["memberships"]
        
        loss = criterion(torch.log(probs + 1e-8), y)
        loss.backward()
        optimizer.step()
        
        preds = probs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)
    return total_loss / total, correct / total

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
    print("PHASE 16: QUANTUM-INSPIRED NEURAL NETWORK (Born Rule)")
    print("=" * 70)
    
    items = list_split(TRAIN_DIR)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    items = [(p, c) for p, c in items if c in c2i]
    tr_items, va_items, _, _ = stratified_val_split(items, c2i, VAL_FRACTION, SEED)
    
    ds_tr = IndexedImageDataset(tr_items, c2i, train_transform(True))
    ds_va = IndexedImageDataset(va_items, c2i, eval_transform())
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=2)
    
    model = Phase16Net(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Warm-start from Phase 14
    ckpt_path = os.path.join(ARTIFACTS, "phase14.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model_state = model.state_dict()
        loaded = 0
        for k, v in ckpt["state_dict"].items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"✓ Warm-started from Phase 14 ({loaded} tensors).")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    EPOCHS = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.NLLLoss() # Because we pass log(probs)
    
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, loader_tr, optimizer, criterion, DEVICE)
        va_acc = evaluate(model, loader_va, DEVICE)
        scheduler.step()
        print(f"ep {epoch:02d} | tr_loss {tr_loss:.4f} | tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f}")
        
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": c2i,
                "arch": "phase16_quantum",
                "config": {"epoch": epoch, "val_acc": va_acc},
            }, os.path.join(ARTIFACTS, "phase16.pt"))
            
    print(f"\n✓ Best val acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()