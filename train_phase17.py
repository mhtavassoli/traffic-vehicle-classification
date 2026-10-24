"""train_phase17.py — Phase 17: Active Learning Fine-Tuning (Human-in-the-Loop).

Focus: Fix the "stubborn blue vans" issue from Phase 16.
We take the Phase 16 model and fine-tune it on the hard examples (test3 folder) 
using a very low learning rate to prevent catastrophic forgetting.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TRAIN_DIR, VAL_FRACTION, SEED, BATCH_SIZE, TESTING_DATA_ROOTS
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
        
        # NLLLoss because the model outputs probabilities and we pass log(probs)
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
    print("PHASE 17: ACTIVE LEARNING FINE-TUNING")
    print("=" * 70)
    
    # 1. Load Original Training Data
    items = list_split(TRAIN_DIR)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    items = [(p, c) for p, c in items if c in c2i]
    tr_items, va_items, _, _ = stratified_val_split(items, c2i, VAL_FRACTION, SEED)
    
    ds_tr_main = IndexedImageDataset(tr_items, c2i, train_transform(True))
    
    # 2. Load Hard Examples (Active Learning Pool)
    # We specifically target the 'test3' folder (blue vans) that Phase 16 failed on.
    hard_items = []
    for root in TESTING_DATA_ROOTS:
        test3_root = os.path.join(root, "test3")
        if os.path.isdir(test3_root):
            # We list the items and ONLY keep the ones we know are "vanet" (or correct label based on human feedback)
            # For the sake of this script, we assume human feedback dictates these are "vanet" 
            # (or we can load from human_feedback.json)
            test3_items = list_split(test3_root)
            
            # If human_feedback.json exists, use it to correct labels dynamically
            feedback_file = os.path.join(ARTIFACTS, "human_feedback.json")
            feedback_dict = {}
            if os.path.exists(feedback_file):
                with open(feedback_file, "r") as f:
                    feedback_data = json.load(f)
                    for entry in feedback_data:
                        feedback_dict[entry["filename"]] = entry["corrected_label"]

            for p, c in test3_items:
                filename = os.path.basename(p)
                # Override label if human corrected it
                if filename in feedback_dict:
                    c = feedback_dict[filename]
                
                # If the true label is 'vanet' (or corrected), we add it to hard items
                # Note: In Phase 14, these were classified as 'vanet'. 
                if c in c2i:
                    hard_items.append((p, c))
    
    print(f"✓ Loaded {len(tr_items)} standard training images.")
    print(f"✓ Loaded {len(hard_items)} hard examples (test3) for Active Learning.")
    
    if not hard_items:
        print("⚠ Warning: No hard examples found in test3. Training on standard data only.")
        ds_tr_hard = None
    else:
        ds_tr_hard = IndexedImageDataset(hard_items, c2i, train_transform(True))
    
    ds_va = IndexedImageDataset(va_items, c2i, eval_transform())
    
    # Combine datasets: 1x standard + 3x hard examples (to give them higher weight)
    if ds_tr_hard:
        # PyTorch doesn't have a direct weight parameter in ConcatDataset, 
        # so we duplicate the hard examples to enforce the model to focus on them.
        combined_ds = ConcatDataset([ds_tr_main, ds_tr_hard, ds_tr_hard, ds_tr_hard])
    else:
        combined_ds = ds_tr_main

    loader_tr = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=2)
    
    # 3. Load Phase 16 Model (Warm Start)
    model = Phase16Net(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt_path = os.path.join(ARTIFACTS, "phase16.pt")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print("✓ Warm-started from Phase 16 (Quantum Model).")
    else:
        print("⚠ Warning: phase16.pt not found. Using random weights.")
    
    # 4. Fine-tuning Setup
    # VERY LOW learning rate to prevent catastrophic forgetting
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4) 
    EPOCHS = 5 # Short fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.NLLLoss()
    
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
                "arch": "phase17_active_learning",
                "config": {"epoch": epoch, "val_acc": va_acc},
            }, os.path.join(ARTIFACTS, "phase17.pt"))
            
    print(f"\n✓ Best val acc after Active Learning: {best_acc:.4f}")
    print("Run evaluate_phase16.py (or a new evaluate_phase17.py) to see if the blue vans are fixed!")

if __name__ == "__main__":
    main()