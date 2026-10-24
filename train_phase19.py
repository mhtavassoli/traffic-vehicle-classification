"""train_phase19.py — Train the Calibrated Quantum Head.

Freezes the backbone and experts, and only trains the temperature parameter
on the validation set to fix overconfidence.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TRAIN_DIR, VAL_FRACTION, SEED, BATCH_SIZE
from data import IndexedImageDataset, eval_transform, stratified_val_split
from audit import list_split
from models_phase19 import Phase19Net

def main():
    print("=" * 70)
    print("PHASE 19: QUANTUM CALIBRATION (Temperature Scaling)")
    print("=" * 70)
    
    items = list_split(TRAIN_DIR)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    items = [(p, c) for p, c in items if c in c2i]
    _, va_items, _, _ = stratified_val_split(items, c2i, VAL_FRACTION, SEED)
    
    ds_va = IndexedImageDataset(va_items, c2i, eval_transform())
    loader_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=2)
    
    model = Phase19Net(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Warm-start from Phase 17
    ckpt_path = os.path.join(ARTIFACTS, "phase17.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("✓ Warm-started from Phase 17.")
    
    # Freeze everything EXCEPT the temperature parameter
    for name, param in model.named_parameters():
        if name != "classifier.temperature":
            param.requires_grad = False
            
    optimizer = torch.optim.Adam([model.classifier.temperature], lr=0.01)
    criterion = nn.NLLLoss()
    
    print(f"Initial Temperature: {model.classifier.temperature.item():.4f}")
    
    # Train temperature for 20 epochs
    model.train() # Keep model.train() because we are optimizing temperature
    for epoch in range(1, 21):
        total_loss = 0
        for x, y in loader_va:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, return_all=True)
            probs = out["memberships"]
            
            # We want to minimize NLL (maximize likelihood)
            loss = criterion(torch.log(probs + 1e-8), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/20 | NLL Loss: {total_loss/len(loader_va):.4f} | Temp: {model.classifier.temperature.item():.4f}")

    # Save Phase 19
    torch.save({
        "state_dict": model.state_dict(),
        "class_to_idx": c2i,
        "arch": "phase19_calibrated",
        "config": {"temperature": model.classifier.temperature.item()},
    }, os.path.join(ARTIFACTS, "phase19.pt"))
    print(f"\n✓ Phase 19 Saved. Final Temperature: {model.classifier.temperature.item():.4f}")

if __name__ == "__main__":
    main()