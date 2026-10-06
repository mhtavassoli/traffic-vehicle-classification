"""smoke_test.py — Quick sanity check before full experiments."""
import os, sys, torch
from config import (TRAIN_DIR, TEST_DIR, UNCLEAN_DIR, TRAIN_DIR_V2,
                    USE_DATASET_V2, DEVICE, NUM_CLASSES, CLASSES)
from audit import run_audit, list_split
from data import (IndexedImageDataset, train_transform, eval_transform,
                  stratified_val_split, BalancedBatchSampler)
from models import TrafficCNN
from train import fit
from torch.utils.data import DataLoader

def main():
    print("=" * 60)
    print("SMOKE TEST — Traffic Vehicle Classification")
    print("=" * 60)

    # 1) Check paths
    print(f"\n[1/6] Checking paths...")
    for name, path in [("TRAIN_DIR", TRAIN_DIR), ("TEST_DIR", TEST_DIR),
                       ("UNCLEAN_DIR", UNCLEAN_DIR)]:
        exists = os.path.isdir(path)
        print(f"  {name}: {path}  {'✓' if exists else '✗'}")
        if not exists:
            print(f"  ERROR: {name} does not exist!")
            sys.exit(1)

    print(f"  USE_DATASET_V2 = {USE_DATASET_V2}")
    if USE_DATASET_V2:
        print(f"  TRAIN_DIR_V2: {TRAIN_DIR_V2}  {'✓' if os.path.isdir(TRAIN_DIR_V2) else '✗'}")

    # 2) Quick audit
    print(f"\n[2/6] Running audit...")
    report, dups, conflicts = run_audit()
    for split, counts in report["counts"].items():
        total = sum(counts.values())
        print(f"  {split}: {total} images across {len(counts)} classes")
    print(f"  Duplicate groups: {report['duplicates']['total_groups']}")
    print(f"  Label conflicts: {len(report['label_conflicts'])}")

    # 3) Build small dataset (just to test)
    print(f"\n[3/6] Building mini dataset...")
    
    # from config import TRAIN_DIR          
    # By commenting the line above,
    # Python will read `TRAIN_DIR` from the global scope (at the top of the file)
    # instead of treating it as a local setting.
    items = list_split(TRAIN_DIR)[:160]  # Only 160 images
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    tr_items, va_items, _, _ = stratified_val_split(items, class_to_idx, 0.2)

    ds_tr = IndexedImageDataset(tr_items, class_to_idx, train_transform(True))
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    tr_loader = DataLoader(ds_tr, batch_size=16, shuffle=True, num_workers=0)
    va_loader = DataLoader(ds_va, batch_size=16, shuffle=False, num_workers=0)
    print(f"  Train: {len(ds_tr)}  Val: {len(ds_va)}")

    # 4) Build model
    print(f"\n[4/6] Building model...")
    model = TrafficCNN(NUM_CLASSES, dropout=0.3, pool_type="max").to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Device: {DEVICE}")
    print(f"  Trainable parameters: {n_params:,}")

    # 5) Train 2 epochs
    print(f"\n[5/6] Training for 2 epochs (smoke)...")
    hist = fit(model, tr_loader, va_loader, epochs=2, lr=1e-3)
    print(f"  Best val acc: {hist['best_val_acc']:.4f}")

    # 6) Predict on one image
    print(f"\n[6/6] Testing predict...")
    from predict import predict
    sample_path = tr_items[0][0]
    result = predict(sample_path, model)
    print(f"  Sample: {os.path.basename(sample_path)}")
    print(f"  Predicted: {result['predicted_class']}  conf={result['confidence']:.3f}")
    print(f"  needs_review: {result['needs_review']}")

    print("\n" + "=" * 60)
    print("✓ SMOKE TEST PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()