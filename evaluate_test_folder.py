"""evaluate_test_folder.py — Final evaluation on mentor's held-out test data.

CRITICAL RULE: This is the ONLY legitimate use of test data.
It should be run ONCE per model, after all training decisions are frozen.
"""
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS,
                    TESTING_DATA_ROOT, TEST_FOLDERS, BATCH_SIZE)
from data import IndexedImageDataset, eval_transform
from audit import list_split
from analysis import load_checkpoint


# ============================================================
# Evaluate one folder
# ============================================================
@torch.no_grad()
def evaluate_folder(model, folder_path, class_to_idx):
    """Run inference on all images in folder_path/class/*.jpg."""
    items = list_split(folder_path)

    # Filter: only known classes
    known_items = [(p, c) for p, c in items if c in class_to_idx]
    unknown_items = [(p, c) for p, c in items if c not in class_to_idx]

    if not known_items:
        return None

    ds = IndexedImageDataset(known_items, class_to_idx, eval_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(1)

    y_true = labels.numpy()
    y_pred = preds.numpy()

    acc = accuracy_score(y_true, y_pred)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Low-confidence analysis
    conf, _ = probs.max(1)
    conf = conf.numpy()
    review_rate = float((conf < 0.70).mean())

    return {
        "folder": os.path.basename(folder_path),
        "n_images": len(known_items),
        "n_unknown_classes": len(unknown_items),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "per_class": {
            CLASSES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(sup[i]),
            } for i in range(len(CLASSES))
        },
        "lowest_recall_class": CLASSES[int(np.argmin(r))],
        "lowest_precision_class": CLASSES[int(np.argmin(p))],
        "review_rate_at_0.70": review_rate,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "confidences": conf.tolist(),
        "probs": probs.numpy().tolist(),
    }


# ============================================================
# Confusion matrix plot
# ============================================================
def plot_confusion_from_results(y_true, y_pred, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                    fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return cm


# ============================================================
# Error gallery
# ============================================================
def plot_test_errors(model, folder_path, class_to_idx, save_path, n=12):
    """Show misclassified images."""
    items = list_split(folder_path)
    known_items = [(p, c) for p, c in items if c in class_to_idx]
    ds = IndexedImageDataset(known_items, class_to_idx, eval_transform())
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(DEVICE)
            probs = torch.softmax(model(x_dev), 1)
            conf, pred = probs.max(1)
            for i in range(x.size(0)):
                if pred[i] != y[i] and len(errors) < n:
                    errors.append((x[i].cpu(), y[i].item(),
                                   pred[i].item(), conf[i].item()))

    if not errors:
        print(f"  No errors in {os.path.basename(folder_path)} 🎉")
        return errors

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (img, true, pred, cf) in enumerate(errors):
        ax = axes[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f"True: {CLASSES[true]}\nPred: {CLASSES[pred]} ({cf:.2f})",
                     color="red", fontsize=9)
        ax.axis("off")
    for i in range(len(errors), len(axes)):
        axes[i].axis("off")
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return errors


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet_ft_v2",
                        help="Checkpoint name in artifacts/")
    parser.add_argument("--arch", default="resnet",
                        choices=["cnn", "resnet"])
    parser.add_argument("--folder", default=None,
                        help="Specific test folder. If None, all folders.")
    args = parser.parse_args()

    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model} ({args.arch})")
    print(f"Testing data root: {TESTING_DATA_ROOT}")

    if not os.path.isdir(TESTING_DATA_ROOT):
        print(f"✗ ERROR: {TESTING_DATA_ROOT} does not exist!")
        print(f"  Please check your .env file or TestingData folder.")
        sys.exit(1)

    # Load model
    model, ckpt = load_checkpoint(args.model, args.arch)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    print(f"✓ Loaded model from {args.model}.pt")

    # Determine folders to evaluate
    if args.folder:
        folders = [args.folder]
    else:
        folders = TEST_FOLDERS
        if not folders:
            # Fallback: look for test0, test1, ... in TESTING_DATA_ROOT
            folders = [os.path.join(TESTING_DATA_ROOT, n)
                       for n in sorted(os.listdir(TESTING_DATA_ROOT))
                       if os.path.isdir(os.path.join(TESTING_DATA_ROOT, n))]

    print(f"\nFound {len(folders)} test folder(s):")
    for f in folders:
        print(f"  • {os.path.basename(f)}")

    # Evaluate each folder
    all_results = []
    output_dir = os.path.join(ARTIFACTS, "test_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    for folder in folders:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {os.path.basename(folder)}")
        print("=" * 70)

        result = evaluate_folder(model, folder, class_to_idx)
        if result is None:
            print(f"  ⚠ Skipped (no known-class images)")
            continue

        print(f"  Images: {result['n_images']}")
        print(f"  Accuracy:        {result['accuracy']:.4f}")
        print(f"  Macro Precision: {result['macro_precision']:.4f}")
        print(f"  Macro Recall:    {result['macro_recall']:.4f}")
        print(f"  Macro F1:        {result['macro_f1']:.4f}")
        print(f"  Lowest recall:   {result['lowest_recall_class']}")
        print(f"  Lowest precision: {result['lowest_precision_class']}")
        print(f"  Review rate (conf<0.70): {result['review_rate_at_0.70']:.3f}")

        # Confusion matrix
        cm_path = os.path.join(output_dir,
                               f"{result['folder']}_confusion.png")
        plot_confusion_from_results(
            result["y_true"], result["y_pred"], cm_path
        )
        print(f"  ✓ Saved confusion matrix: {cm_path}")

        # Error gallery
        err_path = os.path.join(output_dir,
                                f"{result['folder']}_errors.png")
        errs = plot_test_errors(model, folder, class_to_idx, err_path, n=12)
        if errs:
            print(f"  ✓ Saved {len(errs)} errors: {err_path}")

        # Remove heavy fields before saving JSON
        result_light = {k: v for k, v in result.items()
                        if k not in ("y_true", "y_pred", "confidences", "probs")}
        all_results.append(result_light)

        # Save per-folder JSON
        with open(os.path.join(output_dir,
                               f"{result['folder']}_report.json"), "w") as f:
            json.dump(result_light, f, indent=2)

    # Aggregate summary
    if all_results:
        print(f"\n{'=' * 70}")
        print("AGGREGATE SUMMARY")
        print("=" * 70)

        header = f"{'Folder':<12} {'N':>5} {'Acc':>7} {'MacroP':>8} {'MacroR':>8} {'MacroF1':>9}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(f"{r['folder']:<12} {r['n_images']:>5} "
                  f"{r['accuracy']:>7.4f} {r['macro_precision']:>8.4f} "
                  f"{r['macro_recall']:>8.4f} {r['macro_f1']:>9.4f}")

        # Averages
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        avg_f1 = np.mean([r["macro_f1"] for r in all_results])
        print("-" * len(header))
        print(f"{'AVERAGE':<12} {'':>5} {avg_acc:>7.4f} {'':>8} {'':>8} {avg_f1:>9.4f}")

        summary = {
            "model": args.model,
            "arch": args.arch,
            "folders_evaluated": len(all_results),
            "average_accuracy": float(avg_acc),
            "average_macro_f1": float(avg_f1),
            "per_folder": all_results,
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved to {output_dir}/summary.json")

    print(f"\n{'=' * 70}")
    print("✓ FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)
    print("\n⚠️  REMEMBER: This was the ONE legitimate use of test data.")
    print("   Do NOT tune the model based on these results.")
    print("   Any further changes require a NEW held-out test set.")


if __name__ == "__main__":
    main()