"""feature_space_viz.py — Visualize test3 blue vans in 2D via t-SNE."""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from config import (DEVICE, ARTIFACTS, CLASSES, TRAIN_DIR,
                    TESTING_DATA_ROOTS)
from data import IndexedImageDataset, eval_transform
from audit import list_split
from analysis import load_checkpoint


OUT_DIR = os.path.join(ARTIFACTS, "feature_viz")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def extract(model, items, c2i, device):
    ds = IndexedImageDataset(items, c2i, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        f = model.conv1(x)
        f = model.bn1(f); f = model.relu(f); f = model.maxpool(f)
        f = model.layer1(f); f = model.layer2(f); f = model.layer3(f); f = model.layer4(f)
        f = model.avgpool(f).flatten(1)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    model, _ = load_checkpoint("fixmatch_v2", "resnet")
    c2i = {c: i for i, c in enumerate(CLASSES)}

    # Sources
    train_items = [(p, c) for p, c in list_split(TRAIN_DIR) if c in c2i]
    train_items = [(p, c) for p, c in train_items
                   if c in ("ambulance", "vanet")]  # focus on the two

    # Test3 blue vans
    test3_path = None
    for root in TESTING_DATA_ROOTS:
        cand = os.path.join(root, "test3", "ambulance")
        if os.path.isdir(cand):
            test3_path = cand
            break
    test3_items = [(os.path.join(test3_path, f), "ambulance")
                   for f in os.listdir(test3_path)
                   if f != "195654131.jpg"]

    # Extract
    X_train, y_train = extract(model, train_items, c2i, DEVICE)
    X_test3, _ = extract(model, test3_items, c2i, DEVICE)

    # t-SNE on combined
    X_all = np.concatenate([X_train, X_test3])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_all)

    n_train = len(X_train)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot train ambulance
    mask_a = y_train == c2i["ambulance"]
    ax.scatter(X_2d[:n_train][mask_a, 0], X_2d[:n_train][mask_a, 1],
               c="green", label="train / ambulance", s=30, alpha=0.6)

    # Plot train vanet
    mask_v = y_train == c2i["vanet"]
    ax.scatter(X_2d[:n_train][mask_v, 0], X_2d[:n_train][mask_v, 1],
               c="blue", label="train / vanet", s=30, alpha=0.6)

    # Plot test3 blue vans
    ax.scatter(X_2d[n_train:, 0], X_2d[n_train:, 1],
               c="red", marker="X", label="test3 blue vans", s=100)

    ax.legend()
    ax.set_title("t-SNE of feature space: where do test3 blue vans land?")
    fig.savefig(os.path.join(OUT_DIR, "tsne_ambulance_vanet.png"),
                bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved -> {OUT_DIR}/tsne_ambulance_vanet.png")


if __name__ == "__main__":
    main()
