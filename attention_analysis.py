"""attention_analysis.py — Average Grad-CAM across all test3 blue vans.

Purpose: Confirm that the model's attention is systematically on the
LOWER part of blue vans (not on ambulance-specific features).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import DEVICE, ARTIFACTS, CLASSES, TRAIN_DIR, TESTING_DATA_ROOTS
from analysis import load_checkpoint


OUT_DIR = os.path.join(ARTIFACTS, "attention_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_image(path, size=128):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    img = Image.open(path).convert("RGB")
    return img, tf(img).unsqueeze(0).to(DEVICE)


def main():
    model, _ = load_checkpoint("fixmatch_v2", "resnet")
    model.eval()
    gradcam = GradCAM(model, model.layer4[-1])

    # Test3 blue vans
    test3_dir = None
    for root in TESTING_DATA_ROOTS:
        cand = os.path.join(root, "test3", "ambulance")
        if os.path.isdir(cand):
            test3_dir = cand
            break

    test3_files = [os.path.join(test3_dir, f)
                   for f in sorted(os.listdir(test3_dir))
                   if f != "195654131.jpg"]

    # Compute average CAM
    cams = []
    for p in test3_files:
        try:
            _, x = load_image(p)
            cam = gradcam.generate(x, CLASSES.index("ambulance"))
            cams.append(cam)
        except Exception:
            continue

    if not cams:
        print("No valid test3 images found.")
        return

    avg_cam = np.mean(cams, axis=0)
    std_cam = np.std(cams, axis=0)

    # Save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(avg_cam, cmap="jet")
    axes[0].set_title(f"Average CAM (n={len(cams)})")
    axes[0].axis("off")
    axes[1].imshow(std_cam, cmap="hot")
    axes[1].set_title("Std CAM (variance)")
    axes[1].axis("off")
    # Vertical profile (where does attention concentrate?)
    profile = avg_cam.mean(axis=1)  # mean across width
    axes[2].plot(profile, np.arange(len(profile))[::-1])
    axes[2].set_xlabel("Average activation")
    axes[2].set_ylabel("Row (top → bottom)")
    axes[2].set_title("Vertical attention profile")
    axes[2].axhline(y=42, color="r", linestyle="--", label="Top 1/3")
    axes[2].axhline(y=85, color="r", linestyle="--", label="Bottom 1/3")
    axes[2].legend()

    fig.savefig(os.path.join(OUT_DIR, "test3_avg_gradcam.png"),
                bbox_inches="tight", dpi=120)
    plt.close(fig)

    # Quantify
    top_third = profile[:42].mean()
    mid_third = profile[42:85].mean()
    bot_third = profile[85:].mean()
    print("=" * 70)
    print("AVERAGE ATTENTION DISTRIBUTION (test3 blue vans)")
    print("=" * 70)
    print(f"  Top 1/3:    {top_third:.4f}")
    print(f"  Middle 1/3: {mid_third:.4f}")
    print(f"  Bottom 1/3: {bot_third:.4f}")
    print(f"\n  → Attention is {('BOTTOM-heavy' if bot_third > top_third else 'TOP-heavy')}")
    print(f"\n  Saved → {OUT_DIR}/test3_avg_gradcam.png")


if __name__ == "__main__":
    main()