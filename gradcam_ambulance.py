"""gradcam_ambulance.py — Visualize which image regions activate ambulance prediction.

Uses Grad-CAM to show:
    - Real ambulance from train (attention should be on AMBULANCE text / lights)
    - Blue Nissan from test3 (attention should reveal the false feature)

Reference: Selvaraju et al., ICCV 2017.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import DEVICE, ARTIFACTS, CLASSES, TESTING_DATA_ROOTS
from analysis import load_checkpoint


OUT_DIR = os.path.join(ARTIFACTS, "gradcam_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


class GradCAM:
    """Simple Grad-CAM for ResNet18 targeting the last conv block."""
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

        # Global average pool of gradients → weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_image(path, size=128):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.open(path).convert("RGB")
    return img, tf(img).unsqueeze(0).to(DEVICE)


def overlay_cam(img, cam, alpha=0.4):
    img_np = np.asarray(img.resize((128, 128)), dtype=np.float32) / 255.0
    heatmap = plt.cm.jet(cam)[..., :3]
    return (1 - alpha) * img_np + alpha * heatmap


def main():
    print("=" * 70)
    print("GRAD-CAM ANALYSIS")
    print("=" * 70)

    model, _ = load_checkpoint("fixmatch_v2", "resnet")
    model.eval()

    # Target the last conv block
    gradcam = GradCAM(model, model.layer4[-1])

    # Find samples
    real_amb = "dataset_merged/train/ambulance"  # adjust to your path
    # Actually use config
    from config import TRAIN_DIR
    real_amb_dir = os.path.join(TRAIN_DIR, "ambulance")
    real_amb_files = [os.path.join(real_amb_dir, f)
                      for f in sorted(os.listdir(real_amb_dir))[:3]]

    # Blue Nissan from test3
    test3_dir = None
    for root in TESTING_DATA_ROOTS:
        cand = os.path.join(root, "test3", "ambulance")
        if os.path.isdir(cand):
            test3_dir = cand
            break
    test3_files = [os.path.join(test3_dir, f)
                   for f in sorted(os.listdir(test3_dir))
                   if "195654131" not in f][:3]  # skip the real ambulance

    # Generate CAM for each
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, p in enumerate(real_amb_files):
        img, x = load_image(p)
        cam = gradcam.generate(x, CLASSES.index("ambulance"))
        overlaid = overlay_cam(img, cam)
        axes[0, idx].imshow(overlaid)
        axes[0, idx].set_title(f"REAL AMB: {os.path.basename(p)[:18]}",
                                fontsize=8)
        axes[0, idx].axis("off")

    for idx, p in enumerate(test3_files):
        img, x = load_image(p)
        cam = gradcam.generate(x, CLASSES.index("ambulance"))
        overlaid = overlay_cam(img, cam)
        axes[1, idx].imshow(overlaid)
        axes[1, idx].set_title(f"BLUE VAN: {os.path.basename(p)[:18]}",
                                fontsize=8)
        axes[1, idx].axis("off")

    fig.suptitle("Grad-CAM: Where does the model look for 'ambulance'?",
                 fontsize=13)
    fig.savefig(os.path.join(OUT_DIR, "gradcam_comparison.png"),
                bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Saved -> {OUT_DIR}/gradcam_comparison.png")

    print("\nInterpretation:")
    print("  - TOP row (real ambulance): attention should be on AMBULANCE text/lights")
    print("  - BOTTOM row (blue van): if attention is on ROOF/ANGLE, we have feature entanglement")


if __name__ == "__main__":
    main()
