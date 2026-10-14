# فایل: check_unclean_ambulance.py
"""check_unclean_ambulance.py — Direct visual inspection of unclean/ambulance.

This is a sanity check: are the 145+ ambulance samples in unclean real
ambulances, or are they actually blue Nissan vans with wrong labels?
"""
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

from config import UNCLEAN_DIR, ARTIFACTS
from audit import list_split

OUT_DIR = os.path.join(ARTIFACTS, "ambulance_paradox")
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Find all unclean/ambulance images
items = list_split(UNCLEAN_DIR)
unclean_amb = [p for p, c in items if c == "ambulance"]
print(f"Total unclean/ambulance: {len(unclean_amb)}")

# 2) Random sample of 24 (bigger gallery)
random.seed(42)
sample = random.sample(unclean_amb, min(24, len(unclean_amb)))

# 3) Save 4x6 grid
fig, axes = plt.subplots(4, 6, figsize=(20, 14))
axes = axes.flatten()
for idx, p in enumerate(sample):
    img = Image.open(p).convert("RGB").resize((224, 224))
    axes[idx].imshow(img)
    axes[idx].set_title(os.path.basename(p)[:20], fontsize=7)
    axes[idx].axis("off")
for i in range(len(sample), len(axes)):
    axes[i].axis("off")

fig.suptitle("Unclean / ambulance — random 24 samples", fontsize=14)
fig.savefig(os.path.join(OUT_DIR, "unclean_ambulance_check.png"),
            bbox_inches="tight", dpi=100)
plt.close(fig)
print(f"Saved -> {OUT_DIR}/unclean_ambulance_check.png")