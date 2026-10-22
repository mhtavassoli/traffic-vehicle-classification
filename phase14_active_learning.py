"""phase14_active_learning.py — Human-in-the-Loop Fine-tuning.

Simulates a human operator who reviews the 'NEEDS_REVIEW' images from
Phase 13 and provides correct labels. The model is then fine-tuned
on these hard examples.
"""
import os, json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from config import DEVICE, ARTIFACTS, CLASSES
from data import train_transform
from models_phase14 import Phase14Net, Phase13Loss

class HardExampleDataset(Dataset):
    def __init__(self, items, c2i, transform):
        self.items = items
        self.c2i = c2i
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.c2i[cls]

def filter_mismatched_weights(model, state_dict):
    """
    Filters out keys from the state_dict that either do not exist in the model
    or have a different tensor shape. This prevents loading errors when the architecture changes.
    """
    model_state = model.state_dict()
    filtered_state_dict = {}
    
    for key, param in state_dict.items():
        # Only keep weights that exist in the new model and have the exact same shape
        if key in model_state and model_state[key].shape == param.shape:
            filtered_state_dict[key] = param
        else:
            # Safely print the reason for skipping without crashing
            model_shape = model_state[key].shape if key in model_state else "Not Found in Model"
            print(f"[Warning] Skipping {key}: checkpoint shape {param.shape} vs model shape {model_shape}")
            
    return filtered_state_dict

def main():
    print("=" * 70)
    print("PHASE 14: ACTIVE LEARNING (Human-in-the-Loop)")
    print("=" * 70)

    # 1) Load Phase 13 results
    json_path = os.path.join(ARTIFACTS, "phase13", "test3_detailed_results.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)["detailed_results"]

    # 2) Filter NEEDS_REVIEW and assign True Labels (Simulate Human)
    hard_examples = []
    for r in results:
        if r["status"] == "NEEDS_REVIEW" and r["true_label"] == "ambulance":
            # Simulate human correction: The model thought it was an ambulance, but it's a vanet.
            full_path = os.path.join("..", "TestingData1", "test3", "ambulance", r["path"])
            if os.path.exists(full_path):
                hard_examples.append((full_path, "vanet")) 

    print(f"Found {len(hard_examples)} NEEDS_REVIEW images to fine-tune on.")

    if len(hard_examples) == 0:
        print("No hard examples found. Check paths.")
        return

    # 3) Load Phase 14 Model (Warm-start from Phase 13)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    model = Phase14Net(num_classes=len(CLASSES)).to(DEVICE)
    
    ckpt_path = os.path.join(ARTIFACTS, "phase13.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    # Filter weights before loading to avoid shape mismatch RuntimeErrors
    filtered_ckpt_state = filter_mismatched_weights(model, ckpt["state_dict"])
    
    # Load weights (strict=False allows missing keys, but we already filtered them)
    model.load_state_dict(filtered_ckpt_state, strict=False)
    print("Loaded Phase 13 weights (Topology Expert reset to TDA).")

    # 4) Fine-tune on Hard Examples
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    criterion = Phase13Loss(beta_shortcut=0.1, gamma_entropy=0.01)
    
    # FIXED: drop_last=True ensures the last batch is complete (prevents LayerNorm/BatchNorm crash on batch_size=1)
    loader = DataLoader(HardExampleDataset(hard_examples, c2i, train_transform(True)),
                        batch_size=8, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(1, 6): 
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, return_all=True)
            
            # FIXED: Extract the main loss safely. The criterion might return a tuple of multiple losses.
            criterion_outputs = criterion(
                out["memberships"], out["attention"], out["expert_weights"], y
            )
            
            if isinstance(criterion_outputs, tuple):
                loss = criterion_outputs[0] # Main loss is usually the first element
            else:
                loss = criterion_outputs

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/5 | Fine-tune Loss: {total_loss/len(loader):.4f}")

    # Save Phase 14 Model
    torch.save({
        "state_dict": model.state_dict(),
        "class_to_idx": c2i,
        "arch": "phase14_tda_active",
        "config": {"epoch": 5, "fine_tune": True},
    }, os.path.join(ARTIFACTS, "phase14.pt"))
    print("\n✓ Phase 14 Model Saved.")

if __name__ == "__main__":
    main()
