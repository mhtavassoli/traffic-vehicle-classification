"""Single-image inference -> JSON (production path uses CE model)."""
import json, torch
from PIL import Image
from config import CLASSES, CONFIDENCE_THRESHOLD, DEVICE
from data import eval_transform

def predict(image_path, model, class_to_idx=None, threshold=CONFIDENCE_THRESHOLD):
    model.eval()
    tf = eval_transform()
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, 1)[0].cpu()
    conf, idx = probs.max(0)
    predicted = CLASSES[idx.item()]
    return {
        "predicted_class": predicted,
        "confidence": float(conf.item()),
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "needs_review": bool(conf.item() < threshold),
    }

if __name__ == "__main__":
    import sys
    # Example usage (checkpoint loading omitted for brevity)
    # model = torch.load(sys.argv[2]) ...
    print("Call predict(image_path, model) from Python.")