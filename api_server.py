"""api_server.py — FastAPI Backend for the Calibrated Quantum Model (Phase 19)."""
import os
import io
import json
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import transforms
from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from models_phase19 import Phase19Net # Import Phase 19

app = FastAPI(title="Traffic Vehicle Classifier (Phase 19 - Calibrated Quantum)")

print("Loading Phase 19 Model...")
model = Phase19Net(num_classes=NUM_CLASSES).to(DEVICE)
ckpt_path = os.path.join(ARTIFACTS, "phase19.pt")

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"✓ Phase 19 Model Loaded Successfully (Temp: {ckpt['config']['temperature']:.4f}).")
else:
    print("⚠ Warning: phase19.pt not found.")
model.eval()

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# We keep the threshold at 0.25 to see if the calibrated model asks for help.
REJECT_THRESHOLD = 0.25 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x, return_all=True)
            probs = out["memberships"][0]
            
            top2_probs, top2_idx = torch.topk(probs, 2)
            top1_prob = top2_probs[0].item()
            top2_prob = top2_probs[1].item()
            margin = top1_prob - top2_prob
            
            pred_class = CLASSES[top2_idx[0].item()]
            needs_review = margin < REJECT_THRESHOLD
            
            prob_dict = {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))}

        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": round(top1_prob, 4),
            "margin": round(margin, 4),
            "needs_review": needs_review,
            "probabilities": prob_dict,
            "expert_weights": {
                "attention": round(out["expert_weights"][0, 0].item(), 4),
                "topology": round(out["expert_weights"][0, 1].item(), 4)
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/feedback")
async def feedback(filename: str = Form(...), corrected_label: str = Form(...)):
    feedback_file = os.path.join(ARTIFACTS, "human_feedback.json")
    data = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            
    data.append({"filename": filename, "corrected_label": corrected_label})
    
    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=2)
        
    return JSONResponse({"status": "success", "message": "Feedback recorded!"})

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)