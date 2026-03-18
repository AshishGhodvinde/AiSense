import os
import io
import json
import base64
import cv2
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image

from model import load_inference_model
from heuristics import compute_ela, compute_high_freq_noise, compute_exif_heuristics
from explainer import generate_explanation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'weights/best_model.pth'
model = None

# Custom CNN requires 32x32 images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.on_event("startup")
def startup_event():
    global model
    if os.path.exists(model_path):
        model = load_inference_model(model_path, device)

@app.get("/status")
def status():
    return {"status": "ok", "mode": "hybrid-ml-exif", "ml_ready": model is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global model
    if model is None:
        if os.path.exists(model_path):
            model = load_inference_model(model_path, device)
        else:
            return {"error": "ML Model not trained yet. Run train.py first to satisfy the course constraints!"}
            
    content = await file.read()
    filename = file.filename
    
    # 1. EXIF Heuristics (Rule-based Base Layer)
    exif_verdict = compute_exif_heuristics(content, filename)
    exif_is_ai = exif_verdict['isAi']
    exif_confidence = exif_verdict['confidence'] / 100.0
    metadata_reasons = exif_verdict['reasons']
    
    # 2. PyTorch CNN Prediction (ML Layer)
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception as e:
        return {"error": f"Invalid image format: {e}"}
        
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        
    class_idx = preds.item()
    ml_confidence = conf.item()
    
    # Resolve mapping
    mapping_file = 'weights/class_mapping.json'
    ml_is_ai = False
    
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            prediction_label = mapping.get(str(class_idx), "")
            if 'fake' in prediction_label.lower() or 'ai' in prediction_label.lower():
                ml_is_ai = True
    else:
        ml_is_ai = (class_idx == 0)

    # 3. Pixel Heuristics
    ela_cv2 = compute_ela(content)
    ela_variance = float(ela_cv2.var()) if ela_cv2 is not None else 0.0
    noise_score = float(compute_high_freq_noise(content))

    # Calculate unified P(AI) scale
    p_ml = ml_confidence if ml_is_ai else (1.0 - ml_confidence)
    p_exif = exif_confidence if exif_is_ai else (1.0 - exif_confidence)
    
    # Map Pixel heuristics to Probability (0 to 1)
    p_ela = min(1.0, ela_variance / 250.0)
    p_noise = min(1.0, noise_score / 2500.0)
    
    # Core Verdict Logic
    # Mirroring the robust Github repository Metadata approach. The ML model is relegated to a minor 5% supplementary signal.
    # The final overarching verdict heavily prioritizes hard structural EXIF analysis and ELA matrices.
    p_final = (p_exif * 0.75) + (p_ela * 0.20) + (p_ml * 0.05)
        
    final_is_ai = bool(p_final >= 0.5)
    final_confidence = float(p_final if final_is_ai else (1.0 - p_final))
    
    ela_base64 = ""
    if ela_cv2 is not None:
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(ela_cv2, cv2.COLOR_RGB2BGR))
        ela_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 4. Explainability
    explanation = generate_explanation(
        final_is_ai, final_confidence, 
        ml_is_ai, ml_confidence,
        ela_variance, noise_score, metadata_reasons
    )
    
    return {
        "prediction": "AI Generated" if final_is_ai else "Real Image",
        "confidence": final_confidence,
        "explanation": explanation,
        "heuristics": {
            "ela_variance": float(f"{ela_variance:.2f}"),
            "noise_score": float(f"{noise_score:.2f}"),
            "exif_reasons": metadata_reasons
        },
        "ela_image": f"data:image/jpeg;base64,{ela_base64}" if ela_base64 else None
    }
