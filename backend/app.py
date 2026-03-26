import os
import io
import json
import base64
import cv2
import torch
from dotenv import load_dotenv
load_dotenv()   # reads backend/.env → injects SIGHTENGINE_USER / SIGHTENGINE_SECRET
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model import load_inference_model, predict_image
from heuristics import compute_ela, compute_exif_heuristics
from frequency import extract_frequency_features
from noise_analysis import analyze_noise_pattern
from artifact import detect_artifacts
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
model = None

@app.on_event("startup")
def startup_event():
    global model
    # Load production HuggingFace model (downloads automatically on first run)
    model = load_inference_model(None, device)

@app.get("/status")
def status():
    return {"status": "ok", "mode": "multi-branch-fusion-v2", "ml_ready": model is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global model
    if model is None:
        model = load_inference_model(None, device)
        if model is None:
            return {"error": "AI detector model failed to load. Check internet connection for HuggingFace download."}
            
    content = await file.read()
    filename = file.filename
    
    # 1. EXIF Heuristics (Metadata Layer)
    exif_verdict = compute_exif_heuristics(content, filename)
    exif_is_ai = exif_verdict['isAi']
    exif_confidence = exif_verdict['confidence'] / 100.0
    metadata_reasons = exif_verdict['reasons']
    
    # 2. Primary ML Detection (Sightengine API → HuggingFace fallback)
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception as e:
        return {"error": f"Invalid image format: {e}"}
    
    ml_is_ai, ml_confidence, ml_source = predict_image(model, img, raw_bytes=content)

    # 3. Extracted Analytical Features
    freq_prob, freq_features = extract_frequency_features(content)
    noise_prob, noise_features = analyze_noise_pattern(content)
    artifact_prob, artifact_features = detect_artifacts(content)
    
    ela_cv2 = compute_ela(content)
    ela_variance = float(ela_cv2.var()) if ela_cv2 is not None else 0.0

    # Unified P(AI) Variables
    p_ml   = ml_confidence
    p_exif = exif_confidence if exif_is_ai else (1.0 - exif_confidence)

    # 4. ML-Only Fusion
    # Secondary signals (ELA, noise, freq) are too noisy for general photos
    # (e.g., WhatsApp recompression inflates ELA; pro photos have SDXL-like traits)
    # → Use ONLY the ML model score as the decision signal.
    # Small EXIF bonus only if conclusive metadata evidence exists.
    final_score = p_ml

    # Only apply EXIF correction when metadata is CONCLUSIVE
    if p_exif > 0.90:   # AI software tag in EXIF (e.g. "Midjourney")
        final_score = min(0.99, final_score + 0.15)
    elif p_exif < 0.08:  # Full camera chain: Make + Model + GPS + DateOriginal
        final_score = max(0.01, final_score - 0.10)

    # Threshold: 0.70 — require strong confidence before labelling as AI.
    # Real photos score 0.15–0.55 on well-calibrated models.
    # Fully AI images (Midjourney, DALL-E, SD) score 0.75–0.99.
    # The 0.50–0.70 grey zone → Real (we accept missing borderline AI
    # rather than falsely flagging real photos as AI).
    final_is_ai = bool(final_score >= 0.70)

    # Confidence: how sure are we in the ACTUAL verdict direction.
    # Cap at 0.97 so it never reads as an unrealistic "100%".
    # Also apply a soft-scale: linear between 0.50–0.97 so Sightengine's
    # binary 0.01/0.99 outputs produce meaningful variance rather than always 99%.
    raw_conf = float(final_score if final_is_ai else (1.0 - final_score))
    # Re-map [0.5 → 1.0]  to  [0.50 → 0.97]
    final_confidence = min(0.97, 0.50 + (raw_conf - 0.50) * 0.94)
    
    ela_base64 = ""
    if ela_cv2 is not None:
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(ela_cv2, cv2.COLOR_RGB2BGR))
        ela_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 5. Explainability
    explanation = generate_explanation(
        final_is_ai, final_confidence, 
        ml_is_ai, ml_confidence,
        freq_prob, noise_prob, artifact_prob,
        metadata_reasons,
        ela_variance=ela_variance
    )
    
    return {
        "prediction": "AI Generated" if final_is_ai else "Real Image",
        "confidence": final_confidence,
        "explanation": explanation,
        "heuristics": {
            "ela_variance": float(f"{ela_variance:.2f}"),
            "noise_score": float(f"{noise_features.get('noise_variance', 0.0):.2f}"),
            "exif_reasons": metadata_reasons,
            "frequency_ai_prob": float(f"{freq_prob:.2f}"),
            "noise_ai_prob": float(f"{noise_prob:.2f}"),
            "artifact_ai_prob": float(f"{artifact_prob:.2f}")
        },
        "ela_image": ela_base64 if ela_base64 else None
    }
