"""
model.py - AI Detection Engine with Sightengine Primary + HuggingFace Fallback
===============================================================================

PRIMARY: Sightengine API (98.3% accuracy) — fastest, most accurate
FALLBACK: HuggingFace ensemble using MEDIAN (outlier-resistant) of 3 models

Models in fallback:
  1. Organika/sdxl-detector        — good for SD / SDXL
  2. haywoodsloan/ai-image-detector — general real vs AI
  3. dima806/ai_vs_real_image_detection — different architecture, broader data

Using MEDIAN (not average) so one biased model cannot skew the result.
"""

import os
import io
import requests
from transformers import pipeline
import torch

# ── Sightengine credentials ────────────────────────────────────────────────────
SIGHTENGINE_USER   = os.environ.get("SIGHTENGINE_USER", "")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_SECRET", "")
SIGHTENGINE_ENDPOINT = "https://api.sightengine.com/1.0/check.json"

# ── HuggingFace fallback ───────────────────────────────────────────────────────
_detectors   = []
_hf_loaded   = False

# Models with different training data — median makes the ensemble outlier-resistant
_HF_MODEL_IDS = [
    "Organika/sdxl-detector",           # Strong on SD/SDXL; sometimes confuses professional portraits
    "haywoodsloan/ai-image-detector",   # General real-vs-AI, different training source
    "dima806/ai_vs_real_image_detection",  # ViT trained specifically on photo-realistic real vs AI
]


def _load_hf(device_arg: int):
    global _detectors, _hf_loaded
    if _hf_loaded:
        return
    for mid in _HF_MODEL_IDS:
        try:
            p = pipeline("image-classification", model=mid, device=device_arg)
            _detectors.append((mid, p))
            print(f"[AiSense] Loaded HF detector: {mid}")
        except Exception as e:
            print(f"[AiSense] Could not load {mid}: {e}")
    _hf_loaded = True


def load_inference_model(model_path, device):
    device_arg = 0 if str(device) == "cuda:0" else -1
    if SIGHTENGINE_USER and SIGHTENGINE_SECRET:
        print("[AiSense] Sightengine configured — 98.3% accuracy mode active.")
    else:
        print("[AiSense] Sightengine not configured. Loading HuggingFace fallback...")
    _load_hf(device_arg)
    return "ready"


def _parse_hf_score(results) -> float:
    """Convert HuggingFace classification results → AI probability."""
    for r in results:
        label = r["label"].lower()
        score = float(r["score"])
        if any(k in label for k in ["artificial", "fake", "ai", "sdxl", "generated"]):
            return score
        elif any(k in label for k in ["real", "human", "natural", "authentic"]):
            return 1.0 - score
    return 0.5


def _hf_median_score(pil_image) -> float:
    """Run all loaded detectors and return the MEDIAN score (outlier-resistant)."""
    scores = []
    for mid, detector in _detectors:
        try:
            res = detector(pil_image)
            s = _parse_hf_score(res)
            scores.append(s)
            print(f"[AiSense] {mid.split('/')[-1]}: {s:.3f}")
        except Exception as e:
            print(f"[AiSense] Error from {mid}: {e}")
    if not scores:
        return 0.5
    scores.sort()
    n = len(scores)
    # Median: middle value (or average of two middle if even)
    mid_i = n // 2
    if n % 2 == 1:
        return scores[mid_i]
    return (scores[mid_i - 1] + scores[mid_i]) / 2.0


def predict_image(model, pil_image, raw_bytes: bytes = None):
    """
    Returns (is_ai: bool, ai_probability: float, source: str)
    """
    # ── Sightengine primary ────────────────────────────────────────────────────
    if SIGHTENGINE_USER and SIGHTENGINE_SECRET:
        try:
            if raw_bytes is None:
                buf = io.BytesIO()
                pil_image.save(buf, format="JPEG", quality=95)
                raw_bytes = buf.getvalue()
            resp = requests.post(
                SIGHTENGINE_ENDPOINT,
                files={"media": ("img.jpg", raw_bytes, "image/jpeg")},
                data={"models": "genai",
                      "api_user": SIGHTENGINE_USER,
                      "api_secret": SIGHTENGINE_SECRET},
                timeout=15,
            )
            data = resp.json()
            if data.get("status") == "success":
                ai_prob = float(data.get("type", {}).get("ai_generated", 0.5))
                print(f"[AiSense] Sightengine ai_generated={ai_prob:.3f}")
                return ai_prob >= 0.5, ai_prob, "sightengine"
            print(f"[AiSense] Sightengine response error: {data}")
        except Exception as e:
            print(f"[AiSense] Sightengine failed: {e}")

    # ── HuggingFace fallback ───────────────────────────────────────────────────
    _load_hf(-1)
    avg = _hf_median_score(pil_image)
    print(f"[AiSense] HF median score: {avg:.3f}")
    return avg >= 0.5, float(avg), "huggingface"
