from io import BytesIO
from typing import Dict, List
import os
import hashlib

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from server.models.loader import build_eval_transform, load_checkpoint

# OCR Configuration
OCR_ENABLED = True
CONFIDENCE_THRESHOLD = 0.60  # Trigger OCR if confidence < 60%

# Model registry: name -> checkpoint path
MODEL_REGISTRY: Dict[str, str] = {
    "woven_vs_knit": os.path.join("runs", "woven_vs_knit_r50_gpu_e5", "best.pth"),
    "woven_multi": os.path.join("runs", "woven_r50_gpu_e5", "best.pth"),
    "knit_multi": os.path.join("runs", "knit_r50_gpu_e5", "best.pth"),
}

app = FastAPI(title="Swatchon Classifier API", version="0.2")

# Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCache:
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tfm = build_eval_transform(224)

        # Initialize OCR engine (lazy initialization)
        self.ocr = None
        self.ocr_initialized = False

    def get(self, name: str):
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}")
        if name not in self._cache:
            ckpt_path = MODEL_REGISTRY[name]
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            model, classes = load_checkpoint(ckpt_path, self.device)
            self._cache[name] = {"model": model, "classes": classes}
        return self._cache[name]["model"], self._cache[name]["classes"], self.tfm, self.device

    def recognize_ocr(self, img_pil):
        """Run OCR on image and return extracted text using EasyOCR"""
        # Lazy initialization of OCR engine
        if not self.ocr_initialized:
            if not OCR_ENABLED:
                return None
            try:
                import sys
                import easyocr
                sys.stdout.flush()
                sys.stderr.flush()

                # Initialize EasyOCR with English support
                # You can add more languages: ['en', 'ch_sim'] for Chinese
                sys.stderr.write("[OCR] Initializing EasyOCR...\n")
                sys.stderr.flush()

                self.ocr = easyocr.Reader(['en'], gpu=False)
                self.ocr_initialized = True

                sys.stderr.write("[OCR] EasyOCR initialized successfully!\n")
                sys.stderr.flush()
            except Exception as e:
                self.ocr_initialized = True  # Mark as initialized even if failed
                import traceback
                import sys
                sys.stderr.write(f"[OCR] Init failed: {str(e)}\n")
                sys.stderr.write(traceback.format_exc() + "\n")
                sys.stderr.flush()
                return None

        if not self.ocr:
            return None

        try:
            import numpy as np
            import sys

            # Convert PIL Image to numpy array
            img_array = np.array(img_pil)

            sys.stderr.write(f"[OCR] Image shape: {img_array.shape}, dtype: {img_array.dtype}\n")
            sys.stderr.flush()

            # EasyOCR readtext returns: [([bbox], text, confidence), ...]
            result = self.ocr.readtext(img_array)

            sys.stderr.write(f"[OCR] EasyOCR found {len(result)} text regions\n")
            sys.stderr.flush()

            if result and len(result) > 0:
                texts = []
                confidences = []

                for idx, detection in enumerate(result):
                    # detection is a tuple: (bbox, text, confidence)
                    bbox, text, conf = detection
                    sys.stderr.write(f"[OCR] Line {idx}: '{text}' (conf: {conf:.2f})\n")
                    texts.append(text)
                    confidences.append(float(conf))

                sys.stderr.flush()

                if texts:
                    full_text = ' '.join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    sys.stderr.write(f"[OCR] SUCCESS: extracted {len(texts)} items\n")
                    sys.stderr.write(f"[OCR] Text: {full_text[:200]}\n")
                    sys.stderr.flush()
                    return {"text": full_text, "confidence": float(avg_conf)}
            else:
                sys.stderr.write(f"[OCR] No text detected by EasyOCR\n")
                sys.stderr.flush()

        except Exception as e:
            import sys
            import traceback
            sys.stderr.write(f"[OCR] Recognition error: {str(e)}\n")
            sys.stderr.write(traceback.format_exc() + "\n")
            sys.stderr.flush()

        return None


CACHE = ModelCache()

# Optional training hash set for External-only filtering (resolve path from repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAIN_HASHES_PATH = os.path.join(REPO_ROOT, "training", "training_hashes.txt")
TRAIN_HASHES = set()
if os.path.isfile(TRAIN_HASHES_PATH):
    try:
        with open(TRAIN_HASHES_PATH, 'r', encoding='utf-8') as f:
            TRAIN_HASHES = set(line.strip() for line in f if line.strip())
    except Exception:
        TRAIN_HASHES = set()

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


@app.get("/api/models")
def list_models():
    return {
        "models": [
            {"name": k, "checkpoint": v, "classes": (CACHE.get(k)[1] if os.path.exists(v) else None)}
            for k, v in MODEL_REGISTRY.items()
        ],
        "hashes_loaded": len(TRAIN_HASHES)
    }


@app.post("/api/predict")
async def predict(model_name: str = Form(...), files: List[UploadFile] = File(...), external_only: bool = Form(False)):
    model, classes, tfm, device = CACHE.get(model_name)
    results = []
    skipped = 0
    for f in files:
        data = await f.read()
        if external_only and TRAIN_HASHES:
            h = sha256_bytes(data)
            if h in TRAIN_HASHES:
                skipped += 1
                continue
        img = Image.open(BytesIO(data)).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        confidence = float(conf)
        prediction = classes[int(idx)]

        # Check if OCR is needed (confidence below threshold)
        needs_ocr = confidence < CONFIDENCE_THRESHOLD

        results.append({
            "filename": f.filename,
            "pred": prediction,
            "confidence": confidence,
            "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
            "needs_ocr": needs_ocr
        })
    return {"classes": classes, "predictions": results, "skipped": skipped, "hashes_loaded": len(TRAIN_HASHES)}


@app.post("/api/ocr")
async def recognize_label(file: UploadFile = File(...)):
    """OCR recognition for fabric label images"""
    try:
        data = await file.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        ocr_result = CACHE.recognize_ocr(img)

        if ocr_result:
            return {
                "success": True,
                "text": ocr_result["text"],
                "confidence": ocr_result["confidence"]
            }
        else:
            return {
                "success": False,
                "text": None,
                "confidence": None,
                "message": "No text detected in image"
            }
    except Exception as e:
        return {
            "success": False,
            "text": None,
            "confidence": None,
            "message": str(e)
        }


# Mount static files for frontend (after API routes to avoid conflicts)
FRONTEND_DIST = os.path.join(REPO_ROOT, "web", "ant_demo")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
