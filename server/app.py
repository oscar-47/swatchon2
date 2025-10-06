from io import BytesIO
from typing import Dict, List
import os
import hashlib

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from server.models.loader import build_eval_transform, load_checkpoint

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
        results.append({
            "filename": f.filename,
            "pred": classes[int(idx)],
            "confidence": float(conf),
            "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
        })
    return {"classes": classes, "predictions": results, "skipped": skipped, "hashes_loaded": len(TRAIN_HASHES)}

