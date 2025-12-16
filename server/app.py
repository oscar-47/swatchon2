from io import BytesIO
from typing import Dict, List, Optional
import os
import hashlib
from datetime import datetime

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from server.models.loader import build_eval_transform, load_checkpoint
from server.ocr_parser import OCRFieldParser
from server.double_verifier import DoubleStructureVerifier

# OCR Configuration
OCR_ENABLED = True
CONFIDENCE_THRESHOLD = 0.60  # Trigger OCR if confidence < 60%

# Model registry: name -> checkpoint path
MODEL_REGISTRY: Dict[str, str] = {
    "woven_vs_knit": os.path.join("simple_model_v2", "models_stage1", "stage1_knit_woven_best.pth"),
    "woven_multi": os.path.join("simple_model_v2", "models_stage2_woven", "stage2_woven_best.pth"),
    "knit_multi": os.path.join("simple_model_v2", "models_stage2_knit", "stage2_knit_best.pth"),
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
        
        # Initialize OCR parser and double verifier
        self.ocr_parser = OCRFieldParser()
        self.double_verifier = DoubleStructureVerifier(similarity_threshold=0.75)

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


def two_stage_predict(img_pil: Image.Image):
    """
    Two-stage prediction:
    1. Stage 1: Classify as Knit or Woven
    2. Stage 2: Subcategory classification based on stage 1 result
    
    Returns: dict with prediction, confidence, and stage details
    """
    # Stage 1: Knit vs Woven
    model1, classes1, tfm, device = CACHE.get("woven_vs_knit")
    x = tfm(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits1 = model1(x)
        probs1 = torch.softmax(logits1, dim=1)[0]
        conf1, idx1 = torch.max(probs1, dim=0)
    
    stage1_pred = classes1[int(idx1)]
    stage1_conf = float(conf1)
    stage1_probs = {classes1[i]: float(probs1[i]) for i in range(len(classes1))}
    
    # Stage 2: Subcategory classification
    if stage1_pred == "Knit":
        model2, classes2, tfm, device = CACHE.get("knit_multi")
    else:  # Woven
        model2, classes2, tfm, device = CACHE.get("woven_multi")
    
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits2 = model2(x)
        probs2 = torch.softmax(logits2, dim=1)[0]
        conf2, idx2 = torch.max(probs2, dim=0)
    
    stage2_pred = classes2[int(idx2)]
    stage2_conf = float(conf2)
    stage2_probs = {classes2[i]: float(probs2[i]) for i in range(len(classes2))}
    
    # Final prediction: MainType_SubType
    final_pred = f"{stage1_pred}_{stage2_pred}"
    final_conf = stage1_conf * stage2_conf  # Combined confidence
    
    return {
        "prediction": final_pred,
        "confidence": final_conf,
        "stage1": {
            "prediction": stage1_pred,
            "confidence": stage1_conf,
            "probs": stage1_probs
        },
        "stage2": {
            "prediction": stage2_pred,
            "confidence": stage2_conf,
            "probs": stage2_probs
        }
    }

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


@app.post("/api/create_metadata")
async def create_fabric_metadata(
    fabric_image: UploadFile = File(...),
    label_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None)
):
    """
    Complete Fabric Metadata Creation API
    
    Integrates:
    1. Two-stage image classification (Knit/Woven -> Subcategory)
    2. OCR parsing (composition, weight, pattern, etc.)
    3. Optional double-layer verification
    
    Returns complete metadata format.
    """
    try:
        # Step 1: Two-stage image classification
        fabric_data = await fabric_image.read()
        img = Image.open(BytesIO(fabric_data)).convert("RGB")
        
        result = two_stage_predict(img)
        
        # Build metadata structure
        metadata = {
            "fabric_id": hashlib.md5(fabric_data).hexdigest()[:12],
            "timestamp": datetime.now().isoformat(),
            "structure": {
                "primary": result["stage1"]["prediction"],
                "secondary": result["stage2"]["prediction"],
                "full_category": result["prediction"],
                "confidence": round(result["confidence"], 3),
                "stage1_confidence": round(result["stage1"]["confidence"], 3),
                "stage2_confidence": round(result["stage2"]["confidence"], 3),
                "model_version": "stage1+stage2_resnet50",
                "stage1_probabilities": {k: round(v, 3) for k, v in result["stage1"]["probs"].items()},
                "stage2_probabilities": {k: round(v, 3) for k, v in result["stage2"]["probs"].items()}
            },
            "specifications": {},
            "double_check": {
                "performed": False
            }
        }
        
        # Step 2: OCR parsing (if label provided)
        if label_image:
            label_data = await label_image.read()
            label_img = Image.open(BytesIO(label_data)).convert("RGB")
            ocr_result = CACHE.recognize_ocr(label_img)
            
            if ocr_result:
                parsed_fields = CACHE.ocr_parser.parse_all_fields(ocr_result["text"])
                metadata["specifications"] = {
                    "composition": parsed_fields.get("composition"),
                    "weight": parsed_fields.get("weight"),
                    "pattern": parsed_fields.get("pattern"),
                    "width": parsed_fields.get("width"),
                    "thickness": parsed_fields.get("thickness"),
                    "sustainability": parsed_fields.get("sustainability"),
                    "origin": parsed_fields.get("origin")
                }
                metadata["ocr_raw_text"] = ocr_result["text"]
                metadata["ocr_confidence"] = round(ocr_result["confidence"], 3)
        
        # Step 3: Double structure verification (if back image provided)
        if back_image:
            back_data = await back_image.read()
            back_img = Image.open(BytesIO(back_data)).convert("RGB")
            
            verification = CACHE.double_verifier.verify_double_structure(img, back_img)
            metadata["double_check"] = {
                "performed": True,
                "is_double_structure": verification["is_double_structure"],
                "similarity_score": verification["overall_similarity"],
                "confidence_level": verification["confidence_level"],
                "suggestion": verification["suggestion"]
            }
        
        return {
            "success": True,
            "metadata": metadata
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": str(e),
            "traceback": traceback.format_exc()
        }


# Mount static files for frontend (after API routes to avoid conflicts)
FRONTEND_DIST = os.path.join(REPO_ROOT, "production", "web", "ant_demo")
if os.path.isdir(FRONTEND_DIST):
    from fastapi.responses import FileResponse
    
    # Serve v3 as default
    @app.get("/")
    async def read_root():
        return FileResponse(os.path.join(FRONTEND_DIST, "index_v3.html"))
    
    # Serve v2 at /v2
    @app.get("/v2")
    async def read_v2():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
    
    # Serve v3 at /v3 (alias)
    @app.get("/v3")
    async def read_v3():
        return FileResponse(os.path.join(FRONTEND_DIST, "index_v3.html"))
    
    app.mount("/static", StaticFiles(directory=FRONTEND_DIST), name="static")
