# Fabric Classification System

AI-powered fabric classification system with OCR support for textile identification.

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/oscar-47/swatchon2.git
cd swatchon2
```

### 2. Setup Python Environment

**Create Conda Environment:**
```bash
conda create -n swatchon-r50 python=3.11
conda activate swatchon-r50
```

### 3. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn pillow easyocr
```

**Or use production requirements:**
```bash
pip install -r production/requirements.txt
```

### 4. Download Model Files

⚠️ **Model files are NOT included in the repository (270MB total)**

You need to manually place the model files in the `runs/` directory:

```
runs/
├── woven_vs_knit_r50_gpu_e5/
│   └── best.pth          # ~90MB
├── woven_r50_gpu_e5/
│   └── best.pth          # ~90MB
└── knit_r50_gpu_e5/
    └── best.pth          # ~90MB
```

**Download models from:**
- Contact the project owner to get model files
- Or train your own models using the training scripts

### 5. Start Server

**Option 1: Simple Start**
```bash
python start_server.py
```

**Option 2: Production Start**
```bash
cd production
python start.py
```

**Option 3: Direct Uvicorn**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 5000
```

### 6. Access the Application

- **Frontend:** http://localhost:5000
- **API Docs:** http://localhost:5000/docs

## System Requirements

- **Python:** 3.11+
- **CUDA:** 11.8+ (optional, for GPU acceleration)
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 500MB (excluding models)

## Project Structure

```
swatchon2/
├── server/               # FastAPI backend
│   ├── app.py           # Main application
│   └── models/          # Model loader
├── web/                 # Frontend interface
│   └── ant_demo/
│       └── index.html
├── production/          # Clean production environment
├── scripts/             # Data scraping scripts
├── training/            # Model training code
├── tools/               # Utility tools
└── runs/                # Model checkpoints (not in repo)
```

## Features

- **Two-Stage Classification**
  - Stage 1: Woven vs Knit detection
  - Stage 2: Sub-category classification
- **OCR Support**
  - Automatic label text recognition
  - Low-confidence fallback
- **Double Structure Verification**
  - Front/back similarity analysis
- **Web Interface**
  - Mobile camera support
  - Batch processing
  - Real-time predictions

## Troubleshooting

### Model files missing
```
❌ Checkpoint not found: runs/woven_vs_knit_r50_gpu_e5/best.pth
```
**Solution:** Download and place model files in `runs/` directory

### Port already in use
```
⚠️ Port 5000 already occupied
```
**Solution:** 
```bash
# Change port in start_server.py or use:
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### CUDA not available
```
Using device: cpu
```
**Solution:** This is normal. The system works on CPU, just slower.
- For GPU: Install CUDA 11.8+ and PyTorch with CUDA support

### EasyOCR fails to initialize
```
[OCR] Init failed
```
**Solution:**
```bash
pip install easyocr
```

## Development

### Run Tests
```bash
python production/test_upgraded_system.py
```

### Train Models
```bash
python training/train_woven_vs_knit_resnet50.py
```

## License

This project is for internal use only.

## Contact

For model files and support, contact: oscar-47
