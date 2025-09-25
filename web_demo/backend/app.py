from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import hashlib
import time
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Global model cache
MODEL_CACHE = {}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_model(n_classes: int) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, n_classes)
    return model

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_model(ckpt_path: str, device: str = 'auto'):
    if ckpt_path in MODEL_CACHE:
        return MODEL_CACHE[ckpt_path]
    
    dev = torch.device('cuda' if (device == 'auto' and torch.cuda.is_available()) or device == 'cuda' else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=dev)
    classes = ckpt.get('classes', [])
    
    model = build_model(len(classes)).to(dev)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    MODEL_CACHE[ckpt_path] = {
        'model': model,
        'classes': classes,
        'device': dev,
        'transform': build_transform(224)
    }
    return MODEL_CACHE[ckpt_path]

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    models = []
    
    # Check for binary model
    binary_path = '../../runs/woven_vs_knit_r50_gpu_e5/best.pth'
    if os.path.exists(binary_path):
        models.append({
            'id': 'woven_vs_knit',
            'name': 'Woven vs Knit (二分类)',
            'description': '区分机织(Woven)和针织(Knit)面料',
            'type': 'binary',
            'path': binary_path
        })

    # Check for multi-class models
    woven_path = '../../runs/woven_r50_gpu_e5/best.pth'
    if os.path.exists(woven_path):
        models.append({
            'id': 'woven_multi',
            'name': 'Woven 多分类',
            'description': '机织面料细分类别识别',
            'type': 'multiclass',
            'path': woven_path
        })

    knit_path = '../../runs/knit_r50_gpu_e5/best.pth'
    if os.path.exists(knit_path):
        models.append({
            'id': 'knit_multi',
            'name': 'Knit 多分类',
            'description': '针织面料细分类别识别',
            'type': 'multiclass',
            'path': knit_path
        })
    
    return jsonify({'models': models})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict image classification"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        image_data = data.get('image')  # base64 encoded
        
        if not model_id or not image_data:
            return jsonify({'error': 'Missing model_id or image data'}), 400
        
        # Get model info
        models_resp = get_models()
        models = models_resp.get_json()['models']
        model_info = next((m for m in models if m['id'] == model_id), None)
        
        if not model_info:
            return jsonify({'error': 'Model not found'}), 404
        
        # Load model
        model_data = load_model(model_info['path'])
        
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Predict
        transform = model_data['transform']
        model = model_data['model']
        classes = model_data['classes']
        device = model_data['device']
        
        with torch.no_grad():
            x = transform(image).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, min(len(classes), 5))
            
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                predictions.append({
                    'class': classes[int(idx)],
                    'confidence': float(prob),
                    'rank': i + 1
                })
        
        return jsonify({
            'success': True,
            'model': model_info,
            'predictions': predictions,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch predict multiple images"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        images = data.get('images', [])  # list of base64 encoded images
        
        if not model_id or not images:
            return jsonify({'error': 'Missing model_id or images'}), 400
        
        # Get model info
        models_resp = get_models()
        models = models_resp.get_json()['models']
        model_info = next((m for m in models if m['id'] == model_id), None)
        
        if not model_info:
            return jsonify({'error': 'Model not found'}), 404
        
        # Load model
        model_data = load_model(model_info['path'])
        
        results = []
        for i, image_data in enumerate(images):
            try:
                # Decode image
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Predict
                transform = model_data['transform']
                model = model_data['model']
                classes = model_data['classes']
                device = model_data['device']
                
                with torch.no_grad():
                    x = transform(image).unsqueeze(0).to(device)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    
                    # Get top prediction
                    conf, idx = torch.max(probs, dim=0)
                    
                    results.append({
                        'index': i,
                        'success': True,
                        'prediction': {
                            'class': classes[int(idx)],
                            'confidence': float(conf)
                        }
                    })
                    
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'model': model_info,
            'results': results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('../frontend/dist', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend/dist', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
