import argparse
import os
import json
import hashlib
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def scan_inputs(files: List[str], input_dir: str = None, recursive: bool = True, max_samples: int = None) -> List[str]:
    paths = []
    # explicit files
    for f in files or []:
        if os.path.isfile(f) and is_image_file(f):
            paths.append(f)
    # directory
    if input_dir and os.path.isdir(input_dir):
        for root, _, fnames in os.walk(input_dir):
            for fn in fnames:
                p = os.path.join(root, fn)
                if is_image_file(p):
                    paths.append(p)
            if not recursive:
                break
    # unique preserve order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if max_samples is not None and len(uniq) > max_samples:
        uniq = uniq[:max_samples]
    return uniq


def predict(ckpt_path: str, in_paths: List[str], img_size: int = 224, device: str = 'auto') -> Tuple[List[str], List[str], List[float]]:
    dev = torch.device('cuda' if (device == 'auto' and torch.cuda.is_available()) or device == 'cuda' else 'cpu')

    ckpt = torch.load(ckpt_path, map_location=dev)
    classes = ckpt.get('classes')
    if not classes:
        raise SystemExit('Checkpoint missing classes list')
    model = build_model(len(classes)).to(dev)
    model.load_state_dict(ckpt['model'])
    model.eval()

    tfm = build_transform(img_size)

    preds = []
    probs = []
    with torch.no_grad():
        for p in in_paths:
            img = Image.open(p).convert('RGB')
            x = tfm(img).unsqueeze(0).to(dev)
            logits = model(x)
            pr = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(pr, dim=0)
            preds.append(classes[int(idx)])
            probs.append(float(conf))
    return classes, preds, probs


def save_csv(rows: List[List[str]], header: List[str], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')


def make_gallery(out_dir: str, src_paths: List[str], labels: List[str], confs: List[float]):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    entries = []
    for p, lab, conf in zip(src_paths, labels, confs):
        # stable filename
        h = hashlib.md5(p.encode('utf-8')).hexdigest()[:12]
        ext = os.path.splitext(p)[1].lower()
        dst = os.path.join(img_dir, f'{h}{ext}')
        # copy binary
        with open(p, 'rb') as fin, open(dst, 'wb') as fout:
            fout.write(fin.read())
        entries.append((os.path.relpath(dst, out_dir).replace('\\', '/'), lab, conf, os.path.basename(p)))

    # csv
    save_csv(
        rows=[[os.path.basename(src), rel, lab, f'{conf:.4f}'] for (rel, lab, conf, src) in entries],
        header=['source_file', 'gallery_path', 'pred', 'confidence'],
        out_path=os.path.join(out_dir, 'predictions.csv')
    )

    # html
    html = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8"/>',
        '<title>Image Classification Demo</title>',
        '<style>body{font-family:Segoe UI,Arial;padding:16px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px} .card{border:1px solid #ddd;border-radius:8px;overflow:hidden} .card img{width:100%;height:200px;object-fit:cover} .meta{padding:8px;font-size:14px} .pred{font-weight:600} .conf{color:#555}</style>',
        '</head><body>',
        '<h2>Image Classification Demo</h2>',
        '<p>Open this文件将图片和预测并排展示，可直接给老板演示。</p>',
        '<div class="grid">'
    ]
    for rel, lab, conf, src in entries:
        html += [
            '<div class="card">',
            f'<img src="{rel}" alt="{src}">',
            '<div class="meta">',
            f'<div class="pred">预测: {lab}</div>',
            f'<div class="conf">置信度: {conf:.2%}</div>',
            f'<div class="conf">文件: {src}</div>',
            '</div></div>'
        ]
    html += ['</div></body></html>']

    with open(os.path.join(out_dir, 'demo.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    ap = argparse.ArgumentParser(description='Run inference on images using a saved ResNet50 checkpoint')
    ap.add_argument('--ckpt', required=True, help='Path to best.pth checkpoint')
    ap.add_argument('--files', nargs='*', default=[], help='Image files to predict')
    ap.add_argument('--input-dir', default=None, help='Directory with images to predict (recursively)')
    ap.add_argument('--recursive', action='store_true', help='Recurse into subfolders')
    ap.add_argument('--max-samples', type=int, default=None, help='Limit number of images for quick demo')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--out', required=True, help='Output folder for demo artifacts (HTML + CSV + copied images)')
    args = ap.parse_args()

    inputs = scan_inputs(args.files, args.input_dir, args.recursive, args.max_samples)
    if not inputs:
        raise SystemExit('No input images found. Specify --files or --input-dir')

    classes, labels, confs = predict(args.ckpt, inputs, img_size=args.img_size, device=args.device)

    os.makedirs(args.out, exist_ok=True)
    # Save raw predictions json
    with open(os.path.join(args.out, 'raw_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'classes': classes,
            'samples': [
                {'path': p, 'pred': l, 'confidence': c}
                for p, l, c in zip(inputs, labels, confs)
            ]
        }, f, ensure_ascii=False, indent=2)

    make_gallery(args.out, inputs, labels, confs)

    print(f'Demo saved at: {args.out}/demo.html and predictions.csv (classes={classes})')


if __name__ == '__main__':
    main()

