import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Optional

import hashlib
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps, ImageStat

WIKI_API = "https://commons.wikimedia.org/w/api.php"
IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}

TRAIN_HASHES_PATH = os.path.join("training","training_hashes.txt")

def load_train_hashes() -> set:
    s = set()
    if os.path.isfile(TRAIN_HASHES_PATH):
        with open(TRAIN_HASHES_PATH,'r',encoding='utf-8') as f:
            for line in f:
                h=line.strip()
                if h:
                    s.add(h)
    return s


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def wiki_fetch_category_files(cat_title: str, limit: int=60) -> List[Dict]:
    """Return a list of file pages with original image URLs via imageinfo.
    cat_title should be like 'Knit fabrics' or 'Woven textiles'.
    """
    results = []
    params = {
        'action': 'query',
        'generator': 'categorymembers',
        'gcmtitle': f'Category:{cat_title}',
        'gcmnamespace': 6,  # File namespace
        'gcmtype': 'file',
        'gcmlimit': min(50, limit),
        'prop': 'imageinfo',
        'iiprop': 'url|mime',
        'format': 'json'
    }
    cont = {}
    while True and len(results) < limit:
        q = {**params, **cont}
        url = WIKI_API + '?' + urlencode(q)
        req = Request(url, headers={'User-Agent':'Mozilla/5.0 (compatible; DemoPackBot/1.0)'} )
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        pages = data.get('query',{}).get('pages',{})
        for _,page in pages.items():
            iis = page.get('imageinfo')
            if not iis:
                continue
            info = iis[0]
            img_url = info.get('url')
            mime = info.get('mime')
            if not img_url:
                continue
            results.append({'title': page.get('title'), 'url': img_url, 'mime': mime})
            if len(results) >= limit:
                break
        if 'continue' in data:
            cont = data['continue']
        else:
            break
    return results


def pick_first_working(categories: List[str], limit: int) -> List[Dict]:
    for c in categories:
        try:
            out = wiki_fetch_category_files(c, limit=limit)
            if out:
                print(f"Category '{c}' -> {len(out)} files")
                return out
            else:
                print(f"Category '{c}' returned 0 files")
        except Exception as e:
            print(f"Category '{c}' failed: {e}")
    return []


FORBID_WORDS = {
    'person','people','woman','man','maiden','girl','boy','model','portrait','face','hand','hands','armour','armor','museum','statue',
    'dress','skirt','coat','jacket','scarf','sock','socks','hat','bag','carpet','tapestry','loom','weaver','painting','figure',
    'artisan','artisans','at_work','work','architect','engineer','kimono','blog','heritage','festival','street','interior','room','band'
}

ALLOW_HINTS_KNIT = {'knit','knitted','purl','rib','waffle','stockinette','jersey','warp-knit','weft-knit','loop'}
ALLOW_HINTS_WOVEN = {'woven','weave','plain','twill','satin','gauze','basketweave','herringbone','oxford'}
ALLOW_HINTS_GENERIC = {'texture','close-up','close up','detail','macro','pattern'}
BASE_TEXTILE_WORDS = {'fabric','cloth','textile','weave','woven','knit'}

def title_ok(title: str, label_hint: Optional[str]) -> bool:
    t = (title or '').lower()
    if any(w in t for w in FORBID_WORDS):
        return False
    if label_hint == 'Knit':
        if any(w in t for w in ALLOW_HINTS_KNIT):
            return True
    if label_hint == 'Woven':
        if any(w in t for w in ALLOW_HINTS_WOVEN):
            return True
    if any(g in t for g in ALLOW_HINTS_GENERIC) and any(b in t for b in BASE_TEXTILE_WORDS):
        return True
    return False


def good_texture_image(b: bytes) -> bool:
    try:
        im = Image.open(BytesIO(b))
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if min(w, h) < 400:
            return False
        if max(w, h) / max(1, min(w, h)) > 2.5:
            return False
        # analyze central crop
        cw, ch = int(w*0.8), int(h*0.8)
        x0 = (w - cw)//2; y0 = (h - ch)//2
        crop = im.crop((x0, y0, x0+cw, y0+ch)).convert('L').resize((512, 512))
        edges = crop.filter(ImageFilter.FIND_EDGES)
        st = ImageStat.Stat(edges)
        mean_edge = st.mean[0]
        var_edge = st.var[0]
        # thresholds tuned for fabric-like textures
        return mean_edge >= 6.0 and var_edge >= 20.0
    except Exception:
        return False


def download_and_save(entries: List[Dict], out_dir: str, max_per_class: int, train_hashes: set, label_hint: Optional[str]=None) -> List[Dict]:
    ensure_dir(out_dir)
    seen_hashes = set()
    saved = []
    for e in entries:
        if len(saved) >= max_per_class:
            break
        url = e['url']
        title = e.get('title')
        try:
            if not title_ok(title, label_hint):
                print(f"skip (title forbid): {title}")
                continue
            req = Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urlopen(req, timeout=60) as resp:
                ct = resp.headers.get('Content-Type','')
                b = resp.read()
            if not ct.startswith('image/'):
                print(f"skip (not image): {url} -> {ct}")
                continue
            if not good_texture_image(b):
                print(f"skip (not texture close-up): {title}")
                continue
            h = sha256_bytes(b)
            if h in train_hashes:
                print(f"skip (in training): {url}")
                continue
            if h in seen_hashes:
                print(f"skip (dup): {url}")
                continue
            base = re.sub(r'[^a-zA-Z0-9_-]+','_', os.path.basename(url))
            root, ext = os.path.splitext(base)
            ext = ext.lower() if ext.lower() in IMG_EXTS else '.jpg'
            base = root + ext
            out_path = os.path.join(out_dir, base)
            k=1
            while os.path.exists(out_path):
                out_path = os.path.join(out_dir, f"{root}_{k}{ext}")
                k+=1
            with open(out_path,'wb') as f:
                f.write(b)
            seen_hashes.add(sha256_bytes(b))
            saved.append({
                'filename': os.path.basename(out_path),
                'path': out_path.replace('\\','/'),
                'source_url': url,
                'license': 'Wikimedia Commons (see source)'
            })
            print(f"saved: {out_path}")
        except Exception as ex:
            print(f"error downloading {url}: {ex}")
    return saved


def main():
    ap = argparse.ArgumentParser(description='Fetch offline demo pack images from Wikimedia Commons categories')
    ap.add_argument('--out-root', default=os.path.join('web','ant_demo','demo_ext','woven_vs_knit'))
    ap.add_argument('--per-class', type=int, default=40)
    ap.add_argument('--knit-cats', nargs='*', default=['Knit fabrics','Knitted fabrics','Knitting'])
    ap.add_argument('--woven-cats', nargs='*', default=['Woven textiles','Plain weave','Weaving'])
    args = ap.parse_args()

    train_hashes = load_train_hashes()
    print(f"Loaded training hashes: {len(train_hashes)}")

    knit_files = pick_first_working(args.knit_cats, limit=args.per_class*2)
    woven_files = pick_first_working(args.woven_cats, limit=args.per_class*2)

    out_knit = os.path.join(args.out_root,'Knit')
    out_woven = os.path.join(args.out_root,'Woven')

    saved_knit = download_and_save(knit_files, out_knit, args.per_class, train_hashes, label_hint='Knit')
    saved_woven = download_and_save(woven_files, out_woven, args.per_class, train_hashes, label_hint='Woven')

    manifest = {
        'created_at': time.time(),
        'note': 'Offline demo pack for Woven vs Knit from Wikimedia Commons',
        'classes': ['Knit','Woven'],
        'items': [
            *[{'label':'Knit', 'rel_path': os.path.relpath(os.path.join(out_knit,x['filename']), os.path.join('web','ant_demo')).replace('\\','/'), **{k:v for k,v in x.items() if k!='path'}} for x in saved_knit],
            *[{'label':'Woven', 'rel_path': os.path.relpath(os.path.join(out_woven,x['filename']), os.path.join('web','ant_demo')).replace('\\','/'), **{k:v for k,v in x.items() if k!='path'}} for x in saved_woven],
        ]
    }
    man_path = os.path.join('web','ant_demo','demo_ext','woven_vs_knit','manifest.json')
    ensure_dir(os.path.dirname(man_path))
    with open(man_path,'w',encoding='utf-8') as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(f"Wrote manifest: {man_path} with {len(manifest['items'])} items")

if __name__ == '__main__':
    main()

