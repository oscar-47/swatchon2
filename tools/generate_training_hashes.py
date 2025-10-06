import argparse
import hashlib
import os
from typing import List

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_images(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in EXTS:
                out.append(os.path.join(r, fn))
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate SHA256 hashes of training images for External-only demo filter")
    parser.add_argument("paths", nargs="+", help="One or more directories to scan recursively for images")
    parser.add_argument("--out", default=os.path.join("training", "training_hashes.txt"), help="Output text file (one hash per line)")
    args = parser.parse_args()

    all_imgs: List[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            all_imgs.extend(walk_images(p))
        else:
            print(f"Skip non-directory: {p}")
    all_imgs = sorted(set(all_imgs))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path in all_imgs:
            try:
                h = sha256_file(path)
                f.write(h + "\n")
                n += 1
            except Exception as e:
                print(f"Error hashing {path}: {e}")
    print(f"Wrote {n} hashes to {args.out}")

if __name__ == "__main__":
    main()

