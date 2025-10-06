import argparse
import os
from PIL import Image, ImageOps

def convert(src_path: str, dst_path: str | None = None) -> str:
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    root, _ = os.path.splitext(src_path)
    if not dst_path:
        dst_path = root + '.jpg'
        i = 1
        while os.path.exists(dst_path):
            dst_path = f"{root}_{i}.jpg"; i += 1
    im = Image.open(src_path)
    im = ImageOps.exif_transpose(im)
    if im.mode not in ('RGB','L'):
        im = im.convert('RGB')
    im.save(dst_path, 'JPEG', quality=92, optimize=True)
    return dst_path


def main():
    ap = argparse.ArgumentParser(description='Convert JFIF/JPEG-like image to .jpg')
    ap.add_argument('src')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = convert(args.src, args.out)
    print(out)

if __name__ == '__main__':
    main()

