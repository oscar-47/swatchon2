#!/usr/bin/env python3
"""
Scrape professional fabric texture images from GlobalTextiles and TNC.

The collector is tuned for FabricFlow dataset conventions:
  - FabricFlow_Dataset/{L1}/{Class}/{source}/
  - {class_lower}_{pattern}_{source}_{seq:04d}.jpg
  - per-image JSON sidecar
  - CSV logs in scripts/logs/

Notes:
  - globaltextiles search works via:
      https://www.globaltextiles.com/search/product.html?keyword=...
    with pagination on pageId.
  - tnc market search works via:
      https://ml.tnc.com.cn/search/product-c-90-k-{GBK_ENCODED_TERM}-p{page}.html
    after priming a session with https://ml.tnc.com.cn/
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image


DATASET_ROOT = Path("FabricFlow_Dataset")
LOG_ROOT = Path("scripts/logs")
MIN_SIZE = 224
HASH_THRESHOLD = 5
REQUEST_DELAY = 0.4
TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

PROFESSIONAL_SOURCES = ("globaltextiles", "tnc")

BASE_NEGATIVE_TOKENS = (
    "machine",
    "machinery",
    "equipment",
    "sweater",
    "pullover",
    "cardigan",
    "dress",
    "skirt",
    "jacket",
    "coat",
    "boots",
    "boot",
    "shoe",
    "shoes",
    "sock",
    "blanket",
    "bedding",
    "pillow",
    "curtain",
    "bag",
    "toy",
    "throw",
    "yarn",
    "wig",
    "knitting machine",
    "flat knitting",
    "毛衣",
    "针织衫",
    "开衫",
    "连衣裙",
    "半裙",
    "外套",
    "靴",
    "鞋",
    "袜",
    "毛毯",
    "毯",
    "床品",
    "枕",
    "窗帘",
    "玩具",
    "纱线",
    "毛线",
    "机械",
    "机器",
    "设备",
    "针织机",
)


@dataclass(frozen=True)
class ClassSpec:
    name: str
    l1: str
    target: int
    pattern_token: str
    positive_tokens: Tuple[str, ...]
    negative_tokens: Tuple[str, ...]
    queries: Dict[str, Tuple[str, ...]]


CLASS_SPECS: Dict[str, ClassSpec] = {
    "Double_Jersey": ClassSpec(
        name="Double_Jersey",
        l1="KNIT",
        target=400,
        pattern_token="base",
        positive_tokens=(
            "double jersey",
            "double knit",
            "double knitted",
            "双面针织",
            "双面布",
            "双面汗布",
            "平板布",
            "佳积布",
            "罗马布",
        ),
        negative_tokens=("mattress",),
        queries={
            "globaltextiles": ("double jersey fabric", "double jersey", "double knit"),
            "tnc": ("双面针织", "双面布", "双面汗布", "double jersey", "double knit"),
        },
    ),
    "Basket_Hopsack": ClassSpec(
        name="Basket_Hopsack",
        l1="WOVEN",
        target=300,
        pattern_token="base",
        positive_tokens=("hopsack", "basket weave", "basketweave", "panama", "巴拿马"),
        negative_tokens=("needle", "knit", "针织", "pillow", "枕"),
        queries={
            "globaltextiles": ("hopsack", "hopsack fabric", "panama fabric"),
            "tnc": ("巴拿马布", "巴拿马", "hopsack", "panama fabric", "basket weave"),
        },
    ),
    "Cable_Knit": ClassSpec(
        name="Cable_Knit",
        l1="KNIT",
        target=300,
        pattern_token="base",
        positive_tokens=("cable knit", "cable knitted", "麻花", "绞花"),
        negative_tokens=(),
        queries={
            "globaltextiles": ("cable knit fabric", "cable knit"),
            "tnc": ("麻花针织", "麻花", "绞花针织", "麻花提花", "麻花布", "绞花布", "cable knit"),
        },
    ),
    "Purl_Knit": ClassSpec(
        name="Purl_Knit",
        l1="KNIT",
        target=300,
        pattern_token="base",
        positive_tokens=("purl knit", "purl", "反面针织", "反面布"),
        negative_tokens=(),
        queries={
            "globaltextiles": ("purl knit fabric", "purl knit", "purl fabric"),
            "tnc": ("反面针织", "反面布", "purl knit"),
        },
    ),
    "Intarsia": ClassSpec(
        name="Intarsia",
        l1="KNIT",
        target=200,
        pattern_token="intarsia",
        positive_tokens=("intarsia", "嵌花"),
        negative_tokens=("computerized", "flat knit"),
        queries={
            "globaltextiles": ("intarsia fabric", "intarsia"),
            "tnc": ("嵌花", "嵌花针织", "intarsia"),
        },
    ),
    "Raschel": ClassSpec(
        name="Raschel",
        l1="KNIT",
        target=200,
        pattern_token="base",
        positive_tokens=("raschel", "warp knit", "power mesh", "经编", "拉舍尔", "经编网布"),
        negative_tokens=("blanket", "bedding", "毛毯", "床品"),
        queries={
            "globaltextiles": ("raschel", "raschel fabric", "raschel mesh", "warp knit"),
            "tnc": ("经编", "拉舍尔", "经编网布", "raschel", "warp knit"),
        },
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def squish(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    return folded.encode("ascii", "ignore").decode("ascii").lower()


def normalize_url(url: str, base_url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    return urljoin(base_url, url)


def class_token(class_name: str) -> str:
    return class_name.lower()


def filename_for(class_spec: ClassSpec, source: str, seq: int) -> str:
    token = class_token(class_spec.name)
    return f"{token}_{class_spec.pattern_token}_{source}_{seq:04d}.jpg"


def hamming_distance(a: int, b: int) -> int:
    xor = a ^ b
    try:
        return xor.bit_count()
    except AttributeError:
        return bin(xor).count("1")


def dhash_pil(img: Image.Image) -> int:
    resampling = getattr(Image, "Resampling", Image)
    lanczos = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
    gray = img.convert("L").resize((9, 8), lanczos)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        start = row * 9
        current = pixels[start : start + 9]
        for col in range(8):
            bits = (bits << 1) | (1 if current[col] > current[col + 1] else 0)
    return bits


def dhash_path(path: Path) -> Optional[int]:
    try:
        with Image.open(path) as img:
            return dhash_pil(img)
    except Exception:
        return None


def is_duplicate(image_hash: int, existing_hashes: Sequence[int], threshold: int = HASH_THRESHOLD) -> bool:
    return any(hamming_distance(image_hash, other) <= threshold for other in existing_hashes)


def trim_uniform_border(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    corner_samples = [
        pixels[0, 0],
        pixels[width - 1, 0],
        pixels[0, height - 1],
        pixels[width - 1, height - 1],
    ]
    avg = tuple(sum(c[idx] for c in corner_samples) // len(corner_samples) for idx in range(3))

    def is_bg(px: Tuple[int, int, int]) -> bool:
        return sum(abs(int(px[i]) - avg[i]) for i in range(3)) <= 36

    left = 0
    while left < width - 1:
        bg_ratio = sum(1 for y in range(height) if is_bg(pixels[left, y])) / float(height)
        if bg_ratio < 0.985:
            break
        left += 1

    right = width - 1
    while right > left:
        bg_ratio = sum(1 for y in range(height) if is_bg(pixels[right, y])) / float(height)
        if bg_ratio < 0.985:
            break
        right -= 1

    top = 0
    while top < height - 1:
        bg_ratio = sum(1 for x in range(width) if is_bg(pixels[x, top])) / float(width)
        if bg_ratio < 0.985:
            break
        top += 1

    bottom = height - 1
    while bottom > top:
        bg_ratio = sum(1 for x in range(width) if is_bg(pixels[x, bottom])) / float(width)
        if bg_ratio < 0.985:
            break
        bottom -= 1

    if left == 0 and top == 0 and right == width - 1 and bottom == height - 1:
        return rgb

    cropped = rgb.crop((left, top, right + 1, bottom + 1))
    if cropped.width < MIN_SIZE or cropped.height < MIN_SIZE:
        return rgb
    return cropped


def load_existing_hashes(class_spec: ClassSpec) -> List[int]:
    hashes: List[int] = []
    class_dir = DATASET_ROOT / class_spec.l1 / class_spec.name
    for source in PROFESSIONAL_SOURCES:
        src_dir = class_dir / source
        if not src_dir.exists():
            continue
        for image_path in sorted(src_dir.glob("*.jpg")):
            image_hash = dhash_path(image_path)
            if image_hash is not None:
                hashes.append(image_hash)
    return hashes


def professional_count(class_spec: ClassSpec) -> int:
    class_dir = DATASET_ROOT / class_spec.l1 / class_spec.name
    count = 0
    for source in PROFESSIONAL_SOURCES:
        src_dir = class_dir / source
        if src_dir.exists():
            count += len(list(src_dir.glob("*.jpg")))
    return count


def next_sequence(class_spec: ClassSpec, source: str) -> int:
    source_dir = DATASET_ROOT / class_spec.l1 / class_spec.name / source
    source_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{class_token(class_spec.name)}_{class_spec.pattern_token}_{source}_"
    max_seq = 0
    for path in source_dir.glob("*.jpg"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        tail = stem[len(prefix) :]
        if tail.isdigit():
            max_seq = max(max_seq, int(tail))
    return max_seq + 1


class CsvLogger:
    def __init__(self, source: str) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_ROOT / f"{stamp}_{source}_scrape.csv"
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=[
                "timestamp",
                "source",
                "class_name",
                "query",
                "search_url",
                "page",
                "product_url",
                "product_title",
                "image_url",
                "status",
                "reason",
                "filename",
                "width",
                "height",
            ],
        )
        self.writer.writeheader()

    def log(self, **row: object) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class SourceClient:
    source_name: str

    def iter_search_results(self, class_spec: ClassSpec, query: str, max_pages: int) -> Iterable[Dict[str, object]]:
        raise NotImplementedError

    def fetch_product(self, product_url: str) -> Dict[str, object]:
        raise NotImplementedError

    def download_image(self, image_url: str, referer: str) -> bytes:
        raise NotImplementedError


class GlobalTextilesClient(SourceClient):
    source_name = "globaltextiles"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def iter_search_results(self, class_spec: ClassSpec, query: str, max_pages: int) -> Iterable[Dict[str, object]]:
        seen: Set[str] = set()
        for page in range(1, max_pages + 1):
            params = {"keyword": query}
            if page > 1:
                params["pageId"] = str(page)
            url = "https://www.globaltextiles.com/search/product.html"
            response = self.session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            rows: List[Dict[str, object]] = []
            for anchor in soup.select('a[href*="/product/"]'):
                href = normalize_url(anchor.get("href", ""), response.url)
                title = collapse_ws(anchor.get_text(" ", strip=True))
                if not href or not title:
                    continue
                if "/product/" not in href or href in seen:
                    continue
                seen.add(href)
                rows.append(
                    {
                        "page": page,
                        "search_url": response.url,
                        "product_url": href,
                        "title": title,
                    }
                )

            if not rows:
                break

            for row in rows:
                yield row

            time.sleep(REQUEST_DELAY)

    def fetch_product(self, product_url: str) -> Dict[str, object]:
        response = self.session.get(product_url, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = collapse_ws(soup.title.get_text(" ", strip=True) if soup.title else "")
        heading = collapse_ws(" ".join(h.get_text(" ", strip=True) for h in soup.select("h1") if h.get_text(strip=True)))
        if heading:
            title = heading

        image_urls: List[str] = []
        for match in re.findall(r"largeimage\s*[:=]\s*['\"]([^'\"]+)", response.text):
            url = normalize_url(match, response.url)
            if url not in image_urls:
                image_urls.append(url)

        for anchor in soup.select('a[href*="img.globaltextiles.com"]'):
            url = normalize_url(anchor.get("href", ""), response.url)
            if url and url not in image_urls and "_100X100" not in url:
                image_urls.append(url)

        return {"title": title, "product_url": response.url, "image_urls": image_urls}

    def download_image(self, image_url: str, referer: str) -> bytes:
        headers = dict(HEADERS)
        headers["Referer"] = referer
        response = self.session.get(image_url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        return response.content


class TncClient(SourceClient):
    source_name = "tnc"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.get("https://ml.tnc.com.cn/", timeout=TIMEOUT)

    @staticmethod
    def encode_query(query: str) -> str:
        return quote(query.encode("gbk"))

    def iter_search_results(self, class_spec: ClassSpec, query: str, max_pages: int) -> Iterable[Dict[str, object]]:
        seen: Set[str] = set()
        encoded = self.encode_query(query)
        for page in range(1, max_pages + 1):
            url = f"https://ml.tnc.com.cn/search/product-c-90-k-{encoded}-p{page}.html"
            response = self.session.get(
                url,
                headers={"Referer": "https://ml.tnc.com.cn/", **HEADERS},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            rows: List[Dict[str, object]] = []
            for anchor in soup.select('a[href*="/product/"]'):
                href = normalize_url(anchor.get("href", ""), response.url)
                title = collapse_ws(anchor.get_text(" ", strip=True))
                if not href or not title:
                    continue
                if "/product/" not in href or href in seen:
                    continue
                seen.add(href)
                rows.append(
                    {
                        "page": page,
                        "search_url": response.url,
                        "product_url": href,
                        "title": title,
                    }
                )

            if not rows:
                break

            for row in rows:
                yield row

            time.sleep(REQUEST_DELAY)

    def fetch_product(self, product_url: str) -> Dict[str, object]:
        response = self.session.get(product_url, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = collapse_ws(
            " ".join(h.get_text(" ", strip=True) for h in soup.select("h1") if h.get_text(strip=True))
        )
        if not title and soup.title:
            title = collapse_ws(soup.title.get_text(" ", strip=True))

        image_urls: List[str] = []
        for match in re.findall(r"largeimage=['\"]([^'\"]+)", response.text):
            url = normalize_url(match, response.url)
            if url and url not in image_urls:
                image_urls.append(url)

        for img in soup.select("img"):
            src = normalize_url(img.get("src", ""), response.url)
            if not src or "imgtnc.tnccdn.com" not in src:
                continue
            src = src.replace("_100X100", "")
            if src not in image_urls:
                image_urls.append(src)

        return {"title": title, "product_url": response.url, "image_urls": image_urls}

    def download_image(self, image_url: str, referer: str) -> bytes:
        response = self.session.get(image_url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        return response.content


def contains_any(text: str, tokens: Sequence[str]) -> bool:
    lower = collapse_ws(text).lower()
    folded = ascii_fold(lower)
    nospace = squish(text)
    for token in tokens:
        token_lower = token.lower()
        token_nospace = squish(token)
        if re.search(r"[a-z]", token_lower):
            words = [re.escape(part) for part in token_lower.split()]
            pattern = r"\b" + r"[\s/_-]+".join(words) + r"\b"
            if re.search(pattern, lower) or re.search(pattern, folded):
                return True
        elif token_lower in lower or token_nospace in nospace:
            return True
    return False


def title_matches(class_spec: ClassSpec, title: str) -> bool:
    if not title:
        return False
    if not contains_any(title, class_spec.positive_tokens):
        return False
    all_negatives = BASE_NEGATIVE_TOKENS + class_spec.negative_tokens
    if contains_any(title, all_negatives):
        return False
    if class_spec.l1 == "WOVEN" and contains_any(title, ("knit", "knitted", "针织")):
        return False
    if class_spec.l1 == "KNIT" and contains_any(title, ("woven", "梭织")):
        return False
    if class_spec.name == "Cable_Knit" and not contains_any(title, ("knit", "knitted", "针织")):
        return False
    return True


def prepare_image(content: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        with Image.open(io.BytesIO(content)) as img:
            prepared = trim_uniform_border(img)
            prepared = prepared.convert("RGB")
            if prepared.width < MIN_SIZE or prepared.height < MIN_SIZE:
                return None, "too_small"
            return prepared, None
    except Exception:
        return None, "invalid_image"


def save_image(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=95, optimize=True)


def write_metadata(
    out_path: Path,
    *,
    class_spec: ClassSpec,
    source: str,
    query: str,
    search_url: str,
    product_url: str,
    product_title: str,
    image_url: str,
    width: int,
    height: int,
    image_hash: int,
) -> None:
    payload = {
        "source": source,
        "source_site": source,
        "class_name": class_spec.name,
        "l1": class_spec.l1,
        "query": query,
        "search_url": search_url,
        "source_url": product_url,
        "product_title": product_title,
        "image_url": image_url,
        "filename": out_path.name,
        "width": width,
        "height": height,
        "image_hash": image_hash,
        "downloaded_at": utc_now(),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def scrape_source(
    client: SourceClient,
    class_names: Sequence[str],
    max_pages: int,
    delay: float,
    dry_run: bool,
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    logger = CsvLogger(client.source_name)
    try:
        for class_name in class_names:
            class_spec = CLASS_SPECS[class_name]
            total_existing = professional_count(class_spec)
            needed = max(class_spec.target - total_existing, 0)
            class_dir = DATASET_ROOT / class_spec.l1 / class_spec.name / client.source_name
            class_dir.mkdir(parents=True, exist_ok=True)
            seq = next_sequence(class_spec, client.source_name)
            existing_hashes = load_existing_hashes(class_spec)
            seen_products: Set[str] = set()

            class_stats = {
                "existing_professional": total_existing,
                "needed_at_start": needed,
                "downloaded": 0,
                "duplicates": 0,
                "rejected_title": 0,
                "small_or_invalid": 0,
                "errors": 0,
            }
            summary[class_name] = class_stats

            print(
                f"\n[{client.source_name}] {class_spec.name}: "
                f"existing={total_existing}, target={class_spec.target}, remaining={needed}"
            )

            if needed <= 0:
                continue

            for query in class_spec.queries[client.source_name]:
                if class_stats["downloaded"] >= needed:
                    break

                print(f"  query: {query}")
                try:
                    search_rows = client.iter_search_results(class_spec, query, max_pages=max_pages)
                    for row in search_rows:
                        if class_stats["downloaded"] >= needed:
                            break

                        page = int(row["page"])
                        search_url = str(row["search_url"])
                        product_url = str(row["product_url"])
                        result_title = str(row["title"])

                        if product_url in seen_products:
                            continue
                        seen_products.add(product_url)

                        if not title_matches(class_spec, result_title):
                            class_stats["rejected_title"] += 1
                            logger.log(
                                timestamp=utc_now(),
                                source=client.source_name,
                                class_name=class_spec.name,
                                query=query,
                                search_url=search_url,
                                page=page,
                                product_url=product_url,
                                product_title=result_title,
                                image_url="",
                                status="reject",
                                reason="title_filter",
                                filename="",
                                width="",
                                height="",
                            )
                            continue

                        try:
                            product = client.fetch_product(product_url)
                        except Exception as exc:
                            class_stats["errors"] += 1
                            logger.log(
                                timestamp=utc_now(),
                                source=client.source_name,
                                class_name=class_spec.name,
                                query=query,
                                search_url=search_url,
                                page=page,
                                product_url=product_url,
                                product_title=result_title,
                                image_url="",
                                status="error",
                                reason=f"detail_fetch:{type(exc).__name__}",
                                filename="",
                                width="",
                                height="",
                            )
                            continue

                        product_title = str(product.get("title") or result_title)
                        if not title_matches(class_spec, product_title):
                            class_stats["rejected_title"] += 1
                            logger.log(
                                timestamp=utc_now(),
                                source=client.source_name,
                                class_name=class_spec.name,
                                query=query,
                                search_url=search_url,
                                page=page,
                                product_url=product_url,
                                product_title=product_title,
                                image_url="",
                                status="reject",
                                reason="detail_title_filter",
                                filename="",
                                width="",
                                height="",
                            )
                            continue

                        image_urls = list(dict.fromkeys(str(u) for u in product.get("image_urls") or [] if str(u)))
                        if not image_urls:
                            class_stats["errors"] += 1
                            logger.log(
                                timestamp=utc_now(),
                                source=client.source_name,
                                class_name=class_spec.name,
                                query=query,
                                search_url=search_url,
                                page=page,
                                product_url=product_url,
                                product_title=product_title,
                                image_url="",
                                status="error",
                                reason="no_images",
                                filename="",
                                width="",
                                height="",
                            )
                            continue

                        for image_url in image_urls:
                            if class_stats["downloaded"] >= needed:
                                break

                            try:
                                content = client.download_image(image_url, referer=product_url)
                            except Exception as exc:
                                class_stats["errors"] += 1
                                logger.log(
                                    timestamp=utc_now(),
                                    source=client.source_name,
                                    class_name=class_spec.name,
                                    query=query,
                                    search_url=search_url,
                                    page=page,
                                    product_url=product_url,
                                    product_title=product_title,
                                    image_url=image_url,
                                    status="error",
                                    reason=f"image_fetch:{type(exc).__name__}",
                                    filename="",
                                    width="",
                                    height="",
                                )
                                continue

                            prepared, error_reason = prepare_image(content)
                            if prepared is None:
                                class_stats["small_or_invalid"] += 1
                                logger.log(
                                    timestamp=utc_now(),
                                    source=client.source_name,
                                    class_name=class_spec.name,
                                    query=query,
                                    search_url=search_url,
                                    page=page,
                                    product_url=product_url,
                                    product_title=product_title,
                                    image_url=image_url,
                                    status="reject",
                                    reason=error_reason or "invalid_image",
                                    filename="",
                                    width="",
                                    height="",
                                )
                                continue

                            image_hash = dhash_pil(prepared)
                            if is_duplicate(image_hash, existing_hashes):
                                class_stats["duplicates"] += 1
                                logger.log(
                                    timestamp=utc_now(),
                                    source=client.source_name,
                                    class_name=class_spec.name,
                                    query=query,
                                    search_url=search_url,
                                    page=page,
                                    product_url=product_url,
                                    product_title=product_title,
                                    image_url=image_url,
                                    status="reject",
                                    reason="duplicate",
                                    filename="",
                                    width=prepared.width,
                                    height=prepared.height,
                                )
                                continue

                            filename = filename_for(class_spec, client.source_name, seq)
                            out_path = class_dir / filename
                            if not dry_run:
                                save_image(prepared, out_path)
                                write_metadata(
                                    out_path,
                                    class_spec=class_spec,
                                    source=client.source_name,
                                    query=query,
                                    search_url=search_url,
                                    product_url=product_url,
                                    product_title=product_title,
                                    image_url=image_url,
                                    width=prepared.width,
                                    height=prepared.height,
                                    image_hash=image_hash,
                                )
                            existing_hashes.append(image_hash)
                            seq += 1
                            class_stats["downloaded"] += 1
                            logger.log(
                                timestamp=utc_now(),
                                source=client.source_name,
                                class_name=class_spec.name,
                                query=query,
                                search_url=search_url,
                                page=page,
                                product_url=product_url,
                                product_title=product_title,
                                image_url=image_url,
                                status="saved",
                                reason="ok",
                                filename=filename,
                                width=prepared.width,
                                height=prepared.height,
                            )
                            if delay > 0:
                                time.sleep(delay)
                except Exception as exc:
                    class_stats["errors"] += 1
                    logger.log(
                        timestamp=utc_now(),
                        source=client.source_name,
                        class_name=class_spec.name,
                        query=query,
                        search_url="",
                        page="",
                        product_url="",
                        product_title="",
                        image_url="",
                        status="error",
                        reason=f"search:{type(exc).__name__}",
                        filename="",
                        width="",
                        height="",
                    )

            print(
                "  downloaded={downloaded} duplicates={duplicates} "
                "title_rejects={rejected_title} small_or_invalid={small_or_invalid} "
                "errors={errors}".format(**class_stats)
            )
    finally:
        logger.close()
        print(f"\nLog written to {logger.path}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape professional fabric texture images")
    parser.add_argument(
        "--source",
        choices=("globaltextiles", "tnc", "both"),
        required=True,
        help="Which source to scrape.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=sorted(CLASS_SPECS.keys()),
        help="Optional subset of classes to scrape.",
    )
    parser.add_argument("--max-pages", type=int, default=8, help="Max search pages per query.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between saved images.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    invalid = [name for name in args.classes if name not in CLASS_SPECS]
    if invalid:
        print(f"Unknown classes: {', '.join(invalid)}", file=sys.stderr)
        return 2

    sources = [args.source] if args.source != "both" else ["globaltextiles", "tnc"]
    clients: Dict[str, SourceClient] = {
        "globaltextiles": GlobalTextilesClient(),
        "tnc": TncClient(),
    }

    all_summaries: Dict[str, Dict[str, Dict[str, int]]] = {}
    for source in sources:
        summary = scrape_source(
            clients[source],
            class_names=args.classes,
            max_pages=args.max_pages,
            delay=args.delay,
            dry_run=args.dry_run,
        )
        all_summaries[source] = summary

    print("\nSummary")
    for source, source_summary in all_summaries.items():
        print(f"  {source}:")
        for class_name, stats in source_summary.items():
            print(
                f"    {class_name}: downloaded={stats['downloaded']} "
                f"existing={stats['existing_professional']} "
                f"needed={stats['needed_at_start']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
