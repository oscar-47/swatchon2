from __future__ import annotations

import os
import sys
import json
import time
import argparse
import urllib.request
from typing import List, Set, Dict


# 定义所有类别的配置
CATEGORIES = {
    "Plain": {
        "categoryIds": "167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170&sort=&from=/wholesale-fabric"
    },
    "Twill_Weave": {
        "categoryIds": "186,189,175,253,196,194",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=186,189,175,253,196,194&sort=&from=/wholesale-fabric"
    },
    "Satin_Weave": {
        "categoryIds": "254,255,256",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=254,255,256&sort=&from=/wholesale-fabric"
    },
    "Jacquard_Weave": {
        "categoryIds": "184,183",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=184,183&sort=&from=/wholesale-fabric"
    },
    "Pile_Weave": {
        "categoryIds": "188,247",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=188,247&sort=&from=/wholesale-fabric"
    },
    "Dobby": {
        "categoryIds": "171",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=171&sort=&from=/wholesale-fabric"
    },
    "Double_Weave": {
        "categoryIds": "185",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=185&sort=&from=/wholesale-fabric"
    },
    "Eyelet": {
        "categoryIds": "177",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=177&sort=&from=/wholesale-fabric"
    },
    "Ripstop": {
        "categoryIds": "191",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=191&sort=&from=/wholesale-fabric"
    },
    # Locked from SwatchOn filter API: Poplin=168, Gauze=170 (retrieved on 2026-03-03)
    "Poplin": {
        "categoryIds": "168",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=168&sort=&from=/wholesale-fabric"
    },
    "Gauze": {
        "categoryIds": "170",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=170&sort=&from=/wholesale-fabric"
    }
}

CONFIG_PATH = os.path.join("scripts", "config", "targets_phase1_fabricflow.json")

# SwatchOn API base for direct queries (bypasses Nuxt frontend pagination issues)
SWATCHON_API_BASE = "https://api.swatchon.com/api/mall/v1/search/qualities"
SWATCHON_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
PER_PAGE = 48


def parse_only_categories(only_arg: str | None) -> Set[str]:
    """Parse --only argument to a normalized category-name set."""
    if not only_arg:
        return set()
    return {name.strip().lower() for name in only_arg.split(",") if name.strip()}


def load_phase_overrides() -> Dict[str, Dict[str, object]]:
    """Load optional category ID/target overrides from phase config."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        out: Dict[str, Dict[str, object]] = {}
        for item in obj.get("class_plan", []):
            key = item.get("key")
            if not isinstance(key, str):
                continue
            swatchon = item.get("swatchon", {})
            category_ids = swatchon.get("category_ids") if isinstance(swatchon, dict) else None
            out[key] = {
                "target": item.get("target"),
                "category_ids": category_ids,
            }
        return out
    except Exception:
        return {}


def fetch_api_page(category_ids: str, page: int, retries: int = 3) -> dict:
    """Fetch one page of results from SwatchOn search API with retries."""
    url = (
        f"{SWATCHON_API_BASE}?sort=&page={page}&perPage={PER_PAGE}"
        f"&categoryIds={category_ids}&preferredCurrency=usd&shippingCountry=US"
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=SWATCHON_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
            else:
                raise


def scrape_category(category_name: str, category_config: dict, target_count: int = 150, max_pages: int = 50) -> dict:
    """Scrape product links via SwatchOn API (no browser needed)."""

    print(f"\n{'='*80}")
    print(f"  Category: {category_name}")
    print(f"  Target: {target_count} links")
    print(f"  categoryIds: {category_config['categoryIds']}")
    print(f"{'='*80}")

    all_links: Set[str] = set()
    page_results = []
    category_ids = category_config["categoryIds"]
    current_page = 1

    while current_page <= max_pages:
        try:
            data = fetch_api_page(category_ids, current_page)
            total = data.get("total", 0)
            items = data.get("items", [])

            if not items:
                print(f"  [INFO] Page {current_page} returned 0 items, stopping")
                break

            before = len(all_links)
            for item in items:
                landing = item.get("landingUrl", "")
                if landing:
                    full_url = "https://swatchon.com" + landing
                    all_links.add(full_url)

            page_results.append({
                "page": current_page,
                "links_found": len(items),
                "new_unique_links": len(all_links) - before,
                "total_unique_links": len(all_links),
            })
            print(f"  [INFO] Page {current_page}: {len(items)} items, {len(all_links)}/{total} unique links")

            if len(all_links) >= total or current_page * PER_PAGE >= total:
                break
            if len(all_links) >= target_count:
                break

            current_page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  [ERROR] Page {current_page} exception: {e}")
            current_page += 1
            time.sleep(1)
            continue

    return {
        "category": category_name,
        "timestamp": time.time(),
        "target_count": target_count,
        "actual_count": len(all_links),
        "pages_scraped": len(page_results),
        "page_details": page_results,
        "all_links": sorted(list(all_links)),
    }


def main():
    """主函数 - 依次爬取所有分类"""
    parser = argparse.ArgumentParser(description="Scrape SwatchOn woven category links")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Run only selected categories, comma-separated (example: Poplin,Gauze)",
    )
    args = parser.parse_args()
    selected_only = parse_only_categories(args.only)
    overrides = load_phase_overrides()

    # Apply phase config overrides (targets + categoryIds) if present.
    for key in ("Poplin", "Gauze"):
        ov = overrides.get(key) or {}
        cat_ids = ov.get("category_ids")
        if isinstance(cat_ids, list) and cat_ids:
            try:
                cid = ",".join(str(int(x)) for x in cat_ids)
                CATEGORIES[key]["categoryIds"] = cid
                CATEGORIES[key]["url"] = f"https://swatchon.com/wholesale-fabric?categoryIds={cid}&sort=&from=/wholesale-fabric"
            except Exception:
                pass
    
    print("🚀 SwatchOn 全自动分类爬虫启动")
    print("=" * 80)
    
    # 创建输出目录
    base_output_dir = os.path.join(os.getcwd(), "outputs", "categories")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 总体统计
    total_start_time = time.time()
    all_results = {}
    overall_stats = {
        "total_categories": len(CATEGORIES),
        "completed_categories": 0,
        "total_links": 0,
        "start_time": total_start_time,
        "category_results": {}
    }
    
    # Adaptive target counts based on model performance
    # P0 (极差): 500-600, P2 (中等): 300-350, Good/Excellent: 200 (slight buffer)
    CATEGORY_TARGET_COUNTS = {
        "Dobby": 600,            # P0: 21.7% accuracy - CRITICAL!!!
        "Double_Weave": 350,     # P2: 60.9% - moderate improvement
        "Jacquard_Weave": 350,   # P2: 60.9% - moderate improvement
        "Plain": 350,            # P2: 65.2% - moderate improvement
        "Satin_Weave": 350,      # P2: 65.2% - moderate improvement
        "Eyelet": 200,           # Excellent: 95.2% - keep current (buffer)
        "Pile_Weave": 200,       # Excellent: 95.7% - keep current (buffer)
        "Ripstop": 200,          # Good: 87.0% - keep current (buffer)
        "Twill_Weave": 200,      # Good: 87.0% - keep current (buffer)
        "Poplin": 300,           # FabricFlow phase target: 300
        "Gauze": 200,            # FabricFlow phase target: 200
    }
    for key in ("Poplin", "Gauze"):
        ov = overrides.get(key) or {}
        target = ov.get("target")
        if isinstance(target, int) and target > 0:
            CATEGORY_TARGET_COUNTS[key] = target

    run_items = [
        (name, cfg)
        for name, cfg in CATEGORIES.items()
        if not selected_only or name.lower() in selected_only
    ]
    if selected_only and not run_items:
        print(f"[ERROR] No matched categories for --only={args.only}")
        print(f"[INFO] Available: {', '.join(CATEGORIES.keys())}")
        sys.exit(2)

    overall_stats["total_categories"] = len(run_items)

    # 依次爬取每个分类
    for i, (category_name, category_config) in enumerate(run_items, 1):
        try:
            target = CATEGORY_TARGET_COUNTS.get(category_name, 200)
            print(f"\n🏗️  处理分类 {i}/{len(run_items)}: {category_name} (target: {target} links)")

            # 爬取分类
            category_result = scrape_category(category_name, category_config, target_count=target)
            all_results[category_name] = category_result
            
            # 保存分类结果
            category_output_dir = os.path.join(base_output_dir, category_name)
            os.makedirs(category_output_dir, exist_ok=True)
            
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            category_file = os.path.join(category_output_dir, f"{category_name}_links_{timestamp_str}.json")
            
            with open(category_file, "w", encoding="utf-8") as f:
                json.dump(category_result, f, ensure_ascii=False, indent=2)
            
            # 更新总体统计
            overall_stats["completed_categories"] += 1
            overall_stats["total_links"] += category_result["actual_count"]
            overall_stats["category_results"][category_name] = {
                "links_count": category_result["actual_count"],
                "pages_scraped": category_result["pages_scraped"],
                "file_path": category_file
            }
            
            # 输出分类完成信息
            print(f"\n✅ {category_name} 完成!")
            print(f"   📁 保存路径: {category_file}")
            print(f"   📊 链接数量: {category_result['actual_count']}")
            print(f"   📄 爬取页面: {category_result['pages_scraped']}")
            
            # 分类间休息一下
            if i < len(run_items):
                print(f"\n⏱️  休息 3 秒后继续下一个分类...")
                time.sleep(3)
                
        except Exception as e:
            print(f"❌ {category_name} 爬取失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算总耗时
    total_time = time.time() - total_start_time
    overall_stats["end_time"] = time.time()
    overall_stats["total_duration"] = total_time
    
    # 保存总体报告
    report_file = os.path.join(base_output_dir, f"overall_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)
    
    # 输出最终报告
    print(f"\n{'='*80}")
    print("🎊 全部爬取完成！")
    print(f"{'='*80}")
    print(f"📊 总体统计:")
    print(f"   • 处理分类数: {overall_stats['completed_categories']}/{overall_stats['total_categories']}")
    print(f"   • 总链接数: {overall_stats['total_links']}")
    print(f"   • 总耗时: {total_time/60:.1f} 分钟")
    print(f"   • 报告文件: {report_file}")
    
    print(f"\n📋 各分类详情:")
    for category_name, result in overall_stats["category_results"].items():
        print(f"   • {category_name:<15}: {result['links_count']:>3} 链接 ({result['pages_scraped']} 页)")
    
    print(f"\n📁 输出目录: {base_output_dir}")
    print("🎯 所有分类的链接已保存到对应的文件夹中！")


if __name__ == "__main__":
    main()
