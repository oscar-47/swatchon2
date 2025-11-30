import os
import sys
import json
import time
from typing import List, Set, Dict

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


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
    }
}


def build_category_page_url(category_ids: str, page: int) -> str:
    """构建分类页面URL"""
    return f"https://swatchon.com/wholesale-fabric?categoryIds={category_ids}&sort=&page={page}&from=/wholesale-fabric"


def accept_cookies(page) -> None:
    """接受cookies"""
    try:
        print("[debug] 尝试接受 cookies...", flush=True)
        cookie_selectors = [
            "button:has-text('Accept')",
            "button:has-text('I agree')",
            "button:has-text('同意')",
            "text=Accept",
        ]
        
        for sel in cookie_selectors:
            try:
                btn = page.locator(sel).first
                if btn and btn.count() > 0 and btn.is_visible():
                    print(f"[debug] 找到并点击 cookie 按钮", flush=True)
                    btn.click()
                    page.wait_for_timeout(1000)
                    return
            except Exception:
                continue
        print("[debug] 未找到 cookie 按钮", flush=True)
    except Exception as e:
        print(f"[debug] Cookie 处理出错: {e}", flush=True)


def collect_page_links(page, page_num: int, category_name: str) -> List[str]:
    """从单个页面收集链接"""
    print(f"[debug] {category_name} - 开始收集第 {page_num} 页链接...", flush=True)
    
    try:
        # New website structure uses .fabric-card instead of .c-quality
        # Look for <a> tags with href="/fabric/..."
        cards = page.locator("a[href^='/fabric/']")
        total = cards.count()
        print(f"[debug] {category_name} - 第 {page_num} 页找到 {total} 个产品卡片", flush=True)

        if total == 0:
            print(f"[warning] {category_name} - 第 {page_num} 页没有找到产品卡片", flush=True)
            return []

        page_links: List[str] = []

        # 提取每个卡片的链接
        for i in range(total):
            try:
                c = cards.nth(i)
                href = c.get_attribute("href", timeout=2000)

                if href:
                    # 规范化URL
                    if href.startswith("/"):
                        href = "https://swatchon.com" + href
                    elif not href.startswith("http"):
                        href = "https://swatchon.com/" + href

                    page_links.append(href)

            except Exception as e:
                print(f"[debug] {category_name} - 第 {page_num} 页卡片 {i+1} 处理失败: {e}", flush=True)
                continue

        print(f"[success] {category_name} - 第 {page_num} 页成功提取 {len(page_links)} 个链接", flush=True)
        return page_links
        
    except Exception as e:
        print(f"[error] {category_name} - 第 {page_num} 页链接收集失败: {e}", flush=True)
        return []


def scrape_category(category_name: str, category_config: dict, target_count: int = 150, max_pages: int = 20) -> dict:
    """爬取单个分类的所有链接"""
    
    print(f"\n{'='*80}")
    print(f"🎯 开始爬取分类: {category_name}")
    print(f"📊 目标链接数量: {target_count}")
    print(f"🔗 分类URL: {category_config['url']}")
    print(f"{'='*80}")
    
    all_links: Set[str] = set()
    page_results = []
    
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ]
        )
        
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1920, "height": 1080},
        )
        
        page = context.new_page()
        page.set_default_timeout(30000)
        
        current_page = 1
        cookies_accepted = False
        
        while len(all_links) < target_count and current_page <= max_pages:
            try:
                url = build_category_page_url(category_config["categoryIds"], current_page)
                print(f"\n[step] {category_name} - 正在爬取第 {current_page} 页")
                
                # 访问页面
                response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if not response or response.status != 200:
                    print(f"[error] {category_name} - 第 {current_page} 页访问失败，状态码: {response.status if response else 'None'}")
                    current_page += 1
                    continue
                
                # 等待页面加载完成
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except PlaywrightTimeoutError:
                    print(f"[debug] {category_name} - 第 {current_page} 页网络空闲等待超时")
                
                # 等待产品卡片加载
                try:
                    page.wait_for_selector(".c-quality", timeout=10000)
                    time.sleep(2)  # 额外等待确保动态内容加载完成
                except PlaywrightTimeoutError:
                    print(f"[warning] {category_name} - 第 {current_page} 页产品卡片加载超时")
                
                # 第一次访问时接受cookies
                if not cookies_accepted:
                    accept_cookies(page)
                    cookies_accepted = True
                
                # 收集当前页面的链接
                page_links = collect_page_links(page, current_page, category_name)
                
                if not page_links:
                    print(f"[info] {category_name} - 第 {current_page} 页没有找到链接，可能已到最后一页")
                    break
                
                # 添加到总链接集合中（自动去重）
                before_count = len(all_links)
                all_links.update(page_links)
                new_links_count = len(all_links) - before_count
                
                # 记录页面结果
                page_result = {
                    "page": current_page,
                    "url": url,
                    "links_found": len(page_links),
                    "new_unique_links": new_links_count,
                    "total_unique_links": len(all_links)
                }
                page_results.append(page_result)
                
                print(f"[progress] {category_name} - 第 {current_page} 页: 找到 {len(page_links)} 个链接, 新增 {new_links_count} 个唯一链接")
                print(f"[progress] {category_name} - 总进度: {len(all_links)}/{target_count} 个唯一链接")
                
                # 检查是否达到目标
                if len(all_links) >= target_count:
                    print(f"[success] {category_name} - 🎉 已达到目标！共收集 {len(all_links)} 个唯一链接")
                    break
                
                current_page += 1
                
                # 页面间稍作延迟
                time.sleep(1)
                
            except Exception as e:
                print(f"[error] {category_name} - 第 {current_page} 页处理出错: {e}")
                current_page += 1
                continue
        
        context.close()
        browser.close()
        
        # 返回结果
        result = {
            "category": category_name,
            "timestamp": time.time(),
            "target_count": target_count,
            "actual_count": len(all_links),
            "pages_scraped": len(page_results),
            "page_details": page_results,
            "all_links": sorted(list(all_links))
        }
        
        return result


def main():
    """主函数 - 依次爬取所有分类"""
    
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
    }

    # 依次爬取每个分类
    for i, (category_name, category_config) in enumerate(CATEGORIES.items(), 1):
        try:
            target = CATEGORY_TARGET_COUNTS.get(category_name, 200)
            print(f"\n🏗️  处理分类 {i}/{len(CATEGORIES)}: {category_name} (target: {target} links)")

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
            if i < len(CATEGORIES):
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