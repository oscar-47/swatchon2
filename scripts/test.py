import os
import sys
import json
import argparse
import time
from typing import List, Set

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# The base filtered URL provided by the user (woven subcategories)
BASE_FILTERED = (
    "https://swatchon.com/wholesale-fabric?"
    "categoryIds=167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170&sort=&from=/wholesale-fabric"
)


def build_page_url(page: int) -> str:
    """构建页面URL"""
    # 第一页和其他页面都使用统一的格式
    return (
        "https://swatchon.com/wholesale-fabric?"
        f"categoryIds=167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170&sort=&page={page}&from=/wholesale-fabric"
    )


def accept_cookies(page) -> None:
    """接受cookies"""
    try:
        print("[debug] 尝试接受 cookies...", flush=True)
        cookie_selectors = [
            "button:has-text('Accept')",
            "button:has-text('I agree')",
            "button:has-text('同意')",
            "text=Accept",
            "[id*='cookie'] button",
            "[class*='cookie'] button",
            "button[class*='accept']",
            ".cookie-banner button",
            "#cookie-consent button"
        ]
        
        for sel in cookie_selectors:
            try:
                btn = page.locator(sel).first
                if btn and btn.count() > 0 and btn.is_visible():
                    print(f"[debug] 找到并点击 cookie 按钮: {sel}", flush=True)
                    btn.click()
                    page.wait_for_timeout(1000)
                    return
            except Exception as e:
                continue
        print("[debug] 未找到 cookie 按钮", flush=True)
    except Exception as e:
        print(f"[debug] Cookie 处理出错: {e}", flush=True)


def collect_page_links(page, page_num: int) -> List[str]:
    """从单个页面收集链接"""
    print(f"[debug] 开始收集第 {page_num} 页链接...", flush=True)
    
    try:
        # 查找搜索容器
        container = page.locator("div.search-items").first
        if container.count() == 0:
            print(f"[error] 第 {page_num} 页未找到搜索结果容器", flush=True)
            return []
        
        container.wait_for(state="visible", timeout=10000)
        
        # 查找产品卡片
        cards = container.locator(".c-quality")
        total = cards.count()
        print(f"[debug] 第 {page_num} 页找到 {total} 个产品卡片", flush=True)
        
        if total == 0:
            print(f"[warning] 第 {page_num} 页没有找到产品卡片", flush=True)
            return []
        
        page_links: List[str] = []
        
        # 提取每个卡片的链接 - 直接从div的href属性获取
        for i in range(total):
            try:
                c = cards.nth(i)
                
                # 直接从.c-quality div获取href属性
                href = c.get_attribute("href", timeout=2000)
                
                if href:
                    # 规范化URL
                    if href.startswith("/"):
                        href = "https://swatchon.com" + href
                    elif not href.startswith("http"):
                        href = "https://swatchon.com/" + href
                    
                    page_links.append(href)
                    print(f"[debug] 第 {page_num} 页卡片 {i+1} 找到链接: {href}", flush=True)
                else:
                    print(f"[debug] 第 {page_num} 页卡片 {i+1} 没有href属性", flush=True)
                    
            except Exception as e:
                print(f"[debug] 第 {page_num} 页卡片 {i+1} 处理失败: {e}", flush=True)
                continue
        
        print(f"[success] 第 {page_num} 页成功提取 {len(page_links)} 个链接", flush=True)
        return page_links
        
    except Exception as e:
        print(f"[error] 第 {page_num} 页链接收集失败: {e}", flush=True)
        return []


def scrape_multiple_pages(target_count: int = 150, start_page: int = 1, max_pages: int = 20) -> dict:
    """爬取多个页面直到达到目标链接数量"""
    
    all_links: Set[str] = set()  # 使用set自动去重
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
        
        print(f"[info] 开始多页面爬取，目标: {target_count} 个链接", flush=True)
        print("=" * 60)
        
        current_page = start_page
        cookies_accepted = False
        
        while len(all_links) < target_count and current_page <= max_pages:
            try:
                url = build_page_url(current_page)
                print(f"\n[step] 正在爬取第 {current_page} 页", flush=True)
                print(f"[url] {url}", flush=True)
                
                # 访问页面
                response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if not response or response.status != 200:
                    print(f"[error] 第 {current_page} 页访问失败，状态码: {response.status if response else 'None'}", flush=True)
                    current_page += 1
                    continue
                
                # 等待页面加载完成
                print(f"[debug] 第 {current_page} 页等待网络空闲...", flush=True)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except PlaywrightTimeoutError:
                    print(f"[debug] 第 {current_page} 页网络空闲等待超时", flush=True)
                
                # 等待API数据加载（产品列表通常通过API获取）
                print(f"[debug] 第 {current_page} 页等待产品数据加载...", flush=True)
                try:
                    # 等待产品卡片出现
                    page.wait_for_selector(".c-quality", timeout=10000)
                    # 再等待一下确保所有卡片都加载完成
                    time.sleep(2)
                except PlaywrightTimeoutError:
                    print(f"[warning] 第 {current_page} 页产品卡片加载超时", flush=True)
                
                # 第一次访问时接受cookies
                if not cookies_accepted:
                    accept_cookies(page)
                    cookies_accepted = True
                
                # 收集当前页面的链接
                page_links = collect_page_links(page, current_page)
                
                if not page_links:
                    print(f"[warning] 第 {current_page} 页没有找到链接，可能已到最后一页", flush=True)
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
                
                print(f"[progress] 第 {current_page} 页: 找到 {len(page_links)} 个链接, 新增 {new_links_count} 个唯一链接", flush=True)
                print(f"[progress] 总进度: {len(all_links)}/{target_count} 个唯一链接", flush=True)
                
                # 检查是否达到目标
                if len(all_links) >= target_count:
                    print(f"\n[success] 🎉 已达到目标！共收集 {len(all_links)} 个唯一链接", flush=True)
                    break
                
                current_page += 1
                
                # 页面间稍作延迟，避免请求过快
                time.sleep(1)
                
            except Exception as e:
                print(f"[error] 第 {current_page} 页处理出错: {e}", flush=True)
                current_page += 1
                continue
        
        context.close()
        browser.close()
        
        # 返回结果
        result = {
            "timestamp": time.time(),
            "target_count": target_count,
            "actual_count": len(all_links),
            "pages_scraped": len(page_results),
            "page_details": page_results,
            "all_links": sorted(list(all_links))  # 转换为排序的列表
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description="多页面爬取产品链接，达到指定数量后停止")
    parser.add_argument("--target", type=int, default=150, help="目标链接数量 (default: 150)")
    parser.add_argument("--start-page", type=int, default=1, help="起始页面 (default: 1)")
    parser.add_argument("--max-pages", type=int, default=20, help="最大页面数限制 (default: 20)")
    parser.add_argument("--out", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()
    
    print(f"[info] 多页面链接收集器启动", flush=True)
    print(f"[info] 目标链接数量: {args.target}", flush=True)
    print(f"[info] 起始页面: {args.start_page}", flush=True)
    print(f"[info] 最大页面限制: {args.max_pages}", flush=True)
    
    try:
        # 开始爬取
        result = scrape_multiple_pages(
            target_count=args.target,
            start_page=args.start_page,
            max_pages=args.max_pages
        )
        
        # 保存结果
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        
        if args.out:
            out_path = args.out
        else:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"swatchon_links_{args.target}_{timestamp_str}.json")
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n[saved] 结果已保存到: {out_path}", flush=True)
        
        # 输出汇总信息
        print("\n" + "=" * 60)
        print("📊 爬取汇总:")
        print(f"   目标数量: {result['target_count']}")
        print(f"   实际收集: {result['actual_count']}")
        print(f"   爬取页面: {result['pages_scraped']}")
        print(f"   完成率: {result['actual_count']/result['target_count']*100:.1f}%")
        
        print(f"\n📄 页面详情:")
        for page_detail in result['page_details']:
            print(f"   第 {page_detail['page']} 页: {page_detail['links_found']} 个链接 (新增 {page_detail['new_unique_links']} 个)")
        
        print(f"\n🔗 前10个链接示例:")
        for i, link in enumerate(result['all_links'][:10], 1):
            print(f"   {i:2d}. {link}")
        if len(result['all_links']) > 10:
            print(f"   ... 还有 {len(result['all_links']) - 10} 个链接")
            
    except KeyboardInterrupt:
        print(f"\n[info] 用户中断爬取", flush=True)
    except Exception as e:
        print(f"\n[error] 爬取过程出错: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()