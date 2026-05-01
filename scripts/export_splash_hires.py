"""Render mock_demo.html splash page to high-resolution PNG + PDF for Elif's slides.

Usage:
  python scripts/export_splash_hires.py

Outputs:
  outputs/splash_hires.png   (3x device pixel ratio, 3840x2160 base canvas)
  outputs/splash_hires.pdf   (A4 landscape, print-ready)
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "web" / "mock_demo.html"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Prefer live server so fonts / SVG / CSS resolve identically to the user's browser.
SERVER_URL = "http://127.0.0.1:8001/"


def _server_reachable() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def export():
    url = SERVER_URL if _server_reachable() else HTML.as_uri()
    print(f"rendering from {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch()

        # High-DPI PNG for presentation use.
        ctx_png = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=3,  # effective 5760x3240 raster
        )
        page = ctx_png.new_page()
        # Kill auto-dismiss timer BEFORE navigation so splash never hides.
        page.add_init_script("window.setTimeout = () => 0;")
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("#splashPage", state="visible")
        # Wait for web fonts + SVG sprite to finish.
        page.evaluate("document.fonts ? document.fonts.ready : Promise.resolve()")
        page.wait_for_timeout(2000)
        splash = page.locator("#splashPage")
        splash.screenshot(path=str(OUT_DIR / "splash_hires.png"), omit_background=False)
        ctx_png.close()

        # PDF: A4 landscape, print stylesheet.
        ctx_pdf = browser.new_context(viewport={"width": 1920, "height": 1080})
        page2 = ctx_pdf.new_page()
        page2.add_init_script("window.setTimeout = () => 0;")
        page2.goto(url, wait_until="networkidle")
        page2.wait_for_selector("#splashPage", state="visible")
        page2.evaluate("document.fonts ? document.fonts.ready : Promise.resolve()")
        page2.wait_for_timeout(2000)
        page2.evaluate(
            "document.body.innerHTML = document.getElementById('splashPage').outerHTML;"
            "document.body.style.margin='0';"
        )
        page2.pdf(
            path=str(OUT_DIR / "splash_hires.pdf"),
            format="A4",
            landscape=True,
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        ctx_pdf.close()
        browser.close()

    print(f"wrote {OUT_DIR / 'splash_hires.png'}")
    print(f"wrote {OUT_DIR / 'splash_hires.pdf'}")


if __name__ == "__main__":
    export()
