#!/usr/bin/env python
"""
Swatchon 服务器启动脚本（简化版）
"""
import sys
import os

print("=" * 60)
print("🚀 Swatchon 纺织物识别系统")
print("=" * 60)

# 检查必要的包
print("\n📦 检查依赖...")
required = ['torch', 'fastapi', 'uvicorn', 'PIL', 'torchvision', 'easyocr']
missing = []

for pkg in required:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} - 未安装")
        missing.append(pkg)

if missing:
    print(f"\n❌ 缺少依赖: {', '.join(missing)}")
    print("\n请运行:")
    print("  conda activate swatchon-r50")
    print("  pip install " + " ".join(missing))
    sys.exit(1)

# 检查模型文件
print("\n🔍 检查模型文件...")
models = [
    "runs/stage1_knit_woven/best.pth",
    "runs/stage2_woven/best.pth",
    "runs/stage2_knit/best.pth"
]

missing_models = []
for m in models:
    if os.path.exists(m):
        size = os.path.getsize(m) / (1024 * 1024)
        print(f"  ✓ {m} ({size:.1f} MB)")
    else:
        print(f"  ✗ {m} - 文件不存在")
        missing_models.append(m)

if missing_models:
    print(f"\n❌ 缺少模型文件")
    sys.exit(1)

# 检查前端文件
print("\n🌐 检查前端文件...")
frontend = "web/ant_demo/index.html"
if os.path.exists(frontend):
    print(f"  ✓ {frontend}")
else:
    print(f"  ✗ {frontend} - 文件不存在")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有检查通过！")
print("=" * 60)

# 启动服务器
print("\n🚀 启动服务器...")
print("\n访问地址:")
print("  - 前端界面: http://localhost:5005")
print("  - V3 界面: http://localhost:5005/index_v3.html")
print("  - API 文档:  http://localhost:5005/docs")
print("\n按 Ctrl+C 停止服务器\n")

try:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=5005, reload=False)
except KeyboardInterrupt:
    print("\n\n👋 服务器已停止")
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    sys.exit(1)

