#!/usr/bin/env python
"""
测试 production 环境是否完整
"""
import os
import sys

print("=" * 60)
print("🧪 Production 环境完整性测试")
print("=" * 60)

# 测试项目
tests = {
    "必需文件": [
        "server/app.py",
        "server/models/loader.py",
        "web/ant_demo/index.html",
        "runs/woven_vs_knit_r50_gpu_e5/best.pth",
        "runs/woven_r50_gpu_e5/best.pth",
        "runs/knit_r50_gpu_e5/best.pth",
        "training/training_hashes.txt",
        "requirements.txt",
        "README.md"
    ],
    "Python 依赖": [
        "torch",
        "torchvision",
        "fastapi",
        "uvicorn",
        "PIL",
        "easyocr"
    ]
}

all_passed = True

# 测试文件
print("\n📁 测试必需文件...")
for file in tests["必需文件"]:
    if os.path.exists(file):
        size = os.path.getsize(file)
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size / 1024:.1f} KB"
        print(f"  ✓ {file} ({size_str})")
    else:
        print(f"  ✗ {file} - 缺失")
        all_passed = False

# 测试依赖
print("\n📦 测试 Python 依赖...")
for pkg in tests["Python 依赖"]:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} - 未安装")
        all_passed = False

# 测试导入
print("\n🔧 测试模块导入...")
try:
    from server.models.loader import build_model, load_checkpoint
    print("  ✓ server.models.loader")
except Exception as e:
    print(f"  ✗ server.models.loader - {e}")
    all_passed = False

try:
    from server.app import app, CACHE
    print("  ✓ server.app")
except Exception as e:
    print(f"  ✗ server.app - {e}")
    all_passed = False

# 总结
print("\n" + "=" * 60)
if all_passed:
    print("✅ 所有测试通过！环境完整，可以启动服务器。")
    print("\n运行以下命令启动:")
    print("  python start.py")
    print("或双击:")
    print("  启动服务器.bat")
else:
    print("❌ 部分测试失败，请检查上述错误。")
print("=" * 60)

sys.exit(0 if all_passed else 1)

