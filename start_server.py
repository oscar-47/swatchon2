#!/usr/bin/env python
"""
启动脚本 - 用于诊断和启动服务器
"""
import sys
import os

print("=" * 60)
print("🚀 Swatchon 服务器启动诊断")
print("=" * 60)

# 1. 检查Python版本
print(f"\n✓ Python版本: {sys.version}")

# 2. 检查必要的包
print("\n📦 检查依赖包...")
required_packages = {
    'torch': 'PyTorch',
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'PIL': 'Pillow',
    'torchvision': 'TorchVision'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - 未安装")
        missing_packages.append(name)

if missing_packages:
    print(f"\n❌ 缺少以下包: {', '.join(missing_packages)}")
    print("请运行: conda activate swatchon-r50")
    sys.exit(1)

# 3. 检查模型文件
print("\n🔍 检查模型文件...")
model_files = [
    "runs/woven_vs_knit_r50_gpu_e5/best.pth",
    "runs/woven_r50_gpu_e5/best.pth",
    "runs/knit_r50_gpu_e5/best.pth"
]

missing_models = []
for model_path in model_files:
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {model_path} - 文件不存在")
        missing_models.append(model_path)

if missing_models:
    print(f"\n⚠️  警告: 缺少 {len(missing_models)} 个模型文件")
    print("服务器可以启动，但这些模型将不可用")

# 4. 检查前端文件
print("\n🎨 检查前端文件...")
frontend_dist = "production/web/ant_demo"
if os.path.isdir(frontend_dist):
    index_html = os.path.join(frontend_dist, "index_v3.html")
    if os.path.exists(index_html):
        print(f"  ✓ 前端已构建: {frontend_dist}")
    else:
        print(f"  ✗ 缺少 index_v3.html")
else:
    print(f"  ✗ 前端目录不存在: {frontend_dist}")

# 5. 检查端口
print("\n🔌 检查端口 8000...")
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 8000))
sock.close()

if result == 0:
    print("  ⚠️  端口 8000 已被占用")
    print("  请先关闭占用该端口的程序，或使用其他端口")
else:
    print("  ✓ 端口 8000 可用")

print("\n" + "=" * 60)
print("🎯 准备启动服务器...")
print("=" * 60)

# 6. 启动服务器
try:
    import uvicorn
    print("\n启动命令: uvicorn server.app:app --host 0.0.0.0 --port 8000")
    print("\n访问地址:")
    print("  - 前端界面: http://localhost:8000")
    print("  - API文档:  http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务器\n")

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
except KeyboardInterrupt:
    print("\n\n👋 服务器已停止")
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

