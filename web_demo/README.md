# 面料智能分类系统 Web Demo

基于 Vue 3 + Ant Design Vue + Flask 的专业面料分类演示系统。

## 功能特性

- 🎯 **高精度识别**: 基于 ResNet50 深度学习模型，支持二分类和多分类
- ⚡ **快速响应**: GPU 加速推理，单张图片识别时间小于1秒
- 🔧 **专业易用**: 直观的 Web 界面，支持拖拽上传，实时预览结果
- 📊 **批量处理**: 支持多张图片同时识别，提升工作效率
- 📈 **详细报告**: 提供置信度、统计信息和结果导出功能

## 支持的模型

1. **Woven vs Knit (二分类)**: 区分机织(Woven)和针织(Knit)面料
2. **Woven 多分类**: 机织面料细分类别识别 (9类)
3. **Knit 多分类**: 针织面料细分类别识别 (10类)

## 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+
- PyTorch (已在 conda 环境 swatchon-r50 中配置)

### 后端启动

```bash
cd web_demo/backend
conda activate swatchon-r50
pip install flask flask-cors pillow
python app.py
```

后端将在 http://localhost:5000 启动

### 前端启动

```bash
cd web_demo/frontend
npm install
npm run serve
```

前端将在 http://localhost:8080 启动

### 生产部署

```bash
# 构建前端
cd web_demo/frontend
npm run build

# 启动后端 (会自动服务前端静态文件)
cd ../backend
python app.py
```

访问 http://localhost:5000 即可使用完整系统

## 使用说明

### 单图识别
1. 选择识别模型
2. 上传单张面料图片
3. 点击"开始识别"
4. 查看详细预测结果和置信度

### 批量识别
1. 选择识别模型
2. 上传多张面料图片
3. 点击"批量识别"
4. 查看所有结果，支持导出 CSV

## API 接口

### GET /api/models
获取可用模型列表

### POST /api/predict
单图识别
```json
{
  "model_id": "woven_vs_knit",
  "image": "data:image/jpeg;base64,..."
}
```

### POST /api/batch_predict
批量识别
```json
{
  "model_id": "woven_vs_knit", 
  "images": ["data:image/jpeg;base64,...", ...]
}
```

## 技术栈

- **前端**: Vue 3, Ant Design Vue, Axios
- **后端**: Flask, PyTorch, Torchvision
- **模型**: ResNet50 (ImageNet 预训练)

## 演示截图

系统提供专业的用户界面，包括：
- 渐变色主题设计
- 响应式布局
- 拖拽上传功能
- 实时结果展示
- 统计信息面板
- 结果导出功能
