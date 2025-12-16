# Swatchon V3 技术实现说明文档

---

## Topic 1: 自建 Fabric 图像库/训练集（主类 + 子类）

### 当前状态

**已完成：**
- ✅ 两阶段分类模型架构已建立
- ✅ Stage 1 (主类): Knit vs Woven 二分类，测试准确率 **92.4%**
- ✅ Stage 2 (子类): 
  - Knit → 5 类 (French_Terry, Jacquard, Mesh, Rib, Single_Jersey)，测试准确率 **77.6%**
  - Woven → 5 类 (Corduroy, Jacquard, Plain, Satin, Twill)，测试准确率 **79.7%**
- ✅ 训练数据存放在 `fabric_dataset_training/` 目录
- ✅ 模型权重存放在 `simple_model_v2/` 目录

**数据规模估计（基于训练时间和batch设置）：**
- Stage 1: 约 1750 张图（Knit + Woven 各约 875 张）
- Stage 2 Knit: 约 875 张图（5 类，每类约 175 张）
- Stage 2 Woven: 约 875 张图（5 类，每类约 175 张）

### 具体实现思路

```
数据目录结构 (ImageFolder 格式):
fabric_dataset_training/
├── Knit/
│   ├── French_Terry/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── Jacquard/
│   ├── Mesh/
│   ├── Rib/
│   └── Single_Jersey/
└── Woven/
    ├── Corduroy/
    ├── Jacquard/
    ├── Plain/
    ├── Satin/
    └── Twill/
```

**标注规范：**
1. 图片命名：`{source}_{category}_{id}.jpg`（例如 `swatchon_plain_0042.jpg`）
2. 分辨率要求：原图至少 950×950，训练时裁剪到 224×224
3. 图片质量：避免模糊、过曝、严重偏色
4. 去除干扰：裁剪掉标尺、标签区域（左上角 950×950 固定裁剪）

**版本管理方案：**
```
training/training_hashes.txt  # 记录训练集图片的 SHA256 哈希值，用于推理时检测是否为"外部数据"
```

### 代码改动点

| 文件/模块 | 改动内容 |
|-----------|----------|
| `training/train_imagefolder_resnet50.py` | 现有训练脚本，支持任意 ImageFolder 结构 |
| `training/train_woven_vs_knit_resnet50.py` | Stage 1 专用训练脚本 |
| `tools/generate_training_hashes.py` | 生成训练集哈希用于数据溯源 |
| `simple_model_v2/` | 模型权重 + `*_results.json` 记录训练指标和类别列表 |

### 数据/资源依赖

| 需求 | 详情 |
|------|------|
| 数据来源 | Swatchon 网站爬取 + 合作方提供 |
| 标注方式 | 按目录分类（ImageFolder），无需额外标注工具 |
| 算力支持 | GPU 训练约 15-20 分钟/模型（RTX 3090） |
| 人力需求 | 数据清洗和分类校验 1-2 人天 |

---

## Topic 2: 先主类、后子类

### 当前状态

**已完成：**
- ✅ 两阶段级联分类已上线运行
- ✅ API 同时返回 Stage 1 和 Stage 2 结果
- ✅ 前端展示完整分类路径（如 `Woven_Plain`）

**当前输出格式：**
```json
{
  "prediction": "Woven_Plain",
  "confidence": 0.736,
  "stage1": {
    "prediction": "Woven",
    "confidence": 0.92,
    "probs": {"Knit": 0.08, "Woven": 0.92}
  },
  "stage2": {
    "prediction": "Plain",
    "confidence": 0.80,
    "probs": {"Corduroy": 0.02, "Jacquard": 0.05, "Plain": 0.80, "Satin": 0.08, "Twill": 0.05}
  }
}
```

### 具体实现思路

**两阶段推理流程：**
```
输入图片
    ↓
Stage 1 模型 → Knit or Woven (+ confidence)
    ↓
    ├── if Knit → Stage 2 Knit 模型 → 5 子类
    └── if Woven → Stage 2 Woven 模型 → 5 子类
    ↓
最终输出: {主类}_{子类}, confidence = stage1_conf × stage2_conf
```

**核心代码逻辑 (`production/server/app.py`)：**
```python
def two_stage_predict(img_pil):
    # Stage 1: Knit vs Woven
    model1, classes1, tfm, device = CACHE.get("stage1_knit_woven")
    # ... 推理得到 stage1_pred, stage1_conf
    
    # Stage 2: 根据 Stage 1 结果选择对应模型
    if stage1_pred == "Knit":
        model2, classes2, tfm, device = CACHE.get("stage2_knit")
    else:
        model2, classes2, tfm, device = CACHE.get("stage2_woven")
    # ... 推理得到 stage2_pred, stage2_conf
    
    final_conf = stage1_conf * stage2_conf
    return {"prediction": f"{stage1_pred}_{stage2_pred}", ...}
```

### 以后加子类需要的改动

| 场景 | 改动点 |
|------|--------|
| 新增 Knit 子类 (如 Fleece) | 1. 收集 Fleece 数据 → 2. 重训 Stage 2 Knit 模型 → 3. 替换 `stage2_knit_best.pth` |
| 新增 Woven 子类 (如 Denim) | 1. 收集 Denim 数据 → 2. 重训 Stage 2 Woven 模型 → 3. 替换 `stage2_woven_best.pth` |
| 新增主类 (如 Nonwoven) | 1. 重训 Stage 1 为 3 分类 → 2. 新建 Stage 2 Nonwoven 模型 → 3. 修改 `two_stage_predict()` 增加分支 |

### 代码改动点

| 文件 | 说明 |
|------|------|
| `production/server/app.py` | `MODEL_REGISTRY` 增加新模型路径，`two_stage_predict()` 增加新分支 |
| `production/server/models/loader.py` | 无需改动（自动从 checkpoint 读取 classes） |
| `simple_model_v2/` | 新增对应模型目录 |

---

## Topic 3: 多图输入 + 置信度阈值（90/95%）

### 当前状态

**已完成：**
- ✅ 单图分类 + 置信度输出
- ✅ 低置信度（< 60%）时触发 OCR 读取标签验证
- ✅ 支持上传正面图 + 标签图 + 反面图（3 张）
- ✅ 前端 V3 版本已支持多图上传

**当前阈值配置：**
```python
# production/server/app.py
CONFIDENCE_THRESHOLD = 0.60  # 低于60%触发OCR验证
```

### 具体实现思路

**多图融合策略（待实现）：**

| 策略 | 实现方式 | 优点 | 缺点 |
|------|----------|------|------|
| **投票法** | 每张图独立预测，取众数 | 简单稳定 | 无法利用概率信息 |
| **概率平均** | 对所有图的 softmax 概率取平均，再 argmax | 利用概率信息 | 异常图影响大 |
| **最小置信度** | 取所有图中置信度最低的作为最终置信度 | 保守策略，安全 | 可能过于保守 |
| **加权平均** ✅推荐 | 按图片质量/清晰度加权 | 平衡各方因素 | 需要图片质量评估 |

**推荐方案：加权概率平均**
```python
def multi_image_predict(images: List[Image]):
    all_probs = []
    for img in images:
        result = two_stage_predict(img)
        all_probs.append(result["stage1"]["probs"])

    # 平均概率
    avg_probs = {k: np.mean([p[k] for p in all_probs]) for k in all_probs[0]}
    final_pred = max(avg_probs, key=avg_probs.get)
    final_conf = avg_probs[final_pred]

    return final_pred, final_conf
```

**阈值策略配置：**
```python
THRESHOLDS = {
    "high_confidence": 0.95,   # 直接输出，无需验证
    "medium_confidence": 0.90, # 建议人工确认
    "low_confidence": 0.60,    # 触发 OCR 验证
    "reject": 0.40             # 拒绝分类，提示重拍
}
```

### 代码改动点

| 文件 | 改动内容 |
|------|----------|
| `production/server/app.py` | 新增 `multi_image_predict()` 函数 |
| `production/server/app.py` | 修改 `/api/classify` 接口支持多图 |
| `production/web/ant_demo/index_v3.html` | 前端已支持 3 图上传，无需大改 |
| 新建 `production/server/confidence_config.py` | 阈值配置独立文件 |

### API 接口变更

**现有接口：**
```
POST /api/classify
Body: fabric_image (单张)
```

**扩展后：**
```
POST /api/classify
Body:
  - fabric_images[] (多张，1-5张)
  - fusion_strategy: "voting" | "avg_prob" | "min_conf" | "weighted"
Response:
  - prediction, confidence
  - individual_results: [每张图的单独结果]
  - fusion_method: 使用的融合策略
```

---

## Topic 4: 文本标签图提醒 + 图像质量检测

### 当前状态

**已完成：**
- ✅ OCR 功能集成（EasyOCR）
- ✅ 从标签图提取面料成分信息
- ✅ 正反面对比检测 Double 结构（`double_verifier.py`）

**未完成：**
- ❌ 自动检测图片中是否包含标签（而非纯面料）
- ❌ 图像质量检测（模糊、过曝、距离不当）

### 具体实现思路

**1. 标签检测（判断是否误传标签图为面料图）：**

| 方案 | 实现方式 | 准确度 | 速度 |
|------|----------|--------|------|
| OCR 规则 | 检测到大量文字 → 可能是标签图 | 中 | 快 |
| 轻量分类器 | 训练 fabric vs label 二分类模型 | 高 | 中 |
| YOLOv11 检测 | 检测标签位置 | 高 | 中 |

**推荐：OCR + 文字密度规则**
```python
def detect_label_image(img_pil):
    ocr_result = easyocr.readtext(img_pil)
    text_area_ratio = calculate_text_area(ocr_result) / (img_pil.width * img_pil.height)

    if text_area_ratio > 0.15:  # 文字面积超过15%
        return {"is_label": True, "warning": "This appears to be a label image, not fabric texture"}
    return {"is_label": False}
```

**2. 图像质量检测：**

| 指标 | 检测方法 | 阈值建议 |
|------|----------|----------|
| **模糊度** | Laplacian 方差 | < 100 → 模糊警告 |
| **亮度** | 平均像素值 | < 50 或 > 220 → 过暗/过曝 |
| **对比度** | 标准差 | < 30 → 对比度不足 |
| **尺寸** | 分辨率检测 | < 224×224 → 分辨率不足 |

```python
import cv2
import numpy as np

def check_image_quality(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    warnings = []

    # 模糊检测
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        warnings.append({"type": "blur", "message": "Image is too blurry"})

    # 亮度检测
    brightness = np.mean(gray)
    if brightness < 50:
        warnings.append({"type": "dark", "message": "Image is too dark"})
    elif brightness > 220:
        warnings.append({"type": "overexposed", "message": "Image is overexposed"})

    # 对比度检测
    contrast = np.std(gray)
    if contrast < 30:
        warnings.append({"type": "low_contrast", "message": "Image has low contrast"})

    return {
        "quality_score": calculate_quality_score(laplacian_var, brightness, contrast),
        "warnings": warnings,
        "metrics": {"blur": laplacian_var, "brightness": brightness, "contrast": contrast}
    }
```

### Pipeline 位置

```
用户上传图片
    ↓
[1] 图像质量检测 ← 新增
    ├── 通过 → 继续
    └── 警告 → 返回提示，建议重拍
    ↓
[2] 标签/面料检测 ← 新增
    ├── 是面料 → 继续
    └── 是标签 → 提示"请上传面料纹理图"
    ↓
[3] 预处理（裁剪 950×950, resize 224×224）
    ↓
[4] 两阶段分类
    ↓
[5] 置信度判断 → 低置信度触发 OCR
    ↓
输出结果
```

### 代码改动点

| 文件 | 改动内容 |
|------|----------|
| 新建 `production/server/quality_checker.py` | 图像质量检测模块 |
| 新建 `production/server/label_detector.py` | 标签图检测模块 |
| `production/server/app.py` | 在 `/api/classify` 中集成检测逻辑 |
| `production/web/ant_demo/index_v3.html` | 前端展示警告信息 |

### 返回信息格式

```json
{
  "success": true,
  "warnings": [
    {"type": "blur", "message": "Image is slightly blurry", "severity": "low"},
    {"type": "label_detected", "message": "Text detected in fabric area", "severity": "medium"}
  ],
  "quality_score": 0.75,
  "prediction": "Woven_Plain",
  "confidence": 0.85
}
```

---

## Topic 5: ResNet 以外模型 & 交叉验证/集成

### 当前状态

**已完成：**
- ✅ ResNet-50 (ImageNet 预训练) 作为基线模型
- ✅ 单次 train/val/test split (70/15/15)

**未完成：**
- ❌ 其他模型对比实验
- ❌ K-Fold 交叉验证
- ❌ 模型集成

### 具体实现思路

**1. 候选模型对比：**

| 模型 | 参数量 | 预估准确率 | 推理速度 | 训练成本 |
|------|--------|------------|----------|----------|
| **ResNet-50** (当前) | 25M | 92%/78% | 快 | 15 min |
| EfficientNet-B0 | 5M | 预估 +1-2% | 中 | 20 min |
| EfficientNet-B3 | 12M | 预估 +2-3% | 中 | 30 min |
| ConvNeXt-Tiny | 28M | 预估 +2-4% | 中 | 40 min |
| ViT-B/16 | 86M | 预估 +1-3% | 慢 | 60 min |
| Swin-Tiny | 28M | 预估 +2-4% | 慢 | 50 min |

**推荐：EfficientNet-B3 或 ConvNeXt-Tiny**
- 准确率提升明显
- 参数量适中，推理速度可接受
- 预训练权重丰富

**2. 交叉验证方案：**

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate(dataset, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = build_model()
        train_loader = DataLoader(Subset(dataset, train_idx), ...)
        val_loader = DataLoader(Subset(dataset, val_idx), ...)

        # 训练
        best_acc = train(model, train_loader, val_loader)
        fold_results.append({"fold": fold, "val_acc": best_acc})

    mean_acc = np.mean([r["val_acc"] for r in fold_results])
    std_acc = np.std([r["val_acc"] for r in fold_results])

    return {"mean": mean_acc, "std": std_acc, "folds": fold_results}
```

**5-Fold CV 预估时间：**
- Stage 1: 15 min × 5 = 75 min
- Stage 2 Knit: 10 min × 5 = 50 min
- Stage 2 Woven: 7 min × 5 = 35 min
- **总计约 2.5 小时**

**3. 集成方案：**

| 方案 | 实现方式 | 准确率提升 | 推理开销 |
|------|----------|------------|----------|
| **加权平均** ✅推荐 | 多模型 softmax 加权平均 | +1-2% | 2-3x |
| Stacking | 用 meta-classifier 融合 | +2-3% | 3-4x |
| Bagging | 多个相同模型不同初始化 | +0.5-1% | 2x |

**推荐：ResNet-50 + EfficientNet-B3 加权平均**
```python
class EnsembleClassifier:
    def __init__(self, models, weights=None):
        self.models = models  # [resnet50, efficientnet_b3]
        self.weights = weights or [0.5, 0.5]

    def predict(self, img):
        probs_list = []
        for model in self.models:
            logits = model(img)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs)

        # 加权平均
        ensemble_probs = sum(w * p for w, p in zip(self.weights, probs_list))
        return ensemble_probs.argmax(dim=1)
```

### 最终上线选择标准

| 指标 | 权重 | 阈值 |
|------|------|------|
| 准确率 | 40% | 必须 ≥ 当前基线 |
| 推理速度 | 30% | 单张 < 100ms (CPU) |
| 模型大小 | 15% | < 200MB |
| 训练成本 | 15% | < 2 小时 (单模型) |

**决策流程：**
1. 对每个候选模型做 5-Fold CV，取平均准确率
2. 测试推理速度（CPU 和 GPU）
3. 计算综合得分 = 准确率×0.4 + 速度分×0.3 + 模型大小分×0.15 + 训练成本分×0.15
4. 选择综合得分最高且准确率超过基线的模型

### 代码改动点

| 文件 | 改动内容 |
|------|----------|
| `training/train_imagefolder_resnet50.py` | 重构为支持多模型架构 |
| 新建 `training/models/` | 各模型定义 (efficientnet.py, convnext.py, etc.) |
| 新建 `training/cross_validate.py` | K-Fold CV 脚本 |
| 新建 `training/ensemble.py` | 模型集成训练脚本 |
| `production/server/models/loader.py` | 支持加载不同架构模型 |
| `production/server/app.py` | 支持 ensemble 推理 |

### 资源依赖

| 需求 | 详情 |
|------|------|
| 算力 | GPU (RTX 3090+) 用于 CV 和新模型训练 |
| 时间 | 完整对比实验约 1-2 天 |
| 存储 | 每个模型约 100-350MB |

---

## 总结：优先级建议

| 优先级 | Topic | 难度 | 预估工时 | 收益 |
|--------|-------|------|----------|------|
| 🔴 P0 | Topic 4 - 图像质量检测 | 低 | 2-3 天 | 减少低质量输入导致的误分类 |
| 🔴 P0 | Topic 3 - 置信度阈值完善 | 低 | 1-2 天 | 提高可靠性 |
| 🟡 P1 | Topic 1 - 扩充训练数据 | 中 | 1-2 周 | 直接提升准确率 |
| 🟡 P1 | Topic 3 - 多图融合 | 中 | 3-5 天 | 提高鲁棒性 |
| 🟢 P2 | Topic 5 - 模型对比实验 | 中 | 1 周 | 潜在 2-4% 准确率提升 |
| 🟢 P2 | Topic 2 - 新增子类 | 低 | 按需 | 业务需求驱动 |
| 🔵 P3 | Topic 5 - 模型集成 | 高 | 1-2 周 | 1-2% 提升，增加推理开销 |

---

## 附录：项目关键文件清单

```
swatchon2/
├── production/                      # 生产环境
│   ├── server/
│   │   ├── app.py                   # FastAPI 主服务 (核心)
│   │   ├── models/loader.py         # 模型加载器
│   │   ├── ocr_parser.py            # OCR 解析器
│   │   └── double_verifier.py       # 双层结构验证
│   ├── web/ant_demo/
│   │   └── index_v3.html            # V3 前端界面
│   └── simple_model_v2/             # 模型权重
│       ├── models_stage1/
│       ├── models_stage2_knit/
│       └── models_stage2_woven/
├── training/                        # 训练脚本
│   ├── train_imagefolder_resnet50.py
│   └── train_woven_vs_knit_resnet50.py
├── simple_model_v2/                 # 原始模型权重
│   ├── models_stage1/stage1_knit_woven_best.pth
│   ├── models_stage2_knit/stage2_knit_best.pth
│   └── models_stage2_woven/stage2_woven_best.pth
└── tools/                           # 工具脚本
    ├── threshold_analysis.py
    └── generate_training_hashes.py
```

