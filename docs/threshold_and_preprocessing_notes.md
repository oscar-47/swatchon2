# Confidence Threshold & Preprocessing Notes
# 置信度阈值与预处理说明

---

## 1. Confidence Threshold Analysis | 置信度阈值分析

### Why 60%? | 为什么选择60%？

The system uses a **two-stage classification pipeline**:
- **Stage 1**: Binary classification (Knit vs Woven) — ~92% accuracy
- **Stage 2**: Subcategory classification (5 classes each) — ~79% accuracy
- **Final Confidence** = Stage1_Confidence × Stage2_Confidence

系统采用**两阶段分类流程**：
- **第一阶段**：二分类（针织 vs 梭织）— 约92%准确率
- **第二阶段**：细分类（各5个类别）— 约79%准确率
- **最终置信度** = 第一阶段置信度 × 第二阶段置信度

---

### Threshold Comparison Table | 阈值对比表

| Threshold | Use OCR | Use Image Recognition | OCR Trigger Rate | Notes |
|-----------|---------|----------------------|------------------|-------|
| **50%**   | 195     | 805                  | 19.5%            | Too low — misses uncertain predictions |
| **60%**   | 257     | 743                  | **25.7%**        | ✅ **Recommended** — balanced |
| **70%**   | 357     | 643                  | 35.7%            | Higher OCR load (+10%) |
| **80%**   | 641     | 359                  | 64.1%            | Excessive OCR calls (+38%) |
| **90%**   | 947     | 53                   | 94.7%            | Almost all trigger OCR |

| 阈值 | 使用OCR | 使用图像识别 | OCR触发率 | 备注 |
|------|---------|-------------|----------|------|
| **50%** | 195 | 805 | 19.5% | 过低 — 会漏掉不确定的预测 |
| **60%** | 257 | 743 | **25.7%** | ✅ **推荐** — 平衡点 |
| **70%** | 357 | 643 | 35.7% | OCR负载增加10% |
| **80%** | 641 | 359 | 64.1% | OCR调用过多(+38%) |
| **90%** | 947 | 53 | 94.7% | 几乎所有样本都触发OCR |

*Based on 1000 simulated samples using model performance statistics*
*基于1000个模拟样本，使用模型性能统计数据*

---

### Rationale for 60% | 选择60%的理由

1. **Efficiency**: ~75% of samples use fast image recognition directly
2. **Reliability**: Uncertain predictions (confidence < 60%) get OCR verification
3. **Cost-effective**: OCR is computationally expensive; 60% minimizes unnecessary calls
4. **Empirical**: 60% is a common threshold in multi-class classification systems

1. **效率**：约75%的样本直接使用快速的图像识别
2. **可靠性**：不确定的预测（置信度<60%）会触发OCR验证
3. **成本效益**：OCR计算开销大；60%阈值最小化不必要的调用
4. **经验值**：60%是多分类系统中常用的阈值

---

## 2. Image Preprocessing Pipeline | 图像预处理流程

### Step 1: Remove Ruler | 移除标尺
Remove the ruler on the **right edge** of the image.

移除图像**右侧边缘**的标尺。

### Step 2: Remove Label | 移除标签
Use **YOLOv8** to detect and remove the fabric label in the center of the image.

使用 **YOLOv8** 检测并移除图像中央的面料标签。

### Step 3: Crop | 裁剪
Crop the **top-left 950×950 pixel** region as the final input.

裁剪**左上角 950×950 像素**区域作为最终输入。

---

### Preprocessing Summary Table | 预处理流程汇总

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Remove Ruler | Right edge removal |
| 2 | Remove Label | YOLOv8 detection + masking |
| 3 | Crop | Top-left 950×950 px |

| 步骤 | 操作 | 详情 |
|------|------|------|
| 1 | 移除标尺 | 去除右侧边缘 |
| 2 | 移除标签 | YOLOv8检测 + 遮盖 |
| 3 | 裁剪 | 左上角 950×950 像素 |

---

## 3. Data Augmentation | 数据增强

Basic augmentation operations applied during training:

训练过程中应用的基本增强操作：

| Augmentation | Description |
|--------------|-------------|
| Random Horizontal Flip | 随机水平翻转 |
| Random Vertical Flip | 随机垂直翻转 |
| Random Rotation | 随机旋转 (±15°) |
| Color Jitter | 颜色抖动 (brightness, contrast) |
| Random Resized Crop | 随机缩放裁剪 |
| Normalization | ImageNet标准化 (mean, std) |

---

## 4. Model Architecture | 模型架构

- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Stage 1**: Binary classifier (Knit / Woven)
- **Stage 2**: 5-class subcategory classifier (separate for Knit and Woven)

- **主干网络**：ResNet-50（ImageNet预训练）
- **第一阶段**：二分类器（针织/梭织）
- **第二阶段**：5类细分类器（针织和梭织各一个）

