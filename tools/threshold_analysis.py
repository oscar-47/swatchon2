#!/usr/bin/env python3
"""
阈值分析工具 - 分析不同置信度阈值下的OCR触发率

用途：向教授展示为什么选择60%作为阈值，以及不同阈值（60%,70%,80%,90%）
下有多少样本会使用OCR vs 图像识别

使用方法：
    python tools/threshold_analysis.py --ckpt runs/woven_r50_gpu_e5/best.pth --data-dir <path_to_test_images>
    
或者使用模拟数据生成报告：
    python tools/threshold_analysis.py --simulate
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization will be skipped.")


def analyze_thresholds(confidences: List[float], thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> Dict:
    """
    分析不同阈值下的统计数据
    
    Args:
        confidences: 模型置信度列表
        thresholds: 要分析的阈值列表
    
    Returns:
        分析结果字典
    """
    n_total = len(confidences)
    results = {
        "total_samples": n_total,
        "confidence_stats": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
        },
        "thresholds": {}
    }
    
    for threshold in thresholds:
        n_below = sum(1 for c in confidences if c < threshold)
        n_above = n_total - n_below
        
        results["thresholds"][f"{int(threshold*100)}%"] = {
            "threshold": threshold,
            "n_use_ocr": n_below,  # confidence < threshold -> 触发OCR
            "n_use_image_recognition": n_above,  # confidence >= threshold -> 使用图像识别
            "ocr_trigger_rate": round(n_below / n_total * 100, 2) if n_total > 0 else 0,
            "image_recognition_rate": round(n_above / n_total * 100, 2) if n_total > 0 else 0,
        }
    
    return results


def generate_simulated_data(n_samples: int = 500, seed: int = 42) -> Dict:
    """
    基于真实模型性能生成模拟的置信度分布

    根据训练结果：
    - Stage1 (Woven vs Knit): ~92% test accuracy
    - Stage2 Woven: ~80% test accuracy
    - Stage2 Knit: ~78% test accuracy

    两阶段系统的最终置信度 = stage1_conf * stage2_conf
    """
    np.random.seed(seed)

    # Stage 1 模拟: 二分类，置信度通常较高
    # 正确预测 (~92%): 高置信度 beta(10, 2) -> [0.7, 1.0]
    # 错误预测 (~8%): 较低置信度 beta(5, 3) -> [0.5, 0.8]
    n_correct_s1 = int(n_samples * 0.92)
    n_wrong_s1 = n_samples - n_correct_s1

    stage1_high = np.random.beta(10, 2, n_correct_s1) * 0.3 + 0.7  # [0.7, 1.0]
    stage1_low = np.random.beta(5, 3, n_wrong_s1) * 0.3 + 0.5      # [0.5, 0.8]
    stage1_confs = np.concatenate([stage1_high, stage1_low])

    # Stage 2 模拟: 5类分类，置信度分布更广
    # 正确预测 (~79%): 置信度 beta(6, 3) -> [0.5, 1.0]
    # 困难样本 (~21%): 置信度 beta(3, 4) -> [0.2, 0.7]
    n_correct_s2 = int(n_samples * 0.79)
    n_hard_s2 = n_samples - n_correct_s2

    stage2_high = np.random.beta(6, 3, n_correct_s2) * 0.5 + 0.5  # [0.5, 1.0]
    stage2_low = np.random.beta(3, 4, n_hard_s2) * 0.5 + 0.2      # [0.2, 0.7]
    stage2_confs = np.concatenate([stage2_high, stage2_low])

    np.random.shuffle(stage1_confs)
    np.random.shuffle(stage2_confs)

    # 最终置信度 = stage1 * stage2
    final_confs = stage1_confs * stage2_confs

    return {
        "stage1": stage1_confs.tolist(),
        "stage2": stage2_confs.tolist(),
        "final": final_confs.tolist()
    }


def plot_analysis(confidences: List[float], results: Dict, output_path: str):
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 置信度分布直方图
    ax1 = axes[0, 0]
    ax1.hist(confidences, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0.6, color='red', linestyle='--', linewidth=2, label='Current threshold (60%)')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Confidence Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 不同阈值下的OCR触发率条形图
    ax2 = axes[0, 1]
    thresholds_labels = list(results["thresholds"].keys())
    ocr_rates = [results["thresholds"][t]["ocr_trigger_rate"] for t in thresholds_labels]
    img_rates = [results["thresholds"][t]["image_recognition_rate"] for t in thresholds_labels]
    
    x = np.arange(len(thresholds_labels))
    width = 0.35
    bars1 = ax2.bar(x - width/2, ocr_rates, width, label='OCR Triggered', color='coral')
    bars2 = ax2.bar(x + width/2, img_rates, width, label='Image Recognition', color='seagreen')
    
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('OCR vs Image Recognition Rate by Threshold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(thresholds_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, rate in zip(bars1, ocr_rates):
        ax2.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    for bar, rate in zip(bars2, img_rates):
        ax2.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    # 3. 累积分布函数 (CDF)
    ax3 = axes[1, 0]
    sorted_conf = np.sort(confidences)
    cdf = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
    ax3.plot(sorted_conf, cdf, linewidth=2, color='navy')
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = np.searchsorted(sorted_conf, t)
        pct = idx / len(sorted_conf) * 100
        ax3.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
        ax3.annotate(f'{pct:.1f}%', xy=(t, idx/len(sorted_conf)), 
                    xytext=(t+0.02, idx/len(sorted_conf)-0.1), fontsize=9)
    ax3.axvline(x=0.6, color='red', linestyle='--', linewidth=2, label='Current (60%)')
    ax3.set_xlabel('Confidence Threshold', fontsize=12)
    ax3.set_ylabel('Cumulative Proportion (OCR Trigger Rate)', fontsize=12)
    ax3.set_title('Cumulative Distribution - OCR Trigger Rate', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 阈值选择建议表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [['Threshold', 'Use OCR', 'Use Image Rec.', 'OCR Rate']]
    for t in thresholds_labels:
        data = results["thresholds"][t]
        table_data.append([
            t,
            str(data["n_use_ocr"]),
            str(data["n_use_image_recognition"]),
            f'{data["ocr_trigger_rate"]:.1f}%'
        ])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.2, 0.25, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 高亮60%行
    for i in range(4):
        table[(2, i)].set_facecolor('#FFE4E1')
    
    ax4.set_title('Threshold Analysis Summary\n(60% highlighted - current setting)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def print_report(results: Dict):
    """打印分析报告"""
    print("\n" + "="*70)
    print("阈值分析报告 - Threshold Analysis Report")
    print("="*70)

    print(f"\n总样本数 (Total Samples): {results['total_samples']}")

    stats = results['confidence_stats']
    print(f"\n置信度统计 (Confidence Statistics):")
    print(f"  均值 (Mean):     {stats['mean']:.4f}")
    print(f"  标准差 (Std):    {stats['std']:.4f}")
    print(f"  中位数 (Median): {stats['median']:.4f}")
    print(f"  最小值 (Min):    {stats['min']:.4f}")
    print(f"  最大值 (Max):    {stats['max']:.4f}")

    print(f"\n不同阈值下的分布 (Distribution by Threshold):")
    print("-"*70)
    print(f"{'阈值':<10} {'使用OCR':<15} {'使用图像识别':<15} {'OCR触发率':<15}")
    print(f"{'Threshold':<10} {'Use OCR':<15} {'Use Image Rec.':<15} {'OCR Rate':<15}")
    print("-"*70)

    for threshold_name, data in results['thresholds'].items():
        print(f"{threshold_name:<10} {data['n_use_ocr']:<15} {data['n_use_image_recognition']:<15} {data['ocr_trigger_rate']:.1f}%")

    print("-"*70)

    # 60%阈值的合理性分析
    if "60%" in results['thresholds']:
        data_60 = results['thresholds']['60%']
        print(f"\n📊 60%阈值的合理性分析:")
        print(f"   - {data_60['ocr_trigger_rate']:.1f}% 的样本会触发OCR辅助识别")
        print(f"   - {data_60['image_recognition_rate']:.1f}% 的样本直接使用图像识别结果")

        # 比较不同阈值
        if "70%" in results['thresholds'] and "80%" in results['thresholds']:
            d70 = results['thresholds']['70%']
            d80 = results['thresholds']['80%']
            print(f"\n   与其他阈值对比:")
            print(f"   - 如果阈值设为70%: OCR触发率会增加到 {d70['ocr_trigger_rate']:.1f}% (+{d70['ocr_trigger_rate']-data_60['ocr_trigger_rate']:.1f}%)")
            print(f"   - 如果阈值设为80%: OCR触发率会增加到 {d80['ocr_trigger_rate']:.1f}% (+{d80['ocr_trigger_rate']-data_60['ocr_trigger_rate']:.1f}%)")

    print("\n" + "="*70)


def load_predictions_from_checkpoint(ckpt_path: str, data_dir: str, img_size: int = 224) -> Tuple[List[float], List[str]]:
    """从模型检查点加载并进行推理获取置信度"""
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
    except ImportError:
        print("Error: torch/torchvision not installed")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载检查点
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get('classes', [])

    # 构建模型
    from torchvision import models
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    # 构建transform
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 遍历数据目录
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                images.append(os.path.join(root, f))

    if not images:
        print(f"No images found in {data_dir}")
        return [], []

    print(f"Found {len(images)} images")

    confidences = []
    predictions = []

    with torch.no_grad():
        for i, img_path in enumerate(images):
            try:
                img = Image.open(img_path).convert('RGB')
                x = tfm(img).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                conf, idx = torch.max(probs, dim=0)
                confidences.append(float(conf))
                predictions.append(classes[int(idx)])

                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(images)} images...")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return confidences, predictions


def main():
    parser = argparse.ArgumentParser(description='阈值分析工具 - Threshold Analysis Tool')
    parser.add_argument('--ckpt', type=str, help='模型检查点路径')
    parser.add_argument('--data-dir', type=str, help='测试图像目录')
    parser.add_argument('--simulate', action='store_true', help='使用模拟数据（无需真实数据）')
    parser.add_argument('--n-samples', type=int, default=500, help='模拟数据的样本数')
    parser.add_argument('--output', type=str, default='threshold_analysis', help='输出文件前缀')
    parser.add_argument('--thresholds', type=str, default='0.5,0.6,0.7,0.8,0.9', help='要分析的阈值列表')
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(',')]

    if args.simulate:
        print("使用模拟数据进行分析...")
        print(f"模拟样本数: {args.n_samples}")
        sim_data = generate_simulated_data(args.n_samples)
        confidences = sim_data["final"]

        # 打印Stage1和Stage2的独立统计
        print(f"\n模拟的Stage1 (Knit vs Woven) 置信度: mean={np.mean(sim_data['stage1']):.3f}")
        print(f"模拟的Stage2 (细分类) 置信度: mean={np.mean(sim_data['stage2']):.3f}")
        print(f"最终置信度 (Stage1 × Stage2): mean={np.mean(confidences):.3f}")
    elif args.ckpt and args.data_dir:
        print(f"从模型加载推理结果...")
        confidences, predictions = load_predictions_from_checkpoint(args.ckpt, args.data_dir)
        if not confidences:
            print("No predictions generated. Exiting.")
            sys.exit(1)
    else:
        print("请指定 --simulate 或同时指定 --ckpt 和 --data-dir")
        parser.print_help()
        sys.exit(1)

    # 执行分析
    results = analyze_thresholds(confidences, thresholds)

    # 打印报告
    print_report(results)

    # 保存JSON结果
    json_path = f"{args.output}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {json_path}")

    # 生成可视化
    if HAS_MATPLOTLIB:
        plot_path = f"{args.output}_chart.png"
        plot_analysis(confidences, results, plot_path)

    print("\n完成!")


if __name__ == '__main__':
    main()

