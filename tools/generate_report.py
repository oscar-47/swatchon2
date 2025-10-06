import argparse
import json
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(classes)), max(5, 0.5 * len(classes))))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_bar(values: np.ndarray, classes: List[str], out_path: str, title: str, ylabel: str):
    order = np.argsort(values)[::-1]
    values = values[order]
    classes = [classes[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(classes)), 5))
    ax.bar(np.arange(len(classes)), values, color="#4C72B0")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_csv(rows: List[List[str]], header: List[str], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')


def generate(report_json: str, out_dir: str = None):
    with open(report_json, 'r', encoding='utf-8') as f:
        rep = json.load(f)
    cm = np.array(rep['confusion_matrix'], dtype=np.int64)
    classes = rep['classes']
    test_acc = float(rep.get('test_acc', 0.0))
    test_f1 = float(rep.get('test_macro_f1', 0.0))
    best_epoch = rep.get('best_epoch')
    best_val_acc = rep.get('best_val_acc')

    if out_dir is None:
        out_dir = os.path.dirname(report_json)
    ensure_dir(out_dir)

    # Per-class metrics
    tp = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)  # support
    col_sum = cm.sum(axis=0).astype(np.float64)
    support = row_sum
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)  # same as per-class accuracy
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    per_class_acc = recall

    # Save per-class CSV
    rows = []
    for i, c in enumerate(classes):
        fp = col_sum[i] - tp[i]
        fn = row_sum[i] - tp[i]
        rows.append([
            c,
            int(support[i]),
            int(tp[i]),
            int(fp),
            int(fn),
            f"{precision[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{f1[i]:.4f}",
            f"{per_class_acc[i]:.4f}",
        ])
    save_csv(rows, header=["class", "support", "tp", "fp", "fn", "precision", "recall", "f1", "per_class_acc"], out_path=os.path.join(out_dir, 'per_class_metrics.csv'))

    # Plots
    plot_confusion_matrix(cm, classes, os.path.join(out_dir, 'confusion_matrix.png'), title=f"Confusion Matrix (acc={test_acc:.3f}, macroF1={test_f1:.3f})")
    plot_per_class_bar(per_class_acc, classes, os.path.join(out_dir, 'per_class_accuracy.png'), title='Per-class Accuracy (Recall)', ylabel='Accuracy')

    # Simple HTML report
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<title>Classification Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 16px; }}
section {{ margin-bottom: 24px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 8px; border: 1px solid #ddd; text-align: left; }}
</style>
</head>
<body>
<h2>模型分类报告</h2>
<p><b>测试集准确率</b>: {test_acc:.3f} &nbsp; <b>宏平均F1</b>: {test_f1:.3f} &nbsp; <b>最佳验证准确率</b>: {best_val_acc:.3f} (epoch {best_epoch})</p>
<section>
  <h3>混淆矩阵</h3>
  <img src="confusion_matrix.png" alt="confusion matrix"/>
</section>
<section>
  <h3>各类别准确率（召回）</h3>
  <img src="per_class_accuracy.png" alt="per class accuracy"/>
  <p>详表见 <code>per_class_metrics.csv</code></p>
</section>
</body>
</html>
"""
    with open(os.path.join(out_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Saved: {out_dir}/confusion_matrix.png, per_class_accuracy.png, per_class_metrics.csv, report.html")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate confusion matrix and per-class report from test_report.json')
    parser.add_argument('--report', required=True, help='Path to test_report.json')
    parser.add_argument('--out', default=None, help='Output directory (default: same dir as report)')
    args = parser.parse_args()
    generate(args.report, args.out)

