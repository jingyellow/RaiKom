"""
模型评估指标模块
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """训练过程中的指标追踪器"""

    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.total_loss = 0.0
        self.count = 0

    def update(self, preds, labels, loss=None, probs=None):
        """
        更新指标
        preds: 预测类别 (N,)
        labels: 真实标签 (N,)
        loss: 批次损失值
        probs: 预测概率 (N, num_classes)
        """
        self.all_preds.extend(preds.detach().cpu().numpy().tolist())
        self.all_labels.extend(labels.detach().cpu().numpy().tolist())
        if probs is not None:
            self.all_probs.extend(probs.detach().cpu().numpy().tolist())
        if loss is not None:
            self.total_loss += loss * len(labels)
            self.count += len(labels)

    def compute_per_class_metrics(self, labels, preds):
        """计算每个类别的 precision / recall / f1"""
        label_indices = list(range(self.num_classes))
        precision = precision_score(
            labels, preds, average=None, zero_division=0, labels=label_indices
        )
        recall = recall_score(
            labels, preds, average=None, zero_division=0, labels=label_indices
        )
        f1 = f1_score(
            labels, preds, average=None, zero_division=0, labels=label_indices
        )

        per_class = {}
        for i, name in enumerate(self.class_names):
            per_class[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        return per_class

    def compute(self):
        """计算所有指标"""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "per_class": self.compute_per_class_metrics(labels, preds),
        }

        if self.count > 0:
            metrics["loss"] = self.total_loss / self.count

        # Top-k 准确率
        if len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            for k in [1, 3, 5]:
                if k <= self.num_classes:
                    try:
                        metrics[f"top_{k}_accuracy"] = top_k_accuracy_score(labels, probs, k=k)
                    except:
                        pass

        return metrics

    def get_confusion_matrix(self):
        """获取混淆矩阵"""
        return confusion_matrix(self.all_labels, self.all_preds)

    def plot_confusion_matrix(self, save_path=None):
        """绘制混淆矩阵热力图"""
        cm = self.get_confusion_matrix()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    def print_classification_report(self):
        """打印详细分类报告"""
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        print(classification_report(
            self.all_labels, self.all_preds,
            target_names=self.class_names,
            digits=4,
        ))

    @staticmethod
    def print_per_class_metrics(metrics, split_name, class_names):
        """打印各类别的 precision / recall / f1"""
        per_class = metrics.get("per_class", {})
        if not per_class:
            return

        print(f"  [{split_name}] 各类指标:")
        print(f"    {'类别':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for name in class_names:
            cls_metrics = per_class.get(name, {})
            print(
                f"    {name:<10} "
                f"{cls_metrics.get('precision', 0.0):>10.4f} "
                f"{cls_metrics.get('recall', 0.0):>10.4f} "
                f"{cls_metrics.get('f1', 0.0):>10.4f}"
            )
        print(
            f"    {'macro avg':<10} "
            f"{metrics.get('precision_macro', 0.0):>10.4f} "
            f"{metrics.get('recall_macro', 0.0):>10.4f} "
            f"{metrics.get('f1_macro', 0.0):>10.4f}"
        )


def calculate_accuracy(outputs, targets, topk=(1,)):
    """
    计算Top-k准确率
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
