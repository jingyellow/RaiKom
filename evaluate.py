#!/usr/bin/env python3
"""
模型评估脚本
加载训练好的模型，在测试集上进行详细评估
"""
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import build_model
from utils.dataset import WeatherDataset
from utils.transforms import build_transforms
from utils.metrics import MetricsTracker
from utils.device import resolve_device, print_device_info


def evaluate(config_path, checkpoint_path, split="test", device_name=None):
    """评估模型"""
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    requested_device = device_name or config["project"].get("device", "auto")
    device = resolve_device(requested_device)
    print_device_info(device)

    # 加载检查点
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 如果检查点中包含配置，使用检查点中的配置
    if "config" in checkpoint:
        config = checkpoint["config"]

    # 构建模型
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 数据加载
    transform = build_transforms(config, is_train=False)
    dataset = WeatherDataset(
        data_dir=config["dataset"]["data_dir"],
        split=split,
        transform=transform,
        class_names=config["dataset"]["class_names"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["project"]["num_workers"],
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    metrics = MetricsTracker(num_classes=config["dataset"]["num_classes"], class_names=config["dataset"]["class_names"])
    metrics.reset()

    # 评估
    print(f"\n在 {split} 集上评估...")
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            metrics.update(preds, labels, loss.item(), probs)
            total_loss += loss.item() * len(labels)

    # 计算指标
    final_metrics = metrics.compute()

    print("\n" + "=" * 60)
    print(f"评估结果 ({split}集)")
    print("=" * 60)
    print(f"Loss:      {final_metrics['loss']:.4f}")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision_macro']:.4f}")
    print(f"Recall:    {final_metrics['recall_macro']:.4f}")
    print(f"F1-Score:  {final_metrics['f1_macro']:.4f}")

    if "top_1_accuracy" in final_metrics:
        print(f"Top-1 Acc: {final_metrics['top_1_accuracy']:.4f}")
    if "top_3_accuracy" in final_metrics:
        print(f"Top-3 Acc: {final_metrics['top_3_accuracy']:.4f}")

    MetricsTracker.print_per_class_metrics(
        final_metrics, split, config["dataset"]["class_names"]
    )
    metrics.print_classification_report()

    # 保存混淆矩阵
    save_dir = Path(config["logging"]["checkpoint_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics.plot_confusion_matrix(save_path=str(save_dir / f"confusion_matrix_{split}.png"))
    print(f"\n混淆矩阵已保存到: {save_dir / f'confusion_matrix_{split}.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估天气分类模型")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="评估数据集")
    parser.add_argument("--device", type=str, default=None, help="计算设备: auto | cuda | cpu | cuda:0")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU（等同于 --device cuda）")
    args = parser.parse_args()

    device_name = args.device
    if args.gpu:
        device_name = "cuda"

    evaluate(args.config, args.checkpoint, args.split, device_name)
