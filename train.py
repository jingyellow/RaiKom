#!/usr/bin/env python3
"""
天气图像分类模型训练脚本
支持：单GPU训练、混合精度训练、学习率调度、早停
"""
import os
import sys
import yaml
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from models.model import build_model, freeze_backbone
from utils.dataset import build_dataloaders, compute_class_weights, get_train_label_counts
from utils.transforms import build_transforms
from utils.metrics import MetricsTracker, calculate_accuracy
from utils.device import resolve_device, apply_device_settings, configure_cudnn, print_device_info


def get_autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)


def set_seed(seed=42):
    """设置随机种子保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_optimizer(model, config):
    """构建优化器"""
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer_name = config["training"]["optimizer"].lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    """构建学习率调度器"""
    scheduler_name = config["training"]["scheduler"].lower()
    epochs = config["training"]["epochs"]
    lr = float(config["training"]["learning_rate"])
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    total_steps = steps_per_epoch * epochs

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr * 0.01
        )
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=steps_per_epoch * 10, gamma=0.1)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
    else:
        scheduler = None

    return scheduler


def build_criterion(config, train_dataset, num_classes, class_names, device):
    """构建损失函数，可选类别加权"""
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    weight = None

    if config["training"].get("use_class_weights", False):
        weight = compute_class_weights(train_dataset, num_classes).to(device)
        distribution = get_train_label_counts(train_dataset, class_names)

        print("启用类别加权损失:")
        for name in class_names:
            idx = class_names.index(name)
            print(f"  - {name}: {distribution[name]} 张, weight={weight[idx]:.4f}")

    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def get_best_metric_key(config):
    """获取最佳模型选型使用的验证指标"""
    metric_name = config["training"].get("best_metric", "f1_macro")
    if metric_name not in {"f1_macro", "accuracy"}:
        raise ValueError(f"不支持的 best_metric: {metric_name}，可选: f1_macro, accuracy")
    return metric_name


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config):
    """训练一个epoch"""
    model.train()
    metrics = MetricsTracker(num_classes=config["dataset"]["num_classes"], class_names=config["dataset"]["class_names"])
    metrics.reset()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    log_interval = config["logging"]["log_interval"]

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 混合精度训练
        with get_autocast_context(device, scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪
            if config["training"].get("gradient_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config["training"].get("gradient_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()

        # 计算指标
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        metrics.update(preds, labels, loss.item(), probs)

        # 更新进度条
        current_metrics = metrics.compute()
        pbar.set_postfix({
            "loss": f"{current_metrics['loss']:.4f}",
            "acc": f"{current_metrics['accuracy']:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
        })

    return metrics.compute()


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, config, split="Val"):
    """验证/测试"""
    model.eval()
    metrics = MetricsTracker(num_classes=config["dataset"]["num_classes"], class_names=config["dataset"]["class_names"])
    metrics.reset()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{split}]")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with get_autocast_context(device, device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        metrics.update(preds, labels, loss.item(), probs)

        current_metrics = metrics.compute()
        pbar.set_postfix({
            "loss": f"{current_metrics['loss']:.4f}",
            "acc": f"{current_metrics['accuracy']:.4f}",
        })

    final_metrics = metrics.compute()

    return final_metrics, metrics


def main(config_path="configs/config.yaml", device_name=None):
    # 加载配置
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    # 设备设置
    requested_device = device_name or config["project"].get("device", "auto")
    device = resolve_device(requested_device)
    device_settings = apply_device_settings(config, device)
    configure_cudnn(config, device)
    print_device_info(device)
    print(f"  - batch_size: {config['training']['batch_size']}")
    print(f"  - num_workers: {config['project']['num_workers']}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['project']['name']}_{config['model']['backbone']}_{timestamp}"
    checkpoint_dir = Path(config["logging"]["checkpoint_dir"]) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # 数据加载
    print("\n[1/4] 加载数据集...")
    train_transform = build_transforms(config, is_train=True)
    val_transform = build_transforms(config, is_train=False)
    train_loader, val_loader, test_loader, num_classes, class_names, train_dataset = build_dataloaders(config, train_transform, val_transform)

    # 更新配置中的类别数
    config["model"]["num_classes"] = num_classes
    config["dataset"]["num_classes"] = num_classes
    config["dataset"]["class_names"] = class_names

    # 构建模型
    print("\n[2/4] 构建模型...")
    model = build_model(config)
    model = model.to(device)

    # 损失函数（支持类别加权）
    criterion = build_criterion(config, train_dataset, num_classes, class_names, device)
    best_metric_key = get_best_metric_key(config)
    best_metric_label = "F1" if best_metric_key == "f1_macro" else "Acc"

    # 优化器
    optimizer = build_optimizer(model, config)

    # 学习率调度器
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # 混合精度
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 训练状态
    best_score = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = config["training"]["early_stopping_patience"]

    print("\n[3/4] 开始训练...")
    print(f"总Epoch数: {config['training']['epochs']}")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"学习率: {config['training']['learning_rate']}")
    print(f"最佳模型指标: Val {best_metric_label}")
    print(f"早停耐心: {early_stop_patience}")
    print("=" * 60)

    for epoch in range(1, config["training"]["epochs"] + 1):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config
        )

        # 学习率更新
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # 验证
        val_metrics, val_tracker = validate(model, val_loader, criterion, device, epoch, config)

        # ReduceLROnPlateau 调度器
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics[best_metric_key])

        # 打印训练结果
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
        MetricsTracker.print_per_class_metrics(train_metrics, "Train", class_names)
        MetricsTracker.print_per_class_metrics(val_metrics, "Val", class_names)

        # 保存最佳模型
        current_score = val_metrics[best_metric_key]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            patience_counter = 0

            if config["training"]["save_best_only"]:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": val_metrics["accuracy"],
                    "best_val_f1": val_metrics["f1_macro"],
                    "best_metric": best_metric_key,
                    "config": config,
                    "class_names": config["dataset"]["class_names"],
                }, checkpoint_dir / "best_model.pth")
                print(f"  [OK] 保存最佳模型 (Val {best_metric_label}: {best_score:.4f})")
        else:
            patience_counter += 1
            print(f"  - 未提升 ({patience_counter}/{early_stop_patience})")

        # 定期保存检查点
        if epoch % config["logging"]["save_interval"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\n早停触发！最佳验证 {best_metric_label}: {best_score:.4f} (Epoch {best_epoch})")
            break

    print("\n[4/4] 训练完成！")
    print(f"最佳验证 {best_metric_label}: {best_score:.4f} (Epoch {best_epoch})")

    # 最终测试
    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_tracker = validate(model, test_loader, criterion, device, 0, config, split="Test")

    print(f"\n测试结果:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_macro']:.4f}")

    MetricsTracker.print_per_class_metrics(test_metrics, "Test", class_names)
    test_tracker.print_classification_report()
    test_tracker.plot_confusion_matrix(save_path=str(checkpoint_dir / "confusion_matrix.png"))

    print(f"\n所有结果已保存到: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="天气图像分类模型训练")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备: auto | cuda | cpu | cuda:0（指定 GPU 编号）",
    )
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 训练（等同于 --device cuda）")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 训练（等同于 --device cpu）")
    args = parser.parse_args()

    device_name = args.device
    if args.gpu:
        device_name = "cuda"
    elif args.cpu:
        device_name = "cpu"

    main(args.config, device_name)
