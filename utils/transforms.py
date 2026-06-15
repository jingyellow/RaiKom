"""
数据增强与预处理模块
支持 torchvision 和 albumentations 两种增强方式
"""
import torch
import torchvision.transforms as T
from torchvision.transforms import autoaugment
import numpy as np


def build_transforms(config, is_train=True):
    """
    构建数据变换管道
    """
    img_size = config["dataset"]["img_size"]
    aug_cfg = config["augmentation"]

    if is_train:
        transforms = []

        # 基础变换：Resize
        transforms.append(T.Resize((img_size, img_size)))

        # 随机水平翻转
        if aug_cfg.get("horizontal_flip", True):
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        # 随机垂直翻转
        if aug_cfg.get("vertical_flip", False):
            transforms.append(T.RandomVerticalFlip(p=0.5))

        # 随机旋转
        if aug_cfg.get("rotation", 0) > 0:
            transforms.append(T.RandomRotation(degrees=aug_cfg["rotation"]))

        # Color Jitter
        cj = aug_cfg.get("color_jitter", {})
        if cj:
            transforms.append(T.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.1),
            ))

        # AutoAugment (针对ImageNet优化的自动增强策略)
        if aug_cfg.get("use_auto_augment", False):
            transforms.append(autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET))

        # 转换为Tensor
        transforms.append(T.ToTensor())

        # 归一化 (ImageNet预训练模型的标准归一化)
        transforms.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

        # Random Erasing
        if aug_cfg.get("random_erase", 0) > 0:
            transforms.append(T.RandomErasing(p=aug_cfg["random_erase"]))

        return T.Compose(transforms)

    else:
        # 验证/测试变换：仅做Resize和归一化
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def get_inference_transform(img_size=224):
    """
    推理时使用的变换
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
