"""
天气图像数据集加载模块
支持文件夹格式数据集（ImageFolder风格）
"""
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class TransformSubset(torch.utils.data.Dataset):
    """包装 random_split 的结果，使其可以应用不同的 transform"""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class WeatherDataset(Dataset):
    """
    天气图像分类数据集
    目录结构:
        data_dir/
            train/
                cloudy/xxx.jpg
                rainy/xxx.jpg
                ...
            val/
                ...
            test/
                ...
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform

        if class_names is None:
            self.class_names = sorted([
                d.name for d in self.data_dir.iterdir()
                if d.is_dir() and any(
                    f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                    for f in d.glob("*")
                )
            ])
        else:
            self.class_names = class_names

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.samples = []
        self._load_samples()

        print(f"[{split}] 加载完成: {len(self.samples)} 张图像, {self.num_classes} 个类别")

    def _load_samples(self):
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_distribution(self) -> dict:
        counts = {cls: 0 for cls in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


def _is_valid_dir(path):
    """检查路径是否存在且包含图片"""
    if not path or not os.path.exists(path):
        return False
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                return True
    return False


def _collect_train_labels(train_dataset) -> List[int]:
    """从训练集收集标签（兼容 WeatherDataset 与 TransformSubset）"""
    if isinstance(train_dataset, TransformSubset):
        base_dataset = train_dataset.subset.dataset
        return [base_dataset.samples[i][1] for i in train_dataset.subset.indices]
    return [label for _, label in train_dataset.samples]


def get_train_label_counts(train_dataset, class_names: List[str]) -> dict:
    """获取训练集各类别样本数"""
    labels = _collect_train_labels(train_dataset)
    counts = {name: 0 for name in class_names}
    for label in labels:
        counts[class_names[label]] += 1
    return counts


def compute_class_weights(train_dataset, num_classes: int) -> torch.Tensor:
    """根据训练集类别分布计算损失函数权重（少数类权重更高）"""
    labels = _collect_train_labels(train_dataset)
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1

    total = len(labels)
    weights = [total / (num_classes * max(count, 1)) for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(config, train_transform, val_transform):
    """构建训练、验证、测试 DataLoader"""
    data_dir = config["dataset"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["project"]["num_workers"]
    pin_memory = config["project"].get("pin_memory", False)
    split_ratio = config["dataset"].get("train_val_test_split", [0.7, 0.15, 0.15])
    seed = config["project"]["seed"]

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not _is_valid_dir(train_dir):
        raise FileNotFoundError(f"训练数据目录不存在或为空: {train_dir}")

    if not _is_valid_dir(val_dir) or not _is_valid_dir(test_dir):
        print("val 或 test 目录为空，自动从 train 划分...")

        base_dataset = WeatherDataset(
            data_dir=data_dir,
            split="train",
            transform=None,
        )
        num_classes = base_dataset.num_classes
        class_names = base_dataset.class_names
        print(f"[train] 原始数据: {len(base_dataset)} 张图像, {num_classes} 个类别")

        total = len(base_dataset)
        train_size = int(split_ratio[0] * total)
        val_size = int(split_ratio[1] * total)
        test_size = total - train_size - val_size

        train_subset, val_subset, test_subset = random_split(
            base_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_dataset = TransformSubset(train_subset, train_transform)
        val_dataset = TransformSubset(val_subset, val_transform)
        test_dataset = TransformSubset(test_subset, val_transform)

        print(f"[train] 划分后: {len(train_dataset)} 张")
        print(f"[val] 划分后: {len(val_dataset)} 张")
        print(f"[test] 划分后: {len(test_dataset)} 张")
    else:
        train_dataset = WeatherDataset(
            data_dir=data_dir,
            split="train",
            transform=train_transform,
        )
        val_dataset = WeatherDataset(
            data_dir=data_dir,
            split="val",
            transform=val_transform,
        )
        test_dataset = WeatherDataset(
            data_dir=data_dir,
            split="test",
            transform=val_transform,
        )
        num_classes = train_dataset.num_classes
        class_names = train_dataset.class_names

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, num_classes, class_names, train_dataset


def split_dataset(source_dir: str, output_dir: str, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """将原始数据集划分为 train/val/test"""
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = sorted([
            f for f in class_dir.glob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        split_dict = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for split_name, split_images in split_dict.items():
            split_dir = output_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                shutil.copy2(str(img), str(split_dir / img.name))

    print(f"数据集划分完成！输出目录: {output_dir}")
    print(f"划分比例: train={split_ratio[0]}, val={split_ratio[1]}, test={split_ratio[2]}")
