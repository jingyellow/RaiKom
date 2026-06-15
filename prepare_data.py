#!/usr/bin/env python3
"""
数据准备脚本
支持：
1. 下载 Weather Classification 数据集 (Kaggle)
2. 划分数据集为 train/val/test
3. 验证数据集完整性
"""
import os
import argparse
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

from utils.dataset import split_dataset


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """下载文件并显示进度"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_weather_dataset(output_dir="./data"):
    """
    下载天气分类数据集
    使用 Kaggle 的 Weather Classification 数据集
    数据集地址: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

    注意：由于Kaggle需要认证，这里提供手动下载说明
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("天气分类数据集准备")
    print("=" * 60)
    print("\n推荐数据集来源：")
    print("1. Kaggle Weather Classification Dataset")
    print("   https://www.kaggle.com/datasets/jehanbhathena/weather-dataset")
    print("   包含4类: cloudy, rainy, snowy, sunny")
    print()
    print("2. Multi-class Weather Dataset (MWD)")
    print("   https://data.mendeley.com/datasets/4drtyfjtfy/1")
    print("   包含4类: cloudy, rainy, snowy, sunny")
    print()
    print("3. 11-class Weather Dataset")
    print("   https://github.com/wangxiao5791509/RGB_Event_Classification")
    print("   包含11类天气场景")
    print()
    print("=" * 60)
    print("\n手动下载步骤：")
    print("1. 访问上述Kaggle链接")
    print("2. 下载数据集ZIP文件")
    print("3. 解压到 ./data/raw/ 目录")
    print("4. 运行: python prepare_data.py --split")
    print()
    print("或者使用自己的数据集，目录结构应为：")
    print("  data/raw/")
    print("    ├── cloudy/")
    print("    ├── rainy/")
    print("    ├── snowy/")
    print("    └── sunny/")
    print()


def prepare_dataset(raw_dir="./data/raw", output_dir="./data", split_ratio=(0.7, 0.15, 0.15), seed=42):
    """
    划分数据集
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    if not raw_path.exists():
        print(f"错误: 原始数据目录不存在: {raw_dir}")
        print("请先下载数据集到该目录")
        return False

    print(f"\n划分数据集...")
    print(f"原始目录: {raw_dir}")
    print(f"输出目录: {output_dir}")
    print(f"划分比例: Train {split_ratio[0]}, Val {split_ratio[1]}, Test {split_ratio[2]}")

    # 执行划分
    split_dataset(str(raw_path), str(output_path), split_ratio, seed)

    # 验证结果
    print("\n数据集划分结果：")
    for split in ["train", "val", "test"]:
        split_dir = output_path / split
        if split_dir.exists():
            total_images = 0
            class_counts = {}
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*")))
                    class_counts[class_dir.name] = count
                    total_images += count
            print(f"\n  [{split}] 总计: {total_images} 张")
            for cls, count in sorted(class_counts.items()):
                print(f"    - {cls}: {count} 张")

    print(f"\n✅ 数据准备完成！")
    return True


def verify_dataset(data_dir="./data"):
    """验证数据集完整性"""
    data_path = Path(data_dir)

    print("\n验证数据集...")
    issues = []

    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            issues.append(f"缺少目录: {split_dir}")
            continue

        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if len(class_dirs) == 0:
            issues.append(f"{split} 目录下没有类别文件夹")

        for class_dir in class_dirs:
            images = list(class_dir.glob("*"))
            if len(images) == 0:
                issues.append(f"{split}/{class_dir.name} 目录为空")

    if issues:
        print("⚠️ 发现问题：")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ 数据集验证通过！")
        return True


def create_sample_structure(output_dir="./data"):
    """创建示例数据集结构（用于测试）"""
    output_path = Path(output_dir)

    print("\n创建示例数据集结构...")

    # 创建示例类别
    classes = ["cloudy", "rainy", "snowy", "sunny"]

    for split in ["train", "val", "test"]:
        for cls in classes:
            class_dir = output_path / split / cls
            class_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ 示例结构已创建: {output_dir}")
    print("请将图像文件放入对应类别的文件夹中")
    print()
    print("目录结构：")
    print("  data/")
    print("    ├── train/")
    print("    │   ├── cloudy/")
    print("    │   ├── rainy/")
    print("    │   ├── snowy/")
    print("    │   └── sunny/")
    print("    ├── val/")
    print("    │   └── ...")
    print("    └── test/")
    print("        └── ...")


def main():
    parser = argparse.ArgumentParser(description="数据准备脚本")
    parser.add_argument("--download", action="store_true", help="显示数据集下载说明")
    parser.add_argument("--split", action="store_true", help="划分数据集")
    parser.add_argument("--verify", action="store_true", help="验证数据集")
    parser.add_argument("--create-sample", action="store_true", help="创建示例数据集结构")
    parser.add_argument("--raw-dir", type=str, default="./data/raw", help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default="./data", help="输出数据目录")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if args.download:
        download_weather_dataset(args.output_dir)

    if args.create_sample:
        create_sample_structure(args.output_dir)

    if args.split:
        split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)
        if abs(sum(split_ratio) - 1.0) > 0.001:
            print("错误: 划分比例之和必须等于1.0")
            return
        prepare_dataset(args.raw_dir, args.output_dir, split_ratio, args.seed)

    if args.verify:
        verify_dataset(args.output_dir)

    # 如果没有参数，显示帮助
    if not any([args.download, args.split, args.verify, args.create_sample]):
        parser.print_help()
        print("\n示例用法：")
        print("  1. 创建示例结构:  python prepare_data.py --create-sample")
        print("  2. 划分数据集:    python prepare_data.py --split")
        print("  3. 验证数据集:    python prepare_data.py --verify")
        print("  4. 查看下载说明:  python prepare_data.py --download")


if __name__ == "__main__":
    main()
