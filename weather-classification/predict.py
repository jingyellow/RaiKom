#!/usr/bin/env python3
"""
单张图像推理脚本
支持命令行直接推理和批量推理
"""
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from models.model import build_model
from utils.transforms import get_inference_transform


class WeatherPredictor:
    """天气分类预测器"""

    def __init__(self, checkpoint_path, config=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 获取配置
        if config is None:
            if "config" in checkpoint:
                self.config = checkpoint["config"]
            else:
                raise ValueError("请提供配置文件，或检查点中包含配置信息")
        else:
            self.config = config

        # 类别名称
        self.class_names = checkpoint.get("class_names", self.config["dataset"]["class_names"])
        self.num_classes = len(self.class_names)

        # 构建并加载模型
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 图像变换
        self.transform = get_inference_transform(self.config["dataset"]["img_size"])

        print(f"预测器加载完成: {self.config['model']['backbone']}")
        print(f"类别: {self.class_names}")

    @torch.no_grad()
    def predict(self, image_path):
        """
        预测单张图像

        Args:
            image_path: 图像路径或PIL Image对象

        Returns:
            dict: 包含预测结果的字典
        """
        # 加载图像
        if isinstance(image_path, (str, Path)):
            image = Image.open(str(image_path)).convert("RGB")
        else:
            image = image_path

        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.autocast(device_type=self.device.type):
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

        # 获取预测结果
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

        # 所有类别的概率
        all_probs = {cls: probs[0][i].item() for i, cls in enumerate(self.class_names)}

        # 按概率排序
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "predicted_idx": pred_idx,
            "all_probabilities": all_probs,
            "top3": sorted_probs[:3],
        }

    @torch.no_grad()
    def predict_batch(self, image_paths):
        """
        批量预测

        Args:
            image_paths: 图像路径列表

        Returns:
            list: 预测结果列表
        """
        images = []
        for path in image_paths:
            img = Image.open(str(path)).convert("RGB")
            img_tensor = self.transform(img)
            images.append(img_tensor)

        batch = torch.stack(images).to(self.device)

        with torch.autocast(device_type=self.device.type):
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)

        results = []
        for i in range(len(image_paths)):
            pred_idx = torch.argmax(probs[i]).item()
            results.append({
                "image": str(image_paths[i]),
                "predicted_class": self.class_names[pred_idx],
                "confidence": probs[i][pred_idx].item(),
                "all_probabilities": {cls: probs[i][j].item() for j, cls in enumerate(self.class_names)},
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="天气图像分类推理")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--image", type=str, required=True, help="待预测图像路径")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    args = parser.parse_args()

    # 加载配置（可选）
    config = None
    if Path(args.config).exists():
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # 创建预测器
    predictor = WeatherPredictor(args.checkpoint, config, args.device)

    # 预测
    result = predictor.predict(args.image)

    # 输出结果
    print("\n" + "=" * 50)
    print("预测结果")
    print("=" * 50)
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度:   {result['confidence']:.4f}")
    print("\nTop-3 预测:")
    for i, (cls, prob) in enumerate(result['top3'], 1):
        print(f"  {i}. {cls}: {prob:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
