"""
模型定义模块
使用 timm 库构建分类模型，支持多种主干网络
"""
import torch
import torch.nn as nn
import timm


def build_model(config):
    """
    构建分类模型

    支持的主干网络 (timm库):
    - ResNet系列: resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d
    - EfficientNet系列: efficientnet_b0 ~ efficientnet_b7
    - Vision Transformer: vit_base_patch16_224, vit_small_patch16_224, deit_base_patch16_224
    - ConvNeXt: convnext_tiny, convnext_small, convnext_base
    - Swin Transformer: swin_tiny_patch4_window7_224
    - MobileNet: mobilenetv3_large_100, mobilenetv3_small_100
    """
    backbone = config["model"]["backbone"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"]["pretrained"]
    drop_rate = config["model"].get("drop_rate", 0.0)

    # 使用 timm 创建模型
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    print(f"模型构建完成: {backbone}")
    print(f"  - 预训练: {pretrained}")
    print(f"  - 类别数: {num_classes}")
    print(f"  - Dropout: {drop_rate}")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - 可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    return model


def freeze_backbone(model, freeze=True):
    """
    冻结/解冻主干网络参数（用于迁移学习的两阶段训练）
    """
    # 获取分类头之前的所有层
    for name, param in model.named_parameters():
        if "head" not in name and "fc" not in name and "classifier" not in name:
            param.requires_grad = not freeze

    status = "冻结" if freeze else "解冻"
    print(f"主干网络已{status}")

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def get_model_info(model, input_size=(1, 3, 224, 224)):
    """
    获取模型信息摘要
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    # 前向传播测试
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    info = {
        "input_shape": input_size,
        "output_shape": tuple(output.shape),
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    return info
