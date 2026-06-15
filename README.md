# 天气图像分类

基于 PyTorch 和 [timm](https://github.com/rwightman/pytorch-image-models) 的天气场景图像分类项目。支持 EfficientNet、ResNet、ViT、ConvNeXt、Swin Transformer 等多种主干网络，提供数据准备、训练、评估和推理的完整流程。

## 功能特性

- **多模型支持**：通过 timm 一键切换 ResNet、EfficientNet、ViT、ConvNeXt、Swin 等主干
- **迁移学习**：ImageNet 预训练权重，支持冻结主干的两阶段训练
- **自动设备适配**：根据 GPU/CPU 自动调整 batch size、混合精度、cuDNN 等参数
- **数据增强**：AutoAugment、随机翻转/旋转、ColorJitter、Random Erasing
- **训练工具**：混合精度（AMP）、学习率调度、早停、TensorBoard 日志
- **完整评估**：准确率、精确率、召回率、F1、Top-k、混淆矩阵

## 项目结构

```
weather-classification/
├── configs/
│   └── config.yaml          # 训练与推理配置
├── data/
│   ├── train/               # 训练集（按类别分文件夹）
│   ├── val/                 # 验证集
│   └── test/                # 测试集
├── models/
│   ├── __init__.py
│   └── model.py             # 模型构建（timm）
├── utils/
│   ├── dataset.py           # 数据集加载与划分
│   ├── transforms.py        # 数据增强
│   ├── metrics.py           # 评估指标
│   └── device.py            # 设备选择与 GPU/CPU 差异化配置
├── checkpoints/             # 模型检查点
├── results/logs/            # TensorBoard 日志
├── prepare_data.py          # 数据准备
├── train.py                 # 训练
├── evaluate.py              # 评估
├── predict.py               # 推理
├── requirements.txt
├── TUTORIAL.md              # 详细教程
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）

```bash
cd weather-classification

# 创建并激活虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

将图像按类别放入 `data/raw/` 目录：

```
data/raw/
├── cloudy/
├── rainy/
├── snowy/
└── sunny/
```

```bash
# 创建空目录结构（快速测试用）
python prepare_data.py --create-sample

# 划分 train / val / test（默认 7:1.5:1.5）
python prepare_data.py --split

# 验证数据集完整性
python prepare_data.py --verify
```

**推荐公开数据集：**

| 数据集 | 类别 | 链接 |
|--------|------|------|
| Weather Classification (Kaggle) | 4 类 | https://www.kaggle.com/datasets/jehanbhathena/weather-dataset |
| Multi-class Weather Dataset | 4 类 | https://data.mendeley.com/datasets/4drtyfjtfy/1 |

下载后解压到 `data/raw/`，再执行 `--split` 即可。更多步骤见 [TUTORIAL.md](TUTORIAL.md)。

### 2. 训练

```bash
# 自动选择设备（有 GPU 用 GPU，否则 CPU）
python train.py

# 指定 GPU
python train.py --device cuda
python train.py --gpu

# 强制 CPU
python train.py --cpu

# 指定配置文件
python train.py --config configs/config.yaml
```

训练完成后输出：

- 最佳模型：`checkpoints/weather-classification_<backbone>_<timestamp>/best_model.pth`
- 混淆矩阵：`checkpoints/.../confusion_matrix.png`
- TensorBoard 日志：`results/logs/.../`

```bash
tensorboard --logdir=results/logs
```

### 3. 评估

```bash
python evaluate.py --checkpoint checkpoints/.../best_model.pth --split test
python evaluate.py --checkpoint checkpoints/.../best_model.pth --split val --gpu
```

### 4. 推理

```bash
python predict.py --checkpoint checkpoints/.../best_model.pth --image path/to/image.jpg
```

Python API：

```python
from predict import WeatherPredictor

predictor = WeatherPredictor("checkpoints/.../best_model.pth")
result = predictor.predict("image.jpg")
print(result["predicted_class"], result["confidence"])

# 批量推理
results = predictor.predict_batch(["img1.jpg", "img2.jpg"])
```

## 配置说明

主要配置项位于 `configs/config.yaml`：

```yaml
project:
  device: "auto"        # auto | cuda | cpu | cuda:0

device_settings:        # 按设备自动应用
  cpu:
    batch_size: 8
    num_workers: 0
    pin_memory: false
  cuda:
    batch_size: 32
    num_workers: 0
    pin_memory: true

dataset:
  data_dir: "./data"
  num_classes: 4
  class_names: ["cloudy", "rainy", "snowy", "sunny"]
  img_size: 224

model:
  backbone: "efficientnet_b0"
  pretrained: true
  drop_rate: 0.3

training:
  epochs: 50
  learning_rate: 0.0001
  optimizer: "adamw"    # adam | adamw | sgd
  scheduler: "cosine"   # cosine | step | plateau
  early_stopping_patience: 10
```

### 常用主干网络

| 主干 | 参数量 | 特点 |
|------|--------|------|
| `efficientnet_b0` | ~5M | 速度快，适合快速实验 |
| `efficientnet_b3` | ~12M | 速度与精度平衡 |
| `resnet50` | ~26M | 经典稳定 |
| `convnext_tiny` | ~29M | 现代 CNN 架构 |
| `vit_base_patch16_224` | ~86M | 精度高，显存需求大 |
| `swin_tiny_patch4_window7_224` | ~28M | 层次化 Transformer |

修改 `model.backbone` 后重新训练即可切换模型。完整 timm 模型列表见 [timm 文档](https://huggingface.co/docs/timm)。

### 扩展类别

修改 `dataset.num_classes` 和 `dataset.class_names`，同时更新 `model.num_classes`，准备对应数据后重新训练。

## 命令行参数

| 脚本 | 参数 | 说明 |
|------|------|------|
| `train.py` | `--config` | 配置文件路径（默认 `configs/config.yaml`） |
| | `--device` | `auto` / `cuda` / `cpu` / `cuda:0` |
| | `--gpu` / `--cpu` | 快捷指定设备 |
| `evaluate.py` | `--checkpoint` | 模型检查点（必填） |
| | `--split` | `train` / `val` / `test`（默认 `test`） |
| | `--device` / `--gpu` | 计算设备 |
| `predict.py` | `--checkpoint` | 模型检查点（必填） |
| | `--image` | 待预测图像（必填） |
| | `--device` | 计算设备（默认 `cuda`） |
| `prepare_data.py` | `--split` | 划分数据集 |
| | `--verify` | 验证数据集 |
| | `--create-sample` | 创建示例目录结构 |
| | `--download` | 显示数据集下载说明 |
| | `--raw-dir` / `--output-dir` | 原始/输出目录 |
| | `--train-ratio` / `--val-ratio` / `--test-ratio` | 划分比例 |

## 常见问题

**CUDA out of memory**

- 减小 `device_settings.cuda.batch_size`（如 16 或 8）
- 换用更小模型（如 `efficientnet_b0`）
- 减小 `dataset.img_size`

**训练准确率低**

- 确认 `model.pretrained: true`
- 降低学习率（如 `1e-5`）
- 检查数据目录与类别名称是否匹配
- 增加训练轮数

**过拟合**

- 增大 `model.drop_rate` 和 `training.weight_decay`
- 加强数据增强
- 利用早停机制（`early_stopping_patience`）

**CPU 训练慢**

- 项目会自动为 CPU 降低 batch size 并关闭混合精度
- 建议使用 `efficientnet_b0` 等轻量模型

## 进阶用法

- 两阶段迁移学习：使用 `models.model.freeze_backbone()` 先冻结主干训练分类头，再解冻微调
- 批量推理与部署：见 [TUTORIAL.md](TUTORIAL.md) 中的 ONNX 导出、模型量化等章节

## 参考资源

- [timm](https://github.com/rwightman/pytorch-image-models)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [Kaggle Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)

## 许可证

MIT License
