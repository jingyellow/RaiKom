# 🎓 天气图像分类项目完整教程

## 第一步：环境搭建

```bash
# 1. 安装 Python 3.8+
python --version

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(torch.__version__)"
python -c "import timm; print(timm.__version__)"
```

## 第二步：准备数据集

### 2.1 下载公开数据集（推荐）

**Weather Classification Dataset** 是最常用的天气分类数据集：

1. 访问 https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
2. 点击 "Download" 下载 ZIP 文件
3. 解压到项目目录

```bash
# 解压数据集
unzip archive.zip -d data/raw/

# 查看数据结构
ls data/raw/
# 输出: cloudy  rainy  snowy  sunny
```

### 2.2 划分数据集

```bash
# 自动划分为 train/val/test
python prepare_data.py --split --raw-dir ./data/raw --output-dir ./data

# 自定义划分比例
python prepare_data.py --split --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### 2.3 验证数据集

```bash
python prepare_data.py --verify
```

## 第三步：训练模型

### 3.1 快速开始（使用默认配置）

```bash
python train.py
```

### 3.2 修改配置训练不同模型

编辑 `configs/config.yaml`：

```yaml
# 使用 EfficientNet-B3（更高精度）
model:
  backbone: "efficientnet_b3"
  pretrained: true
  drop_rate: 0.3
  num_classes: 4

# 增大批次和训练轮数
training:
  batch_size: 16    # 根据显存调整
  epochs: 100
  learning_rate: 5e-5
```

```bash
python train.py --config configs/config.yaml
```

### 3.3 使用 Vision Transformer

```yaml
model:
  backbone: "vit_base_patch16_224"
  pretrained: true
  drop_rate: 0.1
  num_classes: 4

training:
  batch_size: 16    # ViT 需要更大显存
  epochs: 100
  learning_rate: 1e-5    # ViT 需要更小学习率
  warmup_epochs: 10
```

### 3.4 训练过程监控

```bash
# 启动 TensorBoard（新终端）
tensorboard --logdir=results/logs

# 浏览器访问 http://localhost:6006
```

## 第四步：评估模型

```bash
# 找到最佳模型路径
ls checkpoints/

# 评估测试集
python evaluate.py \
    --checkpoint checkpoints/weather-classification_efficientnet_b0_20260101_120000/best_model.pth \
    --split test

# 评估验证集
python evaluate.py \
    --checkpoint checkpoints/.../best_model.pth \
    --split val
```

## 第五步：推理预测

### 5.1 命令行预测

```bash
python predict.py \
    --checkpoint checkpoints/.../best_model.pth \
    --image path/to/your/image.jpg
```

### 5.2 Python API 调用

```python
from predict import WeatherPredictor

# 加载模型
predictor = WeatherPredictor("checkpoints/.../best_model.pth")

# 单张预测
result = predictor.predict("image.jpg")
print(f"预测结果: {result['predicted_class']} (置信度: {result['confidence']:.2%})")

# 批量预测
import glob
images = glob.glob("test_images/*.jpg")
results = predictor.predict_batch(images)

for r in results:
    print(f"{r['image']}: {r['predicted_class']}")
```

## 第六步：模型优化

### 6.1 导出 ONNX（跨平台部署）

```python
import torch
from models.model import build_model
import yaml

# 加载配置和模型
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

model = build_model(config)
checkpoint = torch.load("checkpoints/.../best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "weather_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("ONNX 模型导出完成！")
```

### 6.2 模型量化（减小体积）

```python
import torch

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), "weather_classifier_quantized.pth")
```

## 常见问题解决

### 问题1：显存不足 (CUDA out of memory)

```yaml
# 解决方案：减小 batch_size
training:
  batch_size: 8   # 甚至 4

# 或使用更小的模型
model:
  backbone: "efficientnet_b0"  # 最小 EfficientNet
```

### 问题2：训练不收敛

```yaml
# 检查以下几点：
# 1. 学习率是否过大
training:
  learning_rate: 1e-5   # 调小学习率

# 2. 数据预处理是否正确
dataset:
  img_size: 224   # 确认与模型输入一致

# 3. 预训练权重是否加载
model:
  pretrained: true   # 必须开启
```

### 问题3：类别不平衡

```python
# 在 train.py 中修改损失函数
from sklearn.utils.class_weight import compute_class_weight

# 计算类别权重
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 使用加权损失
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
```

### 问题4：过拟合

```yaml
# 增加正则化
model:
  drop_rate: 0.5   # 增大 Dropout

training:
  weight_decay: 1e-3   # 增大权重衰减
  label_smoothing: 0.2   # 标签平滑

augmentation:
  use_auto_augment: true   # 启用 AutoAugment
  random_erase: 0.5   # 增大 Random Erasing
```

## 高级技巧

### 技巧1：交叉验证

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    # 为每个 fold 训练一个模型
    # 最后集成预测结果
```

### 技巧2：模型集成

```python
# 加载多个模型
model1 = load_model("checkpoint1.pth")
model2 = load_model("checkpoint2.pth")
model3 = load_model("checkpoint3.pth")

# 集成预测
with torch.no_grad():
    out1 = torch.softmax(model1(images), dim=1)
    out2 = torch.softmax(model2(images), dim=1)
    out3 = torch.softmax(model3(images), dim=1)

    # 平均集成
    ensemble = (out1 + out2 + out3) / 3
    pred = torch.argmax(ensemble, dim=1)
```

### 技巧3：测试时增强 (TTA)

```python
def tta_predict(model, image, predictor):
    """测试时增强预测"""
    predictions = []

    # 原图
    predictions.append(predictor.predict(image))

    # 水平翻转
    from PIL import ImageOps
    flipped = ImageOps.mirror(image)
    predictions.append(predictor.predict(flipped))

    # 取平均
    # ... 实现平均逻辑
```

## 完整训练流程示例

```bash
# 1. 环境准备
pip install -r requirements.txt

# 2. 数据准备
python prepare_data.py --create-sample
# (放入你的图像数据)
python prepare_data.py --split
python prepare_data.py --verify

# 3. 训练
python train.py --config configs/config.yaml

# 4. 评估
python evaluate.py --checkpoint checkpoints/.../best_model.pth --split test

# 5. 推理
python predict.py --checkpoint checkpoints/.../best_model.pth --image test.jpg
```

## 参考资源

- [timm GitHub](https://github.com/rwightman/pytorch-image-models)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Kaggle Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)
