# 训练产物说明

本文档介绍执行 `python train.py` 完成**一次完整训练**后，项目会生成哪些文件、各自用途，以及如何查看和使用。

> 说明：这里的「一轮训练」指从启动 `train.py` 到训练结束（跑满所有 epoch 或触发早停）的完整过程，而非单个 epoch。

## 实验目录命名

每次训练会根据配置和时间戳创建独立的实验目录，命名规则为：

```
{project.name}_{model.backbone}_{YYYYMMDD_HHMMSS}
```

默认配置下的示例：

```
weather-classification_efficientnet_b0_20260613_143052
```

该名称会同时用于 `checkpoints/` 和 `results/logs/` 下的子目录，方便一一对应。

## 产物总览

一次训练结束后，主要产出分布在两个根目录：

```
weather-classification/
├── checkpoints/
│   └── weather-classification_efficientnet_b0_20260613_143052/
│       ├── config.yaml                  # 本次训练使用的配置快照
│       ├── best_model.pth               # 验证集上表现最好的模型（最重要）
│       ├── checkpoint_epoch_5.pth       # 定期保存的中间检查点（可选）
│       ├── checkpoint_epoch_10.pth
│       └── confusion_matrix.png         # 测试集混淆矩阵图
│
└── results/logs/
    └── weather-classification_efficientnet_b0_20260613_143052/
        └── events.out.tfevents.*          # TensorBoard 事件文件
```

---

## 文件详解

### 1. `config.yaml`（配置快照）

**路径：** `checkpoints/<实验名>/config.yaml`

**生成时机：** 训练开始时立即写入。

**内容：** 本次训练实际使用的完整配置，包括数据集路径、类别名称、模型主干、训练超参数、设备设置等。训练过程中若根据实际数据更新了 `num_classes` 和 `class_names`，快照中也会反映最终值。

**用途：**
- 复现本次实验参数
- 对比不同实验的配置差异
- 检查点中已内嵌 `config`，推理时通常不需要单独读取此文件

---

### 2. `best_model.pth`（最佳模型）

**路径：** `checkpoints/<实验名>/best_model.pth`

**生成时机：** 每当验证集准确率超过历史最佳时覆盖保存。

**文件结构：**

```python
{
    "epoch": 23,                          # 取得最佳验证准确率的 epoch
    "model_state_dict": ...,              # 模型权重
    "optimizer_state_dict": ...,          # 优化器状态（用于断点续训）
    "best_val_acc": 0.9523,               # 最佳验证准确率
    "config": { ... },                    # 完整训练配置
    "class_names": ["cloudy", "rainy", "snowy", "sunny"]
}
```

**用途：**
- **推理预测**（最常用）
- **独立评估**
- 断点续训（需自行编写加载逻辑）

**使用示例：**

```bash
# 推理
python predict.py --checkpoint checkpoints/.../best_model.pth --image test.jpg

# 评估
python evaluate.py --checkpoint checkpoints/.../best_model.pth --split test
```

---

### 3. `checkpoint_epoch_N.pth`（周期性检查点）

**路径：** `checkpoints/<实验名>/checkpoint_epoch_5.pth`、`checkpoint_epoch_10.pth` ...

**生成时机：** 每 `logging.save_interval` 个 epoch 保存一次（默认每 5 个 epoch）。

**文件结构：**

```python
{
    "epoch": 5,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "config": { ... }
}
```

与 `best_model.pth` 相比，**不包含** `best_val_acc` 和 `class_names` 字段。

**用途：**
- 查看训练中间阶段的模型状态
- 断点续训
- 若最终模型过拟合，可回退到较早的 epoch

> 若训练在 epoch 3 就因早停结束，则不会产生任何 `checkpoint_epoch_*.pth`（因为尚未到达第一个保存间隔）。

---

### 4. `confusion_matrix.png`（混淆矩阵图）

**路径：** `checkpoints/<实验名>/confusion_matrix.png`

**生成时机：** 训练全部结束后，加载 `best_model.pth` 在**测试集**上评估时生成。

**内容：** 各类别预测结果的混淆矩阵热力图，横轴为预测类别，纵轴为真实类别。

**用途：**
- 直观查看哪些类别容易混淆（如 `cloudy` 与 `sunny`）
- 写入实验报告或论文

---

### 5. TensorBoard 日志

**路径：** `results/logs/<实验名>/events.out.tfevents.*`

**生成时机：** 训练过程中持续写入，训练结束时关闭。

**记录的指标：**

| 曲线名称 | 记录频率 | 含义 |
|----------|----------|------|
| `Train/Loss` | 每 `log_interval` 个 batch | 训练损失 |
| `Train/Accuracy` | 每 `log_interval` 个 batch | 训练准确率（累计） |
| `Train/LR` | 每 `log_interval` 个 batch | 当前学习率 |
| `Val/Loss` | 每个 epoch 结束 | 验证集损失 |
| `Val/Accuracy` | 每个 epoch 结束 | 验证集准确率 |
| `Val/F1_Macro` | 每个 epoch 结束 | 验证集宏平均 F1 |

**查看方式：**

```bash
tensorboard --logdir=results/logs

# 或只看本次实验
tensorboard --logdir=results/logs/weather-classification_efficientnet_b0_20260613_143052
```

浏览器访问 http://localhost:6006 查看训练曲线。

---

## 终端输出（不写入文件）

以下内容仅在终端打印，不会自动保存为文件：

### 每个 epoch 结束时

```
Epoch 12/50
  Train - Loss: 0.2341, Acc: 0.9123, F1: 0.9087
  Val   - Loss: 0.3012, Acc: 0.8945, F1: 0.8901
  [OK] 保存最佳模型 (Val Acc: 0.8945)
```

或验证集未提升时：

```
  - 未提升 (3/10)
```

### 训练结束后（测试集评估）

```
测试结果:
  Accuracy:  0.9012
  Precision: 0.8987
  Recall:    0.8956
  F1-Score:  0.8971

============================================================
Classification Report
============================================================
              precision    recall  f1-score   support

      cloudy     0.9200    0.9100    0.9150        40
       rainy     0.8800    0.9000    0.8900        35
       snowy     0.9100    0.8900    0.9000        38
       sunny     0.8850    0.8820    0.8835        32

    accuracy                         0.9012       145
   macro avg     0.8987    0.8956    0.8971       145
weighted avg     0.9015    0.9012    0.9013       145
```

### 早停触发时

```
早停触发！最佳验证准确率: 0.8945 (Epoch 12)
```

---

## 训练过程中的保存逻辑

```
训练开始
  │
  ├─ 创建实验目录（checkpoints/ + results/logs/）
  ├─ 保存 config.yaml
  └─ 启动 TensorBoard 写入
  │
  ▼
每个 Epoch
  │
  ├─ 训练 → 写入 Train/* 曲线
  ├─ 验证 → 写入 Val/* 曲线
  │
  ├─ 验证准确率创新高？
  │     └─ 是 → 覆盖保存 best_model.pth
  │
  └─ epoch % save_interval == 0？
        └─ 是 → 保存 checkpoint_epoch_N.pth
  │
  ▼
训练结束（跑满 epoch 或早停）
  │
  ├─ 加载 best_model.pth
  ├─ 在测试集上评估
  ├─ 打印 Classification Report（仅终端）
  ├─ 保存 confusion_matrix.png
  └─ 关闭 TensorBoard
```

---

## 相关配置项

在 `configs/config.yaml` 中控制产物行为的参数：

```yaml
training:
  save_best_only: true           # 仅保存最佳模型（best_model.pth）
  early_stopping_patience: 10    # 验证集连续 N 个 epoch 不提升则早停

logging:
  log_dir: "./results/logs"      # TensorBoard 日志目录
  checkpoint_dir: "./checkpoints" # 模型检查点目录
  log_interval: 10               # 每 N 个 batch 记录一次训练曲线
  save_interval: 5               # 每 N 个 epoch 保存一次中间检查点
```

---

## 产物使用速查

| 需求 | 使用产物 | 命令 / 操作 |
|------|----------|-------------|
| 对新图片分类 | `best_model.pth` | `python predict.py --checkpoint ... --image ...` |
| 在测试集上复评 | `best_model.pth` | `python evaluate.py --checkpoint ... --split test` |
| 查看训练曲线 | TensorBoard 日志 | `tensorboard --logdir=results/logs` |
| 分析类别混淆 | `confusion_matrix.png` | 直接打开图片 |
| 复现实验参数 | `config.yaml` | 对比或复制为新的训练配置 |
| 回退到中间 epoch | `checkpoint_epoch_N.pth` | 手动加载权重进行评估或续训 |

---

## 注意事项

1. **`best_model.pth` 是推理和部署的首选**，它保存的是验证集上表现最好的权重，而非最后一个 epoch 的权重。
2. **分类报告不会自动保存为文件**，如需留存请手动复制终端输出，或运行 `evaluate.py` 重新生成评估结果。
3. **每次训练都会新建实验目录**，不会覆盖历史实验，注意定期清理 `checkpoints/` 和 `results/logs/` 以节省磁盘空间。
4. 若训练过程中验证集准确率始终未提升，`best_model.pth` 可能不会被创建；此时训练结束后的测试评估会报错。请检查数据、学习率和预训练权重是否正确加载。
