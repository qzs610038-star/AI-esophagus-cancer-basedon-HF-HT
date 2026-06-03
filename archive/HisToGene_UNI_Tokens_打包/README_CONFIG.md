# HisToGene-UNI Token 序列模型 — 服务器部署配置指南

## 1. 模型概述

本模型基于 H&E 病理切片预测空间转录组 30 条基因通路的 ssGSEA 活性评分。

**架构**: HisToGene-UNI Token 序列（方案B / 方案B-AugMix）
- 使用 UNI2-h ViT-L/14 作为 backbone，提取 65 个 token 特征（1536 维/ token）
- 轻量 Transformer 编码器（1层, 8头, 512 hidden_dim）聚合 token 序列
- 坐标位置编码 + MLP 回归头
- 参数量: ~3.9M（不含 backbone，backbone 仅用于特征提取，不参与训练）

**性能基线**（3折跨患者交叉验证，食管癌 3 患者数据集）:

| Fold | 训练集 | 测试集 | Test PCC |
|------|--------|--------|----------|
| 1    | JFX+LMZ | HYZ | 0.3946 (base) / 0.4142 (AugMix) |
| 2    | HYZ+LMZ | JFX | 0.3424 |
| 3    | HYZ+JFX | LMZ | 0.3731 |
| **3折平均** | | | **0.3812** |

### 1.1 推理部署：优先选用哪个权重？

如果要在**新患者的 H&E 切片上直接预测（不重新训练）**，按优先级排列：

| 优先级 | 权重路径 | 推荐场景 | 理由 |
|--------|----------|----------|------|
| ★★★ | `checkpoints/Fold1_AugMix_Best/best_histogene_uni_tokens_augmix.pth` | **首选** | Val PCC=0.4142（最高单折）；训练时用过 AugMix + MixUp，对染色变异/扫描仪差异的域偏移容忍度更高 |
| ★★☆ | 3折 ensemble（3个 base 权重取均值） | 最稳健 | 三个模型投票，降低单模型偏差，但需推理 3 次 |
| ★☆☆ | `checkpoints/Fold1_JFX_LMZ_to_HYZ/best_histogene_uni_tokens.pth` | 单模型备选 | base 方案，无增强依赖，可作为对照 |

**理论依据**：
- AugMix 在训练时随机采样 H&E 染色增强变体（亮度/对比度/色调扰动），等效于让模型见过多种染色条件的切片。新医院/新扫描仪的切片本质上是一种未见过的染色域，AugMix 训练出的模型天然对此类 domain shift 更鲁棒。
- 3 折中每折训练数据量相同（2 患者），但 Fold1 的测试患者 HYZ 样本量最大，反向说明 Fold1 训练集（JFX+LMZ）学到的表征泛化性最强。

### 1.2 新数据必须重新提取 UNI 特征

**关键事实：模型输入不是原始 H&E 图像，而是 UNI2-h 的 token 特征。**

完整推理链路：

```
新 H&E patch 图像 (.png)
    ↓  UNI2-h ViT-L/14 前向推理（冻结 backbone，只跑一次）
token 特征 [num_tokens, 1536]  →  保存为 .pt 缓存文件
    ↓  HisToGeneUNITokens 模型加载 .pt → 前向推理
30 条通路 ssGSEA 预测值
```

- UNI2-h 是 ViT-L/14 架构的病理 foundation model，将 256×256 patch 编码为 257 个 token（本项目使用 lite 模式截取 65 个），每个 token 1536 维
- HisToGene 模型**不包含图像编码器**，只接受 .pt 文件中的 token 张量
- 因此**任何新数据必须先全部过一遍 UNI2-h 生成 .pt 缓存**，目录结构需符合下方第 4 节的规范

## 2. 文件清单

```
HisToGene_UNI_Tokens/
├── train.py                          # 主训练脚本（支持单患者 / 跨患者 / 3折CV / AugMix）
├── model_uni_tokens.py               # 模型架构定义（HisToGeneUNITokens + LightweightTokenEncoder）
├── dataset_uni_tokens.py             # 基础数据集（UNI token 加载 + 坐标解析）
├── dataset_uni_tokens_augmix.py      # AugMix 增强数据集（H&E 增强 token 随机采样 + MixUp）
├── config_utils.py                   # 路径配置工具（读取 config.yaml，解析患者路径）
├── config.yaml                       # ★ 唯一配置入口 — 服务器部署只需修改此文件
├── notify_utils.py                   # 训练通知工具（桌面 Toast + 状态文件）
├── visualize_results.py              # 结果可视化（训练曲线 + PCC柱状图 + 全报告）
├── requirements.txt                  # Python 依赖
├── README_CONFIG.md                  # 本文件
├── histogene/
│   ├── __init__.py
│   └── utils.py                      # 指标计算（PCC / MAE / R² / MSE）
└── checkpoints/
    ├── Fold1_JFX_LMZ_to_HYZ/         # 3折 CV Fold1: JFX+LMZ→HYZ, Test PCC=0.3946
    │   └── best_histogene_uni_tokens.pth
    ├── Fold2_HYZ_LMZ_to_JFX/         # 3折 CV Fold2: HYZ+LMZ→JFX, Test PCC=0.3424
    │   └── best_histogene_uni_tokens.pth
    ├── Fold3_HYZ_JFX_to_LMZ/         # 3折 CV Fold3: HYZ+JFX→LMZ, Test PCC=0.3731
    │   └── best_histogene_uni_tokens.pth
    └── Fold1_AugMix_Best/            # ★ 最佳单折: AugMix Fold1, Val PCC=0.4142
        └── best_histogene_uni_tokens_augmix.pth
```

## 3. 环境准备

### 3.1 Python 环境

```bash
pip install -r requirements.txt
```

核心依赖: `torch>=2.0.0`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`

### 3.2 UNI2-h 特征提取（必须，一次性预处理）

**无论是训练还是纯推理，新数据都必须先提取 UNI token 特征。** 模型不直接读取 PNG 图像。

UNI2-h 模型权重下载：HuggingFace（需联网，国内用 `hf-mirror.com` 镜像）。

特征提取伪代码：

```python
import torch
from uni2h import UNI2H  # 或你使用的 UNI2-h 加载方式

model = UNI2H("ViT-L/14").cuda().eval()

for patch_path in patches:
    image = load_and_preprocess(patch_path)          # 224×224 或模型要求尺寸
    with torch.no_grad():
        _, tokens = model.forward_features(image)    # tokens: [1, num_tokens, 1536]
    # 如使用 lite 模式（本项目默认 65 token），截断 CLS 后的前 65 个 token
    tokens = tokens[:, 1:66, :].squeeze(0).cpu()     # [65, 1536]
    torch.save(tokens, f"{output_dir}/{stem}.pt")
```

**关键要求**：
- 每个 .pt 文件形状必须为 `[num_tokens, 1536]`（本项目用 65 tokens）
- 文件名必须与对应 PNG patch 的 stem 一致（如 `patch_x4641_y16969.png` → `patch_x4641_y16969.pt`）
- 训练/验证/测试 split 的 .pt 文件放入不同子目录

## 4. 数据目录结构要求

```
<项目根目录>/
├── data/                              # H&E 原始 patch 图像
│   └── patch_noov_spilt/              # （注意：原始拼写，不要修正 typo）
│       ├── PatientA_noov_split/
│       │   ├── train_patches/         # *.png
│       │   └── val_patches/           # *.png
│       ├── PatientB_noov_split/
│       │   ├── train_patches/
│       │   └── val_patches/
│       └── PatientC_noov_split/
│           ├── train_patches/
│           └── val_patches/
├── labels/                            # ssGSEA 标签 CSV
│   ├── PatientA_ssGSEA_zscore.csv
│   ├── PatientB_ssGSEA_zscore.csv
│   └── PatientC_ssGSEA_zscore.csv
├── caches/                            # UNI token 特征缓存
│   └── uni_tokens/
│       ├── PatientA/
│       │   ├── train/                 # *.pt 文件（每个 patch 一个 .pt）
│       │   └── val/
│       ├── PatientB/
│       │   ├── train/
│       │   └── val/
│       └── PatientC/
│           ├── train/
│           └── val/
└── checkpoints/                       # 模型保存目录（自动创建）
```

### 标签 CSV 格式

```csv
,patch_id,tls,tgfb,emt,hypoxia,mhc,icp,ifng,toxic,Glycolysis,...,ECM_Organization
0,patch_x4641_y16969,-0.6259,-0.5149,-0.9186,0.3391,...,-0.7526
1,patch_x4735_y15805,-0.4968,0.8813,-0.8399,-0.9787,...,-1.2477
...
```

- 第一列为索引（自动忽略）
- 第二列为 patch 文件名（不含 .png 后缀）
- 后续 30 列为各通路 Z-score 标准化后的 ssGSEA 评分

## 5. 配置修改 — config.yaml

**这是唯一需要修改的文件。** 以下为关键配置项：

### 5.1 数据路径（paths 节）

```yaml
paths:
  # 取消注释并填写服务器上的实际路径
  patch_base: "/data/server/HE_patches"        # H&E patch 根目录
  ssgsea_base: "/data/server/ssGSEA_scores"    # 标签 CSV 根目录

  caches:
    uni_tokens: "/data/caches/uni_tokens"       # Token 特征缓存根目录
    uni_tokens_aug: "/data/caches/uni_tokens_aug"  # AugMix 增强缓存（可选）

  outputs:
    checkpoints_root: "./checkpoints"           # 模型保存路径
    results_vis: "./results"                    # 可视化输出路径
```

### 5.2 患者配置（patients 节）

为每位患者配置子路径和标签文件名：

```yaml
patients:
  PatientA:                          # 患者名称
    patches_subdir: "PatientA_noov_split"     # patch 子目录名
    labels_csv: "PatientA_ssGSEA_zscore.csv"  # 标签文件名
    # 若某患者数据在特殊位置，可用绝对路径覆盖：
    # patches_dir: "/special/path/to/PatientA"
    # labels_path: "/special/path/to/PatientA_scores.csv"

  PatientB:
    patches_subdir: "PatientB_noov_split"
    labels_csv: "PatientB_ssGSEA_zscore.csv"

  PatientC:
    patches_subdir: "PatientC_noov_split"
    labels_csv: "PatientC_ssGSEA_zscore.csv"
```

### 5.3 交叉验证配置（cross_validation 节）

根据你的患者名称修改：

```yaml
cross_validation:
  folds:
    1: { train: ["PatientB", "PatientC"], test: "PatientA" }
    2: { train: ["PatientA", "PatientC"], test: "PatientB" }
    3: { train: ["PatientA", "PatientB"], test: "PatientC" }
```

### 5.4 训练参数（training 节）

```yaml
training:
  device: "auto"    # "auto" 自动检测 GPU / "cuda" / "cpu"
```

## 6. 训练命令

所有命令在项目根目录（含 train.py 的目录）下执行。

### 6.1 单患者训练

```bash
python train.py --patient PatientA
```

### 6.2 跨患者泛化训练（指定 fold）

```bash
# Fold 1: PatientB+PatientC 训练 → PatientA 测试
python train.py --cross_patient --fold 1

# Fold 2: PatientA+PatientC 训练 → PatientB 测试
python train.py --cross_patient --fold 2

# Fold 3: PatientA+PatientB 训练 → PatientC 测试
python train.py --cross_patient --fold 3
```

### 6.3 完整 3 折交叉验证

```bash
# 依次运行三个 fold
python train.py --cross_patient --fold 1
python train.py --cross_patient --fold 2
python train.py --cross_patient --fold 3
```

### 6.4 启用 AugMix 增强（推荐，性能更优）

```bash
python train.py --cross_patient --fold 1 --use_augmented_tokens --n_augments 3 --aug_sample_prob 0.5 --mixup_alpha 1.0 --mixup_prob 0.5
```

AugMix 关键参数:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_augmented_tokens` | (flag) | 启用 H&E 增强 token |
| `--n_augments` | 3 | 增强变体数量 |
| `--aug_sample_prob` | 0.5 | 训练时选增强版本的概率 |
| `--mixup_alpha` | 1.0 | MixUp Beta 分布参数 (0=禁用) |
| `--mixup_prob` | 0.5 | 每个 batch 应用 MixUp 的概率 |

### 6.5 完整训练超参数

```bash
python train.py --cross_patient --fold 1 \
    --lr 1e-4 \
    --batch_size 64 \
    --num_epochs 150 \
    --early_stop_patience 20 \
    --weight_decay 1e-4 \
    --label_noise 0.0 \
    --gradient_clip 1.0 \
    --feature_dim 1536 \
    --model_dim 1024 \
    --n_pos 128 \
    --n_targets 30 \
    --mlp_dim 2048 \
    --dropout 0.3 \
    --encoder_hidden_dim 512 \
    --n_encoder_layers 1 \
    --n_encoder_heads 8
```

### 6.6 断点续训

```bash
python train.py --cross_patient --fold 1 --resume checkpoints/xxx/resume_xxx.pth
```

### 6.7 纯推理：用预训练权重在新数据上预测（不训练）

如果你的目标是不重新训练、直接用预训练权重在新患者切片上预测 ssGSEA：

**步骤**：
1. 先按 3.2 节提取新数据的 UNI token 特征（.pt 文件）
2. 准备标签 CSV（如果只做推理无 ground truth，可创建占位 CSV，列名和列数保持一致即可）
3. 配置 `config.yaml` 中的患者路径指向新数据
4. 运行推理：

```python
# inference.py — 纯推理脚本，放在项目根目录运行
import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_uni_tokens import HisToGeneUNITokens
from dataset_uni_tokens import HisToGeneUNITokensDataset
from config_utils import get_device, load_config

CKPT = "checkpoints/Fold1_AugMix_Best/best_histogene_uni_tokens_augmix.pth"
DEVICE = get_device(load_config())

# 加载模型
ckpt = torch.load(CKPT, weights_only=False, map_location=DEVICE)
args = ckpt['args']
model = HisToGeneUNITokens(
    feature_dim=args.get('feature_dim', 1536),
    dim=args.get('model_dim', 1024),
    n_pos=args.get('n_pos', 128),
    n_targets=args.get('n_targets', 30),
    mlp_dim=args.get('mlp_dim', 2048),
    dropout=args.get('dropout', 0.3),
    encoder_hidden_dim=args.get('encoder_hidden_dim', 512),
    n_encoder_layers=args.get('n_encoder_layers', 1),
    n_encoder_heads=args.get('n_encoder_heads', 8),
).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
target_cols = ckpt['target_cols']

# 加载新数据集
dataset = HisToGeneUNITokensDataset(
    patches_dir="data/new_patient/patches/",
    feature_cache_dir="caches/uni_tokens/new_patient/",
    labels_csv="labels/new_patient.csv",
    target_cols=target_cols,
    n_pos=args.get('n_pos', 128),
    n_targets=args.get('n_targets', 30),
)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 推理
all_preds = []
with torch.no_grad():
    for tokens, pos_x, pos_y, _ in loader:
        tokens = tokens.to(DEVICE)
        pos_x = pos_x.to(DEVICE)
        pos_y = pos_y.to(DEVICE)
        preds = model(tokens, pos_x, pos_y)
        all_preds.append(preds.cpu().numpy())

preds = np.concatenate(all_preds, axis=0)
df = pd.DataFrame(preds, columns=[f"pred_{c}" for c in target_cols])
df.to_csv("predictions_new_patient.csv", index=False)
print(f"推理完成，保存 {len(df)} 条预测结果")
```

### 6.8 加载预训练权重作为初始化（迁移学习）

在新数据上微调时，用本包中的预训练权重初始化：

```python
pretrained = torch.load("checkpoints/Fold1_AugMix_Best/best_histogene_uni_tokens_augmix.pth",
                         weights_only=False, map_location=device)
model.load_state_dict(pretrained['model_state_dict'], strict=False)
```

## 7. 输出说明

训练完成后，`results/` 目录下生成（按时间戳命名子目录）：

| 文件 | 说明 |
|------|------|
| `predictions.csv` | 所有测试样本的 true_xxx / pred_xxx 预测值 |
| `per_pathway_pcc.csv` | 30 条通路各自的 PCC / R² / MAE |
| `training_curves.png` | Loss + PCC + R² + LR 训练曲线 |
| `pcc_barplot.png` | 逐通路 PCC 柱状图 |
| `full_report.png` | 综合报告（曲线 + 柱状图 + 指标表 + 参数表） |
| `metrics_table.png` | 指标汇总表（PCC / MAE / R²） |
| `model_params.txt` | 模型参数与训练结果文本摘要 |
| `training_history_*.csv` | 逐 epoch 详细训练历史 |

## 8. 训练监控

- **暂停训练**: 在项目根目录创建空文件 `PAUSE_TRAINING`
  - Linux: `touch PAUSE_TRAINING`
- **继续训练**: 删除 `PAUSE_TRAINING` 后使用 `--resume` 恢复
- **通知**: 训练完成/中断时自动写入 `training_status_*.txt` 状态文件

## 9. 常见问题

**Q: 特征维度不匹配 (1536 vs 其他)**
A: UNI2-h ViT-L/14 的 token 维度固定为 1536。若使用其他 backbone，需同时修改
   `--feature_dim` 参数和 dataset 中的维度断言（`dataset_uni_tokens_augmix.py` 第 181 行）。

**Q: 通路数量不是 30？**
A: 修改 `--n_targets` 参数，同时更新 `train.py` 中的 `_PATHWAY_NAMES` 列表和
   `_PATHWAY_PCC` 字典（用于通路加权损失）。

**Q: 如何适配不同 token 数量？**
A: 模型使用全局平均池化聚合 token 序列，对 token 数量不敏感，无需修改模型代码。

**Q: Windows 服务器中文路径问题**
A: 运行时设置环境变量 `PYTHONIOENCODING=utf-8`。

## 10. 关键经验（来自原始项目）

- **val_loss 选最优 epoch**：非 val_pcc 最大，基于验证集 loss 最小值选择最佳模型
- **三层交集过滤**：训练样本 = patches ∩ CSV标签 ∩ token缓存 的交集
- **跨患者泛化衰减**：跨患者相比单患者 PCC 衰减约 31-77%，属正常现象
- **AugMix 提升约 5%**：相比 base 方案，AugMix Fold1 PCC 从 0.3946 提升至 0.4142
- **早停机制**：默认 patience=20 epochs，基于 val_loss 无改善触发
