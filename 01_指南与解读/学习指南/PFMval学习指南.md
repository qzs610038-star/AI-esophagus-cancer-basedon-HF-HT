# PFMval 项目深度学习指南

## 一、项目整体结构

```
d:\AI空间转录病理研究\PFMval_new/
├── README.md                              # 项目说明文档
├── split.py                               # 数据划分脚本（核心数据预处理）
├── zscore.py                              # Z-score标准化脚本
├── HYZ15040-org.zip                       # 原始数据（Patch图片压缩包）
├── HYZ15040_ssGSEA_scores.csv             # 基因集丰富度评分原始文件
├── HYZ15040_ssGSEA_scores_zscore.csv      # Z-score标准化后的评分文件
└── uni2h/                                 # UNI2-h特征提取与训练模块
    ├── uni2h_utils.py                     # 核心工具函数库
    ├── train.py                           # 训练脚本
    └── infer.py                           # 推理脚本
```

**运行时生成的关键目录：**
- `uni2h_cache/HYZ15040/train` 和 `val` — 特征缓存（.pt 文件）
- `checkpoints/HYZ15040/` — 训练检查点
- `res/HYZ15040/` — 推理结果与评估指标

---

## 二、核心代码分析

### 2.1 数据预处理

**split.py** — 空间无重叠划分

- 从文件名（如 `patch_x4641_y16969.png`）解析空间坐标
- 核心算法：逐个候选 Patch 加入验证集时，检查其与已选验证集中所有 Patch 的**欧氏距离 ≥ 350px**，防止"空间泄漏"
- 输出：`train_patches/`（约9806张）和 `val_patches/`（约772张），比例约 9:1

**zscore.py** — ssGSEA 分数标准化

- 对 CSV 最后 8 列（8 个基因集评分）执行 Z-score 标准化：`z = (x - mean) / std`
- 统一量纲，防止某些基因集因数值范围过大而主导训练
- 若某列标准差为 0（全是同一个值），会跳过

### 2.2 特征提取与缓存

**uni2h_utils.py** — 核心工具库，包含 6 大组件：

| 组件 | 作用 |
|------|------|
| `load_uni2h_backbone()` | 从 HuggingFace 加载冻结的 UNI2-h 模型（24层 ViT，1536维输出） |
| `extract_and_cache_features()` | 批量提取 Patch 特征，存为 .pt 文件（支持断点续传） |
| `CachedFeaturePatchDataset` | PyTorch Dataset，从缓存读特征 + 从 CSV 读标签 |
| `BackboneRegressor` | 回归头 MLP：LayerNorm → Linear(1536→256) → GELU → Dropout(0.2) → Linear(256→8) |
| `train_one_epoch() / evaluate()` | 标准 PyTorch 训练/验证循环 |
| `compute_metrics() / pearson_corrcoef()` | 计算 MSE、MAE、R2、PCC 四项指标 |

### 2.3 训练与推理

**train.py** — 完整训练流程：
1. 加载冻结的 UNI2-h → 提取特征并缓存
2. 构建 `CachedFeaturePatchDataset` + DataLoader
3. 创建 `BackboneRegressor`，损失函数 MSELoss，优化器 AdamW
4. 训练循环：含 ReduceLROnPlateau 学习率调度 + 早停（patience=10）
5. 保存最优模型到 `checkpoints/`

**infer.py** — 推理评估：
1. 从 checkpoint 重建模型并加载参数
2. 对验证集批量推理
3. 对每个基因集单独计算 MSE/MAE/R2/PCC + 宏平均
4. 输出逐 Patch 预测结果 CSV 和指标汇总 CSV

---

## 三、完整 Pipeline 机理

```
原始 Patch 图片 + ssGSEA 分数 CSV
        ↓
  split.py ── 空间距离约束(≥350px)划分 train/val
        ↓
  zscore.py ── 对8个基因集评分做 Z-score 标准化
        ↓
  train.py 前半部分 ── UNI2-h 冻结特征提取 → .pt 缓存(每个1536维)
        ↓
  train.py 后半部分 ── 回归头(MLP)微调，早停+LR调度
        ↓
  infer.py ── 验证集推理 → 逐基因集指标 + 宏平均
```

**本质：这是一个迁移学习系统** — 用大规模病理图像预训练的 UNI2-h 作为特征提取器（冻结），只训练一个轻量回归头，将 1536 维视觉特征映射到 8 个基因集评分。

---

## 四、技术栈

| 组件 | 说明 |
|------|------|
| Python 3.10 + PyTorch 2.1.0 + CUDA 11.8 | 核心运行环境 |
| timm ≥ 0.9.8 | 加载 Vision Transformer 架构 |
| huggingface_hub | 下载 UNI2-h 预训练权重 |
| pandas + scikit-learn + Pillow + numpy 1.26.4 | 数据处理与评估 |

UNI2-h 模型关键参数：224×224 输入，patch_size=14，24 层 Transformer，24 个注意力头，1536 维嵌入，SwiGLU 激活。

---

## 五、初学者学习方案

### 第一阶段：基础概念（0.5-1天）

**目标：** 能解释"为什么需要这些步骤"

1. **Patch 概念** — 从全切片扫描（WSI）裁剪的小图块，文件名含坐标信息
2. **ssGSEA 评分** — 8 个基因集（tls、tgfb、emt、hypoxia、icp、ifng、toxic 等）在单样本中的丰富度量化
3. **Z-score 标准化** — 统一量纲，`z = (x-μ)/σ`，深度学习最佳实践

### 第二阶段：数据处理实践（1-2天）

**目标：** 成功运行 split.py 和 zscore.py，理解空间无重叠划分算法

- 精读 `split.py` 的 `find_valid_indices_to_exclude()` 函数（核心算法）
- 理解为何要"空间无重叠"：防止验证时看到训练数据的空间近邻（数据泄漏）
- 用 pandas 验证 CSV 数据质量，检查文件名匹配

### 第三阶段：特征提取与模型架构（2-3天）

**目标：** 理解 ViT → 1536 维特征 → 回归头的完整流程

- 精读 `uni2h_utils.py` 的 `load_uni2h_backbone()` 和 `extract_and_cache_features()`
- 理解冻结 UNI2-h 的原因：数据量不足（~1 万样本）、计算成本高、防止过拟合
- 理解缓存策略：避免每次重新计算特征

### 第四阶段：训练与优化（3-5天）

**目标：** 能训练出收敛的模型，理解训练曲线

- 精读 `train.py` 训练循环
- 掌握：MSELoss（损失）、AdamW（优化器）、ReduceLROnPlateau（学习率调度）、早停
- 关键超参数调优感觉：batch_size、learning_rate、hidden_dim、dropout、patience

### 第五阶段：推理与验证（1-2天）

**目标：** 生成评估报告，能解释模型性能

- 理解 per-target 指标 vs 宏平均
- 判断标准：R2 > 0.5 通常可接受，PCC > 0.7 很好
- 用 matplotlib 绘制预测 vs 真值散点图

---

## 六、关键概念 FAQ

| 问题 | 要点 |
|------|------|
| 350px 距离阈值怎么定的？ | 经验值，保证空间隔离。增大→验证更保守，减小→可能高估 |
| 为什么 Z-score 不用 Min-Max？ | 无界限、不受极端值影响、数学上更友好 |
| 为什么冻结 UNI2-h？ | 数据量不足、计算成本高、过拟合风险大 |
| 回归头为何只一层？ | 1536 维特征已足够丰富，单层 MLP 避免过拟合 |
| MAE vs R2 vs PCC 怎么看？ | MAE 看绝对精度，R2 看方差解释比，PCC 看线性趋势。综合判断最可靠 |

---

## 七、常见坑与解决

1. **HF Token 认证失败** → 注册 HuggingFace 账号并配置 Access Token
2. **坐标解析失败** → 文件名必须符合 `patch_xXXX_yXXX.png` 格式
3. **特征缓存不存在** → 确认 `uni2h_cache/` 路径一致且已运行特征提取
4. **GPU 显存不足** → 降低 batch_size（128 或 64），确保特征用 CPU 加载
5. **严重过拟合** → 增大 dropout（0.3~0.5）、减小 hidden_dim（128）、降低学习率
6. **标签与图片不匹配** → 检查 CSV 第一列 patch_id 是否与文件名一致
7. **Z-score 后有 NaN** → 检查是否有常数列或缺失值
8. **推理指标全 NaN** → 检查预测值是否有常数列或极端异常值

---

## 八、关键参数速查

| 参数 | 文件 | 默认值 | 调优建议 |
|------|------|--------|---------|
| distance_threshold_px | split.py | 350 | 数据特异性强则增大 |
| val_size_fraction | split.py | 0.1 | 一般保持 10% |
| batch_size | train.py | 256 | 显存不足时降低 |
| learning_rate | train.py | 1e-3 | 1e-4 到 1e-2 范围 |
| hidden_dim | train.py | 256 | 128~512 范围调优 |
| dropout | train.py | 0.2 | 过拟合时增大到 0.3~0.5 |
| early_stop_patience | train.py | 10 | 快速迭代降到 5，细调增到 20 |

---

## 九、典型运行命令

```bash
# 1. 准备环境
conda create -n pfmval python=3.10
conda activate pfmval
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas scikit-learn pillow numpy==1.26.4 huggingface_hub timm>=0.9.8

# 2. 数据预处理
cd d:\AI空间转录病理研究\PFMval_new
python split.py                    # 输出: train_patches/, val_patches/
python zscore.py                   # 输出: HYZ15040_ssGSEA_scores_zscore.csv

# 3. 训练 (需修改路径和token)
cd uni2h
python train.py \
  --train_patches_dir D:\path\to\train_patches \
  --val_patches_dir D:\path\to\val_patches \
  --labels_csv D:\path\to\HYZ15040_ssGSEA_scores_zscore.csv \
  --hf_token YOUR_HF_TOKEN \
  --cache_root .\uni2h_cache\HYZ15040

# 4. 推理
python infer.py \
  --split_patches_dir D:\path\to\val_patches \
  --labels_csv D:\path\to\HYZ15040_ssGSEA_scores_zscore.csv \
  --checkpoint_path .\checkpoints\HYZ15040\best_model_uni2h.pth \
  --hf_token YOUR_HF_TOKEN
```

---

## 十、总结

PFMval 项目是一个**端到端的迁移学习系统**，整体学习周期预计 **7-13 天**，核心建议是**边读代码边动手运行**，在实际调试中加深理解。

初学者应按以下顺序掌握：
1. **理解动机**：为什么要做这些预处理和特征提取
2. **数据处理**：能正确运行 split 和 zscore 脚本
3. **特征理论**：理解 UNI2-h 和回归头的设计
4. **训练优化**：掌握超参调优和过拟合诊断
5. **结果分析**：能解释四个评估指标

### 推荐学习资源

**理论基础：**
- Vision Transformer (ViT) 论文：[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Z-score标准化：[Scikit-learn文档](https://scikit-learn.org/stable/modules/preprocessing.html)
- 早停和过拟合：[deeplearning.ai](https://www.deeplearning.ai/) 的相关课程

**工具与库：**
- PyTorch官方教程：https://pytorch.org/tutorials/
- timm文档：https://huggingface.co/docs/timm/index
- Pandas数据处理：https://pandas.pydata.org/docs/

**项目相关：**
- UNI2-h模型卡：https://huggingface.co/MahmoodLab/UNI2-h

---

## 十一、各文件相互关系详解

本章详细梳理项目中 5 个核心 Python 文件之间的调用关系、数据流转以及函数依赖，帮助你建立完整的代码结构认知。

---

### 11.1 文件调用关系图

以下是 5 个 py 文件之间的 import 依赖关系（箭头表示 "import 自"）：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           文件调用关系图                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                                                          │
│   │  split.py    │  ←── 独立脚本，无 import 其他项目文件                      │
│   │  (数据划分)   │                                                          │
│   └──────────────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   输出: train_patches/ 和 val_patches/ 文件夹                                │
│                                                                             │
│   ┌──────────────┐                                                          │
│   │  zscore.py   │  ←── 独立脚本，无 import 其他项目文件                      │
│   │  (标准化)    │                                                          │
│   └──────────────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   输出: *_ssGSEA_scores_zscore.csv                                          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        uni2h_utils.py                               │   │
│   │                    (核心工具函数库)                                  │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  提供功能：                                                  │   │   │
│   │  │  • load_uni2h_backbone()      - 加载 UNI2-h 模型            │   │   │
│   │  │  • extract_and_cache_features() - 特征提取与缓存            │   │   │
│   │  │  • CachedFeaturePatchDataset  - PyTorch Dataset 类          │   │   │
│   │  │  • BackboneRegressor          - 回归头 MLP 模型             │   │   │
│   │  │  • train_one_epoch()          - 单轮训练函数                │   │   │
│   │  │  • evaluate()                 - 评估函数                    │   │   │
│   │  │  • compute_metrics()          - 指标计算函数                │   │   │
│   │  │  • pearson_corrcoef()         - 皮尔逊相关系数计算          │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│          ▲                                          ▲                       │
│          │                                          │                       │
│   import │                                          │ import                │
│          │                                          │                       │
│   ┌──────────────┐                          ┌──────────────┐               │
│   │  train.py    │                          │  infer.py    │               │
│   │  (训练脚本)   │                          │  (推理脚本)   │               │
│   └──────────────┘                          └──────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**执行顺序说明：**

```
Step 1: split.py ──→ 划分数据集
            │
            ▼
Step 2: zscore.py ──→ 标准化标签
            │
            ▼
Step 3: train.py ──→ 训练模型（依赖 uni2h_utils.py）
            │
            ▼
Step 4: infer.py ──→ 推理评估（依赖 uni2h_utils.py）
```

---

### 11.2 文件间数据流转关系

| 步骤 | 输入文件/数据 | 处理脚本 | 输出文件/数据 | 下游使用者 |
|:----:|:-------------|:--------:|:-------------|:----------|
| 1 | `HYZ15040-org.zip` (原始Patch图) | **split.py** | `train_patches/` 文件夹 | train.py |
| 1 | `HYZ15040-org.zip` (原始Patch图) | **split.py** | `val_patches/` 文件夹 | train.py, infer.py |
| 2 | `HYZ15040_ssGSEA_scores.csv` | **zscore.py** | `HYZ15040_ssGSEA_scores_zscore.csv` | train.py, infer.py |
| 3 | `train_patches/` + zscore后的CSV | **train.py** | `uni2h_cache/HYZ15040/train/*.pt` (特征缓存) | train.py (自身使用) |
| 3 | `val_patches/` + zscore后的CSV | **train.py** | `uni2h_cache/HYZ15040/val/*.pt` (特征缓存) | train.py, infer.py |
| 3 | 训练完成的模型参数 | **train.py** | `checkpoints/HYZ15040/best_model_uni2h.pth` | infer.py |
| 4 | `val_patches/` + checkpoint | **infer.py** | `res/HYZ15040/val_predictions_uni2h.csv` | 人工分析 |
| 4 | 推理结果 | **infer.py** | `res/HYZ15040/val_predictions_uni2h_metrics.csv` | 人工分析 |

**数据流转流程图：**

```
原始数据
    │
    ├───→ HYZ15040-org.zip ───────┐
    │                              │
    │                              ▼
    │                        split.py ───→ train_patches/ ─────┐
    │                              │                           │
    │                              └──→ val_patches/ ──────────┤
    │                                                          │
    └───→ HYZ15040_ssGSEA_scores.csv ───→ zscore.py ───→ HYZ15040_ssGSEA_scores_zscore.csv
                                                                  │
                                                                  ▼
    ┌─────────────────────────────────────────────────────────────┘
    │
    ▼
train.py ─────────────────────────────────────────────────────────────┐
    │                                                                 │
    ├──→ 提取特征 → uni2h_cache/HYZ15040/train/*.pt                  │
    │                                                                 │
    ├──→ 提取特征 → uni2h_cache/HYZ15040/val/*.pt                    │
    │                                                                 │
    └──→ 训练模型 → checkpoints/HYZ15040/best_model_uni2h.pth ──────┘
                                                                      │
                                                                      ▼
                                                                infer.py
                                                                      │
                                                                      ├──→ val_predictions_uni2h.csv
                                                                      │
                                                                      └──→ val_predictions_uni2h_metrics.csv
```

---

### 11.3 函数级调用关系

#### 11.3.1 train.py 调用的 uni2h_utils.py 函数/类

| train.py 中的使用位置 | 调用的 uni2h_utils.py 内容 | 用途说明 |
|:---------------------|:--------------------------|:--------|
| 第 12-21 行 import | `CachedFeaturePatchDataset` | 构建 PyTorch Dataset |
| 第 12-21 行 import | `DEFAULT_NUM_TARGETS` | 默认目标数量常量 |
| 第 12-21 行 import | `DEFAULT_TARGET_START_COL` | 默认目标起始列常量 |
| 第 12-21 行 import | `BackboneRegressor` | 回归头模型类 |
| 第 12-21 行 import | `evaluate` | 验证/评估函数 |
| 第 12-21 行 import | `extract_and_cache_features` | 特征提取与缓存 |
| 第 12-21 行 import | `load_uni2h_backbone` | 加载 UNI2-h 模型 |
| 第 12-21 行 import | `train_one_epoch` | 单轮训练函数 |
| 第 60 行 | `load_uni2h_backbone()` | 初始化 UNI2-h backbone |
| 第 69-84 行 | `extract_and_cache_features()` | 提取并缓存 train/val 特征 |
| 第 87-100 行 | `CachedFeaturePatchDataset()` | 创建 train/val Dataset |
| 第 121-126 行 | `BackboneRegressor()` | 创建回归头模型 |
| 第 139 行 | `train_one_epoch()` | 执行训练 |
| 第 140 行 | `evaluate()` | 执行验证 |

#### 11.3.2 infer.py 调用的 uni2h_utils.py 函数/类

| infer.py 中的使用位置 | 调用的 uni2h_utils.py 内容 | 用途说明 |
|:---------------------|:--------------------------|:--------|
| 第 10-19 行 import | `CachedFeaturePatchDataset` | 构建 PyTorch Dataset |
| 第 10-19 行 import | `DEFAULT_NUM_TARGETS` | 默认目标数量常量 |
| 第 10-19 行 import | `DEFAULT_TARGET_START_COL` | 默认目标起始列常量 |
| 第 10-19 行 import | `BackboneRegressor` | 回归头模型类 |
| 第 10-19 行 import | `evaluate` | 评估函数 |
| 第 10-19 行 import | `extract_and_cache_features` | 特征提取与缓存 |
| 第 10-19 行 import | `load_uni2h_backbone` | 加载 UNI2-h 模型 |
| 第 10-19 行 import | `pearson_corrcoef` | 皮尔逊相关系数计算 |
| 第 59 行 | `load_uni2h_backbone()` | 加载 UNI2-h backbone |
| 第 67-74 行 | `extract_and_cache_features()` | 提取并缓存推理特征 |
| 第 77-83 行 | `CachedFeaturePatchDataset()` | 创建推理 Dataset |
| 第 92-97 行 | `BackboneRegressor()` | 重建回归头模型 |
| 第 137 行 | `pearson_corrcoef()` | 计算每个 target 的 PCC |

#### 11.3.3 split.py 和 zscore.py 的独立性

| 文件 | 是否被其他文件 import | 说明 |
|:----|:--------------------|:-----|
| **split.py** | ❌ 否 | 完全独立的预处理脚本，手动执行一次即可，输出文件夹供后续使用 |
| **zscore.py** | ❌ 否 | 完全独立的预处理脚本，手动执行一次即可，输出标准化后的 CSV |

**为什么 split.py 和 zscore.py 是独立的？**

1. **执行时机不同**：这两个脚本只需在数据准备阶段执行一次，生成中间文件后不再修改
2. **无共享代码**：它们的功能非常专一（空间划分、Z-score），不需要复用其他模块的函数
3. **降低耦合**：保持独立使得数据预处理与模型训练解耦，便于调试和复用

---

### 11.4 各解读文档导航

本项目为每个核心代码文件都编写了详细的解读指南，建议按以下顺序阅读：

#### 11.4.1 解读文档清单

| 解读文档 | 对应源码文件 | 主要内容 |
|:--------|:------------|:--------|
| `split_解读指南.md` | `split.py` | 空间无重叠划分算法详解、坐标解析逻辑、距离阈值设置 |
| `zscore_解读指南.md` | `zscore.py` | Z-score 标准化原理、统计量计算、异常值处理 |
| `uni2h_utils_解读指南.md` | `uni2h/uni2h_utils.py` | 特征提取、Dataset 类、回归头模型、评估指标计算 |
| `train_解读指南.md` | `uni2h/train.py` | 训练流程、超参数配置、早停机制、模型保存 |
| `infer_解读指南.md` | `uni2h/infer.py` | 推理流程、指标计算、结果输出 |

#### 11.4.2 推荐阅读顺序

**初学者路线（循序渐进）：**

```
第一阶段：数据准备
    │
    ├──→ split_解读指南.md ──→ 理解数据划分策略
    │
    └──→ zscore_解读指南.md ──→ 理解标签预处理

第二阶段：核心工具
    │
    └──→ uni2h_utils_解读指南.md ──→ 理解所有基础组件

第三阶段：训练与推理
    │
    ├──→ train_解读指南.md ──→ 理解完整训练流程
    │
    └──→ infer_解读指南.md ──→ 理解模型评估方法
```

**快速上手路线（已有基础）：**

```
uni2h_utils_解读指南.md （快速浏览）
    │
    ├──→ train_解读指南.md （重点关注训练逻辑）
    │
    └──→ infer_解读指南.md （重点关注评估逻辑）
```

#### 11.4.3 文档与代码对照速查

| 如果你想了解... | 阅读文档 | 查看源码 |
|:---------------|:--------|:--------|
| 如何避免数据泄漏的空间划分 | `split_解读指南.md` | `split.py` 第 22-96 行 |
| Z-score 标准化的具体实现 | `zscore_解读指南.md` | `zscore.py` 第 101-126 行 |
| UNI2-h 模型如何加载 | `uni2h_utils_解读指南.md` | `uni2h_utils.py` 第 32-70 行 |
| 特征缓存机制 | `uni2h_utils_解读指南.md` | `uni2h_utils.py` 第 138-169 行 |
| Dataset 如何匹配图片和标签 | `uni2h_utils_解读指南.md` | `uni2h_utils.py` 第 173-225 行 |
| 回归头模型结构 | `uni2h_utils_解读指南.md` | `uni2h_utils.py` 第 228-247 行 |
| 训练循环完整逻辑 | `train_解读指南.md` | `train.py` 第 137-190 行 |
| 早停机制实现 | `train_解读指南.md` | `train.py` 第 168-190 行 |
| 推理时如何计算每个 target 的指标 | `infer_解读指南.md` | `infer.py` 第 118-150 行 |

---

### 11.5 总结

通过本章的学习，你应该已经建立了对 PFMval 项目代码结构的完整认知：

1. **split.py 和 zscore.py** 是独立的数据预处理脚本，为后续训练准备数据
2. **uni2h_utils.py** 是核心工具库，被 train.py 和 infer.py 共享使用
3. **train.py 和 infer.py** 分别负责模型训练和推理，都依赖 uni2h_utils.py 提供的基础功能
4. **数据单向流动**：原始数据 → 预处理 → 特征提取 → 训练 → 推理 → 结果输出

建议初学者按照 11.4.2 节推荐的顺序阅读各解读文档，结合实际代码进行学习。
