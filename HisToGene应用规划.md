# HisToGene 模型应用规划与操作细节文档

> 面向 PFMval 项目的完整实施指南

---

## 一、HisToGene 模型概述

### 1.1 论文信息

| 项目 | 内容 |
|------|------|
| **论文标题** | Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors |
| **作者** | Minxing Pang, Kenong Su, Mingyao Li |
| **发表** | bioRxiv (2021), 后被 Nature Communications 等期刊引用 |
| **年份** | 2021 |
| **论文链接** | [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.11.28.470212.full) |
| **代码仓库** | [GitHub: maxpmx/HisToGene](https://github.com/maxpmx/HisToGene) |

### 1.2 核心思想（通俗解释）

HisToGene 是一个基于 **Vision Transformer (ViT)** 的深度学习模型，它的核心思想是：

> **利用病理图像的视觉特征 + 空间位置信息 → 预测该位置的基因表达水平**

想象你在看一张病理切片的照片：
- **传统方法**：只看单个 patch 的图像内容，预测这个位置的基因表达
- **HisToGene 的创新**：不仅看图像，还知道这个 patch 在整张切片上的**坐标位置**，利用空间信息辅助预测

为什么空间信息重要？因为在肿瘤组织中，基因表达往往呈现**空间模式**（比如肿瘤中心 vs 边缘的基因表达不同）。HisToGene 通过位置编码（Position Embedding）将空间坐标融入模型，让模型学会"在哪里"的基因表达应该"是什么样"。

### 1.3 模型架构详解

```
输入图像 (224×224×3)
    ↓
[Patch Embedding] 将图像切分为 32×32 的 patches
    ↓
[位置编码融合] 图像特征 + 空间坐标 (x, y)
    ↓
[ViT 编码器] 8层 Transformer，提取深层特征
    ↓
[MLP 预测头] 将特征映射到基因表达空间
    ↓
输出：基因表达向量 (维度 = 基因数量)
```

#### ViT 编码器配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **Patch Size** | 32×32 | 图像分块大小 |
| **Embedding Dimension** | 1024 | 特征嵌入维度 |
| **Transformer Layers** | 8 | ViT 深度（层数） |
| **Attention Heads** | 16 | 多头注意力头数 |
| **MLP Hidden Dim** | 4096 | 前馈网络隐藏层维度 |
| **Dropout** | 0.1 | Dropout 概率 |
| **Input Image Size** | 224×224 | 输入图像尺寸 |

#### 位置编码方式

HisToGene 的关键创新之一是**空间位置编码**：

- **n_pos = 1024**：位置编码的最大容量（即最多支持 1024 个不同的空间位置）
- 将图像的 (x, y) 坐标映射为可学习的嵌入向量
- 位置编码与图像 patch 的嵌入相加，形成最终的输入表示

这意味着模型不仅学习"图像长什么样"，还学习"在切片的这个位置应该有什么基因表达模式"。

#### 输入/输出维度

| 项目 | 维度/格式 |
|------|----------|
| **输入图像** | (3, 224, 224) - RGB 图像 |
| **输入坐标** | (2,) - (x, y) 空间坐标 |
| **输出** | (n_genes,) - 基因表达向量 |

原始论文中，输出维度等于数据集中的基因数量（如 HER2+ 数据集约 1000+ 个基因）。

### 1.4 与当前 UNI2-h+MLP 方案的关键差异对比表

| 对比维度 | UNI2-h + MLP (当前方案) | HisToGene |
|----------|------------------------|-----------|
| **骨干网络** | UNI2-h (预训练 ViT，冻结) | 自定义 ViT (从头训练) |
| **特征维度** | 1536 维 | 1024 维 |
| **空间信息** | ❌ 不使用 | ✅ 使用位置编码 |
| **输入处理** | 预提取特征 → MLP | 端到端图像 → 基因 |
| **训练方式** | 两阶段 (特征提取 + 回归) | 端到端训练 |
| **参数量** | 小 (仅 MLP: 1536→256→8) | 大 (完整 ViT + MLP) |
| **数据效率** | 高 (预训练特征) | 相对较低 (从头训练) |
| **过拟合风险** | 低 | 高 (需要正则化) |

---

## 二、适配性分析

### 2.1 HisToGene 原始设计 vs 本项目需求的差异表

| 维度 | HisToGene 原始设计 | 本项目需求 | 适配方案 |
|------|-------------------|-----------|---------|
| **预测目标** | 单基因表达 (1000+ 基因) | 8 个通路评分 | 修改输出维度为 8 |
| **输入格式** | 组织切片 + 空间坐标 | Patch PNG (已有坐标) | 从文件名解析坐标 |
| **数据规模** | 多样本，每样本数千 spots | ~1万样本，单样本 | 直接适配，数据量充足 |
| **空间信息** | 必须提供坐标 | 坐标在文件名中 | 提取文件名中的 x, y |
| **图像尺寸** | 224×224 | 与当前一致 | 无需修改 |
| **标签格式** | 基因表达矩阵 | ssGSEA 通路评分 | 直接使用标准化后的评分 |
| **数据分布** | 基因表达计数 (非负) | Z-score 标准化评分 | 使用 Z-score 后的数据 |

### 2.2 需要修改/适配的部分清单

#### 必须修改的部分

1. **输出维度调整**
   - 原始：输出维度 = 基因数量 (1000+)
   - 修改：输出维度 = 8 (通路数量)

2. **输入数据处理**
   - 从 patch 文件名解析 (x, y) 坐标
   - 将坐标归一化到合适范围

3. **损失函数选择**
   - 原始：可能使用 MSE 或负二项损失
   - 考虑：结合数据非正态特点，可能需要 Huber Loss 或加权 MSE

4. **数据加载器**
   - 需要同时返回：图像、坐标、标签

#### 可以直接复用的部分

1. **数据划分**：`split.py` 已经按空间距离划分 train/val
2. **标准化**：`zscore.py` 已经生成标准化后的标签
3. **评估指标**：`uni2h_utils.py` 中的 MSE、MAE、R²、PCC 可以直接使用
4. **训练流程**：早停、学习率调度等逻辑可以参考

### 2.3 可以直接复用的部分

- ✅ 数据划分策略（空间无重叠）
- ✅ Z-score 标准化后的标签文件
- ✅ 评估指标计算函数
- ✅ Patch 图像文件

---

## 三、实施规划

### 阶段 0：环境准备

#### 需要安装的额外依赖

```bash
# 基础依赖（项目已有）
# torch, torchvision, pandas, numpy, scikit-learn, pillow

# HisToGene 可能需要的新依赖
pip install einops           # ViT 常用操作
pip install timm             # 如果要用预训练 ViT（可选）
```

#### HisToGene 代码获取方式

```bash
# 在项目根目录下创建子目录
cd d:\AI空间转录病理研究\PFMval_new
mkdir histogene

# 克隆 HisToGene 仓库（参考用，不直接修改）
# git clone https://github.com/maxpmx/HisToGene.git reference/HisToGene
```

> **注意**：我们**不直接修改**原始 HisToGene 代码，而是**参考其实现**，在 `histogene/` 目录下创建适配后的新版本。

#### 目录结构规划

```
PFMval_new/
├── histogene/              # 新增：HisToGene 适配代码
│   ├── model.py            # ViT + MLP 模型定义
│   ├── dataset.py          # 数据加载器（含坐标解析）
│   ├── train.py            # 训练脚本
│   ├── infer.py            # 推理脚本
│   └── utils.py            # 工具函数
├── checkpoints/            # 已有：模型保存
│   └── HYZ15040/
├── res/                    # 已有：结果输出
└── ...（其他已有文件）
```

---

### 阶段 1：数据准备与适配

#### 现有数据如何转换为 HisToGene 要求的格式

HisToGene 需要三个输入：
1. **图像**：224×224 RGB (已有)
2. **坐标**：(x, y) 归一化后的空间位置 (从文件名解析)
3. **标签**：8 维通路评分 (已有)

#### 具体的数据处理脚本需求

**新建文件：`histogene/dataset.py`**

功能要求：
1. 从 `patch_x4641_y16969.png` 解析 x=4641, y=16969
2. 将坐标归一化到 [0, 1] 或 [0, n_pos-1] 范围
3. 返回 `(image, coordinates, targets)` 三元组

坐标归一化公式建议：
```python
# 方法 1：归一化到 [0, 1]
x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

# 方法 2：映射到整数位置索引（用于位置嵌入查找）
x_idx = int((x - x_min) / bin_size)
y_idx = int((y - y_min) / bin_size)
pos_idx = x_idx * n_y_bins + y_idx  # 展平为 1D 索引
```

#### 标签文件的格式转换

**无需转换**！直接使用 `HYZ15040_ssGSEA_scores_zscore.csv` 即可。

格式要求：
- 第 1 列：patch 文件名（如 `patch_x4641_y16969.png`）
- 第 2-9 列：8 个通路的 Z-score 评分

#### 空间坐标信息如何传递给模型

实现方案：
```python
# 在 Dataset 中
class HisToGeneDataset(Dataset):
    def __getitem__(self, idx):
        # 1. 加载图像
        image = load_image(self.image_paths[idx])
        
        # 2. 解析坐标
        x, y = parse_coordinates_from_filename(filename)
        
        # 3. 加载标签
        targets = self.labels_df.iloc[idx, 1:9].values
        
        return image, (x, y), targets
```

---

### 阶段 2：模型适配与修改

#### HisToGene 代码需要哪些修改（逐项列出）

**1. 模型定义 (`histogene/model.py`)**

```python
class HisToGene(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=32,
                 embed_dim=1024,
                 depth=8,
                 num_heads=16,
                 mlp_dim=4096,
                 n_pos=1024,      # 位置编码容量
                 output_dim=8,    # 修改为 8 个通路
                 dropout=0.1):
        
        # ViT 编码器
        self.patch_embed = PatchEmbedding(...)
        self.pos_embed = nn.Embedding(n_pos, embed_dim)  # 位置嵌入
        self.transformer = TransformerEncoder(...)
        
        # MLP 预测头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_dim)  # 输出 8 维
        )
```

**2. 坐标到位置索引的映射**

```python
def coords_to_pos_index(x, y, x_min, x_max, y_min, y_max, n_pos=1024):
    """
    将 (x, y) 坐标映射到位置索引 [0, n_pos-1]
    """
    # 归一化到 [0, 1]
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    
    # 映射到 [0, n_pos-1]
    x_idx = int(x_norm * (n_pos ** 0.5 - 1))
    y_idx = int(y_norm * (n_pos ** 0.5 - 1))
    
    # 2D 到 1D 的展平
    grid_size = int(n_pos ** 0.5)
    pos_idx = y_idx * grid_size + x_idx
    
    return torch.clamp(pos_idx, 0, n_pos - 1)
```

**3. 损失函数选择建议**

考虑到本项目数据的**非正态分布**和**异常值**问题：

| 损失函数 | 适用性 | 建议 |
|----------|--------|------|
| **MSE** | ⚠️ 对异常值敏感 | 可作为 baseline |
| **Huber Loss** | ✅ 平衡 MSE 和 MAE | **推荐** |
| **Smooth L1** | ✅ 对异常值鲁棒 | 备选 |
| **Weighted MSE** | ✅ 按通路加权 | 考虑不同通路的方差差异 |

Huber Loss 实现：
```python
criterion = nn.HuberLoss(delta=1.0)  # delta 控制过渡点
```

**4. 位置编码适配**

本项目的数据特点：
- 坐标范围：从文件名解析（如 x: 0~10000, y: 0~20000）
- 需要确定 n_pos 的值（建议 1024 或 2048）
- 考虑使用 2D 位置编码（分别对 x 和 y 编码后拼接）

---

### 阶段 3：训练配置

#### 推荐的超参数设置（结合本项目数据量）

| 超参数 | 原始 HisToGene | 本项目建议 | 说明 |
|--------|---------------|-----------|------|
| **Batch Size** | 500 | 64-128 | 数据量较小，适当减小 |
| **Learning Rate** | 0.1 (较大) | 1e-4 ~ 1e-3 | 使用 AdamW，较小学习率 |
| **Epochs** | 100+ | 100-200 | 配合早停使用 |
| **Optimizer** | SGD/Adam | **AdamW** | 带权重衰减 |
| **Weight Decay** | - | 1e-4 | 防止过拟合 |
| **Dropout** | 0.1 | **0.3-0.5** | 数据量小，增大 dropout |
| **LR Scheduler** | - | ReduceLROnPlateau | 学习率衰减 |

#### 训练策略建议

**1. 分层学习率**
```python
# ViT 编码器使用较小学习率
# MLP 头使用较大学习率
param_groups = [
    {'params': model.transformer.parameters(), 'lr': 1e-5},
    {'params': model.mlp_head.parameters(), 'lr': 1e-3}
]
optimizer = AdamW(param_groups, weight_decay=1e-4)
```

**2. 数据增强建议**

由于病理图像的特殊性，建议的增强策略：

| 增强类型 | 参数 | 是否推荐 |
|----------|------|----------|
| **Random Horizontal Flip** | p=0.5 | ✅ 推荐 |
| **Random Vertical Flip** | p=0.5 | ✅ 推荐 |
| **Random Rotation** | 90° 倍数 | ✅ 推荐 |
| **Color Jitter** | 亮度/对比度微调 | ⚠️ 谨慎使用 |
| **Random Crop** | - | ❌ 不推荐（可能切掉关键区域） |

**3. 早停和模型选择策略**

```python
# 早停配置
early_stop_patience = 15      # 比 UNI2-h 更耐心（模型更大）
min_delta = 0.0001            # 最小改善阈值
monitor = 'val_loss'          # 监控验证损失

# 保存最佳模型
save_best_only = True
```

---

### 阶段 4：评估与对比

#### 使用相同的评估指标

复用 `uni2h_utils.py` 中的评估函数：
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- PCC (Pearson Correlation Coefficient)

#### 与现有 UNI2-h+MLP 的对比方案

**对比维度**：

1. **整体性能对比**
   - 宏平均指标（8 个通路的平均）
   - 训练时间对比
   - 推理速度对比

2. **逐通路性能分析**
   - 每个通路的 R² 和 PCC
   - 哪些通路 HisToGene 表现更好？

3. **空间预测可视化**
   - 将预测结果映射回切片坐标
   - 对比真实值 vs 预测值的空间分布

#### 逐通路性能分析

特别关注以下通路的表现差异：

| 通路 | 数据特点 | 预期表现 |
|------|----------|----------|
| **mhc** | 最接近正态 | 两个模型都应该较好 |
| **ifng** | 异常值 19% | HisToGene 可能更鲁棒（如果用好损失函数） |
| **hypoxia** | 勉强可用 | 对比哪个模型更稳定 |

---

## 四、操作细节

### 步骤 1：创建项目目录结构

```powershell
# 在 PowerShell 中执行
cd d:\AI空间转录病理研究\PFMval_new

# 创建 HisToGene 目录
mkdir histogene

# 创建子目录
mkdir histogene\checkpoints
mkdir histogene\results
```

**预期输出**：
```
histogene/
├── checkpoints/     # 模型检查点
├── results/         # 推理结果
└── （代码文件）
```

### 步骤 2：创建数据集类

**新建文件：`histogene/dataset.py`**

```python
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def parse_coordinates_from_filename(filename):
    """从文件名解析坐标"""
    match = re.search(r'patch_x(\d+)_y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class HisToGeneDataset(Dataset):
    """
    HisToGene 数据集类
    返回: (image, pos_index, targets)
    """
    def __init__(self, patches_dir, labels_csv, n_pos=1024, 
                 transform=None, target_cols=8):
        self.patches_dir = patches_dir
        self.n_pos = n_pos
        self.transform = transform or self.default_transform()
        
        # 加载标签
        self.labels_df = pd.read_csv(labels_csv)
        
        # 获取所有 patch 文件
        self.patch_files = []
        self.coordinates = []
        
        for fname in os.listdir(patches_dir):
            if fname.endswith('.png'):
                x, y = parse_coordinates_from_filename(fname)
                if x is not None and y is not None:
                    self.patch_files.append(fname)
                    self.coordinates.append((x, y))
        
        # 计算坐标范围用于归一化
        self.x_coords = [c[0] for c in self.coordinates]
        self.y_coords = [c[1] for c in self.coordinates]
        self.x_min, self.x_max = min(self.x_coords), max(self.x_coords)
        self.y_min, self.y_max = min(self.y_coords), max(self.y_coords)
        
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def coords_to_pos_index(self, x, y):
        """将坐标映射到位置索引"""
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        
        grid_size = int(np.sqrt(self.n_pos))
        x_idx = int(x_norm * (grid_size - 1))
        y_idx = int(y_norm * (grid_size - 1))
        
        pos_idx = y_idx * grid_size + x_idx
        return min(max(pos_idx, 0), self.n_pos - 1)
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.patches_dir, self.patch_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 获取坐标索引
        x, y = self.coordinates[idx]
        pos_idx = self.coords_to_pos_index(x, y)
        
        # 加载标签（假设 CSV 第 1 列是文件名，第 2-9 列是标签）
        patch_name = self.patch_files[idx]
        row = self.labels_df[self.labels_df.iloc[:, 0].str.contains(
            patch_name.replace('.png', ''))]
        
        if len(row) == 0:
            raise ValueError(f"Label not found for {patch_name}")
        
        targets = row.iloc[0, 1:9].values.astype(np.float32)
        
        return image, torch.tensor(pos_idx), torch.tensor(targets)
```

### 步骤 3：创建模型定义

**新建文件：`histogene/model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    """图像分块嵌入"""
    def __init__(self, img_size=224, patch_size=32, in_channels=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)  # (B, embed_dim, 7, 7)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, 49, embed_dim)
        return x


class HisToGeneModel(nn.Module):
    """
    HisToGene 模型适配版
    输出维度改为 8（通路数量）
    """
    def __init__(self, 
                 img_size=224,
                 patch_size=32,
                 in_channels=3,
                 embed_dim=1024,
                 depth=8,
                 num_heads=16,
                 mlp_dim=4096,
                 n_pos=1024,
                 output_dim=8,  # 修改为 8 个通路
                 dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 可学习的位置嵌入（空间坐标）
        self.pos_embed = nn.Embedding(n_pos, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP 预测头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, img, pos_idx):
        """
        Args:
            img: (B, 3, 224, 224)
            pos_idx: (B,) 位置索引
        Returns:
            (B, output_dim)
        """
        B = img.shape[0]
        
        # Patch embedding
        x = self.patch_embed(img)  # (B, n_patches, embed_dim)
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)
        
        # 添加位置嵌入（基于空间坐标）
        pos_emb = self.pos_embed(pos_idx)  # (B, embed_dim)
        pos_emb = pos_emb.unsqueeze(1)  # (B, 1, embed_dim)
        x = x + pos_emb  # 广播到所有 patches
        
        # Transformer
        x = self.transformer(x)  # (B, n_patches+1, embed_dim)
        
        # 取 CLS token 输出
        cls_out = x[:, 0]  # (B, embed_dim)
        
        # MLP 预测
        out = self.mlp_head(cls_out)  # (B, output_dim)
        
        return out
```

### 步骤 4：创建训练脚本

**新建文件：`histogene/train.py`**

```python
import argparse
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import HisToGeneModel
from dataset import HisToGeneDataset


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 逐通路计算
    n_targets = y_true.shape[1]
    per_target_r2 = []
    per_target_pcc = []
    
    for j in range(n_targets):
        yt, yp = y_true[:, j], y_pred[:, j]
        if np.std(yt) > 0:
            per_target_r2.append(r2_score(yt, yp))
        else:
            per_target_r2.append(np.nan)
        
        if np.std(yt) > 0 and np.std(yp) > 0:
            per_target_pcc.append(np.corrcoef(yt, yp)[0, 1])
        else:
            per_target_pcc.append(np.nan)
    
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': np.nanmean(per_target_r2),
        'pcc': np.nanmean(per_target_pcc),
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    for images, pos_idx, targets in dataloader:
        images = images.to(device)
        pos_idx = pos_idx.to(device)
        targets = targets.to(device)
        
        outputs = model(images, pos_idx)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())
    
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = running_loss / len(dataloader)
    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, pos_idx, targets in dataloader:
            images = images.to(device)
            pos_idx = pos_idx.to(device)
            targets = targets.to(device)
            
            outputs = model(images, pos_idx)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = running_loss / len(dataloader)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, 
                       default=r'D:\PycharmProjects\AIPath-data\patch\HYZ15040\train_patches')
    parser.add_argument('--val_dir', type=str,
                       default=r'D:\PycharmProjects\AIPath-data\patch\HYZ15040\val_patches')
    parser.add_argument('--labels_csv', type=str,
                       default=r'D:\PycharmProjects\AIPath-data\HYZ15040_ssGSEA_scores_zscore.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_pos', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--checkpoint_dir', type=str, default='./histogene/checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建数据集
    train_dataset = HisToGeneDataset(args.train_dir, args.labels_csv, n_pos=args.n_pos)
    val_dataset = HisToGeneDataset(args.val_dir, args.labels_csv, n_pos=args.n_pos)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # 创建模型
    model = HisToGeneModel(output_dim=8, dropout=args.dropout, n_pos=args.n_pos).to(device)
    
    # 损失函数 - 使用 Huber Loss 对异常值更鲁棒
    criterion = nn.HuberLoss(delta=1.0)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['loss'])
        
        print(f'Train | loss={train_metrics["loss"]:.4f} mae={train_metrics["mae"]:.4f} '
              f'r2={train_metrics["r2"]:.4f} pcc={train_metrics["pcc"]:.4f}')
        print(f'Val   | loss={val_metrics["loss"]:.4f} mae={val_metrics["mae"]:.4f} '
              f'r2={val_metrics["r2"]:.4f} pcc={val_metrics["pcc"]:.4f}')
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
            'val_pcc': val_metrics['pcc'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # 早停检查
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'args': vars(args),
                'best_val_loss': best_val_loss,
                'epoch': epoch + 1
            }, os.path.join(args.checkpoint_dir, 'best_model_histogene.pth'))
            print(f'*** New best model saved ***')
        else:
            patience_counter += 1
            print(f'No improvement ({patience_counter}/{args.patience})')
            
        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 保存训练历史
    pd.DataFrame(history).to_csv(
        os.path.join(args.checkpoint_dir, 'training_history.csv'), index=False)
    print(f'\nTraining completed. Best val loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
```

### 步骤 5：运行训练

```powershell
# 在 PowerShell 中执行
cd d:\AI空间转录病理研究\PFMval_new

# 运行训练
python histogene/train.py `
    --train_dir "D:\PycharmProjects\AIPath-data\patch\HYZ15040\train_patches" `
    --val_dir "D:\PycharmProjects\AIPath-data\patch\HYZ15040\val_patches" `
    --labels_csv "D:\PycharmProjects\AIPath-data\HYZ15040_ssGSEA_scores_zscore.csv" `
    --batch_size 64 `
    --epochs 150 `
    --lr 1e-4 `
    --dropout 0.3 `
    --patience 15
```

**预期输出**：
```
Using device: cuda
Train samples: 9806
Val samples: 772

Epoch 1/150
Train | loss=0.8234 mae=0.6543 r2=0.1234 pcc=0.3456
Val   | loss=0.6543 mae=0.5432 r2=0.2345 pcc=0.4567
*** New best model saved ***
...
```

### 步骤 6：创建推理脚本

**新建文件：`histogene/infer.py`**

参考 `uni2h/infer.py` 的结构，创建类似的推理脚本，输出预测结果和评估指标。

### 可能遇到的问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **显存不足** | ViT 参数量大 | 减小 batch_size 到 32 或 16 |
| **过拟合严重** | 数据量相对模型太小 | 增大 dropout (0.5)，添加 L2 正则化 |
| **坐标解析错误** | 文件名格式不匹配 | 检查正则表达式 `patch_x(\d+)_y(\d+)` |
| **标签找不到** | CSV 文件名格式不一致 | 检查 CSV 第 1 列是否包含 patch 文件名 |
| **训练不稳定** | 学习率太大 | 减小 lr 到 1e-5，使用梯度裁剪 |
| **某些通路预测很差** | 数据分布问题 | 尝试 Weighted Loss 或单独训练 |

---

## 五、目录结构规划

### 新增文件和目录的完整规划

```
d:\AI空间转录病理研究\PFMval_new\
├── histogene/                          # 【新增】HisToGene 实现
│   ├── __init__.py
│   ├── model.py                        # ViT + MLP 模型定义
│   ├── dataset.py                      # 数据加载器（含坐标解析）
│   ├── train.py                        # 训练脚本
│   ├── infer.py                        # 推理脚本
│   ├── utils.py                        # 工具函数
│   ├── checkpoints/                    # 模型保存
│   │   └── best_model_histogene.pth
│   └── results/                        # 推理结果
│       └── predictions.csv
│
├── uni2h/                              # 【已有】UNI2-h 实现
│   ├── train.py
│   ├── infer.py
│   └── uni2h_utils.py
│
├── split.py                            # 【已有】数据划分
├── zscore.py                           # 【已有】标准化
├── HYZ15040_ssGSEA_scores_zscore.csv   # 【已有】标签文件
│
└── HisToGene应用规划.md                # 【本文档】
```

### 建议的项目结构

保持现有文件不变，所有 HisToGene 相关代码放在独立的 `histogene/` 目录下，便于：
1. 与 UNI2-h 方案并行对比
2. 独立开发和调试
3. 避免污染现有代码

---

## 六、风险评估与应对

### 6.1 数据量是否足够训练 ViT

**风险等级**：⚠️ 中等

**分析**：
- HisToGene 原始论文使用多样本，每个样本数千 spots
- 本项目：~1万样本，单一样本
- ViT 参数量：~10M+（取决于配置）
- 样本/参数比：~1万/10M = 0.001（偏低）

**应对策略**：
1. ✅ 使用较大的 dropout (0.3-0.5)
2. ✅ 使用强正则化（weight_decay=1e-4）
3. ✅ 使用早停（patience=15）
4. ✅ 考虑数据增强（翻转、旋转）
5. ⚠️ 如果过拟合严重，考虑减小模型（depth=4-6）

### 6.2 过拟合风险和应对策略

**风险等级**：🔴 高

**应对策略**：

| 策略 | 具体做法 | 预期效果 |
|------|----------|----------|
| **增大 Dropout** | 0.1 → 0.3-0.5 | 减少神经元共适应 |
| **权重衰减** | weight_decay=1e-4 | 限制权重大小 |
| **早停** | patience=15 | 防止训练过度 |
| **数据增强** | 翻转、旋转 | 有效增加数据多样性 |
| **简化模型** | depth=8 → 4-6 | 减少参数量 |
| **标签平滑** | 软化硬标签 | 减少过拟合置信度 |

### 6.3 训练时间和计算资源估算

**硬件假设**：NVIDIA RTX 4090 (24GB) 或同级

| 配置 | 预估时间/epoch | 预估总时间 |
|------|---------------|-----------|
| Batch=64, Epochs=150 | ~30-60 秒 | 1-2.5 小时 |
| Batch=32, Epochs=150 | ~40-80 秒 | 1.5-3 小时 |

**显存占用估算**：
- 模型参数：~50 MB
- 中间激活：~2-4 GB (取决于 batch_size)
- 总占用：~4-6 GB

### 6.4 数据分布问题（非正态、异常值）的处理建议

针对本项目数据的特殊性质：

| 问题 | 影响 | 建议方案 |
|------|------|----------|
| **ifng 异常值 19%** | MSE 损失会被极端值主导 | 使用 Huber Loss 或 Smooth L1 |
| **严重右偏** | 模型可能偏向预测低值 | 考虑对数变换后再标准化 |
| **不同通路方差差异大** | 模型偏向大方差通路 | 使用 Weighted MSE |

**Weighted MSE 实现建议**：
```python
# 根据各通路的标准差计算权重
# 标准差大的通路（如 mhc）权重小，标准差小的通路（如 icp）权重大
weights = 1.0 / stds  # stds 是各通路的标准差
weights = weights / weights.sum() * 8  # 归一化

def weighted_mse_loss(pred, target, weights):
    se = (pred - target) ** 2
    weighted_se = se * weights.unsqueeze(0)
    return weighted_se.mean()
```

---

## 七、时间线估算

### 各阶段预计耗时

| 阶段 | 任务 | 预计耗时 | 依赖 |
|------|------|----------|------|
| **阶段 0** | 环境准备、依赖安装 | 30 分钟 | - |
| **阶段 1** | 创建 dataset.py | 1-2 小时 | 阶段 0 |
| **阶段 2** | 创建 model.py | 2-3 小时 | 阶段 1 |
| **阶段 3** | 创建 train.py | 2-3 小时 | 阶段 2 |
| **阶段 4** | 调试运行、修复问题 | 2-4 小时 | 阶段 3 |
| **阶段 5** | 完整训练（150 epochs） | 2-3 小时 | 阶段 4 |
| **阶段 6** | 创建 infer.py、评估 | 1-2 小时 | 阶段 5 |
| **阶段 7** | 与 UNI2-h 对比分析 | 1-2 小时 | 阶段 6 |

**总计**：约 **12-20 小时**（不含训练等待时间）

### 里程碑节点

| 里程碑 | 完成标准 | 预计时间 |
|--------|----------|----------|
| **M1** | dataset.py 可正常加载数据 | 阶段 1 结束 |
| **M2** | model.py 前向传播无错误 | 阶段 2 结束 |
| **M3** | 完成第一个 epoch 训练 | 阶段 4 结束 |
| **M4** | 获得最佳模型 | 阶段 5 结束 |
| **M5** | 完成评估对比报告 | 阶段 7 结束 |

---

## 附录：快速参考

### HisToGene 关键参数速查表

| 参数 | 原始值 | 本项目建议值 | 说明 |
|------|--------|-------------|------|
| img_size | 224 | 224 | 输入图像尺寸 |
| patch_size | 32 | 32 | Patch 大小 |
| embed_dim | 1024 | 1024 | 嵌入维度 |
| depth | 8 | 8 (或 6) | Transformer 层数 |
| num_heads | 16 | 16 | 注意力头数 |
| mlp_dim | 4096 | 4096 | MLP 隐藏层 |
| n_pos | 1024 | 1024 | 位置编码容量 |
| output_dim | 1000+ | 8 | 输出维度 |
| dropout | 0.1 | 0.3-0.5 | 正则化 |

### 命令速查

```powershell
# 训练
python histogene/train.py --batch_size 64 --epochs 150 --lr 1e-4 --dropout 0.3

# 推理
python histogene/infer.py --checkpoint_path histogene/checkpoints/best_model_histogene.pth
```

---

> **文档版本**：v1.0  
> **创建日期**：2026-04-15  
> **适用项目**：PFMval - HisToGene 适配规划
