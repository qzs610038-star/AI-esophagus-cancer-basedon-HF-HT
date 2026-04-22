# EGNv2 模型部署方案

## 1. 模型简介

### 1.1 EGNv2 架构概述

**EGNv2**（Exemplar Guided Graph Neural Network v2）是由 Yan Yang 等人于 2024 年发表在 Pattern Recognition 上的空间转录组基因表达预测模型。

**核心组件：**
- **特征提取器**：ResNet-50（比 HisToGene 的 ViT 更轻量，参数量约 23M vs 86M）
- **骨干网络**：GraphSAGE 图卷积网络，用于建模 patch 间的空间关系
- **代表学习（Exemplar Learning）**：构建代表库（exemplar library），学习 patch-to-exemplar 的相似性关系
- **预测头**：MLP 回归头，输出基因表达或通路评分

**核心创新：**
1. **代表学习机制**：从训练集中选取代表性样本构建 exemplar library，测试时通过相似性检索进行预测
2. **图卷积建模**：将每个 patch 视为图节点，通过 KNN 或空间邻接构建图结构，利用 GraphSAGE 聚合邻居信息
3. **空间感知**：显式建模 patch 间的空间依赖关系，而非仅依赖坐标嵌入

### 1.2 与 HisToGene 的差异对比

| 维度 | HisToGene | EGNv2 |
|------|-----------|-------|
| **特征提取** | ViT（ViT-Large 变体，patch_size=16） | ResNet-50（轻量级 CNN） |
| **空间建模** | 坐标嵌入（x_embed + y_embed） | 图卷积（GraphSAGE）+ KNN 图 |
| **核心机制** | Transformer 自注意力 | Exemplar Learning + 图消息传递 |
| **参数量** | ~86M（dim=1024, depth=8, heads=16） | ~35M（ResNet-50 + GraphSAGE） |
| **训练数据需求** | 较大（易过拟合） | 中等（代表学习提供正则化） |
| **推理开销** | 低（单次前向） | 中（需构建图 + KNN 检索） |
| **跨数据集泛化** | PCC=0.53（Nature Comm 2025 基准） | PCC=0.53（并列最优） |

### 1.3 性能基准

根据 Nature Communications 2025 年的跨方法基准测试（11 种方法）：
- **EGNv2 在 HER2ST 数据集**：PCC=0.28（排名第一）
- **跨数据集泛化能力**：PCC=0.53（与 HisToGene 并列最优）

---

## 2. 数据适配性分析

### 2.1 数据格式兼容性

#### 2.1.1 当前数据格式（PFMval 项目）

**输入数据：**
- **图像**：224×224 H&E patch（PNG 格式），文件名格式 `patch_x{y}_y{x}.png`
- **坐标**：嵌入在文件名中，通过正则解析 x, y 像素坐标
- **标签**：ssGSEA 通路评分（30 个通路），Z-score 标准化，CSV 格式

**HisToGene 数据流：**
```
PNG Image → Resize(224,224) → Normalize(ImageNet) → ViT PatchEmbed
                                    ↓
Coordinate → x_embed/y_embed → 加到 CLS token
                                    ↓
                            Transformer → MLP Head → 30-dim output
```

#### 2.1.2 EGNv2 输入要求

根据论文和 GitHub 仓库信息，EGNv2 期望的输入格式：

**训练阶段：**
- **图像**：H&E patch（论文使用 224×224 或 299×299）
- **空间坐标**：用于构建 KNN 图或空间邻接图
- **基因表达/通路评分**：作为回归目标

**推理阶段：**
- **图像**：待预测的 H&E patch
- **图结构**：需要预先构建 KNN 图（基于特征相似性或空间距离）
- **代表库**：从训练集构建的 exemplar library

#### 2.1.3 适配性评估

| 数据维度 | 匹配度 | 说明 |
|----------|--------|------|
| **图像格式** | ✅ 完全兼容 | 224×224 PNG 可直接输入 ResNet |
| **坐标解析** | ✅ 完全兼容 | 现有 `parse_coordinates()` 可直接复用 |
| **标签格式** | ✅ 完全兼容 | 30 通路 Z-score 评分可直接作为回归目标 |
| **图构建** | ⚠️ 需适配 | EGNv2 需要显式构建图结构（KNN 或空间邻接） |
| **代表库** | ⚠️ 需新增 | 需要实现 exemplar library 的构建和管理 |

**结论**：当前数据格式与 EGNv2 输入要求高度兼容，主要适配工作集中在图结构构建和代表库管理。

### 2.2 数据量适配性

#### 2.2.1 当前数据规模

| 数据集 | 训练样本 | 验证样本 | 总计 |
|--------|----------|----------|------|
| HYZ15040 | ~2,124 (80%) | ~531 (20%) | 2,655 |
| JFX0729 | ~6,270 (80%) | ~1,568 (20%) | 7,838 |
| LMZ12939 | ~6,010 (80%) | ~1,503 (20%) | 7,513 |
| **联合训练** | **~14,404** | **~3,602** | **18,006** |

#### 2.2.2 模型参数量对比

| 模型 | 参数量 | 每样本参数量比 |
|------|--------|----------------|
| HisToGene | ~86M | HYZ: 40K/参数，联合: 209/参数 |
| EGNv2 | ~35M | HYZ: 99K/参数，联合: 514/参数 |

#### 2.2.3 过拟合风险评估

**HisToGene 现状：**
- HYZ15040 训练 PCC 0.65+，验证 PCC 0.51，Gap ~0.14（明显过拟合）
- 联合训练 Gap 减小，但仍存在过拟合

**EGNv2 预期优势：**
1. **ResNet 更轻量**：参数量减少 60%，降低过拟合风险
2. **代表学习正则化**：Exemplar learning 引入隐式数据增强效果
3. **图卷积平滑**：邻居聚合起到隐式正则化作用

**风险等级评估：**
- HYZ15040（2K 样本）：⚠️ 中等风险（EGNv2 更轻量，风险低于 HisToGene）
- JFX0729/LMZ12939（7K+ 样本）：✅ 低风险
- 联合训练（14K+ 样本）：✅ 低风险

#### 2.2.4 代表库构建策略

**策略选项：**

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **A. 全量代表库** | 使用全部训练样本作为 exemplars | 数据量小（<5K）时 |
| **B. K-means 聚类** | 对训练集特征聚类，取聚类中心作为代表 | 数据量大时，减少计算开销 |
| **C. 随机采样** | 随机选取固定数量（如 1024）样本 | 快速原型验证 |

**推荐策略：**
- HYZ15040：策略 A（全量，2K 样本可接受）
- JFX0729/LMZ12939：策略 B（K-means 聚类到 2048 个代表）
- 联合训练：策略 B（聚类到 4096 个代表）

### 2.3 预期优势

#### 2.3.1 相比 HisToGene 的潜在改进

1. **过拟合缓解**
   - ResNet-50 参数量仅为 ViT 的 40%
   - Exemplar learning 的隐式正则化效果
   - 图卷积的邻居平滑效应

2. **空间依赖建模**
   - HisToGene 仅通过坐标嵌入引入空间信息
   - EGNv2 通过图卷积显式建模 patch 间关系
   - 对空间相关性强的通路（如免疫浸润相关通路）可能有更好预测

3. **跨数据集泛化**
   - 论文报告跨数据集 PCC=0.53，与 HisToGene 持平
   - 代表学习机制可能对域偏移更鲁棒

#### 2.3.2 预期性能目标

基于 HisToGene 基线（Val PCC: 0.52-0.61），EGNv2 目标：

| 数据集 | HisToGene Val PCC | EGNv2 目标 Val PCC | 挑战 |
|--------|-------------------|-------------------|------|
| HYZ15040 | 0.5164 | 0.50-0.55 | 数据量小，过拟合风险 |
| JFX0729 | 0.6050 | 0.58-0.65 | 数据充足，有望提升 |
| LMZ12939 | 0.5287 | 0.52-0.58 | 数据充足，有望提升 |
| 联合训练 | 0.5569 | 0.55-0.62 | 数据充足，图结构更复杂 |

### 2.4 预期挑战

#### 2.4.1 技术挑战

| 挑战 | 影响 | 应对策略 |
|------|------|----------|
| **代表库构建开销** | 训练前需提取全部训练集特征，增加准备时间 | 预计算并缓存特征，增量更新 |
| **图结构定义** | KNN 图 vs 空间邻接图的选择影响性能 | 实验对比两种图构建策略 |
| **KNN 检索效率** | 大规模代表库时检索开销增加 | 使用 FAISS 加速，或限制代表数量 |
| **超参调优** | GraphSAGE 层数、KNN 的 k 值等新增超参 | 网格搜索 + 早停 |

#### 2.4.2 数据适配挑战

| 挑战 | 影响 | 应对策略 |
|------|------|----------|
| **从单基因到多通路** | 原论文针对单基因预测，需适配多通路回归 | 修改输出头为 30-dim，使用 MSELoss |
| **图批处理** | 不同 batch 的图结构可能不同 | 使用 PyG 的 Batch 处理或固定图结构 |
| **坐标归一化** | EGNv2 可能需要不同的坐标处理方式 | 实验对比原始坐标 vs 归一化坐标 |

#### 2.4.3 工程挑战

| 挑战 | 影响 | 应对策略 |
|------|------|----------|
| **PyTorch Geometric 依赖** | 新增依赖库，需验证兼容性 | 测试 torch_geometric 与现有环境兼容性 |
| **显存占用** | 图卷积可能增加显存需求 | 监控显存，必要时减小 batch size |
| **训练时间** | 图构建和 KNN 检索增加训练时间 | 预计算图结构，使用高效 KNN 库 |

---

## 3. 部署方案设计

### 3.1 目录结构

遵循"只新增不修改"原则，EGNv2 独立目录结构：

```
PFMval_new/
├── egnv2/                          # 新建目录
│   ├── __init__.py
│   ├── model.py                    # EGNv2 模型定义
│   │   ├── ResNetFeatureExtractor  # ResNet-50 特征提取
│   │   ├── GraphSAGEBackbone       # GraphSAGE 图卷积
│   │   ├── ExemplarLibrary         # 代表库管理
│   │   └── EGNv2Model              # 完整模型
│   ├── dataset.py                  # 数据加载器
│   │   ├── EGNv2Dataset            # 基础数据集
│   │   └── GraphDataset            # 图结构数据集（PyG）
│   ├── exemplar_builder.py         # 代表库构建工具
│   │   ├── build_exemplar_library  # 构建代表库
│   │   └── extract_features        # 特征提取
│   ├── graph_builder.py            # 图结构构建
│   │   ├── build_knn_graph         # KNN 图构建
│   │   └── build_spatial_graph     # 空间邻接图构建
│   ├── train.py                    # 训练脚本
│   ├── infer.py                    # 推理脚本
│   ├── utils.py                    # 工具函数
│   ├── checkpoints/                # 模型保存
│   │   └── {dataset_name}/
│   ├── exemplar_libs/              # 代表库缓存
│   │   └── {dataset_name}_exemplars.pth
│   └── results_vis/                # 可视化结果
│       └── {dataset_name}_{timestamp}/
├── histogene/                      # 现有（不修改）
├── uni2h_new/                      # 现有（不修改）
├── visualize_results.py            # 复用
├── notify_utils.py                 # 复用
└── config.yaml                     # 复用
```

### 3.2 集成方式

#### 3.2.1 与现有组件的复用关系

| 组件 | 复用方式 | 说明 |
|------|----------|------|
| `visualize_results.py` | 直接调用 | 训练完成后调用 `generate_full_report()` |
| `notify_utils.py` | import | 训练完成/异常时发送通知 |
| `config.yaml` | 读取 | 通过 `config_utils.py` 读取数据路径 |
| `config_utils.py` | import | 统一配置管理 |

#### 3.2.2 配置管理

**方案 A：复用现有 config.yaml（推荐）**
- 优点：配置集中，无需新增文件
- 实现：EGNv2 训练脚本通过 `config_utils.get_data_paths()` 读取路径

**方案 B：独立配置文件 egnv2/config.yaml**
- 优点：EGNv2 专属配置隔离
- 缺点：增加配置分散度

**推荐**：方案 A，保持配置集中化。

### 3.3 关键适配点

#### 3.3.1 数据加载器适配

**输入格式统一：**
```python
# EGNv2Dataset 输出格式与 HisToGeneDataset 保持一致
# 返回: (image, coord_features, targets)
# - image: (3, 224, 224) Tensor
# - coord_features: (2,) Tensor (x, y) 或图节点索引
# - targets: (30,) Tensor
```

**图结构处理：**
```python
# GraphDataset (PyTorch Geometric Data)
# - x: 节点特征 (N, feature_dim)
# - edge_index: 边索引 (2, num_edges)
# - y: 节点标签 (N, 30)
# - pos: 节点坐标 (N, 2) 用于空间图构建
```

#### 3.3.2 代表库构建流程

```python
# Step 1: 特征提取
def extract_features(model, dataloader, device):
    """提取训练集所有样本的 ResNet 特征"""
    features = []
    coords = []
    targets = []
    for images, _, _, labels in dataloader:
        with torch.no_grad():
            feat = model.feature_extractor(images.to(device))
        features.append(feat.cpu())
        targets.append(labels)
    return torch.cat(features), torch.cat(targets)

# Step 2: 代表库构建（K-means 聚类）
def build_exemplar_library(features, targets, n_exemplars=2048):
    """使用 K-means 选取代表性样本"""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_exemplars, random_state=42)
    kmeans.fit(features.numpy())
    # 取每个聚类中心最近的样本作为代表
    exemplars = ...
    return exemplars
```

#### 3.3.3 图结构定义

**方案 A：KNN 图（基于特征相似性）**
```python
def build_knn_graph(features, k=10):
    """基于特征相似性构建 KNN 图"""
    from sklearn.neighbors import kneighbors_graph
    adj = kneighbors_graph(features, k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    return edge_index
```

**方案 B：空间邻接图（基于坐标距离）**
```python
def build_spatial_graph(coords, radius=200):
    """基于空间距离构建半径图"""
    # 距离小于 radius 的节点间建立边
    edge_index = radius_graph(coords, r=radius, loop=False)
    return edge_index
```

**推荐**：优先尝试方案 B（空间邻接图），更符合病理图像的空间连续性假设。

#### 3.3.4 损失函数和评估指标统一

**与 HisToGene 对齐：**
```python
# 损失函数
import torch.nn as nn
criterion = nn.HuberLoss(delta=1.0)  # 与 HisToGene 一致

# 评估指标（utils.py 中实现）
def compute_metrics(y_true, y_pred):
    """与 HisToGene utils.py 相同的指标计算"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc = pearson_corrcoef(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'pcc': pcc}
```

**predictions.csv 格式对齐：**
```python
# 与 visualize_results.py 期望的格式一致
# 列名: true_{pathway}, pred_{pathway}
# 例如: true_KEGG_GLYCOLYSIS_GLUCONEOGENESIS, pred_KEGG_GLYCOLYSIS_GLUCONEOGENESIS
```

### 3.4 训练参数推荐

#### 3.4.1 基于论文默认参数

根据 EGNv2 论文和 GitHub 仓库的默认配置：

| 参数 | 论文默认值 | PFMval 推荐值 | 说明 |
|------|-----------|---------------|------|
| **backbone** | ResNet-50 | ResNet-50 | 保持不变 |
| **graph_layers** | 2 | 2 | GraphSAGE 层数 |
| **hidden_dim** | 512 | 512 | 图卷积隐藏维度 |
| **k_neighbors** | 10 | 10 | KNN 图的 k 值 |
| **n_exemplars** | 1024 | 2048 | 代表库大小（根据数据量调整） |
| **learning_rate** | 1e-3 | 1e-4 | 配合 AdamW，更保守 |
| **batch_size** | 32 | 64 | 与 HisToGene 一致 |
| **epochs** | 100 | 150 | 配合早停 |
| **dropout** | 0.2 | 0.3 | 数据量小，增加正则化 |

#### 3.4.2 与 HisToGene 配置对比

| 参数 | HisToGene | EGNv2 | 差异说明 |
|------|-----------|-------|----------|
| 特征提取 | ViT-Large (86M) | ResNet-50 (23M) | EGNv2 更轻量 |
| 空间建模 | 坐标嵌入 | GraphSAGE | EGNv2 显式图建模 |
| 核心机制 | Transformer | Exemplar + GNN | 完全不同的范式 |
| 学习率 | 1e-4 | 1e-4 | 保持一致 |
| batch_size | 64 | 64 | 保持一致 |
| dropout | 0.3 | 0.3 | 保持一致 |
| optimizer | AdamW | AdamW | 保持一致 |
| scheduler | ReduceLROnPlateau | ReduceLROnPlateau | 保持一致 |

---

## 4. 实施步骤

### Step 1: 环境准备（预计 0.5 天）

**1.1 安装依赖**
```bash
# 在现有 conda 环境中安装 PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 安装 FAISS 用于高效 KNN 检索（可选但推荐）
pip install faiss-gpu  # 或 faiss-cpu

# 验证安装
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

**1.2 验证 GPU 兼容性**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from torch_geometric.nn import GraphSAGE; print('PyG OK')"
```

**【用户确认点 1】** 是否接受安装 torch_geometric 作为新依赖？

### Step 2: 代码适配（预计 2 天）

**2.1 克隆 EGN 仓库并分析代码**
```bash
cd d:\AI空间转录病理研究\PFMval_new
git clone https://github.com/Yan98/EGN.git egnv2/egn_source
```

**2.2 实现核心模块**

| 文件 | 功能 | 预估代码量 |
|------|------|-----------|
| `egnv2/model.py` | ResNet + GraphSAGE + Exemplar | ~200 行 |
| `egnv2/dataset.py` | 数据加载 + 图构建 | ~150 行 |
| `egnv2/exemplar_builder.py` | 代表库构建 | ~100 行 |
| `egnv2/graph_builder.py` | KNN/空间图构建 | ~80 行 |
| `egnv2/utils.py` | 指标计算 | ~50 行 |

**2.3 适配多通路输出**
```python
# 修改原论文的单基因输出为 30 通路输出
class EGNv2Model(nn.Module):
    def __init__(self, n_pathways=30, ...):
        ...
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_pathways)  # 30-dim output
        )
```

### Step 3: 代表库构建（预计 0.5 天）

**3.1 预计算训练集特征**
```bash
python egnv2/exemplar_builder.py \
    --patches_dir ./data_new_3ST/patch_noov_spilt/HYZ15040_noov_split/train_patches \
    --labels_csv ./data_new_3ST/ssGSEA_zscore/HYZ15040_ssGSEA_zscore.csv \
    --output ./egnv2/exemplar_libs/HYZ15040_exemplars.pth
```

**3.2 验证代表库质量**
- 检查代表样本分布是否覆盖各区域
- 验证特征空间聚类效果

### Step 4: 训练（预计 3-5 天）

**4.1 单患者训练**
```bash
# HYZ15040
python egnv2/train.py --dataset_name HYZ15040 --n_exemplars 2048

# JFX0729
python egnv2/train.py --dataset_name JFX0729 --n_exemplars 4096

# LMZ12939
python egnv2/train.py --dataset_name LMZ12939 --n_exemplars 4096
```

**4.2 联合训练**
```bash
python egnv2/train.py --multi_patient \
    --patient_dirs ./data_new_3ST/patch_noov_spilt/HYZ15040_noov_split/train_patches \
                   ./data_new_3ST/patch_noov_spilt/JFX0729_noov_split/train_patches \
                   ./data_new_3ST/patch_noov_spilt/LMZ12939_noov_split/train_patches \
    --patient_csvs ./data_new_3ST/ssGSEA_zscore/HYZ15040_ssGSEA_zscore.csv \
                  ./data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore.csv \
                  ./data_new_3ST/ssGSEA_zscore/LMZ12939_ssGSEA_zscore.csv
```

**【用户确认点 2】** 训练顺序偏好？建议：HYZ → JFX → LMZ → 联合

### Step 5: 评估与对比（预计 1 天）

**5.1 推理生成 predictions.csv**
```bash
python egnv2/infer.py --dataset_name HYZ15040 \
    --checkpoint ./egnv2/checkpoints/HYZ15040/best_egnv2.pth
```

**5.2 生成可视化报告**
```bash
python visualize_results.py \
    --model_name "EGNv2_HYZ15040" \
    --history_csv egnv2/training_history_HYZ15040.csv \
    --predictions_csv egnv2/checkpoints/HYZ15040/predictions.csv \
    --output_dir egnv2/results_vis \
    --prefix HYZ15040
```

**5.3 与 HisToGene 对比分析**
- 汇总各数据集 Val PCC 对比
- 逐通路 PCC 差异分析
- 可视化结果并排对比

---

## 5. 风险评估与应对

| 风险 | 影响 | 概率 | 应对策略 |
|------|------|------|----------|
| **PyG 安装失败** | 阻塞 | 中 | 提供 CPU 版本备选；使用 conda 安装 |
| **显存不足** | 阻塞 | 中 | 减小 batch_size；使用梯度累积 |
| **图构建过慢** | 延迟 | 高 | 预计算并缓存图结构；使用 FAISS 加速 KNN |
| **代表库效果差** | 性能下降 | 中 | 调整 n_exemplars；尝试不同聚类策略 |
| **过拟合仍严重** | 性能不达预期 | 中 | 增加 dropout；早停策略调优；数据增强 |
| **与 HisToGene 持平/更差** | 投入产出比低 | 中 | 设定明确的止损条件（如联合训练 PCC<0.50 则暂停） |
| **空间图构建错误** | 预测异常 | 低 | 可视化图结构验证；单元测试覆盖 |

---

## 6. 时间规划

| 阶段 | 预计时间 | 产出 | 依赖 |
|------|----------|------|------|
| **Step 1: 环境准备** | 0.5 天 | 安装 PyG，验证 GPU | 无 |
| **Step 2: 代码适配** | 2 天 | egnv2/ 目录完整代码 | Step 1 |
| **Step 3: 代表库构建** | 0.5 天 | 3 个数据集的代表库缓存 | Step 2 |
| **Step 4: 训练** | 3-5 天 | 训练好的模型 + 训练历史 | Step 3 |
| **Step 5: 评估对比** | 1 天 | 可视化报告 + 对比分析 | Step 4 |
| **总计** | **7-9 天** | - | - |

**里程碑检查点：**
- **M1（第 2.5 天）**：HYZ15040 训练完成，验证基础功能正常
- **M2（第 5 天）**：3 个单患者训练完成，评估是否继续联合训练
- **M3（第 7 天）**：联合训练完成，生成最终对比报告

---

## 7. 决策点汇总

| 序号 | 决策项 | 选项 | 推荐 |
|------|--------|------|------|
| 1 | 图构建策略 | A. KNN 图 / B. 空间邻接图 | B（空间邻接图） |
| 2 | 代表库大小（HYZ） | A. 1024 / B. 2048 / C. 全量 | B（2048） |
| 3 | 代表库构建策略 | A. K-means / B. 随机采样 / C. 全量 | A（K-means） |
| 4 | 训练顺序 | A. 单患者→联合 / B. 联合优先 | A（单患者优先） |
| 5 | 止损条件 | 联合训练 Val PCC < ? | 0.50 |

---

## 附录：参考资源

- **EGNv2 论文**：Yan Yang et al., "Spatial transcriptomics analysis of gene expression prediction using exemplar guided graph neural network", Pattern Recognition 2024
- **GitHub 仓库**：https://github.com/Yan98/EGN
- **HisToGene 基线**：见 `histogene/training_history_*.csv`
- **PFMval 项目配置**：`config.yaml`

---

*文档版本：v1.0*
*生成时间：2026-04-20*
*作者：AI Assistant*
