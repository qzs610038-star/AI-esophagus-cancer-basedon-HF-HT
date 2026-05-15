# HisToGene UNI训练结果对比分析

<cite>
**本文档引用的文件**
- [HisToGene_UNI训练结果对比分析.md](file://HisToGene_UNI训练结果对比分析.md)
- [README.md](file://README.md)
- [train_uni.py](file://histogene/train_uni.py)
- [infer_uni.py](file://histogene/infer_uni.py)
- [model_uni.py](file://histogene/model_uni.py)
- [dataset_uni.py](file://histogene/dataset_uni.py)
- [train.py](file://egnv2/train.py)
- [infer.py](file://egnv2/infer.py)
- [model.py](file://egnv2/model.py)
- [dataset.py](file://egnv2/dataset.py)
- [training_history_HYZ15040_UNI.csv](file://egnv2/checkpoints/results_vis/HYZ15040_UNI_20260424_231853/training_history_HYZ15040_UNI.csv)
- [training_history_JFX0729_UNI.csv](file://egnv2/checkpoints/results_vis/JFX0729_UNI_20260424_233219/training_history_JFX0729_UNI.csv)
- [training_history_LMZ12939_UNI.csv](file://egnv2/checkpoints/results_vis/LMZ12939_UNI_20260424_233145/training_history_LMZ12939_UNI.csv)
- [training_history_HYZ15040_UNI.csv](file://histogene/checkpoints/results_vis/HYZ15040_UNI_20260422_232743/training_history_HYZ15040_UNI.csv)
- [training_history_JFX0729.csv](file://histogene/checkpoints/results_vis/JFX0729_20260416_224437/training_history_JFX0729.csv)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构概览](#架构概览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考量](#性能考量)
8. [故障排除指南](#故障排除指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介
本报告对比分析了HisToGene UNI（方案A）与原始HisToGene（端到端ViT）两种模型变体在三个独立数据集上的训练与验证表现。研究重点关注：

- **模型架构对比**：UNI2-h预提取特征 vs 端到端ViT图像编码器
- **性能指标对比**：Val PCC、Val R²、Val Loss、训练速度等
- **训练效率分析**：参数量减少94.4%，训练速度提升45.3%
- **过拟合风险评估**：Train PCC与Val PCC差距平均增加30.9%

## 项目结构
该项目采用模块化设计，包含三个主要子系统：

```mermaid
graph TB
subgraph "HisToGene UNI系统"
A[train_uni.py] --> B[model_uni.py]
A --> C[dataset_uni.py]
D[infer_uni.py] --> B
D --> C
end
subgraph "EGNv2系统"
E[train.py] --> F[model.py]
E --> G[dataset.py]
H[infer.py] --> F
end
subgraph "数据集"
I[HYZ15040]
J[JFX0729]
K[LMZ12939]
end
A --> I
A --> J
A --> K
E --> I
E --> J
E --> K
```

**图表来源**
- [train_uni.py:1-737](file://histogene/train_uni.py#L1-L737)
- [train.py:1-675](file://egnv2/train.py#L1-L675)

**章节来源**
- [README.md:1-44](file://README.md#L1-L44)

## 核心组件

### HisToGene UNI架构组件
HisToGene UNI采用"特征提取+坐标编码+回归头"的三层架构：

```mermaid
classDiagram
class HisToGeneUNI {
+int feature_dim
+int dim
+int n_pos
+int n_targets
+int mlp_dim
+float dropout
+proj : Sequential
+x_embed : Embedding
+y_embed : Embedding
+head : Sequential
+forward(features, pos_x, pos_y)
+count_parameters()
}
class HisToGeneUNIDataset {
+str feature_cache_dir
+str patches_dir
+int n_pos
+list samples
+dict label_map
+get_coord_stats()
+__getitem__(idx)
}
HisToGeneUNI --> HisToGeneUNIDataset : "使用"
```

**图表来源**
- [model_uni.py:14-67](file://histogene/model_uni.py#L14-L67)
- [dataset_uni.py:23-203](file://histogene/dataset_uni.py#L23-L203)

### EGNv2架构组件
EGNv2采用"ResNet特征提取+图卷积+代表库融合"的多层架构：

```mermaid
classDiagram
class ResNetFeatureExtractor {
+int freeze_layers
+features : Sequential
+avgpool : AdaptiveAvgPool2d
+_freeze_layers(n)
+forward(x)
}
class ExemplarLibrary {
+Tensor features
+Tensor targets
+NearestNeighbors _nn
+retrieve(query_features, k)
+get_features(indices)
+get_targets(indices)
+save(path)
+load(path)
}
class EGNv2Model {
+int in_dim
+int hidden_dim
+int n_targets
+int graph_layers
+int k_exemplars
+feature_proj : Sequential
+graph_convs : ModuleList
+graph_norms : ModuleList
+exemplar_fuse : Sequential
+regressor : Sequential
+forward(node_features, edge_index, exemplar_agg_features)
}
ResNetFeatureExtractor --> ExemplarLibrary : "生成特征"
EGNv2Model --> ExemplarLibrary : "使用"
```

**图表来源**
- [model.py:15-211](file://egnv2/model.py#L15-L211)

**章节来源**
- [model_uni.py:1-67](file://histogene/model_uni.py#L1-L67)
- [dataset_uni.py:1-203](file://histogene/dataset_uni.py#L1-L203)
- [model.py:1-211](file://egnv2/model.py#L1-L211)

## 架构概览

### 训练流程对比

```mermaid
sequenceDiagram
participant A as "HisToGene UNI训练"
participant B as "HisToGene UNI推理"
participant C as "EGNv2训练"
participant D as "EGNv2推理"
A->>A : UNI特征缓存检查
A->>A : 数据加载(HisToGeneUNIDataset)
A->>A : 模型前向传播
A->>A : 损失计算与反向传播
A->>A : 早停与检查点保存
B->>B : 加载最佳模型
B->>B : UNI特征缓存检查
B->>B : 数据加载(HisToGeneUNIDataset)
B->>B : 预测与指标计算
C->>C : 特征提取(ResNet)
C->>C : 图构建与代表库
C->>C : 模型前向传播
C->>C : 损失计算与反向传播
C->>C : 早停与检查点保存
D->>D : 加载最佳模型
D->>D : 缓存特征加载
D->>D : 图数据构建
D->>D : 预测与指标计算
```

**图表来源**
- [train_uni.py:293-737](file://histogene/train_uni.py#L293-L737)
- [infer_uni.py:98-218](file://histogene/infer_uni.py#L98-L218)
- [train.py:226-675](file://egnv2/train.py#L226-L675)
- [infer.py:48-148](file://egnv2/infer.py#L48-L148)

## 详细组件分析

### 训练历史数据分析

#### HYZ15040数据集对比

```mermaid
flowchart TD
A["HisToGene UNI<br/>HYZ15040_UNI"] --> B["HisToGene<br/>HYZ15040"]
C["训练轮次"] --> D["PCC提升<br/>+0.0609 (11.8%)"]
E["训练速度"] --> F["加速<br/>51.4% (37→19)"]
G["验证样本"] --> H["UNI: 17 vs 原始: 265"]
B --> I["验证集过小<br/>统计可信度低"]
A --> J["特征缓存匹配问题"]
style I fill:#ffcccc
style J fill:#ffcccc
```

**图表来源**
- [training_history_HYZ15040_UNI.csv:1-21](file://histogene/checkpoints/results_vis/HYZ15040_UNI_20260422_232743/training_history_HYZ15040_UNI.csv#L1-L21)
- [training_history_JFX0729.csv:1-44](file://histogene/checkpoints/results_vis/JFX0729_20260416_224437/training_history_JFX0729.csv#L1-L44)

#### JFX0729数据集对比

```mermaid
graph LR
subgraph "HisToGene UNI<br/>JFX0729_UNI"
A1["Val PCC: 0.6114"]
A2["Val R²: 0.3742"]
A3["Val Loss: 0.2725"]
A4["训练轮次: 21"]
end
subgraph "HisToGene<br/>JFX0729"
B1["Val PCC: 0.6041"]
B2["Val R²: 0.3521"]
B3["Val Loss: 0.2782"]
B4["训练轮次: 42"]
end
A1 --> C["+0.0073 (+1.2%)"]
A2 --> D["+0.0221 (+6.3%)"]
A3 --> E["-0.0057 (-2.0%)"]
A4 --> F["加速50.0% (42→21)"]
```

**图表来源**
- [training_history_JFX0729_UNI.csv:1-122](file://egnv2/checkpoints/results_vis/JFX0729_UNI_20260424_233219/training_history_JFX0729_UNI.csv#L1-L122)
- [training_history_JFX0729.csv:1-44](file://histogene/checkpoints/results_vis/JFX0729_20260416_224437/training_history_JFX0729.csv#L1-L44)

#### LMZ12939数据集对比

```mermaid
graph TB
subgraph "HisToGene UNI<br/>LMZ12939_UNI"
L1["Val PCC: 0.5385<br/>+0.0098 (1.9%)"]
L2["Val R²: 0.2781<br/>+0.0248 (9.8%)"]
L3["Val Loss: 0.2872<br/>-0.0077 (2.6%)"]
L4["训练轮次: 21<br/>加速34.4%"]
end
subgraph "HisToGene<br/>LMZ12939"
M1["Val PCC: 0.5287"]
M2["Val R²: 0.2533"]
M3["Val Loss: 0.2949"]
M4["训练轮次: 32"]
end
style L1 fill:#ccffcc
style L2 fill:#ccffcc
style L3 fill:#ccffcc
style L4 fill:#ccffcc
```

**图表来源**
- [training_history_LMZ12939_UNI.csv:1-103](file://egnv2/checkpoints/results_vis/LMZ12939_UNI_20260424_233145/training_history_LMZ12939_UNI.csv#L1-L103)

### 关键性能指标对比

| 模型 | 数据集 | 总Epoch | 最佳Epoch | Val PCC | Val R² | Val Loss | Train PCC | 过拟合Gap | 参数量 | 训练样本 |
|------|--------|---------|-----------|---------|--------|----------|-----------|-----------|--------|----------|
| HisToGene-UNI | HYZ15040_UNI | 19 | 4 | **0.5773** | 0.2177 | **0.2587** | 0.8137 | 0.2364 | 4.0M | 2215 |
| HisToGene-UNI | JFX0729_UNI | 21 | 6 | **0.6114** | **0.3742** | **0.2725** | 0.8442 | 0.2328 | 4.0M | 7055 |
| HisToGene-UNI | LMZ12939_UNI | 21 | 6 | **0.5385** | **0.2781** | **0.2872** | 0.8288 | 0.2903 | 4.0M | 6762 |
| 原始HisToGene | HYZ15040 | 37 | 22 | 0.5164 | 0.2257 | 0.2869 | 0.7238 | 0.2074 | 70.6M | 2390 |
| 原始HisToGene | JFX0729 | 42 | 27 | 0.6041 | 0.3521 | 0.2782 | 0.7955 | 0.1914 | 70.6M | 7055 |
| 原始HisToGene | LMZ12939 | 32 | 17 | 0.5287 | 0.2533 | 0.2949 | 0.7135 | 0.1848 | 70.6M | 6762 |

**章节来源**
- [HisToGene_UNI训练结果对比分析.md:66-75](file://HisToGene_UNI训练结果对比分析.md#L66-L75)

## 依赖关系分析

### 模块间依赖关系

```mermaid
graph TD
subgraph "HisToGene UNI"
A1[train_uni.py] --> B1[model_uni.py]
A1 --> C1[dataset_uni.py]
A1 --> D1[uni2h_utils.py]
D1 --> E1[UNI2-h模型]
end
subgraph "EGNv2"
A2[train.py] --> B2[model.py]
A2 --> C2[dataset.py]
A2 --> F2[exemplar_builder.py]
F2 --> G2[ResNet特征提取]
end
subgraph "通用工具"
H1[config_utils.py]
H2[notify_utils.py]
H3[visualize_results.py]
end
A1 --> H1
A2 --> H1
A1 --> H2
A2 --> H2
A1 --> H3
A2 --> H3
```

**图表来源**
- [train_uni.py:6-31](file://histogene/train_uni.py#L6-L31)
- [train.py:7-41](file://egnv2/train.py#L7-L41)

### 数据流依赖

```mermaid
flowchart LR
subgraph "数据准备"
A["PNG图像"] --> B["坐标解析"]
B --> C["标签匹配"]
end
subgraph "HisToGene UNI"
D["UNI特征缓存(.pt)"] --> E["HisToGeneUNIDataset"]
E --> F["HisToGeneUNI模型"]
end
subgraph "EGNv2"
G["ResNet特征提取"] --> H["图构建"]
H --> I["Exemplar库"]
I --> J["EGNv2Model"]
end
C --> D
C --> G
F --> K["训练历史CSV"]
J --> K
```

**图表来源**
- [dataset_uni.py:15-139](file://histogene/dataset_uni.py#L15-L139)
- [dataset.py:17-101](file://egnv2/dataset.py#L17-L101)

**章节来源**
- [train_uni.py:1-737](file://histogene/train_uni.py#L1-L737)
- [train.py:1-675](file://egnv2/train.py#L1-L675)

## 性能考量

### 训练效率分析
- **参数量对比**：UNI版本4.0M vs 原始版本70.6M，减少94.4%
- **训练速度**：平均加速45.3%，最佳Epoch从22-27提前到4-6
- **内存占用**：UNI版本显著降低GPU内存需求
- **推理速度**：UNI版本推理速度提升约2-3倍

### 过拟合风险分析
- **Gap趋势**：Train PCC与Val PCC差距平均增加30.9%
- **数据集影响**：小样本数据集（HYZ15040）过拟合风险更高
- **正则化需求**：需要增强Dropout、权重衰减等正则化措施

### 数据质量考量
- **特征缓存一致性**：UNI特征与图像文件名匹配问题
- **验证集规模**：HYZ15040验证集仅17样本，统计意义有限
- **数据分布**：不同数据集间存在样本分布差异

## 故障排除指南

### 常见问题及解决方案

#### UNI特征缓存问题
```mermaid
flowchart TD
A["特征缓存缺失"] --> B["检查缓存目录"]
B --> C{"缓存文件存在?"}
C --> |否| D["重新提取特征"]
C --> |是| E["检查文件完整性"]
E --> F{"文件格式正确?"}
F --> |否| G["删除损坏文件"]
F --> |是| H["检查权限设置"]
```

**图表来源**
- [train_uni.py:51-79](file://histogene/train_uni.py#L51-L79)

#### 训练中断处理
- **断点续训**：支持从最佳检查点恢复训练
- **暂停信号**：检测到暂停信号时自动保存resume checkpoint
- **早停机制**：基于验证损失的早停策略

#### 推理结果验证
- **模型兼容性**：确保推理时使用的参数与训练时一致
- **数据一致性**：验证集与训练集坐标统计信息匹配
- **指标计算**：使用相同的评估指标进行结果对比

**章节来源**
- [train_uni.py:521-636](file://histogene/train_uni.py#L521-L636)
- [infer_uni.py:108-131](file://histogene/infer_uni.py#L108-L131)

## 结论

### 核心发现
1. **性能保持**：HisToGene UNI在三个数据集上的Val PCC均保持或提升，平均提升+4.9%
2. **效率显著提升**：参数量减少94.4%，训练速度平均提升45.3%
3. **过拟合挑战**：Train PCC与Val PCC差距平均增加30.9%，需要加强正则化
4. **数据集依赖**：小样本数据集（HYZ15040）结果需谨慎解读

### 推荐策略
1. **短期优化**：增强正则化、降低学习率、修复验证集样本匹配问题
2. **中期探索**：开展HisToGene-UNI三患者联合训练实验
3. **长期发展**：探索保留patch token序列的方案B/C

### 实施建议
- 优先修复HYZ15040验证集样本匹配问题
- 增大Dropout至0.3-0.5，调整权重衰减至1e-3
- 开展HisToGene-UNI联合训练实验验证泛化性能
- 探索保留patch token序列的混合方案

## 附录

### 技术规格对比

| 组件 | HisToGene UNI | 原始HisToGene | EGNv2 |
|------|---------------|---------------|-------|
| 输入类型 | 1536维UNI特征 | 224×224 RGB图像 | 224×224 RGB图像 |
| 特征提取器 | UNI2-h(冻结) | ViT(训练) | ResNet-50(可冻结) |
| 参数量 | 4.0M | 70.6M | ~25M |
| 训练速度 | 快 | 慢 | 中等 |
| 过拟合风险 | 中等 | 低 | 中等 |
| 推理速度 | 快 | 慢 | 中等 |

### 未来发展方向
1. **模型集成**：结合多种预训练模型的特征表示
2. **自监督学习**：探索无监督特征学习方法
3. **知识蒸馏**：从大型模型迁移到轻量化模型
4. **在线学习**：支持增量学习和持续更新