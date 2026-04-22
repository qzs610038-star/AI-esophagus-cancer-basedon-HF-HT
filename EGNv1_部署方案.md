# EGN-v1 模型部署方案

> **版本澄清**：EGN-v1 采用 ViT-Large（大模型，86M参数），EGN-v2/EGGN 改用 ResNet-50（轻量化，23M参数）。v2 是 v1 的轻量化改进版本。

## 1. EGN-v1 模型简介

### 1.1 论文信息

**标题**：Exemplar Guided Deep Neural Network for Spatial Transcriptomics Analysis of Gene Expression Prediction

**作者**：Yan Yang, Md Zakir Hossain, Eric A. Stone, Shafin Rahman

**发表**：IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023

**论文链接**：https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Exemplar_Guided_Deep_Neural_Network_for_Spatial_Transcriptomics_Analysis_of_WACV_2023_paper.pdf

**ArXiv**：https://arxiv.org/abs/2210.16721

**GitHub 仓库**：https://github.com/Yan98/EGN

### 1.2 核心思想和创新点

EGN-v1（Exemplar Guided Network）提出了一个创新的范式来直接从组织切片图像预测基因表达。核心创新包括：

#### **创新一：代表库学习机制**

- 从训练集中自动选取代表性样本（exemplar），构建代表库
- 推理时，通过与代表库中最相似样本的检索，获取其已知的基因表达值
- 代表库作为一种 记忆机制，避免了从零训练的必要

#### **创新二：图卷积网络建模空间关系**

- 将每个 patch 视为图的节点
- 通过特征相似性构建 KNN 图，建立节点间的连接关系
- 使用 GCN（图卷积网络）聚合邻近 patch 的信息

#### **创新三：三层融合架构**

组合方式：图像特征提取 → 代表库检索 → 图卷积聚合 → 预测

---

## 2. EGN-v1 vs EGNv2 架构对比

### 2.1 核心架构对比表

| **维度** | **EGN-v1（WACV 2023）** | **EGNv2/EGGN（PR 2024）** | **改进说明** |
|---------|-------------------------|--------------------------|------------|
| 特征提取器 | **ViT-Large**（~86M参数，dim=1024, depth=8, heads=16） | ResNet-50（~23M参数） | v2 参数**减少** 73%，更轻量高效 |
| 骨干 GNN | GCN（图卷积网络，转导学习） | GraphSAGE（采样聚合，归纳学习） | v2 支持小批量归纳学习，推理更灵活 |
| 图构建方式 | KNN 图（基于特征相似性） | 空间半径图（基于坐标距离） | v2 更符合病理组织空间连续性 |
| 代表库大小 | 固定（通常全量） | 灵活（支持 K-means 聚类） | v2 可根据数据量动态调整 |
| Exemplar 机制 | Exemplar Bridging 块，嵌入 ViT 中间层 | K-means 聚类 + 相似性加权融合 | v2 融合方式更复杂 |
| 残差连接 | 无 | 有（图卷积层间） | v2 梯度传播更畅通 |
| 归一化层 | BatchNorm | LayerNorm | v2 对小批量不敏感 |
| 参数总量 | ~86M（主要来自 ViT） | ~23M（主要来自 ResNet-50） | v2 模型容量减少 73% |

### 2.2 EGN-v1 的潜在优势和劣势

#### **潜在优势**

1. **ViT 特征表达力强**：86M 参数的 ViT-Large 理论上具有更强的特征提取能力，可能捕捉更丰富的病理学语义
2. **KNN 图的功能关联性**：基于特征相似性构建的 KNN 图可捕捉功能相关的 patch，补足空间图的不足
3. **全量代表库**：小数据集上包含所有训练信息，无信息损失

#### **潜在劣势**

1. **参数量过大导致过拟合风险**：86M vs 23M，在当前小数据集（2K-7K样本）上过拟合风险显著更高
2. **GCN 的转导学习限制**：转导学习要求推理时图结构固定，不适合多患者动态场景
3. **ViT 计算开销大**：86M 参数带来更高的训练和推理计算成本
4. **KNN 图依赖高质量特征**：若 ViT 特征在病理域迁移不佳，会导致错误的邻接关系

---

## 3. EGN-v1 与当前数据的适配性分析

### 3.1 预期性能

基于 EGNv2 的基线（PCC 0.4245），以及 v1 在参数和架构上的差异：

| 数据集 | EGNv2 PCC | EGN-v1 预期 | 置信度 | 理由 |
|--------|-----------|-----------|--------|------|
| HYZ15040 | 0.4048 | 0.38~0.42 | 中 | 86M参数在小数据集上过拟合风险高 |
| JFX0729 | 0.4445 | 0.40~0.44 | 中 | 数据量较大，ViT特征优势可能部分显现 |
| LMZ12939 | 0.3837 | 0.35~0.39 | 低 | 样本最少，过拟合风险最高 |
| 联合训练 | 0.4245 | 0.38~0.42 | 低 | GCN转导学习限制多患者场景 |

**保守结论**：EGN-v1 在当前数据上**整体不会优于 EGNv2**。v1 参数量比 v2 多 3.7 倍，在小数据集（2K-7K样本）上过拟合风险显著更高；v1 的 GCN 转导学习不如 v2 的 GraphSAGE 归纳学习适合多患者联合训练。预期 v1 在单训上可能接近或略低于 v2，联合训练可能明显不如 v2。

### 3.2 v1 不推荐部署的核心原因

1. **参数量过大导致严重过拟合风险**：86M 参数 vs 23M 参数，在当前小数据集（2K-7K样本）上，v1 的过拟合风险显著高于 v2
2. **GCN 转导学习限制多患者场景**：v1 的 GCN 要求推理时图结构固定，不适合多患者动态联合训练场景
3. **ViT 计算开销大**：86M 参数带来更高的训练和推理成本，投入产出比低
4. **ViT 在病理域的迁移效果不确定**：v1 的 ViT 虽参数多，但在病理图像上的特征质量未必优于专门调优的 ResNet-50

> **但也要指出**：v1 的 ViT 特征提取思路本身有参考价值，若能解决过拟合问题（如使用预训练 ViT、增加正则化），可能值得探索。

---

## 4. 部署方案推荐

### 4.1 核心建议

**❌ 不推荐单独部署 EGN-v1**

理由：
- **参数量过大**：86M vs 23M，在当前小数据集上过拟合风险显著更高
- **GCN 转导学习限制**：不适合多患者动态联合训练场景
- **计算开销高**：ViT 带来更高的训练和推理成本，投入产出比低
- **v2 是 v1 的改进版**：v2 通过轻量化设计（ResNet-50 + GraphSAGE）解决了 v1 的核心问题

> 若要探索 ViT 特征提取的潜力，建议基于 EGNv2 框架升级特征提取器（如 UNI、DINO-v2），而非回退到 v1 架构。

### 4.2 推荐方案 A：EGNv2 + 特征提取器升级

**优先级最高**

升级路线：ResNet-50 (ImageNet) → UNI / CONCH / DINO-v2 (病理专用)

预期效果：
- 联合训练 PCC：0.4245 → 0.48~0.52
- 预期提升：+0.05~0.10

### 4.3 推荐方案 B：EGNv2 + 混合图构建

**优先级次高**

在 EGNv2 基础上，同时构建空间图和 KNN 特征图：

空间半径图（基于坐标）+ KNN 特征图（基于特征）= 混合图

预期效果：
- 额外提升：+0.01~0.03
- 特别有利于：免疫通路

---

## 5. 如果决定实验 EGN-v1 的实施步骤

**前置条件**：仅在 EGNv2 + UNI 特征升级达不到预期（PCC < 0.48）时，才考虑尝试 EGN-v1。

### 5.1 代码获取和新建目录结构

`
egnv1/                          # 新建目录
├── __init__.py
├── model.py                    # EGNv1 模型定义（GCN 替代 GraphSAGE）
├── dataset.py                  # 数据加载（复用 EGNv2 大部分）
├── exemplar_builder.py         # 代表库构建（复用 EGNv2 逻辑）
├── graph_builder.py            # KNN 图构建（新增）
├── train.py                    # 训练脚本（基于 EGNv2 改造）
├── infer.py                    # 推理脚本
├── utils.py                    # 工具函数
├── checkpoints/                # 模型保存
└── results_vis/                # 可视化结果
`

### 5.2 关键参数建议

| 参数 | EGN-v1 建议值 | 说明 |
|-----|-------------|------|
| backbone | ViT-Large | 官方参数：dim=1024, depth=8, heads=16, mlp_dim=4096, patch_size=32 |
| graph_layers | 2 | 保持同 v2 |
| hidden_dim | 1024 | 匹配 ViT 输出维度 |
| k_exemplars | 10 | 保持同 v2 |
| graph_type | knn（纯 KNN） 或 hybrid（混合） | 试验对比 |
| batch_size | 16 | 相比 v2 的 64 大幅降低（ViT 显存占用大） |
| dropout | 0.5 | 相比 v2 的 0.3 显著提高（抑制过拟合） |
| learning_rate | 1e-5 | 更保守（大模型需更稳定优化） |
| weight_decay | 1e-4 | 增加正则化抑制过拟合 |

---

## 6. 成本-收益分析

### 6.1 EGN-v1 部署的投入产出比

| 投入项 | 成本 | 评估 |
|--------|------|------|
| 环境搭建 | 0.5 天 | 快 |
| 代码适配 | 1.5~2 天 | 中等（主要改 GCN 部分） |
| 代表库构建 | 0.5 天 | 快 |
| 单模型训练 | 2~3 天 | 中等 |
| 评估对比 | 1 天 | 快 |
| **总计** | **5.5~7 天** | - |

### 6.2 预期收益

- **最乐观**：EGN-v1 PCC = 0.44，投入 7 天换 +0.01 PCC → 投入产出比低
- **现实**：EGN-v1 PCC = 0.40，投入 7 天换 -0.02 PCC → 负收益
- **对比 UNI 升级**：投入 3~4 天，预期 PCC +0.05~0.10 → 投入产出比高

**结论**：如果时间有限（< 15 天），应优先做 EGNv2 + UNI，而不是 EGN-v1。

---

## 7. 总体推荐

### 推荐优化路线（按优先级）

1. **第一步（3~4 天）**：EGNv2 + UNI 病理基础模型特征提取器升级
   - 预期联合 PCC：0.4245 → 0.48~0.52

2. **第二步（2~3 天）**：EGNv2 + 混合图构建（空间图 + KNN 功能图）
   - 预期联合 PCC：+0.01~0.03，累计 0.50~0.55

3. **第三步（1~2 天）**：超参数微调 + 早停策略优化
   - 预期联合 PCC：+0.005~0.01，累计 0.51~0.56

### EGN-v1 的应用场景

- 作为学术参考，理解代表学习的演进
- 作为混合图构建的灵感来源（参考 KNN 图构造）
- 仅在所有 EGNv2 优化都失效后尝试

---

## 8. 参考资源

### 论文和代码

| 资源 | 链接 |
|-----|------|
| EGN-v1 原文 | https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Exemplar_Guided_Deep_Neural_Network_for_Spatial_Transcriptomics_Analysis_of_WACV_2023_paper.pdf |
| EGN-v1 ArXiv | https://arxiv.org/abs/2210.16721 |
| EGNv2 论文 | https://www.sciencedirect.com/science/article/pii/S0031320324000584 |
| GitHub 仓库 | https://github.com/Yan98/EGN |

### 相关技术文档

- PyTorch Geometric GCN：https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.GCNConv
- PyTorch Geometric GraphSAGE：https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.SAGEConv
- UNI 病理基础模型：https://github.com/mahmoodlab/UNI

---

*文档版本：v1.0*
*生成时间：2026-04-20*
*作者：AI 研究分析员*
*适用项目：PFMval_new 空间转录病理图像分析*

