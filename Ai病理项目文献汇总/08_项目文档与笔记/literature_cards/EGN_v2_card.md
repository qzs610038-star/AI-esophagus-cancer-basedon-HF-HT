# [文献卡片] LIT-01-002: EGN / EGN-v2 原型引导网络

- **文献标题**: Exemplar Guided Deep Neural Network for Spatial Transcriptomics Analysis (WACV 2023 / PR 2026)
- **作者/机构**: Yang et al.
- **发表期刊/年份**: WACV 2023 / Pattern Recognition 2026 (EGN-v2)
- **文件路径与本地源码**: 
  - PDF: [EGN Yang WACV 2023.pdf](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/Ai%E7%97%85%E7%90%86%E9%A1%B9%E7%9B%AE%E6%96%87%E7%8C%AE%E6%B1%87%E6%80%BB/01_HE映射空间转录组模型/EGN%20Yang_Exemplar_Guided_Deep_Neural_Network_for_Spatial_Transcriptomics_Analysis_of_WACV_2023_paper.pdf)
  - 本地 PyTorch 源码: `Ai病理项目文献汇总/09_EGN-v2补充材料/EGN-v2/`

---

## 1. 核心技术痛点与研究动机
在临床应用中，直接黑盒预测基因表达难以被病理医生信任。同时，病理图像的表型异质性极高，传统 MLP 模型缺乏在有限切片样本下进行**跨切片表型对齐与原型泛化**的能力。EGN 提出利用 **Exemplar (典型范例/原型)** 词典来引导和约束网络预测。

## 2. 网络架构与核心机制（内容严格校验核对）
- **双分支架构 (Dual-Branch Architecture)**:
  1. **局部图像特征编码器 (Query Encoder)**: 提取目标 HE Tile 的深层特征 $f_{query}$。
  2. **Exemplar 原型匹配分支 (Exemplar Guidance Branch)**: 维持一个包含代表性病理形态与基因表达配对的原型库 $\mathcal{B} = \{(e_{img}^k, e_{exp}^k)\}_{k=1}^K$。
- **交叉注意力与相似度打分 (Cross-Attention & Similarity Weighting)**:
  - 计算 Query 特征与每个 Exemplar 的余弦相似度: $w_k = \text{Softmax}(\text{CosSim}(f_{query}, e_{img}^k) / \tau)$。
  - 将 Exemplar 表达特征 $e_{exp}^k$ 按权重 $w_k$ 加权融合，作为基准指导输出。
- **损失设计**:
  - 重构 Loss (MSE/PCC) + Exemplar 特征对齐 Loss (Contrastive Loss)。

## 3. 对 PFMval Phase 2 MPP 预测的借鉴与反向优化价值
- **核心契合点**: 食管癌存在明确的表型亚型（如基底样型、分化良好型、免疫浸润型等）。引入 EGN 的 **Exemplar 原型引导机制**，不仅能大幅提升少量样本下的回归稳定度，还能向临床提供直观的相似度诊断依据（如“本区域与标准食管癌免疫高表达范例相似度为 88%”）。
- **工程优势**: `09_EGN-v2补充材料/` 内部已包含完整的 PyTorch 核心模块，迁移成本极低。
