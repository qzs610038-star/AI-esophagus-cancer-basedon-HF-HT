# [文献卡片] LIT-01-001: Hist2ST 空间转录组预测网络

- **文献标题**: Hist2ST: Spatial transcriptomics prediction from histology images
- **作者/机构**: Zeng et al., 中山大学 / 华南理工大学
- **发表期刊/年份**: Nature Machine Intelligence (2023)
- **文件路径**: [Hist2ST.pdf](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/Ai%E7%97%85%E7%90%86%E9%A1%B9%E7%9B%AE%E6%96%87%E7%8C%AE%E6%B1%87%E6%80%BB/01_HE映射空间转录组模型/Hist2ST.pdf)
- **开源代码**: https://github.com/diannaa/Hist2ST

---

## 1. 核心技术痛点与研究动机
传统方法（如 ST-Net, DeepST）通常将 Whole Slide Image (WSI) 切分为孤立的 HE Patch/Tile，单独拟合对应位置的基因表达。这种做法忽略了**组织微环境中的细胞间通信与空间邻域依赖（Spatial Neighborhood Continuity）**。Hist2ST 旨在结合 Patch 图像特征与 2D 物理空间坐标，显式建模跨 Spot 空间上下文。

## 2. 网络架构与核心机制（内容严格校验核对）
- **输入维度**: 
  - HE Patch/Tile 特征阵列 $\mathbf{X} \in \mathbb{R}^{N \times D_{img}}$ ($N$ 为单张切片 Spot 总数，Visium 标准下约为几千个 spot)。
  - 2D Spot 物理坐标阵列 $\mathbf{P} \in \mathbb{R}^{N \times 2}$。
- **特征提取 Backbone**: CNN (ResNet/Swin) 提取单 Tile 局部表型特征。
- **空间邻域建模模块**:
  1. **空间位置编码 (Spatial Embedding)**: 利用 2D 相对物理坐标嵌入。
  2. **Self-Attention & Graph Learning**: 构建 KNN 空间拓扑图，利用 Transformer / GCN 在邻域 Spot 之间传递自注意力信息。
- **预测头与 Loss 函数**:
  - **Zero-Inflated Negative Binomial (ZINB) Loss** 或 MSE + Cosine Similarity Loss，解决单细胞/Visium 表达数据的单细胞稀疏性与 dropout 现象。

## 3. 对 PFMval Phase 2 MPP 预测的借鉴与反向优化价值
- **核心契合点**: Phase 2 现行单 Tile MLP 忽略了邻域 Spot 的空间连续性。引入 Hist2ST 的 **Spatial-Transformer 邻域聚合模块** 能显著平滑空间噪音，提升 PCC。
- **需要踩坑与改进的地方**: Visium Spot 是 100μm，而本食管癌项目如果使用更高分辨率（如 Visium HD / 50μm）或局部大阵列，全图 Self-Attention 的 $O(N^2)$ 计算复杂度极高，需要换为 **局部滑动窗口 3x3/5x5 卷积注意力或 GCN 邻域图传递**。
