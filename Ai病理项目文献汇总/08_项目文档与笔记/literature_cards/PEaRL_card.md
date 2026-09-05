# [文献卡片] LIT-01-003: PEaRL 通路增强表达预测网络

- **文献标题**: PEaRL: Pathway-Enhanced Representation Learning for Gene and Pathway Expression Prediction
- **作者/机构**: Majumder et al., WACV 2026
- **发表期刊/年份**: WACV (2026)
- **文件路径**: [Majumder_PEaRL.pdf](file:///D:/AI%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%97%85%E7%90%86%E7%A0%94%E7%A9%B6/PFMval_new/Ai%E7%97%85%E7%90%86%E9%A1%B9%E7%9B%AE%E6%96%87%E7%8C%AE%E6%B1%87%E6%80%BB/01_HE映射空间转录组模型/Majumder_PEaRL_Pathway-Enhanced_Representation_Learning_for_Gene_and_Pathway_Expression_Prediction_WACV_2026_paper.pdf)

---

## 1. 核心技术痛点与研究动机
在从 HE 图像预测高维 spatial RNA-seq 时，常规模型通常直接拟合几千个单基因（Gene-level）。然而：
1. **单基因高噪声与稀疏性**: 单个基因在切片上的表达易受测序噪声干扰，缺乏稳定性；
2. **生物学可解释性弱**: 直接回归数千维标量难以反映生物学功能模块（如 p53, EMT, 免疫浸润等通路）。
PEaRL 引入 **Pathway-Enhanced (通路增强)** 机制，使用先验通路知识辅助约束底层图像特征空间。

## 2. 网络架构与核心机制（内容严格校验核对）
- **先验矩阵引入**: 提取 MSigDB / KEGG 通路数据库，构建二进制/权重先验矩阵 $\mathbf{W}_{pathway} \in \mathbb{R}^{G \times K}$ ($G$ 个基因，$K$ 个通路)。
- **多任务联合预测头 (Multi-Task Prediction Heads)**:
  - **主分支 (Gene Head)**: 预测高维基因表达向量 $\mathbf{\hat{y}}_{gene} \in \mathbb{R}^{G}$。
  - **辅助分支 (Pathway Head)**: 预测低维通路得分向量 $\mathbf{\hat{y}}_{pathway} \in \mathbb{R}^{K}$。
- **联合损失函数**:
  $$\mathcal{L}_{PEaRL} = \mathcal{L}_{MSE}(\mathbf{\hat{y}}_{gene}, \mathbf{y}_{gene}) + \alpha \cdot \mathcal{L}_{MSE}(\mathbf{\hat{y}}_{pathway}, \mathbf{y}_{pathway}) + \beta \cdot \mathcal{L}_{consistency}$$
  通过生物学通路一致性损失，强迫特征空间在低维生物学流形上收敛。

## 3. 对 PFMval Phase 2 MPP 预测的借鉴与反向优化价值
- **核心契合点**: 食管癌预后与响应高度依赖关键通路（如 p53, Wnt, Notch, 免疫微环境）。Phase 2 MPP 正处于单基因预测 PCC 偏低、raw $R^2$ 为负的困境。引入 **PEaRL 通路物理约束** 能为神经网络引入强强的先验正则化，大幅提升模型的泛化性与 $R^2$ 指标。
- **创新亮点**: 论文发表中可直接作为“生物学先验导向的多任务增强网络”的重大创新。
