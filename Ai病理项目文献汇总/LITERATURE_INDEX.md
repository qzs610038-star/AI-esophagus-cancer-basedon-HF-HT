# 📚 Ai病理项目文献汇总 - 全量结构化目录索引 (LITERATURE_INDEX)

> **版本号**: v1.2  
> **更新时间**: 2026-08-03  
> **根路径**: `D:\AI空间转录病理研究\PFMval_new\Ai病理项目文献汇总`  
> **检索说明**: 本索引专为人类研究员及 AI Agent 设计。包含 11 个子目录分类、80+ 篇文献/笔记、98个即插即用AI模块及相关开源代码库的结构化导览。

---

## 目录分类映射与检索标签 (Categories Overview)

| 目录编号 | 目录名称 | 文件数/规模 | 核心定位与使用场景 | Agent 检索 Tag 关键词 |
|:--:|---|:--:|---|---|
| **01** | [01_HE映射空间转录组模型](./01_HE映射空间转录组模型) | 13 | **算法核心库**：HE图像预测基因/通路表达SOTA模型 | `SpatialTranscriptomics`, `HE2ST`, `Hist2ST`, `EGN`, `PEaRL`, `FmH2ST`, `GHIST` |
| **02** | [02_病理基础模型_FoundationModels](./02_病理基础模型_FoundationModels) | 5 | **Teacher Backbone 库**：固定 UNI2-h 为特征提取基模 | `FoundationModel`, `UNI2h`, `TeacherModel`, `Prov-GigaPath`, `Virchow`, `CTransPath` |
| **03** | [03_多实例学习MIL与WSI分类](./03_多实例学习MIL与WSI分类) | 5 | **WSI聚合算法**：弱监督病理分类与生存分析 | `MIL`, `WeaklySupervised`, `CLAM`, `TransMIL`, `ABMIL`, `DSMIL` |
| **04** | [04_空间转录组方法与工具](./04_空间转录组方法与工具) | 8 | **生物技术与分析工具**：Visium, Single-Cell, Deconvolution | `VisiumHD`, `cell2location`, `Seurat`, `UCell`, `Deconvolution` |
| **05** | [05_多模态融合与预后预测](./05_多模态融合与预后预测) | 4 | **多模态与预后**：病理+基因组跨模态生存预测 | `Multimodal`, `PathomicFusion`, `HE2RNA`, `Survival` |
| **06** | [06_食管癌流行病学与临床背景](./06_食管癌流行病学与临床背景) | 3 | **临床背景**：食管鳞癌(ESCC)与腺癌(EAC)临床统计 | `EsophagealCancer`, `ESCC`, `GLOBOCAN`, `Epidemiology` |
| **07** | [07_图像预处理与工作流工具](./07_图像预处理与工作流工具) | 2 | **图像预处理**：染色归一化与图像配准 | `Preprocessing`, `ColorNormalization`, `Macenko`, `QuPath` |
| **08** | [08_项目文档与笔记](./08_项目文档与笔记) | 19+ | **项目知识库**：摘要汇编(JSON/Word)、报告与文献卡片 | `Abstracts`, `ProjectNotes`, `BibTeX`, `LiteratureCards` |
| **09** | [09_EGN-v2补充材料](./09_EGN-v2补充材料) | 8 | **代码与复现资产**：EGN-v2 PyTorch源码与PR补充材料 | `EGNv2`, `SourceCode`, `ExemplarGuided`, `PyTorch` |
| **10** | [10_未使用文献](./10_未使用文献) | 9 | **备选文献库**：拓展延伸与备选文献 | `Backup`, `Auxiliary` |
| **11** | [98个即插即用的AI模块-阿飞](./98个即插即用的AI模块-阿飞) | 98 模块 | **即插即用算子库**：Mamba/CNN/Graph/Tabular/Attention轻量扩展组件 | `PlugAndPlay`, `MambaIR`, `ConvSSM`, `MambaTab`, `AKConv`, `Attention` |

---

## 详细文件清单与元数据索引 (File Registry)

### 01_HE映射空间转录组模型 (13 篇)
- `Hist2ST.pdf`: [Hist2ST: 结合 Transformer 与 Graph 邻域建模预测空间转录组] (*Nature Machine Intelligence 2023*)
- `EGN Yang_Exemplar_Guided_Deep_Neural_Network_for_Spatial_Transcriptomics_Analysis_of_WACV_2023_paper.pdf`: [EGN: 基于 Exemplar 引导的双分支可解释空间转录组预测] (*WACV 2023*)
- `Majumder_PEaRL_Pathway-Enhanced_Representation_Learning_for_Gene_and_Pathway_Expression_Prediction_WACV_2026_paper.pdf`: [PEaRL: 通路增强表示学习与多任务预测] (*WACV 2026*)
- `FmH2ST foundation model-based spatial transcriptomics.pdf`: [FmH2ST: 基于病理基础模型 (UNI) Adapter 微调的空间表达映射] (*2024*)
- `AI-Driven Spatial Transcriptomics Unlocks Large-Scale Breast.pdf`: [Path2Space: DINOv2 Backbone + MIL 端到端预后与空间映射] (*2025 bioRxiv*)

### 02_病理基础模型_FoundationModels (固定教师基模 UNI2-h)
- `[12]_Towards a general-purpose foundation model for computational pathology.pdf`: [UNI / UNI2-h: 项目指定固定教师模型 (Teacher Model) 用于 H&E 特征预提取] (*Nature Medicine 2024*)
- `[15]_A whole-slide foundation model for digital pathology from real-world data.pdf`: [Prov-GigaPath: 17万 WSI 预训练整张切片 Foundation Model] (*Nature 2024*)

### 03_多实例学习MIL与WSI分类 (Phase 3 CLAM)
- `[7]_Attention-based deep multiple instance learning.pdf`: [Attention-based MIL：CLAM 注意力 bag 聚合的直接方法基础] (*ICML 2018*)
- `[8]_Clinical-grade computational pathology using weakly supervised deep learning on whole slide images.pdf`: [大规模弱监督 WSI 分类背景] (*Nature Medicine 2019*)
- `[9]_Data-efficient and weakly supervised computational pathology on whole-slide images.pdf`: [CLAM 原论文与本次 Phase 3 主模型直接依据] (*Nature Biomedical Engineering 2021*)
- `[11]_TransMIL Transformer based correlated multiple instance learning for whole slide image classification.pdf`: [空间/相关性 MIL 邻近对照；当前基线包未实现] (*NeurIPS 2021*)
- `Pancancer outcome prediction via a unified weakly supervised.pdf`: [PROGPATH 病理-临床多模态预后邻近工作；终点不同] (*Signal Transduction and Targeted Therapy 2025*)
- Phase 3 CLAM 逐篇关系、DOI 和 metadata-only 候选见 [PHASE3_CLAM_LITERATURE_REGISTRY_20260803.md](./PHASE3_CLAM_LITERATURE_REGISTRY_20260803.md)。

### 05_多模态融合与预后预测 (Phase 3 融合背景)
- `[6]_Pathomic fusion an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis.pdf`: [Pathomic Fusion：病理-组学融合背景；当前代码不是原模型逐式复现] (*IEEE TMI*)
- Cross-attention、External Attention、Low-rank Fusion 与 FiLM 的 metadata-only 条目已登记到 Phase 3 CLAM 专表及 `literature_manifest.json`。

### 11_98个即插即用的AI模块-阿飞 (98 算子库)
- `98个即插即用模块_完整版.md`: [包含了 Mamba/SSM 空间序列模块、卷积增强模块 AKConv/ACNet、表格数据分析 MambaTab、注意力网络组件等 98 个可复用算子]
- **针对 MPP Phase 2 的核心推荐算子**:
  - `ConvSSM` / `MambaIR`: 用于 2D 物理空间长距离邻域建模。
  - `MambaTab`: 用于高维通路与表格特征流形拟合。
  - `AKConv`: 不规则形状邻域卷积，适应真实切片非规则 Spot 边缘。

---

## Agent 使用本目录指南 (Agent Quickstart)
1. **查某算法原理/结构**：直接阅读 `Ai病理项目文献汇总/08_项目文档与笔记/literature_cards/` 中对应的文献卡片。
2. **查即插即用轻量模块**：检索 `98个即插即用的AI模块-阿飞/98个即插即用模块_完整版.md`。
3. **针对 Phase 2 MPP 预测改造**：直接参阅 `Ai病理项目文献汇总/Phase2_MPP预测改进模块选型与风险评估说明书.md`。
