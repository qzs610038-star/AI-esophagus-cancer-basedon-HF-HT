# Phase 3 CLAM 关联文献注册表（2026-08-03）

> 适用范围：`reference_implementations/phase3_clam_escc_20260801/`。本表登记“代码直接依据、代码内候选、邻近对照”三种关系；登记不代表本项目已复现实验结果。

## A. 直接依据

| ID | 文献/资源 | 本地文件 | 与基线的关系 | 状态 |
|---|---|---|---|---|
| LIT-03-001 | Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018 | `03_多实例学习MIL与WSI分类/[7]_Attention-based deep multiple instance learning.pdf` | gated attention MIL 基础 | local_pdf_verified |
| LIT-03-003 | Lu et al. Data-efficient and weakly supervised computational pathology on whole-slide images. 2021. DOI 10.1038/s41551-020-00682-w | `03_多实例学习MIL与WSI分类/[9]_Data-efficient and weakly supervised computational pathology on whole-slide images.pdf` | CLAM 主模型与上游代码直接依据 | local_pdf_verified |
| LIT-02-002 | Chen et al. Towards a general-purpose foundation model for computational pathology. 2024. DOI 10.1038/s41591-024-02857-3 | `02_病理基础模型_FoundationModels/[12]_Towards a general-purpose foundation model for computational pathology.pdf` | UNI 家族论文背景 | local_pdf_verified |
| LIT-02-003 | MahmoodLab UNI2-h model card | 无本地 PDF | 当前 1536 维 UNI2-h 特征架构、用法与许可证权威来源 | metadata_only |
| LIT-03-006 | Liu et al. Pseudo-Bag Mixup Augmentation for MIL-based WSI Classification | 无本地 PDF | `SameLabelPseMix` 候选增强来源；主方案未启用 | metadata_only |
| LIT-03-007 | Berrada et al. Smooth Loss Functions for Deep Top-k Classification. ICLR 2018 | 无本地 PDF | `smooth-topk` 环境依赖；README 主命令使用 CE 时不激活 | metadata_only |

## B. 代码内融合候选的来源/背景

| ID | 文献/资源 | 本地文件 | 与基线的关系 | 状态 |
|---|---|---|---|---|
| LIT-05-001 | Chen et al. Pathomic Fusion. DOI 10.1109/TMI.2020.3021387 | `05_多模态融合与预后预测/[6]_Pathomic fusion an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis.pdf` | 病理-组学融合背景；当前代码不是原模型复现 | local_pdf_verified |
| LIT-05-002 | Vaswani et al. Attention Is All You Need. NeurIPS 2017 | 无本地 PDF | `nn.MultiheadAttention` 的方法背景 | metadata_only |
| LIT-05-003 | Guo et al. Beyond Self-Attention: External Attention Using Two Linear Layers for Visual Tasks. DOI 10.1109/TPAMI.2022.3211006 | 无本地 PDF | `ExternalAttention` 候选来源 | metadata_only |
| LIT-05-004 | Liu et al. Efficient Low-rank Multimodal Fusion With Modality-Specific Factors. DOI 10.18653/v1/P18-1209 | 无本地 PDF | `LowRankFusion` 候选背景；未证明为逐式复现 | metadata_only |
| LIT-05-005 | Perez et al. FiLM. DOI 10.1609/aaai.v32i1.11671 | 无本地 PDF | `film` 候选来源 | metadata_only |

## C. 邻近对照

| ID | 文献/资源 | 本地文件 | 与基线的关系 | 状态 |
|---|---|---|---|---|
| LIT-03-002 | Campanella et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. DOI 10.1038/s41591-019-0508-1 | `03_多实例学习MIL与WSI分类/[8]_Clinical-grade computational pathology using weakly supervised deep learning on whole slide images.pdf` | 大规模弱监督 WSI 临床背景 | local_pdf_verified |
| LIT-03-004 | Shao et al. TransMIL. NeurIPS 2021 | `03_多实例学习MIL与WSI分类/[11]_TransMIL Transformer based correlated multiple instance learning for whole slide image classification.pdf` | patch 相关性/空间 MIL 对照；当前包未实现 | local_pdf_verified |
| LIT-03-005 | Yuan et al. PROGPATH. DOI 10.1038/s41392-025-02374-w | `03_多实例学习MIL与WSI分类/Pancancer outcome prediction via a unified weakly supervised.pdf` | 病理-临床多模态预后邻近工作，终点不同 | local_pdf_verified |

## 官方来源

- [CLAM 论文](https://www.nature.com/articles/s41551-020-00682-w)
- [CLAM 官方代码](https://github.com/mahmoodlab/CLAM)
- [Attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html)
- [UNI 论文](https://www.nature.com/articles/s41591-024-02857-3)
- [UNI2-h 模型卡](https://huggingface.co/MahmoodLab/UNI2-h)
- [PseMix 官方代码](https://github.com/liupei101/PseMix)
- [TransMIL 论文](https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- [Low-rank Multimodal Fusion](https://aclanthology.org/P18-1209/)
- [FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671)
- [Smooth Top-k](https://openreview.net/forum?id=Hk5elxbRW)
