# SpaMIL 论文逐段拆解与 Phase 3 融合方案对比学习指南

> **创建日期**：2026-08-03  
> **适用范围**：PFMval Phase 3 ESCC pCR/MPR 疗效预测、MPP2 虚拟 ssGSEA 分数融合路线改进  
> **论文索引**：*SpaMIL: Multiple instance learning with spatial transcriptomics for interpretable patient-level predictions: application in glioblastoma* (bioRxiv, 2025)  
> **论文链接**：[DOI: 10.1101/2025.10.13.682206](https://doi.org/10.1101/2025.10.13.682206) | [bioRxiv 页面](https://www.biorxiv.org/content/10.1101/2025.10.13.682206v1) | [GitHub (owkin/SpaMIL)](https://github.com/owkin/SpaMIL)

---

## 📌 概述与前言

在 PFMval 项目 Phase 3 的最新基线探索中（参考 `reference_implementations/phase3_clam_escc_20260801`），我们在食管鳞癌（ESCC）新辅助治疗响应（pCR/MPR）预测中发现了一个关键现象：
> **仅病理图像特征（UNI2-h 1536维）与 Phase 2 模型预测出的虚拟 ssGSEA 分数（30维）直接进行特征拼接或晚期融合（如 Cross-Attention / Adaptive Fusion）时，AUC 相比纯病理图像没有显著改善，在部分 fold 上甚至出现微幅下降。**

本学习指南引入最新发表的 **SpaMIL** (Owkin, 2025) 论文，深入剖析其“多模态知识蒸馏（MabMIL）”设计思路，并结合项目 Phase 2 & Phase 3 的现状，探究我们当前面临问题的深层原因并提出下一代架构演进建议。

---

## 💡 第一章：我们实质上希望通过 ssGSEA 分数对预测结果做哪些改良？

在 Phase 3 的设计初衷中，引入 Phase 2 预测的 30 维虚拟 ssGSEA 分数主要寄希望于实现以下 3 项改良：

1. **补充高阶生物学功能与细胞微环境先验**：
   * H&E 染色病理图像（UNI2-h 特征）展现的是**组织形态学**（如细胞核大小、极性、淋巴细胞浸润程度），难以直接提取高阶分子功能。
   * 30 维 ssGSEA 分数涵盖了 DNA 修复、缺氧、免疫激活、代谢等高阶生物学通路，希望以**分子功能先验**弥补形态学特征的盲区。
2. **指导 MIL（如 CLAM）注意力机制聚焦关键区域**：
   * 一张整切片（WSI）包含数千个 Patch，其中大量属于正常基质或无应答相关的背景区域。
   * 希望分子通路得分能够协助 Attention 机制更快、更准地定位与治疗响应（pCR）高度相关的“关键肿瘤微环境 Patch”。
3. **提升模型的跨中心/跨设备泛化能力**：
   * 病理图像易受切片染色厚度、扫描仪型号及批次效应干扰。
   * 希望相对稳定的分子通路表示能够作为跨中心泛化的“锚点”，降低模型在小样本切片上的过拟合风险。

---

## 📖 第二章：SpaMIL 论文设计思路逐段拆解与关键信息提取

<details>
<summary><b>点击展开：SpaMIL 论文原文关键段落提取与逐段设计思路解析</b></summary>

### 论文元信息与资源链接
* **论文题目**：*SpaMIL: Multiple instance learning with spatial transcriptomics for interpretable patient-level predictions: application in glioblastoma*
* **预印本发布时间**： bioRxiv, 2025
* **官方资源链接**：
  * [bioRxiv 论文原文页面](https://www.biorxiv.org/content/10.1101/2025.10.13.682206v1)
  * [DOI 解析链接](https://doi.org/10.1101/2025.10.13.682206v1)
  * [Owkin 官方开源代码库 (owkin/SpaMIL)](https://github.com/owkin/SpaMIL)

---

### 1. Abstract & Introduction（摘要与研究背景段落）
* **原文关键信息**：
  > *"Spatial transcriptomics (SpT) enables spatially resolved molecular profiling, offering unprecedented resolution into tumor microenvironments. However, SpT is restricted by high cost and technical complexity, preventing widespread clinical use. Standard histology (H&E) is ubiquitous but lacks direct molecular information. Here, we present SpaMIL..."*
* **思路拆解**：
  * **临床痛点**：空间转录组（SpT）能够精确测定肿瘤微环境（TME）中每个位置的基因表达和细胞分布，是建立预后模型的“金标准”，但测序昂贵、无法常规临床应用；H&E 染色便宜易得，但看不到直接的分子表达。
  * **Core Idea**：能否设计一种机制，在训练阶段借用 SpT 的强分子表达信息来训练模型，但在推理阶段**脱离 SpT 数据**，仅凭 H&E 切片就做到同样高精度的预测？

---

### 2. Method 1: abMIL Architecture（单模态空间转录组 MIL 架构段落）
* **原文关键信息**：
  > *"abMIL is an attention-based multiple instance learning model specifically designed for spatial transcriptomic spots. Each spot is treated as an instance, with deconvolution-derived cell type proportions or pathway embeddings as inputs to predict patient-level overall survival."*
* **思路拆解**：
  * **多实例范式映射**：整张切片为一个 Bag，每一个 SpT 测序 Spot 为一个 Instance。
  * **输入与目标**：输入 Spot 级的分子/细胞解卷积特征，直接学习预测患者总生存期（OS/C-index）。
  * **作用与地位**：abMIL 在 SpaMIL 框架中作为**Teacher（教师模型）**，用以提供纯分子视角下的“金标准表征能力”。

---

### 3. Method 2: MabMIL Multimodal Distillation（多模态知识蒸馏架构段落）[最关键机制]
* **原文关键信息**：
  > *"To enable deployment on standard H&E stained slides without requiring SpT data at test time, MabMIL adopts a multimodal distillation framework. During training, both SpT and paired H&E images are fed into the network. Knowledge distillation aligns the representation learned by the H&E image encoder with the molecular representation from the SpT branch."*
* **思路拆解**：
  * **双流训练（Dual-stream Training）**：训练阶段，网络同时吃进 SpT 空间基因表达与相对应的 H&E 图像 Patch。
  * **知识蒸馏（Knowledge Distillation）**：并非简单的多模态特征拼接，而是计算损失函数（如 MSE 或 KL 散度），**强制 Student 支路（仅基于 H&E 图像的编码器）去模仿 Teacher 支路（基于 SpT）提取的高阶分子表征**。
  * **推理脱离（Inference Decoupling）**：测试/部署时，**彻底关停 SpT 支路**！只输入 H&E 切片，经过蒸馏强化后的 H&E 编码器能够自动提取出包含“分子感知的形态特征”。

---

### 4. Results & Interpretability: Shapley Attribution（结果与 Shapley 可解释性段落）
* **原文关键信息**：
  > *"MabMIL achieved a C-index of 0.72 on glioblastoma (GBM) overall survival prediction using H&E slides alone. By integrating Shapley values with cell-type deconvolution, SpaMIL revealed that high-risk predictions strongly attribute to specific macrophage and immune cell spatial densities."*
* **思路拆解**：
  * **性能评估**：在胶质母细胞瘤（GBM）的 MOSAIC 队列中，脱离 SpT 数据后仅凭 H&E 图像就实现了 0.72 的预测精度（C-index），证明了蒸馏机制的有效性。
  * **生物学可解释性**：使用 Shapley 归因值将模型关注的图像 Patch 映射回空间细胞解卷积结果，证实模型选出的高权重区域确实是巨噬细胞和免疫细胞聚集的肿瘤微环境关键区域。

</details>

---

## ⚖️ 第三章：现有 Phase 3 方案与 SpaMIL 论文方案的深度对比

将我们当前的 Phase 3 方案（`reference_implementations/phase3_clam_escc_20260801`）与 SpaMIL 的核心逻辑进行对比：

| 比较维度 | 现有 Phase 3 方案 (`phase3_clam_escc`) | SpaMIL 论文方案 (`MabMIL`) | 关键差异解读 |
| :--- | :--- | :--- | :--- |
| **融合/集成范式** | **硬拼接 / 晚期特征融合**<br>(`[1536 + 30]` 拼接，或 Cross-Attention / Adaptive Fusion) | **跨模态表征知识蒸馏**<br>(Teacher-Student Loss 对齐，不改变输入特征维度) | 现有方案将分子作为**硬性输入**；SpaMIL 将分子作为**训练导师/监督约束**。 |
| **推理期数据依赖** | **强依赖虚拟 ssGSEA 分数**<br>(推理时必须同时输入 H&E 和预测的 30 维基因分数) | **零分子依赖 (Zero-Omics at Test Time)**<br>(推理时**仅需 H&E 切片图像**) | 现有方案严重累积了 Phase 2 模型预测的上游误差；SpaMIL 在推理阶段完全不受分子估计偏差影响。 |
| **基因数据来源与性质** | **虚拟 / 拟合 ssGSEA 分数**<br>(来自 Phase 2 UNI2-h + MLP 预测，PCC≈0.60~0.65) | **真实空间转录组 (SpT) 实测数据**<br>(包含物理空间坐标与测序真值) | 现有方案的分子输入自带噪声；SpaMIL 蒸馏源是纯净的实验测序真值。 |
| **信息流动方向** | 图像特征 (1536维) 与 基因特征 (30维) 在分类 Head 中**竞争注意力** | 分子特征通过损失函数 **单向约束与引导** 病理图像编码器的参数更新 | 现有方案易发生高维特征吞噬低维特征；SpaMIL 的分子信息通过 Loss 显式注入。 |

---

## 🔍 第四章：定位 Phase 3 拼接/融合 AUC 无改善甚至下降的 3 个核心原因

结合 Phase 2 模型表现与 Phase 3 代码实现（`concat_clam_features.py`, `simple_escc_baseline.py`），定位出以下 3 个可能原因：

### 原因 1：二次拟合误差积聚（Error Cascading）与“内生伪信号”污染
* **现状事实**：我们在 Phase 3 使用的 30 维 ssGSEA 分数，**并非实验测序真值**，而是 Phase 2 利用 UNI2-h + MLP 预测出的“虚拟分数”（Phase 2 模型在外部验证集上的 PCC 约 0.60~0.65，残差与误差较大）。
* **深层机制**：在 Phase 3 中把这些由 H&E 特征预测出来的“估计分数”再次与原始 H&E 特征拼接，相当于**让模型去消化一个由自身衍生出来、且带有估计偏差的低维投影**。
* **结果**：这种“硬拼接”未能提供外部新信息，反而把 Phase 2 模型的预测偏差作为噪声引入了 Phase 3 MIL 模型中，干扰了 CLAM 原本对高维图像形态特征的学习。

### 原因 2：维度极度失衡与表征稀释（Dimension Mismatch & Representation Dominance）
* **现状事实**：图像 Patch 特征高达 **1536 维**（UNI2-h），而 ssGSEA 分数仅为 **30 维**。
* **深层机制**：
  1. **直接拼接 (`[1536 + 30 = 1566]`)**：在后续全连接层计算中，30 维基因特征仅占总维度的 **1.9%**，在梯度更新中极易被 1536 维的高维图像特征“淹没与稀释”，无法起到预期的引导作用。
  2. **交叉注意力 (`Cross-Attention`)**：若强行以 30 维基因特征作为 Query 去查询 1536 维图像特征，由于 30 维信息量太低且带有预测误差，会形成信息瓶颈（Information Bottleneck），反而破坏了图像特征自身丰富的形态学表达。

### 原因 3：融合范式错位——“硬输入”引发快捷学习（Shortcut Learning）
* **深层机制**：
  * SpaMIL 论文的根本逻辑表明：**分子信息是极佳的“导师（Supervision/Loss 约束）”，而非合适的“输入（Input）”**。
  * 当把 30 维数值作为硬性输入送入分类器时，弱监督 MIL 模型倾向于采取“快捷学习”策略（试图从简单标量中找规律），而放弃深度挖掘 Patch 级复杂的肿瘤空间异质性；而 SpaMIL 通过蒸馏损失，强制要求图像编码器在不改变输入的前提下自我演进，吸收分子表征。

---

## 🚀 第五章：Phase 3 借鉴 SpaMIL 的下一代改进探索路线

为解决当前 Phase 3 拼接/融合停滞不前的困境，建议从以下 3 个方向开展下一步探索：

```
                              【Phase 3 架构演进对比】
                              
  现状方案（硬拼接）:   [H&E (1536)] + [虚拟ssGSEA (30)]  ──► [CLAM] ──► AUC微降/无改善 (误差累积)
                                      
  改进路线 A（辅助重构）: [H&E (1536)] ──► [CLAM Encoder] ──┬─► 主任务 Loss (pCR/MPR)
                                                             └─► 辅助 Loss (预测 30维 ssGSEA)
                                                             
  改进路线 B（不确定性加权）: [H&E] + [虚拟ssGSEA] ──► [Gate(置信度W)] ──► 过滤低PCC通路后再融合
```

### 1. 路线 A：多任务辅助重构损失 (Auxiliary Reconstruction Loss) [最推荐]
* **做法**：取消特征拼接！输入恢复为纯粹的 **1536 维 H&E 特征**。
* **损失设计**：在 CLAM 的 Bag-level 或 Instance-level 特征层上挂载一个轻量 Head（如单层 MLP），去预测 30 维 ssGSEA 分数，并计算辅助 MSE 损失。
  $$\text{Loss}_{\text{total}} = \text{Loss}_{\text{pCR (主分类损失)}} + \lambda \cdot \text{Loss}_{\text{ssGSEA (辅助重构损失)}}$$
* **优势**：
  1. 推理阶段零分子依赖，完全不受 Phase 2 分数不确定性的污染；
  2. 利用辅助损失强迫 CLAM 图像编码器隐式吸收分子通路的表达规律。

### 2. 路线 B：基于置信度的软门控融合 (Uncertainty-Gated Soft Fusion)
* **做法**：若仍希望保留输入融合，必须引入 **Phase 2 预测通路的置信度门控（Gate）**。
* **机制**：计算每个通路预测的置信度/PCC。对 PCC > 0.85 的高质量通路放大融合权重，对预测偏差较大的通路进行动态 Mask 或降权，防止低质量预测倒灌噪声。

---

## 🌟 第六章：对比 SpaMIL，我们的数据优势与更进一步的创新方向

相较于 SpaMIL 论文（面对通用的 GBM 生存期预测），PFMval 项目在数据与前置积累上具备显著优势，能够支持我们做出更具学术竞争力的创新：

### 1. 我们的数据与基础设施优势
1. **精准且经权威专家核验的临床硬终点**：
   * SpaMIL 针对的是总生存期（OS），受年龄、合并症、后续治疗等强混杂因素干扰，噪声极高。
   * PFMval 拥有 **329 张 ESCC 切片/约78例患者** 的食管鳞癌新辅助治疗响应（pCR/MPR）确切病理结局，这是直接反映肿瘤微环境应答的“硬终点”。
2. **高维度、强语义的超大病理基础模型特征 (UNI2-h 1536维)**：
   * SpaMIL 依赖的是较早期的 ResNet50 或常规编码器，形态学表示能力有限。
   * 我们的 Phase 3 特征建立在最新的 **UNI2-h（1536维）** 基础模型之上，图像 Patch 表征本身已具备极高辨识度。
3. **经过严密审计与去泄露处理的精选生物学通路 (MPP1-5)**：
   * SpaMIL 面向成千上万个原始基因表达，噪声大且难以解释。
   * PFMval 拥有**标准化划分、严禁重叠审计、训练集单独计算 z-score 的 30 维精选 ssGSEA 通路**，具备极高的生物学可解释性。

### 2. 支持我们做出更进一步的 3 项创新点
* **创新点 1：基于预测不确定性感知的噪噪表征蒸馏 (Uncertainty-Aware Noisy-Target Distillation, UAND)**
  * *背景*：SpaMIL 假定蒸馏源（SpT 实测值）无噪。而在我们的真实临床落地中，分子预测值包含 Phase 2 的估算残差。
  * *创新机制*：引入 Phase 2 通路预测的不确定性/残差估计。对预测精度高（PCC > 0.85）的通路设定强蒸馏 Loss 权重，对不确定性大的通路动态衰减蒸馏梯度，实现**抗噪声的鲁棒蒸馏**。
* **创新点 2：生物学通路 Token 化的双阶分层蒸馏 (Hierarchical Pathway Token Distillation)**
  * *创新机制*：在 Patch 级利用 30 维 ssGSEA 通路重构蒸馏约束局部微环境，在 Bag 级利用跨模态 Attention 建模通路 Token 与图像 Patch Token 的全局交互，实现“局部微环境蒸馏 + 全局 pCR 疗效预测”的双阶层次化特征蒸馏。
* **创新点 3：跨模态对比表征对齐 (Contrastive Latent Alignment, 借鉴 BLEEP)**
  * *创新机制*：放弃标量级的硬回归或硬拼接，而是将 1536 维图像特征与 30 维 ssGSEA 通路特征通过双塔网络映射至统一的隐空间（Joint Latent Space），通过 InfoNCE 对比学习拉近“同位置图像与通路”的表征距离，消除维度失衡导致的信号稀释。

---

## 📚 第七章：同类前沿顶刊/顶会论文借鉴列表（解决当前瓶颈）

针对我们面临的“多模态拼接无效、推理期基因数据不确定性”问题，以下 3 篇来自 Nature / MICCAI / NeurIPS 的顶级论文提供了极佳的解决思路：

### 1. SurvPath (Nature Machine Intelligence, 2024 / MICCAI 2023)
* **论文题目**：*SurvPath: Transpath-guided Multimodal Survival Prediction with Missing Omics*
* **核心思路**：不将组学数据简单降维成标量拼接，而是将其Token化为**语义清晰的生物学通路 Token（Pathway Tokens）**，在 Transformer 中与病理图像 Patch Tokens 进行密集 Cross-Attention 交互。
* **对 PFMval 的借鉴**：将我们的 30 维 ssGSEA 分数映射为 30 个 Pathway Tokens，而不是单层向量拼接，解决 1536 维稀释 30 维的问题。

### 2. BLEEP (NeurIPS 2023 / Bioinformatics 2024)
* **论文题目**：*BLEEP: Bi-modal Latent Embedding for Expression Prediction from Histology Images*
* **核心思路**：采用类似 CLIP 的**双塔对比学习（Bi-modal Contrastive Learning）**，构建病理图像与基因表达的统一隐空间（Joint Latent Space），拉近二者表征。
* **对 PFMval 的借鉴**：利用对比损失（InfoNCE）让 UNI2-h 特征在不更改输入的前提下隐式向 30 维 ssGSEA 表征靠拢，从根源上避免“硬拼接 `[1536+30]`”引起的二次拟合噪声。

### 3. THuNDER (MICCAI 2024)
* **论文题目**：*Transcriptomic-guided Histology Understanding via Noise-robust Distillation for Cancer Prognosis*
* **核心思路**：专门解决“用带有预测误差/噪声的基因组学数据指导病理图像模型”的问题，提出了**噪声鲁棒蒸馏损失（Noise-robust Distillation Loss）** 与动态残差重加权机制。
* **对 PFMval 的借鉴**：完美匹配 Phase 2 虚拟 ssGSEA 存在预测残差（PCC≈0.60~0.65）的现状！通过引入残差不确定性感知，衰减估计偏差较大通路的蒸馏权重，提升模型的鲁棒性。

---

## 🎯 结论与行动建议

1. **短期止损**：暂停简单的 `concat_clam_features.py`（硬拼接 `[1536+30]`）尝试，避免在已有偏估算特征上浪费计算资源。
2. **中期探索**：在 `reference_implementations/phase3_clam_escc_20260801/main.py` 中引入**路线 A（辅助重构损失）**与 **THuNDER 噪声鲁棒蒸馏**，验证 Loss 约束范式能否超越单纯的 H&E 病理基线。

