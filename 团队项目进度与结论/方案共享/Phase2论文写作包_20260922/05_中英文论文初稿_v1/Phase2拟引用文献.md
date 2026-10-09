# Phase2 拟引用文献

核对日期：2026-09-22。本文献表只为 Phase2 初稿提供可追溯的背景与方法来源；不替代本项目的实验配置、数据划分、训练过程或已接纳结果证据。`当前稿引用`指可以进入本轮中英文初稿的引用，`候选`指需以实际采用的方法或图件为前提。

| ID | 类型与当前状态 | 经核验出处与来源 | 可支撑的正文论断 | 不支持的论断 |
| --- | --- | --- | --- | --- |
| [P2-R1] | 已发表；当前稿引用 | He B, Bergenstråhle L, Stenbeck L, *et al.* **Integrating spatial gene expression and breast tumour morphology via deep learning.** *Nature Biomedical Engineering* 4, 827–834 (2020). DOI: [10.1038/s41551-020-0578-x](https://doi.org/10.1038/s41551-020-0578-x). [出版页](https://www.nature.com/articles/s41551-020-0578-x) | ST-Net 是从 H&E 图像预测空间基因表达的早期代表性工作，可用于交代研究问题与乳腺癌场景。 | 不能证明本项目模型、通路聚合、数据划分或性能优于该文。 |
| [P2-R2] | 已发表；当前稿引用 | Zeng Y, Wei Z, Yu W, Yin R, *et al.* **Spatial transcriptomics prediction from histology jointly through Transformer and graph neural networks.** *Briefings in Bioinformatics* 23(5), bbac297 (2022). DOI: [10.1093/bib/bbac297](https://doi.org/10.1093/bib/bbac297). [出版页](https://academic.oup.com/bib/article/23/5/bbac297/6645485) | Hist2ST 将图像表征与图结构用于空间转录组预测，可作为图模型相关工作的背景。 | 不能作为本项目的实现、超参数或外部测试结果依据。 |
| [P2-R3] | 已发表；当前稿引用 | Chen RJ, Ding T, Lu MY, Williamson DFK, Jaume G, *et al.* **Towards a general-purpose foundation model for computational pathology.** *Nature Medicine* 30, 850–862 (2024). DOI: [10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3). [出版页](https://www.nature.com/articles/s41591-024-02857-3) | UNI 是计算病理学通用基础模型的已发表来源，可支撑对 UNI 编码器背景的说明。 | 不能将其表述为 UNI2-h 的专门论文，也不能证明本项目中 UNI/UNI2-h/Virchow2 的相对性能。 |
| [P2-R4] | 预印本；当前稿引用 | Zimmermann E, Vorontsov E, Viret J, Casson A, Zelechowski M, *et al.* **Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology.** arXiv:2408.00738v3 (2024). DOI: [10.48550/arXiv.2408.00738](https://doi.org/10.48550/arXiv.2408.00738). [arXiv 原文](https://arxiv.org/abs/2408.00738) | Virchow2 的模型名称、混合倍率自监督训练定位和作者声明的模型设计背景。 | 截至核对日未找到可确认的正式期刊版本；不能写成已发表论文，也不能作为本项目结果的独立验证。 |
| [P2-R5] | 官方模型说明，非论文；当前稿引用（方法来源） | MahmoodLab. **UNI2-h model card.** Hugging Face，访问于 2026-09-22；DOI：无。 [官方模型卡](https://huggingface.co/MahmoodLab/UNI2-h)；[官方代码库](https://github.com/mahmoodlab/UNI)。模型卡说明 UNI2-h 的 ViT-H/14 架构、DINOv2 训练路线、1536 维特征接口，并将其“Paper”链接指向 [P2-R3]。 | 本项目使用 UNI2-h 时，模型名称、官方加载接口和公开模型说明。 | UNI2-h 没有在该模型卡中给出独立论文；不得把 [P2-R3] 写为 UNI2-h 专文，亦不得以模型卡替代本项目训练细节。 |
| [P2-R6] | 已发表；候选，**仅在正文确认原始 raw30 通路分数以 ssGSEA 聚合时启用** | Barbie DA, Tamayo P, Boehm JS, *et al.* **Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1.** *Nature* 462, 108–112 (2009). DOI: [10.1038/nature08460](https://doi.org/10.1038/nature08460). [PubMed 原始记录（PMID: 19847166）](https://pubmed.ncbi.nlm.nih.gov/19847166/) | 文中所用 ssGSEA 的原始方法出处。 | 不能证明本项目使用了 ssGSEA；在 raw30 聚合方法未核准前不得进入方法段。 |
| [P2-R7] | 已发表；候选，**仅在正文确认使用 GSVA 时启用** | Hänzelmann S, Castelo R, Guinney J. **GSVA: gene set variation analysis for microarray and RNA-Seq data.** *BMC Bioinformatics* 14, 7 (2013). DOI: [10.1186/1471-2105-14-7](https://doi.org/10.1186/1471-2105-14-7). [出版页](https://link.springer.com/article/10.1186/1471-2105-14-7) | GSVA 的逐样本基因集评分方法来源。 | 不能与 ssGSEA 混称；未实际采用 GSVA 时不得引用为本项目方法。 |
| [P2-R8] | 预印本；候选，当前无 UMAP 图，不进入本轮实证 | McInnes L, Healy J, Melville J. **UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.** arXiv:1802.03426 (2018). [arXiv 原文](https://arxiv.org/abs/1802.03426)；DOI：未列。 | 若以后制作 UMAP 嵌入图，可作为降维可视化方法来源。 | 不能支持当前未绘制的 UMAP，也不能把二维可视化解释为模型性能或因果机制。 |
| [P2-R9] | 预印本；可选背景，不进入当前实证 | Shulman ED, Campagnolo EM, Lodha R, Cantore T, Hu T, *et al.* **Path2Space: An AI Approach for Cancer Biomarker Discovery Via Histopathology Inferred Spatial Transcriptomics.** bioRxiv 2024.10.16.618609 (posted 2024-10-18). DOI: [10.1101/2024.10.16.618609](https://doi.org/10.1101/2024.10.16.618609). [bioRxiv 原文](https://www.biorxiv.org/content/10.1101/2024.10.16.618609v1) | 可用于补充“由组织病理图像推断空间转录组并用于生物标志物研究”的相关背景。 | 是预印本，不能写成同行评审结论；不能支持 Phase2 的性能、临床结论或比较优越性。 |

## 建议的参考文献格式

<a id="p2-r1"></a>

**[P2-R1]** He B, Bergenstråhle L, Stenbeck L, *et al.* Integrating spatial gene expression and breast tumour morphology via deep learning. *Nat Biomed Eng.* 2020;4:827–834. doi:[10.1038/s41551-020-0578-x](https://doi.org/10.1038/s41551-020-0578-x).

<a id="p2-r2"></a>

**[P2-R2]** Zeng Y, Wei Z, Yu W, Yin R, *et al.* Spatial transcriptomics prediction from histology jointly through Transformer and graph neural networks. *Brief Bioinform.* 2022;23(5):bbac297. doi:[10.1093/bib/bbac297](https://doi.org/10.1093/bib/bbac297).

<a id="p2-r3"></a>

**[P2-R3]** Chen RJ, Ding T, Lu MY, Williamson DFK, Jaume G, *et al.* Towards a general-purpose foundation model for computational pathology. *Nat Med.* 2024;30:850–862. doi:[10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3).

<a id="p2-r4"></a>

**[P2-R4]** Zimmermann E, Vorontsov E, Viret J, Casson A, Zelechowski M, *et al.* Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology. arXiv:2408.00738v3 [Preprint]. 2024. doi:[10.48550/arXiv.2408.00738](https://doi.org/10.48550/arXiv.2408.00738).

<a id="p2-r5"></a>

**[P2-R5]** MahmoodLab. [UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h). Hugging Face. Accessed September 22, 2026. **官方模型说明，非论文。**

<a id="p2-r6"></a>

**[P2-R6]** Barbie DA, Tamayo P, Boehm JS, *et al.* Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1. *Nature.* 2009;462:108–112. doi:[10.1038/nature08460](https://doi.org/10.1038/nature08460). **仅在采用 ssGSEA 时保留。**

<a id="p2-r7"></a>

**[P2-R7]** Hänzelmann S, Castelo R, Guinney J. GSVA: gene set variation analysis for microarray and RNA-Seq data. *BMC Bioinformatics.* 2013;14:7. doi:[10.1186/1471-2105-14-7](https://doi.org/10.1186/1471-2105-14-7). **仅在采用 GSVA 时保留。**

<a id="p2-r8"></a>

**[P2-R8]** McInnes L, Healy J, Melville J. [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426). arXiv:1802.03426 [Preprint]. 2018. **仅在绘制 UMAP 时保留。**

<a id="p2-r9"></a>

**[P2-R9]** Shulman ED, Campagnolo EM, Lodha R, Cantore T, Hu T, *et al.* Path2Space: An AI Approach for Cancer Biomarker Discovery Via Histopathology Inferred Spatial Transcriptomics. bioRxiv 2024.10.16.618609 [Preprint]. 2024. doi:[10.1101/2024.10.16.618609](https://doi.org/10.1101/2024.10.16.618609). **可选背景文献。**

## 待正文落笔前确认

- [P2-R6]、[P2-R7] 必须按最终冻结的 raw30 通路聚合算法二选一或同时保留；目前本表不据此推断。
- [P2-R8] 对应的 UMAP 图不在本轮首批图中，故不应进入本轮实证结果或图注。
- [P2-R4] 与 [P2-R9] 均为预印本；投稿前应再次核对是否出现正式发表版本，并相应替换引文类型与链接。
