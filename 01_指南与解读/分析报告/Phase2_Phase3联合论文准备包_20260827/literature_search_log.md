# Phase2 / Phase3 联合论文准备包：文献检索日志

- 检索日期：2026-08-27
- 最后核验日期：2026-08-28
- 检索模式：lit-review（面向论文新颖性与证据边界）
- 项目范围：PFMval 的 H&E/WSI→基因或通路表达预测、pathway-aware / contrastive representation、图像—通路融合与临床结局、患者级验证、校准和负对照
- 交付文件：`novelty_matrix.csv`、`references.bib` 与 `literature_search_log.md`

## SR1：检索与获取（Search and Retrieval）

### 来源范围

仅检索或保留以下原始、正式来源：

- PubMed / PubMed Central 的期刊元数据或全文页；
- Nature Portfolio、BMJ、Oxford Academic、Elsevier、IEEE 等期刊官网；
- IEEE/CVF Open Access 与 NeurIPS Proceedings 官方论文页；
- DOI 可解析的正式出版记录；
- 若无正式发表版本但对当前临床问题非常接近，允许保留预印本并明确降级。

未使用博客、商业解读、百科、新闻稿或聚合站作为证据。搜索引擎结果只用于定位原始页面，不作为事实来源。

### 检索主题与布尔式

以下布尔式在官方域名限定下组合使用：

```text
("H&E" OR histology OR WSI) AND
("spatial gene expression" OR transcriptomics) AND
(prediction OR imputation OR reconstruction)

(pathway-aware OR pathway-informed OR transcriptomics-guided) AND
(histopathology OR WSI) AND
(contrastive OR representation OR fusion)

("histology-derived transcriptomics" OR "imputed transcriptomics") AND
(response OR prognosis OR survival OR pCR)

("clinical prediction model" OR "medical AI") AND
("patient-level validation" OR calibration OR Brier OR "decision curve")

("negative control" OR "shuffled labels" OR falsification) AND
(prediction OR bias OR validation)
```

### 实际执行过的核心检索式

1. `site:nature.com H&E spatial gene expression prediction histology 2024 2025 benchmark`
2. `site:openaccess.thecvf.com spatial gene expression prediction histology 2024 TRIPLEX`
3. `site:pubmed.ncbi.nlm.nih.gov histology spatial transcriptomics gene expression prediction H&E 2023 2024 2025`
4. `site:proceedings.neurips.cc HEST-1k spatial transcriptomics histology 2024`
5. `site:pubmed.ncbi.nlm.nih.gov BLEEP spatial gene expression histology 2023`
6. `site:pubmed.ncbi.nlm.nih.gov HisToGene THItoGene spatial gene expression histology`
7. `site:nature.com spatial gene expression histology iStar 2024`
8. `site:pubmed.ncbi.nlm.nih.gov SEPAL spatial gene expression histology local graph`
9. `site:openaccess.thecvf.com transcriptomics guided pathology contrastive learning TANGLE CVPR 2024`
10. `site:nature.com histopathology transcriptomics contrastive representation learning 2024 2025`
11. `site:pubmed.ncbi.nlm.nih.gov pathology transcriptomics contrastive learning whole slide 2024`
12. `site:openaccess.thecvf.com pathology gene expression contrastive multimodal learning 2023 2024`
13. `site:nature.com DeepPT ENLIGHT histopathology transcriptomics treatment response Nature Cancer 2024`
14. `site:pubmed.ncbi.nlm.nih.gov histopathology predicted transcriptomics treatment response DeepPT ENLIGHT`
15. `site:nature.com histology predicted spatial gene expression prognosis survival 2024 2025 IGI-DL`
16. `site:pubmed.ncbi.nlm.nih.gov whole slide image genomics multimodal survival pathway 2023 2024`
17. `site:bmj.com TRIPOD+AI 2024 prediction model reporting guideline`
18. `site:bmj.com clinical prediction model evaluation discrimination calibration Brier 2024`
19. `site:pubmed.ncbi.nlm.nih.gov patient-level cross validation pathology AI data leakage whole slide`
20. `site:pubmed.ncbi.nlm.nih.gov negative controls machine learning medical imaging clinical prediction`
21. `site:bmj.com evaluating clinical prediction models discrimination calibration clinical utility 2024 Riley Van Calster`
22. `site:pubmed.ncbi.nlm.nih.gov calibration clinical prediction models slope intercept Brier 2019 Van Calster`
23. `site:pubmed.ncbi.nlm.nih.gov precision recall plot more informative ROC imbalanced datasets 2015`
24. `site:pubmed.ncbi.nlm.nih.gov decision curve analysis net benefit 2006 Vickers Elkin`
25. `site:pubmed.ncbi.nlm.nih.gov Pathomic Fusion histopathology genomic features prognosis 2020`
26. `site:openaccess.thecvf.com multimodal co-attention transformer histology genomics survival MCAT ICCV 2021`
27. `site:pubmed.ncbi.nlm.nih.gov PORPOISE multimodal survival pathology genomics 2022`
28. `site:pubmed.ncbi.nlm.nih.gov SurvPath pathology genomics pathways survival prediction 2024`
29. `"SurvPath" pathology pathways survival prediction`
30. `"Deep Biological Pathway Informed" pathology genomic survival prediction`
31. `site:pubmed.ncbi.nlm.nih.gov pathway informed pathology genomic multimodal survival prediction`
32. `site:nature.com pathway informed histopathology genomic survival prediction`
33. `site:pubmed.ncbi.nlm.nih.gov negative controls confounding bias observational studies Lipsitch 2010`
34. `site:pubmed.ncbi.nlm.nih.gov negative control outcomes exposures machine learning medical prediction`
35. `site:bmj.com negative control outcome epidemiology guidance`
36. `site:pubmed.ncbi.nlm.nih.gov patient-level split data leakage whole slide histopathology machine learning`
37. `site:nature.com patient-level data leakage computational pathology validation`
38. `site:pubmed.ncbi.nlm.nih.gov clustered data cross-validation patient level medical machine learning leakage`
39. `site:pubmed.ncbi.nlm.nih.gov concordance correlation coefficient Lin 1989 biometrics`
40. `site:pubmed.ncbi.nlm.nih.gov "Multimodal deep learning: An improvement in prognostication or a reflection of batch effect?"`
41. `site:pubmed.ncbi.nlm.nih.gov/39005487 "receiver operating characteristic curve accurately assesses imbalanced datasets"`
42. `site:openaccess.thecvf.com/content/WACV2026 PEaRL "Pathway-Enhanced Representation Learning"`
43. `site:doi.org/10.1109/WACV61042.2026.00777 PEaRL`
44. `site:academic.oup.com/bioinformatics 10.1093/bioinformatics/btag643 DeepPathway`
45. `site:academic.oup.com/bioinformatics/advance-articles DeepPathway "Accepted Manuscript"`
46. `site:openaccess.thecvf.com/content/ICCV2025W/CVAMD "A Large-Scale Benchmark of Cross-Modal Learning"`
47. `site:proceedings.neurips.cc/paper_files/paper/2023 BLEEP "Bi-modal Contrastive Learning"`
48. `site:bmj.com/content/384/bmj-2023-074820 Riley Archer "external validation"`
49. `site:nature.com/articles/s41591-024-02857-3 UNI computational pathology`
50. `site:nature.com/articles/s41467-025-56618-y "best fold" "test set"`

另外执行了题名、DOI、作者、年份和出版状态的逐篇核验查询。检索与核验查询合计 78 个。

## SE2：筛选与纳排（Screening and Eligibility）

### 纳入标准

1. 直接研究 H&E/WSI→空间或bulk基因表达、通路表达或派生表达表征；
2. 研究 pathway-aware、transcriptomics-guided 或视觉—组学对比表征；
3. 研究真实或合成表达与病理图像融合后的疗效、预后或分级；
4. 提供患者级验证、校准、区分度、临床效用、不确定性或负对照的方法依据；
5. 优先纳入 2024–2026 文献；早期奠基论文仅在定义指标、任务范式或关键反例时保留；
6. 能从官方页核验题名、年份、来源及 DOI 或官方 URL。

### 排除标准

1. 仅做空间转录组聚类、解卷积或差异分析，未涉及图像预测或本项目评价边界；
2. 新闻、博客、非官方综述摘要、商业宣传或无法回链原始论文的二手页面；
3. 已有正式发表版时的旧预印本；
4. 仅报告图块级随机拆分且无法判断患者泄漏，又没有独立方法价值；
5. 无法核实 DOI、正式出版状态或关键方法细节；
6. 与 PFMval 的通路连续回归、同源派生融合或患者级结局评价没有可解释关系。

### 筛选结果

- 搜索接口返回约 300 条结果卡片；该数包含同一论文的 HTML、PDF、PubMed、引用页及重复搜索结果，因此不是去重后的论文数。
- 最终纳入：44 条唯一来源。
- 其中 2024–2026：32 条。
- 同行评议期刊或会议：43 条。
- 预印本：1 条（乳腺癌治疗特异反应研究，已在矩阵中标为 `preprint` 和 C 级证据）。
- 正式来源门槛：满足“至少25条正式来源”；实际为 43 条正式同行评议来源。
- Riley、Archer 等的 BMJ 2024 外部验证论文已作为 S30 存在，因此保持单条、不重复计数；新增 UNI 2024 正式论文，闭合 PEaRL 所用图像编码器的来源链并维持 44 条唯一来源。

由于检索阶段遵守“不写项目文件”的约束，未保存逐条搜索结果卡片或逐条排除表；本日志不伪造无法逐项复核的排除原因计数。

## EE3：证据抽取（Evidence Extraction）

每篇来源抽取以下信息：

- 题名、年份、正式来源、DOI 和官方 URL；
- WHY：论文解决的问题；
- HOW：主要数据和方法；
- WHAT：主要评价或结论；
- 与 PFMval 的任务、样本、输出单位和模态独立性差异；
- 可支持的新颖性表述；
- 禁止外推或需要负对照才能成立的主张；
- 同行评议状态和与当前主张的适配度。

矩阵保留 `evidence_strength` 的文字化证据强度，同时新增两个逐行必填字段：

- `study_design_level`：项目内研究设计类型标记；VI 表示原始经验研究或方法验证，VII 表示报告、偏倚评估或预测模型评价指南。它不是期刊等级，也不由发表进度决定。
- `source_fitness_grade`：来源对 PFMval 具体主张的适配等级，使用 A / B / C 并可带 `+/-` 修饰。

`evidence_strength` 与 `source_fitness_grade` 均不以期刊名替代质量判断：

- A：多数据集、外部验证、直接系统基准或权威方法指南；
- B：同行评议但数据域窄、样本较少、参考库依赖或仅间接支持；
- C：预印本、评论性材料或只能支持问题合理性。

DeepPathway 的 `accepted manuscript` 仅记录在 `peer_review_status`（发表生命周期）中；其研究设计标为 VI、来源适配等级为 B，未因 accepted 状态升级。

## CB4：主张边界（Claim Boundaries）

1. 多数表达预测论文是基因级、大队列、多切片任务；不能据此预先承诺 7 位患者、30 通路的性能提升。
2. SurvPath、MCAT、Pathomic Fusion 和 PORPOISE 使用独立实测组学；若 PFMval 通路由同一 H&E 派生，不应称为独立多模态证据。
3. PathGen 是最接近“同源派生表达再与图像融合”的正式先例，但仍需 image-only、容量匹配、标签/通路置乱和批次负对照，才能区分信息增量与额外容量。
4. PCC 只度量线性关联；连续通路预测还应报告 MAE/RMSE、R²、CCC 和校准信息，并按患者平衡汇总。
5. AUROC 与 AUPRC 回答不同问题：AUROC反映排序且对 prevalence 不变；AUPRC反映目标队列下少数阳性检出表现。二者均不能替代 Brier 和校准。
6. 负对照支持 falsification（证伪检查）逻辑，但不能替代患者级拆分、真正外部验证或充分样本量。
7. S03 的 H&E-to-SGE benchmark 若以 test set 选择最佳 fold，可能产生乐观偏差；引用排行榜或最优数值时必须披露并核查其 fold 选择流程。

## AD5：AI 辅助披露（AI-Assistance Disclosure）

本检索、初筛、题名归一化、WHY/HOW/WHAT 摘要和证据边界整理由 AI 辅助完成。所有纳入条目的题名、来源、年份、DOI 或官方 URL 均回链至 PubMed、期刊官网、CVF 或 NeurIPS 官方页。AI 未代替同行评议，也未对无法从官方页面核实的模型细节作肯定判断。

建议论文方法或补充材料披露：

> Literature searching and initial evidence extraction were assisted by an AI system. All included citations and claim boundaries were manually traceable to publisher, PubMed, CVF, NeurIPS, or DOI records as of 28 August 2026.

## QA6：已知局限（Quality Assurance and Limitations）

- 文献明显偏向乳腺癌、TCGA、Visium 和公开大型队列；食管癌与7患者设定的直接证据稀少。
- 许多既有方法采用留一切片而非留一患者；不能将其结果与患者级验证直接等价比较。
- BLEEP、HECLIP、OmiCLIP/Loki、mclSTExp 等检索式方法可能依赖参考表达库；引用时必须披露该依赖。
- S03 报告最优 fold 的 test-set 选择流程存在潜在乐观偏差；在复现作者评价代码或逐项核对补充材料前，不把该最优值当作无偏性能估计。
- `references.bib` 使用稳定键；少数超长作者列表以 `and others` 缩写，正式投稿前可由 DOI 元数据管理器补全，但 DOI、URL 和题名已核验。
- 本次未访问项目服务器、未下载训练数据、未运行模型，也未据文献生成可比较实验结论。
