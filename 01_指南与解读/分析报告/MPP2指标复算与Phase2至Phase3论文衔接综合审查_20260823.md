# MPP2指标复算与Phase 2至Phase 3论文衔接综合审查

> 创建日期：2026-08-23  
> 生命周期：`pending_user_review`  
> 文档性质：只读审查与论文预调研；不改变实验结果状态，不替代医生后续提供的论文模板  
> 关联文件一：[MPP2首批评价指标复算报告](MPP2首批评价指标复算报告_20260823.md)  
> 关联文件二：[MPP2评价指标与同类论文原始数据对比分析](MPP2评价指标与同类论文原始数据对比分析_20260823.md)  
> 修改边界：本次没有修改上述两份文件，也没有写入experiment Registry、document Registry或current state；未执行哈希校验。  
> 检索边界：截至2026-08-23的定向证据审查，纳入同行评审论文、会议论文、公开基准和明确标注的预印本；不是系统综述或Meta分析。  

## 1. 审查结论

### 1.1 总体判定

**CONDITIONAL GO（有条件通过）**。

两份报告可继续作为项目内部参考，MPP2核心复算数值未发现真实性错误；但在完成本报告列出的状态纠偏、统计限制披露和可比性降级前，不应直接用于论文中的“显著领先”“SOTA”或“临床增益”表述。

### 1.2 四个问题的直接回答

| 审查问题 | 判定 | 核心回答 |
|---|---|---|
| 1. 数据真实性 | 基本可信 | MPP2核心数值可从原始预测文件独立复算；关键论文与多数精确值能回到正式论文表格、官方Source Data或作者代码。未发现伪造数据，但部分数值的证据类型需要重新标注。 |
| 2. 覆盖面 | 当前不足 | 现有对比对H&E→基因表达覆盖较多，但直接通路预测、公开大基准以及pCR/MPR衔接证据不足。本文补为三层文献结构。 |
| 3. 差异区分 | 主体正确，仍需收紧 | 原报告已多次声明不可直接横比，但仍需明确区分基因与通路、单患者空间外推与跨患者外部验证、标准SSIM与项目自定义SSIM、论文表格值与教程/图中估读值。 |
| 4. Phase 2服务Phase 3 | 可形成清晰路线 | Phase 2应证明通路分数“可信、稳定、可迁移且提供互补表征”；Phase 3再证明其是否给pCR主终点带来增量判别、校准或临床净获益。仅报告Phase 2 PCC不足以完成这条证据链。 |

### 1.3 当前最重要的论文边界

Phase 3没有真实转录组，1536维UNI2-h图像特征和30维预测ssGSEA分数都由同一张H&E切片产生。因此：

- 推荐称为 **H&E衍生的形态—通路双表征融合**（H&E-derived morphology–pathway dual-representation fusion）；
- 可以称“生物学引导的特征增强”或“同源双分支融合”；
- 不宜不加限定地称为“图像+真实组学多模态融合”；
- Phase 2的真实转录组监督证明的是上游代理表征有效性，不等于Phase 3存在独立分子模态。

这不是命名细节，而是审稿人判断“新增信息”还是“同一图像的再次编码”的基础。

另外，pCR（病理完全缓解）和MPR（主要病理缓解）是**新辅助治疗反应终点**，不是总生存期、无病生存期等预后终点。合并论文应使用“treatment response prediction”或“pathological response prediction”，避免笼统写“prognosis prediction”。

## 2. 审查对象与证据边界

### 2.1 本地证据层级

| 证据 | 当前状态 | 本报告如何使用 |
|---|---|---|
| accepted/accepted-smoke原始预测 | 已接受的输入证据 | 用于核验复算值；`accepted-smoke`只代表治理状态，不代表科学优越性 |
| 两份MPP2报告 | document Registry中为`historical_reference / reference / active` | 可作活跃参考，但不是accepted实验结果 |
| 本次新报告 | `pending_user_review` | 用户审核前不登记为active reference |
| 外部论文正式表格/Source Data | 较高证据 | 可精确引用，但必须保留其队列、任务与聚合定义 |
| 作者教程、代码输出 | 中等证据 | 标为“作者教程/代码值”，不得冒充正式benchmark表 |
| 论文图中人工读取 | 较低证据 | 仅可标为“图中估读值”并附原图或论文链接；不用于精确胜负 |
| 预印本 | 前沿但未同行评审 | 明确标“预印本/未同行评审”，不能与正式论文混写状态 |

### 2.2 文档状态漂移

两份原报告页首仍写`pending_user_review`，第一份报告第12节还写“未成为active reference”；但当前`project_state/document_registry.json`已将两份文件登记为`scope=historical_reference`、`authority=reference`、`lifecycle=active`，验证时间为2026-08-23 23:14:49。

这是**文档叙述与机器状态不一致**，不影响已复算数值本身，但会误导后续Agent或作者。由于用户要求不改原文件，本报告作为关联补充说明：

> 两份原报告当前是active reference；其中新增指标仍不自动成为accepted实验结果。原报告关于“未登记active reference”的句子仅代表其撰写时状态，不再代表当前状态。

## 3. 第一份报告：复算真实性与统计审查

### 3.1 可取之处

1. **核心数值可独立复现。**从FBR外部XZY原始预测直接计算得到：逐通路平均PCC `0.519386`、CCC `0.398858`、z-RMSE `0.767225`、raw R² `-0.088017`，与报告一致；pooled PCC为`0.654896`。
2. **公式和聚合顺序总体正确。**PCC、CCC、z-RMSE、raw R²实现与报告定义一致；PCC患者聚合使用Fisher-z。
3. **患者等权而非spot等权。**这能避免spot数多的患者支配内部结果。
4. **证据不足处有主动降级。**Huber缺少患者/spot标识，因此只做逐通路描述，没有伪造患者或空间置信区间。
5. **空间指标命名较诚实。**使用`masked_SSIM_PFMval`而非直接冒充论文默认SSIM；内部队列因不存在完整7×7窗口返回NA是合理结果。
6. **负面结果未隐藏。**raw R²均值为负、LoRA外部不利、Huber pooled PCC下降均被保留，有利于避免选择性汇报。

### 3.2 需要补充或收紧之处

| 级别 | 问题 | 审查结论与建议 |
|---|---|---|
| Major | 外部仅一个XZY患者 | 2048坐标块Bootstrap反映单患者内部空间不确定性，不能替代跨患者、跨中心外部验证；论文应写“external slide/patient evaluation”，不要泛化成“external cohort validation”。 |
| Major | raw R²为`-0.088017` | 说明按原始数值误差衡量，模型平均不优于以真实均值作常数预测的基准。PCC较高与R²为负可以同时成立，论文必须并列披露，不能只报PCC。 |
| Major | 多重比较未校正 | 30通路、多个指标、多个实验臂存在多重比较。当前“4/30、17/30、19/30改善”只能作描述；未做FDR/Holm校正前不得称“显著改善的通路数”。 |
| Major | 自定义SSIM不可跨论文排名 | 当前SSIM使用稀疏网格、完整7×7窗口、固定`data_range=6`且不插值，是项目冻结协议值。它可作内部比较，但不能用`0.309`对外宣称高于HEtoSGEBench的`0.223`。 |
| Moderate | SSIM Bootstrap单位未充分展开 | PCC/CCC/z-RMSE外部使用21个2048坐标块；SSIM实际只有11个有效block。建议任何置信区间表同时给`raw_units`。 |
| Moderate | 两种Δ估计量容易混淆 | headline逐通路平均Δ与空间块Bootstrap中心值不是同一个estimand（目标估计量）；二者略有不同不是复算错误，需分列命名。 |
| Moderate | Moran’s I定义需更透明 | 代码枚举右/下无向rook邻边，再在分子和`s0`中按双向权重折算；数学上可复现，但论文方法应写清。 |
| Moderate | 最小测试覆盖有限 | 当前测试覆盖完美预测、仿射偏移、患者等权、配对键和随机种子；没有覆盖SSIM/Moran公式、空间块独立性与多重比较。不得把“最小测试通过”写成“所有统计性质已验证”。 |
| Minor | pooled PCC容易显得更高 | pooled PCC把不同通路和位置汇总，受通路间动态范围影响；不应与逐通路平均PCC混为同一指标。论文主表优先逐通路/逐患者分布，pooled值只作补充。 |

### 3.3 第一份报告可保留的核心表述

> 在冻结MPP2协议下，FBR在单个外部XZY患者的30条通路上取得逐通路平均PCC 0.519、CCC 0.399和z-RMSE 0.767；但raw R²均值为-0.088，提示相关性与绝对数值拟合仍不一致。空间块区间仅刻画该患者内部空间不确定性，不构成跨患者外部泛化证据。

## 4. 第二份报告：论文数据真实性、覆盖面与可比性

### 4.1 可取之处

1. **已经找到最接近的直接任务。**PEaRL同样由H&E预测ssGSEA通路活性，比普通基因表达预测更适合做量级参考。
2. **没有简单制作“排行榜”。**报告明确禁止`0.519 vs 0.281`、`0.309 vs 0.223`式的直接领先表述。
3. **保留了原始任务差异。**对基因/通路、数据集、分辨率、标准化、切分、聚合和指标定义均有提醒。
4. **低证据数值已有部分标注。**DeepPathway数值明确来自作者教程而非正式benchmark表。
5. **对LoRA和Huber的外部证据没有反推本项目成功。**SEAL等论文说明路线有先例，但没有被用来掩盖MPP2本地不利结果。

### 4.2 真实性问题不是“假数据”，而是“来源类型不够统一”

| 当前内容 | 审查结果 | 需要补充的来源标签 |
|---|---|---|
| PEaRL Table 2/4 | 正式WACV 2026会议论文，可核验 | `同行评审会议论文；正式表格值` |
| HEtoSGEBench聚合指标 | Nature Communications 2025、官方Source Data/代码可核验 | `同行评审；Source Data重算值`，不是所有数都直接印在正文表格 |
| DeepPathway数值 | 任务存在；数值来自作者教程 | `预印本；作者教程保存输出；非正式benchmark表` |
| SEAL | 2026 arXiv | `预印本；未同行评审` |
| DeepSpot-M | 2026 medRxiv | `预印本；未同行评审` |
| 图中曲线值 | 当前报告基本未强行估读 | 若后续加入，必须标`图中估读值`并链接到原图/论文 |

### 4.3 HEtoSGEBench的NRMSE定义需要纠偏

第二份报告把引用的`NRMSE-SD`概括为“按真实表达标准差”。官方代码实际上同时计算：

- `nrmse_range`：RMSE除以观测范围；
- `nrmse_sd`：RMSE除以观测标准差。

论文方法对NRMSE的总述使用range归一化，而Source Data/代码列又包含SD版本。因此以后必须写明具体列名和来源，例如“HEtoSGEBench官方代码的`nrmse_sd`列”，不能只写“论文NRMSE”。官方核验入口：[Nature Communications论文](https://www.nature.com/articles/s41467-025-56618-y)、[官方代码](https://github.com/SydneyBioX/HEtoSGEBench/blob/main/benchmark%20pipeline/00-CombineDat.Rmd)。

### 4.4 覆盖面缺口

现有报告对“基因表达预测方法”覆盖尚可，但仍缺：

1. **直接通路层：**除PEaRL和DeepPathway外，应补BiSCALE等能够连接预测表达、通路与下游表型的工作；
2. **大规模公开基准：**HEST-1k/HEST-Benchmark能够提供跨器官、跨癌种的PFM+Ridge参照；
3. **下游临床层：**缺少H&E衍生分子签名→pCR、H&E直接pCR，以及形态/分子融合评价；
4. **报告规范层：**缺少TRIPOD+AI、PROBAST、DeLong比较、校准和决策曲线等临床预测报告依据；
5. **同源表征边界：**没有讨论“预测ssGSEA也是由H&E产生”，因此Phase 3不是真实双模态组学融合。

## 5. 三层文献证据矩阵

说明：下表优先回答“这篇文献能支持哪一句话”，不把不同任务数值排成统一榜单。`同行评审`包括期刊和正式会议；`预印本`均未同行评审。

### 5.1 L1 直接空间基因/通路预测（Direct Spatial Prediction）

| 文献 | 状态 | 任务与可引用结论 | 与MPP2的距离 |
|---|---|---|---|
| [PEaRL, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Majumder_PEaRL_Pathway-Enhanced_Representation_Learning_for_Gene_and_Pathway_Expression_Prediction_WACV_2026_paper.html) | 同行评审会议 | H&E→ssGSEA通路；乳腺最佳通路PCC `0.5055±0.0271` | **最近**；但队列、通路、切分和聚合不同 |
| [BiSCALE, Advanced Science 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13159098/) | 同行评审期刊 | 同时预测bulk与spot表达，并检验通路恢复、风险分层和细胞身份 | 可支持“上游表达预测应连接下游表型”，不是同一30通路任务 |
| [DeepPathway, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.07.21.665956v1) | 预印本 | H&E→UCell通路分数 | 直接任务接近；教程值证据低于正式表格 |
| [Digital Modeling of Spatial Pathway Activity, arXiv 2025](https://arxiv.org/abs/2512.09003) | 预印本 | 直接建模空间通路活性 | 可作前沿方法线索，暂不用于精确数值比较 |
| [HEtoSGEBench, Nature Communications 2025](https://www.nature.com/articles/s41467-025-56618-y) | 同行评审期刊 | 11种H&E→空间基因方法、5个SRT数据集、28项指标及TCGA外部测试 | 基因级综合基准；不能与通路PCC直接排名 |
| [HEST-1k, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60a899cc31f763be0bde781a75e04458-Abstract-Datasets_and_Benchmarks_Track.html) | 同行评审会议/基准 | 1,229个ST样本、26器官；公开PFM+Ridge基准 | 可支持UNI2-h等图像表征参照；目标为50个HVG |
| [ST-Net, Nature Biomedical Engineering 2020](https://doi.org/10.1038/s41551-020-0578-x) | 同行评审期刊 | 早期H&E→空间基因表达，患者留出 | 基因级历史基线 |
| [Hist2ST, Briefings in Bioinformatics 2022](https://doi.org/10.1093/bib/bbac297) | 同行评审期刊 | Transformer/图结构空间基因预测 | 基因级、模型结构参考 |
| [BLEEP, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract.html) | 同行评审会议 | 图像—表达对比学习与检索式预测 | 基因级；可说明跨模态对齐路线 |
| [TRIPLEX, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chung_Accurate_Spatial_Gene_Expression_Prediction_by_Integrating_Multi-Resolution_Features_CVPR_2024_paper.html) | 同行评审会议 | 多尺度图像上下文预测空间表达 | 基因级；可说明上下文建模 |
| [mclSTExp, Briefings in Bioinformatics 2024](https://doi.org/10.1093/bib/bbae551) | 同行评审期刊 | 多层对比学习的空间基因预测 | 基因级；PEaRL表格中的对照方法 |
| [MISO/LAMIL, Nature Communications 2025](https://www.nature.com/articles/s41467-025-66691-y) | 同行评审期刊 | 大规模组织学→空间基因，强调局部空间上下文与OOD | 基因级；可支持跨样本泛化和空间上下文 |

### 5.2 L2 H&E→转录组/分子表型与通路解释（Molecular Phenotyping）

| 文献 | 状态 | 主要用途 | 对MPP2/Phase 3的启示 |
|---|---|---|---|
| [HE2RNA, Nature Communications 2020](https://doi.org/10.1038/s41467-020-17678-4) | 同行评审期刊 | WSI→bulk RNA表达 | 支持H&E含转录组信号，但不是空间通路任务 |
| [Wang et al., Cancer Research 2021](https://doi.org/10.1158/0008-5472.CAN-21-0482) | 同行评审期刊 | 病理图像预测分子特征 | 可支持形态—分子关联，不支持MPP2数值领先 |
| [Diao et al., Nature Communications 2021](https://doi.org/10.1038/s41467-021-21896-9) | 同行评审期刊 | WSI预测基因表达、分子通路和突变 | 支持通路解释层写法 |
| [SEQUOIA, Nature Communications 2024](https://doi.org/10.1038/s41467-024-54182-5) | 同行评审期刊 | WSI→转录表达，并强调跨癌种迁移 | 支持外部泛化和分子签名派生 |
| [DeepPT/ENLIGHT, Nature Cancer 2024](https://doi.org/10.1038/s43018-024-00793-2) | 同行评审期刊 | 图像衍生转录组/通路用于治疗反应预测 | **Phase 2→临床终点的关键先例**；两阶段证据链，不是空间真值同任务 |
| [Wei et al., American Journal of Pathology 2023](https://doi.org/10.1016/j.ajpath.2023.06.004) | 同行评审期刊 | 由病理预测通路/分子分类 | 适合引用通路分类AUROC思路 |
| [Mondol et al., Cancers 2023](https://doi.org/10.3390/cancers15092569) | 同行评审期刊 | 组织学图像与转录组关联 | 作为邻近证据，不能替代空间验证 |
| [PEKA, MICCAI 2025](https://papers.miccai.org/miccai-2025/0905-Paper4791.html) | 同行评审会议 | PFM参数高效适配用于空间表达 | 仅支持参数高效适配路线有先例，不支持本项目LoRA有效 |
| [SEAL, arXiv 2026](https://arxiv.org/abs/2602.14177) | 预印本 | 大规模PFM适配、线性探针/LoRA/损失消融 | 可解释为何要做适配消融；未同行评审 |
| [DeepSpot-M, medRxiv 2026](https://doi.org/10.64898/2026.06.19.26356060) | 预印本 | LoRA与跨癌种空间基因外部验证 | 可作最新路线参考；未同行评审 |

### 5.3 L3 pCR/MPR临床预测、融合与报告规范（Clinical Response and Fusion）

| 文献 | 状态 | 可核验信息 | 对本项目的服务方式 |
|---|---|---|---|
| [TIGER/Howard et al., medRxiv 2025](https://pubmed.ncbi.nlm.nih.gov/40909848/) | **预印本** | 1,940例；H&E衍生775个基因签名；HER2阴性pCR AUROC `0.794`，临床模型`0.704`，`p=0.001`，并有4个外部队列 | **最接近“图像衍生分子签名→pCR”**；可支持设计，不可冒充同行评审定论 |
| [Zhang et al., Journal of Translational Medicine 2021](https://pubmed.ncbi.nlm.nih.gov/34399795/) | 同行评审期刊 | H&E pCR-score AUC `0.847`；与sTIL/亚型融合AUC `0.890`，基线`0.839`，`P=0.001` | 支持报告融合ΔAUC，而非只报融合模型单一AUC |
| [PROACTING, Breast Cancer Research 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644597/) | 同行评审期刊 | 术前常规活检H&E预测乳腺NAC后pCR | 支持图像单模态基线与可解释组织学特征 |
| [Mao et al., Science Advances 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12042891/) | 同行评审期刊 | 多源特征用于乳腺NAC反应预测 | 支持临床/病理融合框架和外部验证设计 |
| [Cross-modal pathology+ultrasound pCR, 2024](https://pubmed.ncbi.nlm.nih.gov/39237596/) | 同行评审期刊 | 病理与超声跨模态pCR预测 | 真正异源模态案例；与本项目同源双表征不同 |
| [ESCC radiopathomics pCR, 2025](https://pubmed.ncbi.nlm.nih.gov/41423263/) | 同行评审期刊 | 内外部pCR性能分层报告 | 可参考外部队列AUC与过拟合披露 |
| [CLAM, Nature Biomedical Engineering 2021](https://www.nature.com/articles/s41551-020-00682-w) | 同行评审期刊 | 弱监督注意力多实例学习 | 支持slide级聚合架构，不直接证明pCR增益 |
| [MCAT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_Images_ICCV_2021_paper.html) | 同行评审会议 | 图像与真实组学co-attention | 融合结构参考；其真实组学模态与本项目预测ssGSEA不同 |
| [Pathomic Fusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC10339462/) | 同行评审期刊 | 病理、基因组等真实异源模态融合 | 只能作架构背景，不可直接声称同类型多模态 |
| [PORPOISE](https://www.sciencedirect.com/science/article/pii/S1535610822003178) | 同行评审期刊 | 可解释的病理—分子多模态预后建模 | 支持可解释融合与缺失模态讨论 |
| [TRIPOD+AI, BMJ 2024](https://www.bmj.com/content/385/bmj.q902) | 同行评审报告规范 | AI临床预测模型透明报告 | 用作Phase 3论文报告清单 |
| [PROBAST](https://www.probast.org/wp-content/uploads/2020/02/PROBAST_20190515.pdf) | 风险评估工具 | 偏倚与适用性评估 | 约束队列、预测因子、结局和分析设计 |
| [DeLong et al., 1988](https://pubmed.ncbi.nlm.nih.gov/3203132/) | 同行评审统计方法 | 相关ROC曲线AUC比较 | 用于同一测试集上融合与单分支的ΔAUROC比较 |
| [BMJ预测模型评价指南, 2024](https://www.bmj.com/content/384/bmj-2023-074819) | 同行评审方法指南 | 区分判别、校准和临床效用 | 约束Phase 3不能只报AUC |
| [Decision Curve Analysis](https://www.bmj.com/content/352/bmj.i6) | 同行评审方法文章 | 决策曲线与净获益 | 仅在有合理临床阈值和外部验证时使用 |

## 6. 为什么不能把这些论文做成统一排名

跨论文比较至少应同时匹配以下九项：

| 维度 | MPP2当前定义 | 常见论文差异 | 对结论的影响 |
|---|---|---|---|
| 预测目标 | 固定30条ssGSEA通路 | 全基因、HVG、SVG、UCell或基因签名 | 通路通常比稀疏单基因更平滑，PCC不可直接排名 |
| 监督真值 | Phase 2真实转录组派生ssGSEA | bulk RNA、Visium基因、单细胞或伪标签 | 估计对象不同 |
| 预测单位 | patch/spot | patch、spot、slide、patient | 样本量和独立性不同 |
| 数据切分 | 内部6患者，外部XZY单患者 | 随机spot、slide留出、patient留出、跨中心 | 随机spot切分通常更乐观 |
| 聚合 | 患者等权、逐通路平均、另报pooled | spot等权、基因中位数、最佳子集 | 相同PCC名义下estimand不同 |
| 预处理 | 受保护ssGSEA/z-score | log、rank、min-max、UCell等 | 改变误差尺度与相关性 |
| PCC定义 | 逐通路空间PCC/pooled PCC | 逐基因PCC、across-gene PCC | 不能横向相减 |
| NRMSE定义 | 训练期每通路σ | 观测range或观测SD | 数值分母不同 |
| SSIM定义 | 稀疏masked、7×7、range=6 | 插值、背景填充、默认range | 数值不可直接排名 |

因此，当前最稳健的跨论文结论是“处于有竞争力的参考量级”，不是“领先”。

## 7. Phase 2如何为Phase 3服务

### 7.1 论文证据链

建议把合并论文写成四段连续论证：

1. **真实监督：**在Phase 2有真实转录组的患者中，H&E能够恢复30条空间通路分数；
2. **空间与外部稳健性：**恢复的不只是全局均值，而是患者内排序、绝对误差和空间结构，并在独立患者/队列保持；
3. **同源互补表征：**预测通路分数与1536维形态嵌入都来自H&E，但前者提供生物学约束或压缩后的通路表征；
4. **临床增量：**在Phase 3只有H&E的队列中，通路分支相对图像单分支是否改善pCR预测，并检验校准和临床净获益。

如果第4段未改善，第1–2段仍能支撑“可解释的空间通路重建”；但不能支撑“通路分支提高pCR预测”。

### 7.2 当前代码核验结果

用户口述的“逐patch融合”与本地实现一致：

- [`0801-CLAM_README.md`](../../团队项目进度与结论/yzq/0801-CLAM/0801-CLAM_README.md)记录`N×1536`图像特征和`gene_dim=30`；
- [`convert_patch_scores_to_clam.py`](../../团队项目进度与结论/yzq/0801-CLAM/convert_patch_scores_to_clam.py)按patch坐标把通路分数重排到参考特征顺序；
- [`concat_clam_features.py`](../../团队项目进度与结论/yzq/0801-CLAM/concat_clam_features.py)沿特征维拼接两个张量；
- [`model_clam.py`](../../团队项目进度与结论/yzq/0801-CLAM/models/model_clam.py)把每行前1536维和后30维拆成两支，再进行融合和CLAM注意力池化。

这证明**代码设计存在**，不证明Phase 3训练已成功、坐标全量匹配或融合AUC已经改善。原用户路径中的`PFMval\_new`为拼写错误，实际核验路径是`PFMval_new`。

代码中的`gene_dim=30`/`gene_h`是历史变量名；按当前方案其语义是30维**通路分数**而非30个基因。论文、配置说明和后续图表应统一称`pathway scores`，避免实现命名反向误导研究对象。

Phase 2和Phase 3来自同一家医院，使染色流程相近成为合理先验，但尚不能自动视为已证明的批次一致性。后续至少记录并比较取材时点（治疗前/后）、标本类型、固定方式、染色批次、扫描仪型号、倍率和像素分辨率；这些变量若不同，仍可能造成域偏移。

### 7.3 Phase 2现在可以报告的指标

| 层级 | 建议指标 | 用途 | 当前状态 |
|---|---|---|---|
| 主指标 | 逐通路、逐患者PCC和CCC，给中位数/IQR/95% CI | 同时回答排序与一致性 | PCC/CCC已有；外部仅单患者 |
| 主要误差 | z-RMSE与raw R² | 防止只有相关性而绝对值错误 | 已有；raw R²必须并列 |
| 辅助排序 | Spearman、top-k通路重叠、通路排名一致性 | 更贴近Phase 3对相对高低的利用 | 建议新增，当前不是已完成结果 |
| 空间结构 | 项目SSIM、Moran’s I差异、热点区域重叠 | 证明空间图样而非仅均值 | 部分已有；跨论文需降级 |
| 稳定性 | patch→slide聚合后的通路均值/分位数稳定性 | 对接CLAM的slide级预测 | 建议新增 |
| 迁移性 | 跨患者、跨批次、跨扫描仪性能与下降幅度 | 回答Phase 3域转移 | 当前不足，XZY仅单患者 |
| 互补性 | 通路分支与图像分支的相关、冗余和条件增量 | 证明不是简单重复编码 | 建议在Phase 3消融中完成 |

### 7.4 Phase 3未来必须计算的指标

#### 主终点：pCR

1. 图像单分支、预测ssGSEA单分支、简单后融合、逐patch融合四组；用户已计划前三组，建议加入“简单后融合”作为低复杂度对照；
2. AUROC及95% CI；
3. AUPRC及95% CI，尤其在pCR比例不平衡时；
4. 融合相对图像单分支的ΔAUROC和配对DeLong检验；
5. 预先固定阈值下的敏感度、特异度、PPV、NPV和F1；
6. 校准截距、校准斜率、校准曲线和Brier score；
7. 有明确决策阈值时再报告DCA净获益。

#### 次终点：MPR

- 作为次要或探索性终点单独报告，不能与pCR混合标签；
- 样本量不足时应给置信区间并明确探索性，不以不显著等同于无效。

#### 必要的负对照

- 在患者间打乱30维ssGSEA与图像的对应关系；
- 使用随机通路或不相关通路集；
- 保持模型容量相近的随机30维/主成分对照；
- 检查融合增益是否仅来自额外参数量。

这些对照能回答审稿人最可能提出的问题：增益来自通路生物学，还是来自增加了30个由同一H&E计算的数值和更多参数。

### 7.5 最小论文指标组合

在实验成本受限时，建议至少保留：

| 环节 | 最小必报集合 |
|---|---|
| Phase 2主表 | PCC、CCC、z-RMSE、raw R²；患者/通路分布；95% CI；真实与预测空间图示 |
| Phase 2补充 | 项目SSIM、Moran’s I、30通路逐项值、聚合与Bootstrap定义 |
| Phase 3主表 | 四组消融的AUROC、AUPRC、95% CI；融合相对图像单分支的ΔAUROC |
| Phase 3校准 | Brier score、校准截距/斜率、校准图 |
| Phase 3稳健性 | 外部/时间留出、pCR比例、亚型分层、负对照 |
| 解释性 | 关键通路的方向、注意力区域、通路分支对单病例预测的贡献 |

## 8. 两种论文表述路线

### 8.1 强主张路线：目前证据不足，保留为未来条件路线

只有同时满足以下条件，才考虑写“improves”或“outperforms”：

1. 在同一预先固定测试集上，逐patch融合相对图像单分支有正ΔAUROC；
2. 95% CI和配对检验支持该差异，不依赖单次随机种子；
3. AUPRC与校准不恶化；
4. 外部或时间留出队列方向一致；
5. 打乱/随机通路负对照不能复制增益；
6. Phase 2至少有跨患者证据，而不只是单个XZY患者空间块。

当前不能写：

- “MPP2显著优于PEaRL或HEtoSGEBench”；
- “预测转录组为pCR提供了独立组学信息”；
- “融合模型具有临床效用”；
- “外部队列验证成功”（当前外部只有一个患者）。

### 8.2 保守路线：当前更稳妥

建议主叙事：

> 我们首先在具有配对空间转录组的Phase 2队列中，评估由常规H&E推断30条通路活性的可行性、误差与空间一致性；随后在仅具有H&E和治疗反应标签的Phase 3队列中，将该通路表征作为生物学引导的H&E衍生特征，与UNI2-h形态表征进行同源双分支融合，并检验其相对图像单分支对pCR预测的增量价值。

若融合效果不理想，可如实写：

> 预测通路表征提高了模型解释的生物学可读性，但在当前样本量和队列下未证明其对pCR判别具有稳定增量；结果提示上游通路保真度、域迁移和同源特征冗余仍是限制因素。

这一路线仍能形成完整论文问题，不需要把弱消融包装成成功。

## 9. 对两份原报告的建议处理方式

用户已要求不修改原文件，因此建议把本报告作为“审查附录/勘误与扩展入口”。后续如用户决定正式修订，才另开任务处理原报告。

### DVC1 数据验证收口（Data Verification Closure，数据验证收口）

- 保留现有核心数值；
- 补充多重比较、SSIM 11个block、两类Δ估计量和Moran权重定义；
- 明确XZY是单患者而非外部队列。

### SRC2 来源分类（Source Classification，来源分类）

- 给每条外部数值增加`正式表格 / Source Data重算 / 作者代码 / 教程 / 图中估读`标签；
- 给每篇文献增加`同行评审 / 预印本`标签；
- 图中数值一律写“图中估读值”并附原图或论文链接。

### CMP3 可比性门控（Comparability Gate，可比性门控）

- 只有通路目标、队列切分、聚合和指标定义近似匹配时才允许并列表格；
- PEaRL可作最近量级参考，但不是统计对照组；
- HEtoSGEBench、HEST等放入邻近基准层，不计算“领先百分比”。

### PBG4 阶段桥接（Phase Bridging，阶段桥接）

- Phase 2增加排序、聚合稳定性和跨患者迁移指标；
- Phase 3按图像、通路、简单融合、逐patch融合进行消融；
- 把ΔAUROC、校准、AUPRC和负对照设为“增量价值”的判定门。

## 10. 最终风险登记

| 风险 | 当前等级 | 是否阻断内部参考 | 是否阻断强论文主张 |
|---|---|---|---|
| 核心复算数值错误 | 未发现 | 否 | 否 |
| 文档状态叙述过期 | Moderate | 否，本报告已补充 | 是，若不纠偏 |
| 单外部患者 | Major | 否 | 是 |
| raw R²为负 | Major科学限制 | 否 | 是，若只报PCC |
| 多重比较未校正 | Major | 否，限描述性 | 是，若称显著 |
| 自定义SSIM跨论文比较 | Major | 否 | 是 |
| 预印本与正式论文混标 | Moderate | 否 | 是，若不标状态 |
| 同源双表征误称真实多模态 | Major | 否 | 是 |
| Phase 3仅报AUC | Major | 否 | 是，临床预测报告不足 |
| 消融效果弱 | 结果风险而非真实性问题 | 否 | 不阻断保守路线；阻断强路线 |

## 11. 审查后的简明结论

1. MPP2的`PCC 0.519386 / CCC 0.398858 / z-RMSE 0.767225 / raw R² -0.088017`是真实可复算的，不是凭空引用；但它们共同描述一个“相关性尚可、绝对值拟合仍弱”的模型。
2. 原论文对比没有发现明显伪造，但来源层级与审稿状态标记不够统一；尤其要区分正式表格、Source Data重算、作者教程和预印本。
3. PEaRL是最近的直接任务参照，BiSCALE、HEtoSGEBench和HEST补足外部基准；DeepPT/ENLIGHT和TIGER补足“图像衍生通路/签名→治疗反应”的桥梁。
4. Phase 2最重要的新增价值不是再堆一个相关系数，而是证明通路表征能在患者间迁移、经slide聚合稳定，并在Phase 3相对图像单分支提供可检验的增量。
5. 当前最推荐的论文定位是：**真实转录组监督的空间通路重建 + 仅H&E队列中的同源形态—通路双表征pCR预测**。pCR为主终点，MPR为次要/探索性终点。
