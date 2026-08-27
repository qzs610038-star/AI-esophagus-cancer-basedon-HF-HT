# 从 H&E 衍生空间通路表征到治疗反应桥接：证据受限的双阶段计算病理框架

> 稿件状态：`pending_user_review`；通用期刊型 IMRaD 初稿。
> 底层已接纳证据快照：本地 `main` 提交 `9859a3c`；探索性复算版本：`3d0cad7`；论文包基线提交：`216083f`。稿件仍待用户审核。
> 本稿不是投稿定稿；伦理审批编号、作者、机构、队列完整纳排标准及目标期刊格式待作者补充。

## 摘要

### 背景

从常规苏木精-伊红（H&E）图像推断空间基因或通路表达，为缺少空间转录组测量的队列提供了生物学表征候选。然而，相关性较高并不等于绝对一致性或跨患者泛化；在拟议的端到端设计中，由同一 H&E 图像衍生的形态与通路分支也不构成独立的图像—组学多模态证据。本研究建立的是受证据边界约束的 Phase 2→Phase 3 论文叙事，而非已完成流程。

### 方法

Phase 2 使用已接纳的 W007 四臂实验：冻结基线复放（FBR）、仅残差对照（RCC，保留历史臂代码）、硬配对对比残差（HCR）和连续通路几何对比残差（CPGCR）。四臂共享冻结基线和 30 维通路输出；CPGCR 以连续通路距离构造软关系目标，但推理仅需图像。Phase 3 使用 W004 四臂受限结果组：绝对值输入（A001）、切片内序数输入（A002）、真实坐标空间平滑（A003）和坐标打乱空间对照（A004）。我们区分 per-fit 均值与重复行 pooled 指标，补充 Pearson 相关系数（PCC）、一致性相关系数（CCC）、z-RMSE、raw R²、Spearman、AUROC、AUPRC、Brier 及校准诊断。外部工作采用定向来源核验，不构成系统综述。

### 结果

W007 单随机种子四臂结果均已正式接纳，但差异小且方向混合。内部验证中 RCC 的患者等权 z-MSE 最低（0.378252），外部单病例 E1 中 HCR 的 pooled PCC 最高（0.655982），而 CPGCR 的 z-MSE（0.656832）、raw MAE（1173.902）和平均逐通路 raw R²（−0.080119）数值最优；没有任何一臂支配所有指标。W004 中，A002 相对 A001 的病理完全缓解（pCR）平均 per-fit AUC 差值为 +0.023182，主要病理缓解（MPR）为 +0.003958；A003 相对 A004 的差值分别为 +0.002955 和 −0.000625。该既有结果组保持 `COMPATIBILITY_ONLY`：使用 legacy z-score 输入，且重复切片级留出存在患者重叠，因此不构成患者独立的 out-of-fold（OOF，折外）预测、融合协同或临床效用证据。

### 结论

作者将 CPGCR 定位为待验证的 Phase 2 方法候选，而不是性能最佳模型；其 30 维输出只具有未来接口的形状兼容性，尚无 W007→W004 数据绑定。W004 的 ordinal 信号值得进入新的患者独立验证，但当前空间对照不支持稳定空间增益。患者独立 Phase 3、校准、容量匹配负对照和新鲜外部验证是证据升级的必要条件。

**关键词：** 计算病理；空间转录组；通路表达；对比学习；治疗反应；pCR；校准；患者级验证

## 1. 引言

空间转录组把组织形态与局部分子状态联系起来，但其成本、组织消耗与技术要求限制了大规模临床应用。ST-Net、TRIPLEX、HEST-1k 及近期单细胞/生成式工作表明，H&E 形态中包含可用于预测局部表达的信号。然而，2025 年 Nature Communications 基准比较 11 种方法、5 个数据集和 28 项指标后强调：单一相关系数不足以判断转化价值，跨数据集、下游任务和多维评价可能改变模型排序（[官方论文](https://www.nature.com/articles/s41467-025-56618-y)）。

通路级目标有两个潜在优势。第一，通路聚合比单基因稀疏表达更接近生物过程；第二，通路可以作为 H&E 表征中的生物学瓶颈。PEaRL 已在 WACV 2026 正式发表，直接探索 H&E 到基因与通路表达的预测（[CVF](https://openaccess.thecvf.com/content/WACV2026/html/Majumder_PEaRL_Pathway-Enhanced_Representation_Learning_for_Gene_and_Pathway_Expression_Prediction_WACV_2026_paper.html)）。DeepPathway 于 2026 年 8 月以 Bioinformatics accepted manuscript 上线，进一步说明通路级对比框架是当前研究方向。但 HESCAPE 的反例提示，跨模态对比预训练可能改善某些下游分类，却降低直接表达预测；因此“使用对比学习”本身不是性能改善证据。

另一项关键问题是上游代理分子表征能否支持临床结局。ENLIGHT-DeepPT 证明了“H&E→推断转录组→治疗反应”的两阶段研究范式在大规模多癌种和独立队列中具有先例（[Nature Cancer](https://pubmed.ncbi.nlm.nih.gov/38961276/)）。但 MCAT、SurvPath、Pathomic Fusion 等工作使用独立测量的真实组学；本项目拟议的 future integration 若由同一 H&E 同时产生形态和通路分支，更准确的术语是“H&E 衍生的形态—通路双表征”，而非独立模态。当前 W004 legacy 输入的完整上游来源链未在本任务中重新核验，因此该同源表述不作为 W004 已证实事实。

本研究围绕三个问题组织现有证据：（1）连续通路几何对比方法在严格匹配四臂中的实现与性能边界是什么；（2）PCC、CCC、误差与校准如何共同揭示模型所学的是趋势还是绝对一致性；（3）absolute、ordinal 和空间反事实输入能否为 Phase 3 pCR/MPR 提供受限桥接信号。我们的目标不是以现有小样本证明临床效用，而是建立可审查、可复算、不会越界的双阶段论文准备框架。

## 2. 方法

### 2.1 研究设计与证据层级

本研究是对既有本地实验的治理化回顾与探索性再分析，不连接项目服务器、不下载模型或训练数据，也不重新训练模型。项目事实的来源顺序为：Experiment Registry 已接纳结果、本地 `pending_user_review` 报告与新复算、Google Drive 设计建议、官方外部文献。该顺序是事实来源与生命周期管理，不是临床科学证据等级；`accepted` 只表示项目内治理接纳。外部研究另按研究设计等级与来源适配度评价。

W007 为项目内接纳的单随机种子 Phase 2 证据；W004 为项目内接纳且当前结果组限定为 `COMPATIBILITY_ONLY` 的四臂组；所有新复算均为 `exploratory_reanalysis / pending_user_review`。W004 的审批、attempt 和事务细节保留在内部审核索引，不进入科学主文。

### 2.2 Phase 2：W007 四臂表征学习

每个 spot 输入 1536 维冻结病理基础模型特征，输出 30 维通路 z-score 预测。FBR 复放已接纳的冻结基线 B0。RCC 在 B0 上加入 `1536→256` 图像投影器和零初始化 `256→30` 残差头；它是仅残差协议对照，不能单独证明纯容量因果效应。HCR 在相同预测结构上加入双向硬配对 InfoNCE。CPGCR 将真实 30 维通路向量之间的均方距离转换为 batch 内软目标分布，以连续关系取代“一正样本/其余全负”的假设。HCR/CPGCR 的真实通路编码器只用于训练，推理仍为 image-only。

绝对回归损失按患者等权计算。每个 batch 包含 6 名患者、每名患者 5 个 spot；优化器为 Adam，学习率 1×10⁻⁴；最多 50 个 epoch，patience 10；checkpoint 只依据 internal patient-balanced z-MSE 选择，外部病例 E1 不参与选模。完整训练源码绑定历史 source commits，当前 `main` 不包含完整可执行 W007 训练树。

### 2.3 Phase 2 评价指标

逐通路、逐患者计算 Pearson 相关系数（PCC）、一致性相关系数（CCC）、z-RMSE、raw R² 和 Spearman，再以患者等权方式汇总。PCC 衡量线性趋势；CCC 同时惩罚均值和尺度偏差。诊断性差距定义为 `PCC−CCC` 和 `PCC²−raw R²`。均方误差分解为均值偏移、尺度差异和相关结构误差。坐标完整时才计算 Moran’s I 或 hotspot 指标；缺失时记录 `not_computable`，不重建受保护数据。

### 2.4 Phase 3：W004 四臂桥接

项目材料将 W004 定位为食管鳞状细胞癌（ESCC）新辅助治疗反应任务；关键合同记录 78 位患者与 307 张有效切片。终点为病理完全缓解（pCR）和主要病理缓解（MPR），但病理阈值、评估者、盲法、事件数、中心及时间范围尚待作者补充，本文不作推测。

W004 对 pCR 和 MPR 分别训练 4 seeds×5 folds，即每个终点每臂 20 fits，两个终点合计每臂 40 fits；20 个 fit 也不是 20 位独立患者。每张切片的 patch bag 包含 1536 维图像特征、30 维通路输入和坐标。双分支映射至 hidden 128，经 adaptive gate 融合、多实例学习（MIL）注意力聚合并输出二分类 logit。使用患者加权二元交叉熵（BCE），AdamW 学习率 1×10⁻³，最多 20 epoch，patience 5，并按 validation BCE 选模。

A001 使用 absolute legacy z-score 输入；A002 在每张切片内对每条通路转换为百分位秩；A003 按真实坐标执行 k=5、lambda=0.5 的 kNN 平滑；A004 先确定性打乱坐标，再执行相同平滑。主要预设比较为 A002−A001 和 A003−A004。

### 2.5 Phase 3 评价与统计边界

每个 fit 计算 AUROC、AUPRC 和 Brier score；四臂差异在相同 task/seed/fold 上配对。重复行 pooled ROC/PR 仅作为描述性视图，与 mean per-fit 指标分开。按 `case_id` 聚合的分析是 post hoc，不能称患者独立 OOF。校准截距和斜率仅在标签与概率满足计算条件时提供，并在标题中注明患者非独立。

20 个 seed-fold 不是 20 个独立患者，不用于普通 t 检验。当前没有可信的患者级 OOF，因此不执行 DeLong 或临床决策曲线主张。未来验证应按患者完整分组，并采用患者级配对 cluster bootstrap。

### 2.6 文献检索与报告规范

检索日期为 2026-08-27，采用定向来源核验而非系统综述。共执行 78 个标题/主题核验查询，纳入 44 篇；32 篇发表于 2024–2026，43 篇已同行评审，唯一预印本单独降级。来源限于 PubMed、期刊官网、DOI、CVF、NeurIPS Proceedings 和 BMJ。

TRIPOD+AI（[BMJ](https://www.bmj.com/content/385/bmj-2023-078378)）与 PROBAST+AI 仅用于稿件准备参考；正式逐条 TRIPOD+AI 映射和 PROBAST+AI 四域问答尚未完成。两者不互相替代，也不因使用清单而自动证明低偏倚。

### 2.7 伦理、隐私与数据可用性占位

伦理委员会全称、批准号、批准日期、回顾性二次分析范围、书面知情同意或豁免依据、去标识化流程和数据访问权限尚未由作者提供，投稿前为阻断项。公开稿仅使用“外部病例 E1”，不公开内部患者代码。本文未向 Google Drive 或公开 Web 服务写入患者级文件；AI 平台的数据处理合规性及机构政策仍须由项目负责人确认。

## 3. 结果

### 3.1 W007 四臂没有总体性能赢家

**Table 1. W007 已接纳四臂指标；single seed，外部仅一例。**

| 臂 | Internal patient-balanced z-MSE ↓ | Internal pooled PCC ↑ | External E1 z-MSE ↓ | External pooled PCC ↑ | External raw MAE ↓ | External mean pathway raw R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| FBR | 0.380743 | 0.797126 | 0.662077 | 0.654896 | 1176.211 | −0.088017 |
| RCC | **0.378252** | **0.798154** | 0.659534 | 0.653758 | 1177.107 | −0.087571 |
| HCR | 0.379641 | 0.797706 | 0.658109 | **0.655982** | 1174.598 | −0.082654 |
| CPGCR | 0.378847 | 0.798000 | **0.656832** | 0.654932 | **1173.902** | **−0.080119** |

RCC 在内部绝对误差上数值最好，HCR 在外部病例 E1 的 pooled PCC 上最高，CPGCR 在 E1 多个误差类指标上数值较好。差异均小，且 E1 只有一名患者；因此结果支持“指标特异性变化”，不支持总体优胜。作者基于连续通路几何和匹配消融把 CPGCR 定位为待验证方法候选；30 维输出仅具未来接口形状兼容性，不表示 Phase 3 已实证绑定。

### 3.2 多指标再分析的角色

新复算先在每个 patient×pathway 内计算指标，再对患者等权，并将 PCC/CCC、z-RMSE、raw R²、Spearman、PCC−CCC、PCC²−R² 和 MSE 分解写入 `experiments/explorations/phase2_phase3_reanalysis_v002/`。内部 6 位患者、30 个通路的等权均值中，RCC 的 PCC/CCC/raw R² 数值最高（0.6504/0.6112/0.2058），CPGCR 为 0.6503/0.6103/0.1990；差异很小。外部单病例 E1 中，四臂逐通路平均 PCC 为 0.5165–0.5194、CCC 为 0.3978–0.3993，raw R² 均为负（−0.0880 至 −0.0801）。这些指标为 `pending_user_review` 的探索性派生结果，不改变 W007 Registry，也不能产生新的“赢家”。

W007 坐标完整，840 个 patient×pathway×arm/split 单元可计算 Moran’s I；热点仅为无置换检验的象限计数。它们描述单个病例内部空间结构，不推断外部患者总体。患者平衡图见 `figures/figure3_w007_patient_balanced_metrics.png`。

### 3.3 W004 ordinal 信号强于空间信号，但仅限兼容性证据

**Table 2. W004 固定四臂 AUROC 对比；`COMPATIBILITY_ONLY`。**

| 比较 | 终点 | 对照均值 AUC | 处理均值 AUC | 差值 |
|---|---|---:|---:|---:|
| A002 ordinal − A001 absolute | pCR | 0.838636 | 0.861818 | **+0.023182** |
| A002 ordinal − A001 absolute | MPR | 0.871667 | 0.875625 | +0.003958 |
| A003 spatial identity − A004 shuffle | pCR | 0.837273 | 0.840227 | +0.002955 |
| A003 spatial identity − A004 shuffle | MPR | 0.872292 | 0.871667 | −0.000625 |

pCR ordinal 差值支持进一步患者独立验证；MPR 差值较弱。空间 identity 与 shuffle 的差值接近零且跨终点方向不一致，不支持稳定空间增益。Registry 中臂级 AUPRC/Brier 随四臂组在 `COMPATIBILITY_ONLY` 范围内接纳；本次新生成的 pooled、case-aggregated 和校准结果仍为探索性待审。pooled test 描述中，ORD 的 pCR AUPRC/Brier 为 0.8236/0.1361，ABS 为 0.8132/0.1458；MPR 对应差异很小。由于 620 行来自重复 fit、不是 620 位独立患者，不能作患者级推断。W004 prediction CSV 不含坐标列，因此 Moran’s I/hotspot 登记为 `not_computable`，没有重建受保护数据。

### 3.4 新颖性位于“连续通路几何 + 受约束残差 + 证据桥接”的组合

文献已覆盖通路级预测（PEaRL、DeepPathway）、跨模态表示（BLEEP、TANGLE、OmiCLIP）、空间表达基准（HEST-1k、HE-to-SGE benchmark）和预测表达到临床结局（ENLIGHT-DeepPT、PathGen）。因此不能主张“首次使用通路对比学习”。本项目可检验的差异在于：（1）连续 30 维 ssGSEA 几何软目标；（2）accepted B0 冻结锚点与零起点残差；（3）FBR/RCC/HCR/CPGCR 匹配消融；（4）显式区分相对趋势、绝对一致性和下游治疗反应桥接；（5）对阴性和受限证据采用治理化分层。

## 4. 讨论

本研究最重要的发现不是某一臂“获胜”，而是不同评价维度揭示了不同问题。W007 中 RCC、HCR 和 CPGCR 的数值赢家不同，提示小样本下的相关性、绝对误差与逐通路 R² 不能互相替代。PCC 可能在整体平移或缩放后仍较高；因此 CCC、R² 和误差分解是判断“趋势已学到但数值未校准”的必要补充。Lin 的 CCC 和连续结局验证方法学均支持这一评价框架（[Lin 1989](https://pubmed.ncbi.nlm.nih.gov/2720055/)）。

W004 的 ordinal pCR 差值提供了一个值得验证的假设：当跨患者绝对幅度不稳定时，切片内相对通路排序可能保留更多可迁移信息。然而，当前协议的患者重叠使该结果容易受到伪重复与病例特征记忆影响。它只能说明四臂兼容性比较中的数值信号，不能证明新患者受益。A003/A004 结果进一步说明，真实坐标平滑并未相对坐标打乱产生跨终点一致增益；这项阴性结果应保留，而不是只展示 pCR 的微小正差值。

双阶段叙事还面临同源信息问题。SurvPath、MCAT 和 Pathomic Fusion 使用真实独立组学，而拟议的 W007→Phase 3 流程将由 H&E 推断通路分支。PathGen 是更接近的同源合成表达先例。当前 W004 legacy 输入来源链尚未重新核验；未来若要声称通路分支提供增量，需先绑定来源，再设置 image-only、pathway-only、direct concat、learned fusion，以及 PCA-30、random-30、患者置乱通路和通路身份置乱等负对照。

TRIPOD+AI 要求透明报告参与者、数据划分、性能、亚组和开放性；PROBAST+AI 进一步要求评估参与者与数据、预测因子、结局和分析偏倚。当前最主要的高风险点包括患者非独立拆分、外部仅一名患者、校准不足、多重比较和当前 main 缺少完整 W007 训练树。保留这些限制能使论文形成可信的“方法候选 + 受限桥接 + 明确验证路线”，而不是夸大的临床宣称。

## 5. 局限性

1. W007 只有一个正式随机种子；internal 为 6 名患者，外部病例 E1 为 1 名患者。
2. W004 是重复切片级留出，病例在数据子集间重叠；case 聚合仅为 post hoc。
3. W004 使用 legacy z-score compatibility 输入，未绑定 W007 CPGCR 输出。
4. 当前结果不能证明 raw MPP2、融合协同、患者独立泛化或临床效用。
5. 新复算是派生探索证据，用户审核前不进入 Registry。
6. 当前 `main` 不含完整 W007 训练源码，需未来恢复 source-bound tree 并做独立复现检查。
7. 伦理审批编号、知情同意/豁免、标本时间点、扫描设备和批次信息尚待作者补充。

## 6. 结论与后续工作

现阶段最安全的结论是：作者把 CPGCR 定位为具有明确生物学动机、实现完整的 Phase 2 待验证方法候选，但它没有显示跨指标一致的性能优势；其 30 维输出只具未来接口形状兼容性，尚无下游数据绑定。W004 ordinal pCR 信号支持开展新协议验证，而空间反事实不支持稳定空间增益。完成多 seed W007、Phase 3 患者级 OOF、容量匹配负对照和新鲜外部验证后，才能考虑升级为“性能增量”或“临床价值”主张。

## 数据、代码与 AI 辅助披露

- 本稿仅使用本地已存在的 Registry、回传包、预测表、代码/历史 Git 对象和只读 Google Drive 文档；没有连接项目服务器。
- 新脚本与输出位于 `scripts/explorations/phase2_phase3_reanalysis_v002/` 和 `experiments/explorations/phase2_phase3_reanalysis_v002/`。
- 文献核验、代码静态审查、复算脚本和稿件起草使用 AI 辅助；作者/项目负责人承担最终事实、伦理与表述审核责任。是否满足机构对 AI 处理本地人类数据的政策，仍须投稿前确认。
- 未推送 Gitee/GitHub，未写 Google Drive，未修改原始 W004 回传包或受保护资产。

## 参考文献

正文引用的正式条目与 44 篇完整核验库见 `references.bib`；检索式、出版状态与纳排记录见 `literature_search_log.md`。核心来源包括 PEaRL（WACV 2026）、DeepPathway（Bioinformatics 2026 accepted manuscript）、HESCAPE（ICCV Workshops 2025）、BLEEP（NeurIPS 2023）、H&E→SGE benchmark（Nature Communications 2025）、TRIPOD+AI 与 PROBAST+AI（BMJ）。
