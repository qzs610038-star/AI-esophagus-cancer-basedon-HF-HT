# PFMval 未来探索池

更新时间：2026-09-04  
定位：保存尚未进入正式执行主线、但可能对后续数据或论文有价值的研究构想。本文档不是实验批准单，也不构成结果证据；实验状态与正式结论仍以 `experiments/experiment_registry.json` 为准。

## 使用方式

- 每个方向只维护：研究假说、已有依据、最小验证、启动条件、相关资料和当前处置。
- 状态分为：`后续优先`、`条件性储备`、`暂缓`、`阴性归档`、`已转正式计划`。
- 只有用户决定开展并补齐数据与评价协议后，才将条目升级为正式 `planned` 实验。
- 云端或历史对话中的旧优先级不得覆盖用户最新决定；未运行方案不得写成实验结论。

## 当前处置总览

| 探索方向 | 状态 | 与当前论文关系 | 建议动作 |
|---|---|---|---|
| 基因预测后重建通路 | 已转正式计划 | Phase 2 分配任务 | 补齐基因集、表达矩阵、ID 映射和评分协议后执行 |
| 稀疏训练—稠密测试 | 已转正式计划 | Phase 2 分配任务 | 先冻结规则稀疏与等量随机对照 |
| 多病理模型公平比较 | 已转正式计划 | Phase 2 分配任务 | 先确定模型短名单和统一协议 |
| 趋势—尺度解耦与校准诊断 | 后续优先 | 可直接强化现有结果解释 | 优先利用现有预测结果复算，不新增训练 |
| H&E 染色—频域鲁棒性 | 后续优先 | 本篇论文可不纳入 | 先做低成本机制诊断，再决定是否训练 |
| 通路子空间分解与残差融合 | 后续优先 | 更适合 Phase 3 或后续论文 | 先验证子空间是否稳定、可解码、可擦除 |
| Phase 3 raw／校准值／ordinal 表征 | 条件性储备 | 下游临床桥接 | 等患者级 OOF 与正式通路输出可用后开展 |
| 标签标准化与密集空间推断 | 条件性储备 | 为基因重建和密度实验提供方法参考 | 先统一术语和标签合同，不单独立项 |
| 基因类别诊断与专家模型 | 条件性储备 | 后续扩展 | 先分析 HVG/SVG/零表达等失败类型，再决定是否使用 MoE |
| 六折留一患者验证 | 暂缓 | 可能增强严谨度 | 等周六组会决定，当前不纳入执行基线 |
| LoRA、Dropout、Huber、Ridge 旧路线 | 阴性归档 | 可作为负结果或路线排除依据 | 不在相同合同下继续反复调参 |

## 已转正式计划

### 基因预测后重建通路

- **研究问题**：直接由 H&E 预测 30 条通路，与先预测通路成员基因再重建通路得分，哪条路线更准确、更稳定？
- **当前状态**：已登记为 `phase2_gene_reconstruction_comparison_v001_20260904`，尚未运行。
- **缺失输入**：基因集成员表、spot/bin 表达矩阵、基因 ID 映射、通路评分软件与参数、背景基因宇宙。
- **最小验证**：固定同一测试集，比较 Direct 与 Gene-mediated 两条路径的逐通路 PCC、CCC、误差和空间热点一致性。
- **启动条件**：李博士确认数据与评分合同；若 ssGSEA 依赖全基因排序，不能只用通路成员基因替代背景基因宇宙。
- **相关资料**：
  - [实验注册器](../experiment_registry.json)
  - [Phase 2 与 Phase 3 工作计划原文](../../团队项目进度与结论/lzd/Phase2与Phase3工作计划原文_20260903.md)
  - [Phase 2 补充任务盘点](../../团队项目进度与结论/方案共享/Phase2补充任务现状盘点与Phase3并行推进建议_20260903.md)
  - [论文基线](../../团队项目进度与结论/lzd/基于HE图像的空间通路活性重建用于增强食管鳞癌新辅助免疫治疗疗效预测.md)

### 稀疏训练—稠密测试

- **研究问题**：Visium HD 的空间稠密性是否提供超出训练样本数量本身的增益？
- **当前状态**：已登记为 `phase2_spatial_density_comparison_v001_20260904`，尚未运行。
- **最小验证**：在同一训练/测试合同下比较 Dense、Structured-sparse、Random-matched 三组；测试集始终保持稠密。
- **关键控制**：规则稀疏组与随机组保留相同 patch 数；先完成数据划分，再进行稀疏化；不得预设 Masked SSIM 一定显著占优。
- **启动条件**：明确棋盘格/单轴间隔规则、边界处理、保留率、随机对照重复次数及稠密测试集定义。
- **相关资料**：
  - [实验注册器](../experiment_registry.json)
  - [图像分辨率方案与后续结果](../../团队项目进度与结论/qzs/2026-08-23_图像分辨率方案成果共享_已审核/图像分辨率方案与后续进展.md)
  - [Phase 2 指标学习指南](../../01_指南与解读/学习指南/Phase2_空间转录组核心评估指标体系与论文对比学习指南_20260904.md)
  - [Google Drive：Phase 2 指标与补充实验建议](https://docs.google.com/document/d/1uj4k-4WLgqby13vYYY-ERsTjbrvljmmq8WTwycbK3wc)
  - 历史任务：`梳理 Phase 2 根据分工和论文大纲的后续安排`（任务 ID：`01a06300-3dbb-7f10-b4a3-d10abc5e5a67`）

### 多病理模型公平比较

- **研究问题**：不同病理基础模型在相同数据、回归头、预算和评价协议下，是否存在稳定差异？
- **当前状态**：已登记为 `phase2_pathology_model_benchmark_v001_20260904`，尚未运行。
- **最小验证**：优先选择 2–3 个权重可获得、运行环境可复现的模型；固定 split、目标顺序、标准化、回归头和指标。
- **启动条件**：确定模型短名单、权重与许可证、输入分辨率、特征维数适配和共同训练预算。
- **相关资料**：
  - [实验注册器](../experiment_registry.json)
  - [Beachum 2026 与 Wang 2025 对比报告](../../01_指南与解读/分析报告/Beachum2026与Wang2025_Phase2模型演化及项目实验对比报告_20260822.md)
  - [Google Drive：Wang 2025 学习笔记](https://docs.google.com/document/d/1RjRLV7wvkxavRuj5nN1kUrRu6bliDm6sTpN5FqZNV3c)
  - [Google Drive：模块 3 学习进度总结](https://docs.google.com/document/d/1kIVsVpc2qeGCGnd8ggN7NfEpqEb6U0udxPMwIC_XudA)

## 后续优先探索

### 趋势—尺度解耦与校准诊断

- **研究假说**：模型较好保留患者内空间高低趋势，但跨患者绝对均值、方差和振幅恢复较弱。
- **已有依据**：MPP2 探索性复算出现逐通路 PCC 高于 CCC、raw R² 为负的现象；W007 四臂模型排序随指标变化且差异很小。
- **证据边界**：派生指标仍是探索性复算；XZY 是单外部患者，不能提供跨患者总体证据。
- **最小验证**：使用现有预测文件统一计算患者等权逐通路 PCC、CCC、raw R²、校准斜率/截距、PCC–CCC gap、PCC²–R² gap 和 MSE 分解。
- **启动条件**：统一指标实现、目标顺序、训练期标准化参数和聚合口径；不需要新增训练。
- **相关资料**：
  - [MPP2 首批评价指标复算报告](../../01_指南与解读/分析报告/MPP2首批评价指标复算报告_20260823.md)
  - [MPP2 指标与同类论文对比](../../01_指南与解读/分析报告/MPP2评价指标与同类论文原始数据对比分析_20260823.md)
  - [Phase 2 至 Phase 3 综合审查](../../01_指南与解读/分析报告/MPP2指标复算与Phase2至Phase3论文衔接综合审查_20260823.md)
  - [复算探索产物](mpp2_metric_recalculation_v001/)
  - [Phase 2/3 重分析产物](phase2_phase3_reanalysis_v002/)
  - [Google Drive：联合论文证据升级参考稿](https://docs.google.com/document/d/1QIhkmjh7fIWKrfkKzNb0wtl0j9cmmDK37_r1Kulh294)
  - [Google Drive：Phase 2 指标建议](https://docs.google.com/document/d/1uj4k-4WLgqby13vYYY-ERsTjbrvljmmq8WTwycbK3wc)

### H&E 染色—频域鲁棒性

- **研究假说**：染色风格在光密度空间可能近似低秩，在频谱中常表现为低频幅度变化占优；同 patch 的技术风格变化不应显著改变通路预测。
- **最小验证**：先对同一批 patch 生成 H/E 浓度、颜色基、gamma、白平衡和轻度傅里叶幅度变体，测量像素频段能量以及 UNI2-h 特征差分的有效秩。
- **后续训练条件**：只有诊断支持可控技术扰动后，才比较染色增强、预测一致性、傅里叶幅度增强及其组合；频域方法不能使用 XZY 统计进行选择。
- **停止条件**：若频域增强不能稳定优于单纯染色增强，或技术子空间与通路相关子空间高度重叠，则不推进复杂频域网络或硬投影删除。
- **相关资料**：
  - [Google Drive：H&E 染色与跨批次噪声探索](https://docs.google.com/document/d/197J4sy81oq3I8G1T22rdKDMFg47CY1mCoWBoe-svjg4)
  - [Google Drive：UNI2-h 特征子空间分解](https://docs.google.com/document/d/1WObzCnYTZgnTjZa18M1-0kg366SM-tqWue9QMlUChWA)
  - 历史 GPT 对话：`HE染色噪声分析`（对话 ID：`6a9264c6-2490-83ea-b887-10961c625ae1`）
  - [在线 patch 数据集](../../dataset_mpp_online.py)
  - [UNI2-h 特征工具](../../uni2h/uni2h_utils.py)
  - [在线/缓存特征一致性检查](../../scripts/check_mpp_online_cache_parity.py)

### 通路子空间分解与残差融合

- **研究假说**：UNI2-h 特征中同时存在通路可解码信息与通路之外的补充形态；直接拼接完整图像特征与预测通路会重复输入部分同源信息。
- **推荐主方法**：使用训练数据拟合 `X→G` 的 Ridge 解码器，对权重做 SVD，构造通路子空间与正交残差；PLS 或 LEACE 仅作为稳健性对照。
- **最小验证**：先证明通路子空间可解码、残差中的通路可解码性下降、不同患者折的子空间方向相对稳定，再比较完整特征、通路分数、直接拼接、残差融合和等秩随机投影。
- **启动条件**：Phase 2 通路输出合同稳定；Phase 3 使用患者级划分；所有 scaler、投影、秩选择和分类器调参都只在训练折完成。
- **停止条件**：若残差仍能轻易解码通路，或随机/PCA 擦除能够复现收益，则不能主张“通路特异去冗余”。
- **相关资料**：
  - [Google Drive：UNI2-h 特征子空间分解](https://docs.google.com/document/d/1WObzCnYTZgnTjZa18M1-0kg366SM-tqWue9QMlUChWA)
  - [Google Drive：Phase 2—Phase 3 整合研究方案](https://docs.google.com/document/d/1zwqecm8ZmtEM4EZv6YX3xS6wyUCzeN6K2BwMVxfr03w)
  - [Google Drive：联合论文证据升级参考稿](https://docs.google.com/document/d/1QIhkmjh7fIWKrfkKzNb0wtl0j9cmmDK37_r1Kulh294)
  - 历史 GPT 对话：`消融实验设计`（对话 ID：`6a92880d-a680-83e9-93d4-0f7779a8589b`）
  - [Phase 2 至 Phase 3 综合审查](../../01_指南与解读/分析报告/MPP2指标复算与Phase2至Phase3论文衔接综合审查_20260823.md)

## 条件性储备

### Phase 3 raw、校准值与 ordinal 表征

- **研究假说**：如果 Phase 2 更稳定地恢复相对排序而不是绝对尺度，ordinal 或仅用训练数据拟合的 calibrated absolute 可能比 raw 通路值更适合治疗反应预测。
- **已有依据**：W004 中 ordinal 相对 absolute 的 pCR mean-per-fit AUROC 出现 `+0.023182` 的兼容性数值信号。
- **证据边界**：W004 使用 legacy 输入和患者可能重叠的切片级留出，属于 `COMPATIBILITY_ONLY`；不能解释为患者独立泛化或临床效用。
- **最小验证**：严格患者级 OOF 下比较 image-only、pathway-only、直接拼接、raw、calibrated absolute、ordinal、raw+rank，并加入 PCA-30、random-30、患者间通路置乱及通路身份置乱。
- **启动条件**：冻结 Phase 2 推理接口、患者级 split、每名患者唯一 OOF 概率和预设患者级聚合方式。
- **相关资料**：
  - [实验注册器](../experiment_registry.json)
  - [Google Drive：Phase 3 指标与后续建议](https://docs.google.com/document/d/1GKOiohCvCewKD6sAN0eMLohqQmM3Aud8EOTZ-5C1I98)
  - [Google Drive：联合论文证据升级参考稿](https://docs.google.com/document/d/1QIhkmjh7fIWKrfkKzNb0wtl0j9cmmDK37_r1Kulh294)
  - [Phase 2/3 重分析报告](phase2_phase3_reanalysis_v002/review_report.md)
  - 历史 GPT 对话：`完善AI空转项目论文初稿方案`（对话 ID：`6a915148-4a38-83ea-a059-630ecbdc0139`）

### 标签标准化与密集空间推断

- **研究假说**：表达标签的 library-size normalization、log 变换和计数分布约束会影响“相对趋势—绝对尺度”的可恢复性；重叠滑窗可以提高预测地图密度，但不会产生新的独立分子测量。
- **处理方式**：作为基因重建、稀疏/稠密实验的协议参考，不独立作为模型创新。
- **最小验证**：记录原始计数、标准化表达、ssGSEA 输入和训练期尺度之间的映射；将 HisToGene 式结果称为“密集空间推断”，避免称为真实分子超分辨率。
- **相关资料**：
  - [Google Drive：图像直接预测组学（下）](https://docs.google.com/document/d/1lbrHFgCHDKprW-fZ0KuoTtq5CryjSfwcHWrAFMEWpAc)
  - [Google Drive：Wang 2025 学习笔记](https://docs.google.com/document/d/1RjRLV7wvkxavRuj5nN1kUrRu6bliDm6sTpN5FqZNV3c)
  - 历史 GPT 对话：`Histogene与His2ST分析`（对话 ID：`6a92ab82-93c0-83ea-9cc7-8bdcd60947f6`）
  - 历史 GPT 对话：`分析标准化与超分辨率`（对话 ID：`6a92ab96-a3c8-83e9-9619-6085a0d510cd`）

### 基因类别诊断与专家模型

- **研究假说**：不同类别基因或通路的可预测性可能由表达变异、空间自相关、零表达比例和所需感受野决定；单一架构可能不是所有目标的共同最优解。
- **最小验证**：先按 HVG、SVG、零表达比例、Moran's I、通路类别和基线难度分层分析误差，再判断局部 CNN、空间图模型或范例检索是否各有稳定优势。
- **启动条件**：基因级输出路线可运行，且目标数量足以支撑分层分析。
- **停止条件**：若不同类别没有稳定的架构优势，不进入 MoE（Mixture of Experts，混合专家）训练；当前患者数较少，不优先训练多个专家和路由器。
- **相关资料**：
  - [Google Drive：Wang 2025 学习笔记](https://docs.google.com/document/d/1RjRLV7wvkxavRuj5nN1kUrRu6bliDm6sTpN5FqZNV3c)
  - [Google Drive：模块 3 学习进度总结](https://docs.google.com/document/d/1kIVsVpc2qeGCGnd8ggN7NfEpqEb6U0udxPMwIC_XudA)
  - [Beachum 2026 与 Wang 2025 对比报告](../../01_指南与解读/分析报告/Beachum2026与Wang2025_Phase2模型演化及项目实验对比报告_20260822.md)
  - [Google Drive：Wang 2025 benchmark PDF](https://drive.google.com/file/d/1Xrbf_hGzjKUF76ro9ALGkk27L3CWRpge)

## 暂缓

### 六折留一患者验证

- **价值**：比同患者空间块验证更直接回答跨患者泛化，也可与稀疏训练设计组合。
- **当前决定**：用户已明确暂不加入，等待周六组会与项目成员讨论。
- **恢复条件**：组会明确是否开展、主结果还是补充结果、训练预算及其对 Phase 3 进度的影响。
- **相关资料**：
  - [项目当前状态](../../project_state/current_state.json)
  - [用户决定记录](../../project_state/directives.jsonl)
  - [Google Drive：Phase 2 指标建议](https://docs.google.com/document/d/1uj4k-4WLgqby13vYYY-ERsTjbrvljmmq8WTwycbK3wc)
  - 历史任务：`梳理 Phase 2 根据分工和论文大纲的后续安排`（任务 ID：`01a06300-3dbb-7f10-b4a3-d10abc5e5a67`）

## 阴性归档

### LoRA、Dropout、Huber、Ridge 旧路线

- **当前结论**：相关实验已提供重要负结果或边界信息，但没有显示足以取代修复后冻结 MPP2 基线的稳定外部增益。
- **处理方式**：保留原始结果与条件，作为路线排除依据和论文阴性结果；除非数据规模、监督目标或适配方法发生实质变化，不在相同合同下继续重复调参。
- **相关资料**：
  - [图像分辨率方案与后续结果](../../团队项目进度与结论/qzs/2026-08-23_图像分辨率方案成果共享_已审核/图像分辨率方案与后续进展.md)
  - [W007 核心创新方法方案](../../01_指南与解读/部署方案/W007_Phase2核心创新方法_论文强化与必要补充实验部署方案_20260826.md)
  - [实验注册器](../experiment_registry.json)

## 重新评估规则

出现以下任一情况时，可重新审查对应方向：

- 获得新的患者、中心、染色批次、基因表达或外部验证数据；
- 当前三项 Phase 2 补充实验暴露新的主要误差来源；
- Phase 3 完成严格患者级 OOF，并能够验证通路表征的真实增量；
- 用户明确要求将某个储备方向升级为正式实验。

重新评估不自动授权训练。升级时应先把研究问题、输入合同、主比较、评价指标、停止条件和证据边界写入正式实验记录。
