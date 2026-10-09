# Phase 2 中英文论文初稿 v2

日期：2026-09-28。本版保留 v1 的首批三臂、训练密度结果及五张图，并加入第二批六编码器空间臂和任务二单次基因预测后通路重建结果。v1 目录保持原样。

| 文件 | 内容 |
|---|---|
| [中文初稿](Phase2中文初稿.md) | 摘要、背景、结果、讨论、方法及图表说明 |
| [English draft](Phase2_English_Draft.md) | 与中文稿对应的英文正文和图注 |
| [拟引用文献](Phase2拟引用文献.md) | 既有文献及 H-optimus-0、H-optimus-1、Phikon-v2 官方来源 |

两稿引用首批 [五张图](../02_图表配置/首批图片_v1/README.md)，新增[图 P2-3S 六编码器](../02_图表配置/第二批图片_v2/exports/Fig_P2_3S_six_encoders.png)和[图 P2-5 基因路线](../02_图表配置/第二批图片_v2/exports/Fig_P2_5_gene_route.png)。表 P2-5 汇总第二批六种冻结编码器的空间臂三种子结果；表 P2-4 将任务二一次来源运行与既有 UNI2-h 空间模型三种子结果描述性并列。论文采用两种 PCC 定义：患者—通路等权均值和整体展平相关；三种子误差为样本标准差，任务二单次结果无跨运行误差线。

数值依据[第二批本地汇总](../../../../experiments/results/phase2_backbone_spatial_ablation_batch2_20260919/20260927_154318_475_2242f53b/analysis/six_model_summary.csv)、[任务二本地汇总](../../../lzd/task2_gene_pathway_v001/delivery_20260923/analysis/task2_summary.csv)及[描述性并列表](../../../lzd/task2_gene_pathway_v001/delivery_20260923/analysis/baseline_descriptive_comparison.csv)。既有 V003 通路目标与任务二的 319 基因背景 ssGSEA/ES 重构评分分别在方法中说明；两路线的图仅作描述性并列。第二批结果已登记为 accepted；任务二为 partial/accepted，正文仅纳入首批单次运行。更详细的科研有效性审查和补证事项记录在写作包根目录的内部[严谨性与待补证据说明](../严谨性与待补证据说明.md)，不作为论文正文。

图号及文献号为 Phase 2 临时编号，合稿时再统一。