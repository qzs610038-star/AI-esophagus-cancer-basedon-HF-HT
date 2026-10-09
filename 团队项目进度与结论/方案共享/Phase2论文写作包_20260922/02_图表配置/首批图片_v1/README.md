# Phase 2 首批图片 · 中文审阅版 v1

首版日期：2026-09-22。已制作4张主图和1张补充图。先看PNG确认信息、颜色和布局；修改文字或布局可用SVG，插入论文可用矢量PDF。图号为写作包工作编号。

| 图片 | 本版内容 | 高清预览 | 可编辑矢量 | 论文插图 |
|---|---|---|---|---|
| 图P2-1 | 数据集合、全视野流程、选模边界和下游接口 | [PNG](exports/Fig_P2_1_data_flow.png) | [SVG](exports/Fig_P2_1_data_flow.svg) | [PDF](exports/Fig_P2_1_data_flow.pdf) |
| 图P2-2 | H/C/B结构、同切片邻域、孤立点处理与三臂矩阵 | [PNG](exports/Fig_P2_2_model_structure.png) | [SVG](exports/Fig_P2_2_model_structure.svg) | [PDF](exports/Fig_P2_2_model_structure.pdf) |
| 图P2-3 | 内部/外部双PCC，三编码器三臂；同种子空间分支增量 | [PNG](exports/Fig_P2_3_编码器与空间分支.png) | [SVG](exports/Fig_P2_3_编码器与空间分支.svg) | [PDF](exports/Fig_P2_3_编码器与空间分支.pdf) |
| 图P2-4 | 稠密、规则隔点、随机等量；等更新次数主分析 | [PNG](exports/Fig_P2_4_训练密度消融.png) | [SVG](exports/Fig_P2_4_训练密度消融.svg) | [PDF](exports/Fig_P2_4_训练密度消融.pdf) |
| 图P2-4S | 同轨迹前50个完整训练轮次的敏感性分析 | [PNG](exports/Fig_P2_4S_等50轮敏感性分析.png) | [SVG](exports/Fig_P2_4S_等50轮敏感性分析.svg) | [PDF](exports/Fig_P2_4S_等50轮敏感性分析.pdf) |

配套[完整图注](首批图片图注.md)可供后续合稿。全套图片也可通过[浏览图集](图集.html)顺序查看。

## 这一版的表现方式

- 示意图用方框、箭头和抽象邻域；不引入病理底图或未确认的物理尺度。
- 结果图用空心均值标记、样本标准差误差线和浅色逐种子点。样本SD描述种子45/46/47之间的训练波动，不是患者置信区间。浅色点的纵向错开只为避免遮挡。
- 同一指标的内部/外部面板使用相同坐标范围；两种PCC分开。PCC是相关系数，本版使用局部刻度范围的点图，完整刻度明确标注，不用柱长暗示从零开始的幅度。
- 图P2-3的E/F面板按相同种子计算spatial−no_b，再汇总均值与样本SD；没有使用两组SD相减，也没有新增显著性检验。
- 图P2-4的点阵仅表达采样方式，不是真实切片坐标；训练数量取自实际记录。图结构统计已经定位，本批先聚焦主性能与选模敏感性，后续可增加邻域连接面板。

## 数据与可编辑材料

[数据来源及核对记录](data/source_manifest.json)逐项列明Registry、运行汇总、接纳日期和筛选条件。制图表保留全部绘制数值，便于改配色或重排面板：

- [主实验汇总](data/main_summary.csv)、[主实验逐种子](data/main_per_seed.csv)；仅最终消融，3编码器×3臂×3种子。
- [密度汇总](data/density_summary.csv)、[密度逐种子](data/density_per_seed.csv)；两种选择器分别保存。
- [空间增量汇总](data/paired_spatial_gain_summary.csv)、[空间增量逐种子](data/paired_spatial_gain_per_seed.csv)；从已接纳逐种子指标派生。
- [训练图结构统计](data/density_graph.csv)；本批未画入主图。

`source`目录保留制图和图集生成的辅助源文件。它们只读取现有小型记录并生成图片，不是正式实验训练包。字体为Microsoft YaHei（微软雅黑）；PNG为300 dpi，SVG保留文本节点，PDF使用矢量图元和嵌入字体。SVG在缺少该字体的电脑上可能替换字体，此时优先使用PDF查看原排版。

## 已完成检查

2026-09-22已逐张查看5张PNG，检查中文、箭头、坐标刻度、图例、数值与误差线。主实验指标与运行汇总一致；密度主选择器登记与运行指标的最大浮点差异约为1.16×10⁻¹¹。每个结果标记的均值和样本SD均与对应三种子一致，详见[渲染检查](render_checks.json)。PNG尺寸、PDF页数与SVG文本节点完成检查。

图P2-5待任务二结果，图P2-6待空间图输入与选择规则，图P2-7待UMAP表征和参数配置；本批未绘制这三项。原始实验、权重、结果和登记未修改。
