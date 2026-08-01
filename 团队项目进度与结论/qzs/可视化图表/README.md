# qzs 可视化图表

本目录集中存放 qzs 项目进展文档直接引用的图表。图表按实验分目录保存，复制的既有图表保留原文件名，新绘图表同时优先保存 PNG 和 SVG。

| 子目录 | 内容 | 主要数据来源 |
|---|---|---|
| `01_基线与数据修复` | MPP2 修复前后指标比较 | `experiments/experiment_registry.json` |
| `02_LoRA配对` | S0 冻结对照与 S1 LoRA 诊断图 | 既有 LoRA 报告绘图数据与预测补充材料 |
| `03_LoRA_dropout` | dropout=0、0.10 与冻结对照比较 | Registry 与各实验 `result.json` |
| `04_Ridge校准` | Ridge 外部指标、逐通路变化和空间区间 | r003 `metrics.json`、CSV 与空间分析 JSON |
| `05_MSE与Huber` | MSE/Huber pooled PCC、通路方向统计和平均逐通路 Raw R² | Registry、G14 回执和科学记录 |

所有图表都必须与正文保持相同的证据限制。例如，“17/30 通路改善”只是方向计数，不能替代总体 pooled PCC 的比较。
