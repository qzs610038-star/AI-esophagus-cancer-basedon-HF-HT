# 历史 319 基因候选

本目录保存旧任务二从 319 个 pre-ssGSEA 基因生成的 30 维候选标签及相关记录。它仅用于定位聚合口径差异，不是当前主线参考真值，也不是已接纳的新真值；即使后续聚合成功复现 V003，本目录仍保留为历史诊断证据。

## 内容

- `generated_task2_common_labels/`：旧直接臂预生成的候选 30 维标签。
- `common_task2_pathway_scale.json`：候选 ES 的训练集逐通路标准化参数。
- `direct_target_spec.json`、`gene_target_spec.json`：30 维和 319 维目标顺序。
- `gene_model_scale.json`：319 基因训练尺度。
- `pathway_reconstruction_metrics.json`：旧基因臂记录的 GSEApy 版本、输入尺度、基因背景和聚合结果。
- `direct_metrics.json`、`gene_metrics.json`：旧两臂结果上下文；均未因此变成当前任务二的 accepted 结果。

已记录的历史实现是 GSEApy 1.3.1 `ssgsea`、rank、ES、`min_size=1`，背景为输入的 319 基因。原登记映射 `D:\AIPatho\Patch\genes_tag\gene_to_pathway_mapping.csv` 未包含在本地包中，因此通路成员、别名和缺失处理需由队友在其工作环境中核实；不要求将完整 mapping 回传。

不要从此目录读取“默认正确”的算法参数，不要未经核查就把旧标签作为两臂共享真值，也不要覆盖 `../current_phase2_labels/`。新实验须核实真实基因/聚合算法，只生成一份冻结的重构真值供两臂读取；V003 仍为独立主线对照，基因臂仅对预测基因聚合。
