# 当前主线与 319 候选的比较材料

这里是只读历史诊断证据，不是训练结果。现行规则以包根交接合同为准：核实输入与算法后，用真实基因只生成一次供两臂共享的重构真值；V003 是主线参考和差异诊断对象，并非必需逐值复现的启动门槛。不同目标的分数不能直接比较。

- `task2_true_label_comparison_30_pathways.csv`：30 行逐通路统计。
- `task2_true_label_comparison_30_pathways.png`：比较图。
- `task2_true_label_comparison_summary.json`：内部/外部整体摘要和旧两臂真值漂移。
- `task2_true_label_source_manifest.json`：六键连接、来源文件和“不计算哈希”记录。
- `task2_ssgsea_truth_gap.md`：差异解释和待核清单。
- `专项审计报告_20260919.md`：完整专项审计结论。
- `label_audit.py`、`source_config.json`：原比较脚本和当时的源路径配置，仅供追溯；移动后不是开箱即跑入口。

比较只加载 true 标签，不加载预测列。内部 1,078 点、外部 1,039 点按 `patient_id, patch_id, x, y, split, slide_id` 一对一连接。结果证明差异存在，但不能单独判定是哪一个聚合步骤造成。
