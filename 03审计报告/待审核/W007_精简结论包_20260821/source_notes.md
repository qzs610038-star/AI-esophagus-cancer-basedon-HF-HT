# W007 精简结论包：来源与图表映射

> 本文件是待审包的支持说明，不是正式项目结论。

| 图 | 分析问题 | 图形 | 数据 | 支持的结论 |
|---|---|---|---|---|
| 图1 | 是否有跨指标一致赢家 | 4×5 排名热图 | accepted Registry external metrics | 无一致赢家；名次不代表显著性 |
| 图2 | 训练目标是否转化为验证收益 | 多序列训练曲线 | 3 份 training_history.csv + FBR 参考 | 最佳点后反弹/辅助目标脱节 |
| 图3 | z-score 是否为外部负 R² 主因 | 横向诊断条形图 | FBR XZY_predictions.csv | 域偏移与振幅压缩更贴近观测 |

复算脚本：`analysis_w007.py`；完整派生数据：`analysis_summary.json`；报告源：`artifact.json`。

证据分级：W007 四臂结果为 `accepted`；本包的科学归因与措辞为 `pending_user_review`。
