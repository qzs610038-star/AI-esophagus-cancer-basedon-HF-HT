# Gemini MPP 报告生成输入说明（2026-07-13）

## 1. 报告事实边界

报告必须以 `experiments/experiment_registry.json` 中 `evidence_status=accepted` 且 `provenance_complete=true` 的 repaired-data 结果为事实源。`experiments/experiment_dashboard.md` 仅是派生视图。历史 contaminated 结果只能明确标注为背景对照，不得与 repaired-data 结果混写成同一批次结论。

统一数据资产：

- `data_manifest_id`: `barcode-repair-20260711-d626ad8-v003:1204018178a4d355`
- `source_repair_evidence_id`: `barcode-repair-20260711-d626ad8-v003`
- 内部验证用于 checkpoint 选择；external MPP2/XZY 仅在选择完成后评估。
- MPP3、MPP5 保留 `bbox_embargo`。

## 2. Repaired frozen formal 结果

| 训练 MPP | Best epoch | Internal val PCC | External XZY PCC | Raw MAE | Raw R2 | Accepted bundle |
|---|---:|---:|---:|---:|---:|---|
| MPP1 | 16 | 0.759747 | 0.6900 | 1360.9619 | -0.3719 | `automation/results/mpp1-repair-v003-recheck-20260711/attempt-fixed-eol-20260711203345` |
| MPP2 | 15 | 0.7971 | 0.6549 | 1176.2114 | -0.0880 | `automation/results/mpp2-repair-v003-frozen-20260711-r002/attempt-003` |
| MPP3 | 11 | 0.822454 | 0.6436 | 1209.3894 | -0.1265 | `automation/results/mpp3-repair-v003-recheck-20260711/attempt-fixed-eol-20260711204215` |
| MPP4 | 12 | 0.820883 | 0.6151 | 1053.1715 | 0.1090 | `automation/results/mpp4-repair-v003-recheck-20260711/attempt-fixed-eol-20260711204235` |
| MPP5 | 10 | 0.834697 | 0.6072 | 1045.6602 | 0.0811 | `automation/results/mpp5-repair-v003-recheck-20260711/attempt-fixed-eol-20260711204417` |

不要仅按单一指标给出“最佳 MPP”结论：MPP1 的 PCC 最高，但 MAE/R2 最差；MPP4/5 的 MAE/R2 更好，但 PCC 较低。报告应把这种指标权衡写清楚。

## 3. MPP2 paired LoRA smoke

| 指标 | S0 frozen-continue | S1 LoRA r=8 | S1-S0 |
|---|---:|---:|---:|
| External PCC | 0.644546 | 0.646439 | +0.001893 |
| Raw MAE | 1215.284 | 1206.826 | -8.458 (-0.696%) |
| Raw R2 | -0.144913 | -0.135891 | +0.009022 |

必须同时写明：

- 这是 seed 42、最多 3 epoch 的 paired smoke，不是正式多 seed 训练。
- PCC 增益 `+0.001893` 低于预注册候选有效门 `+0.02`。
- 空间 cluster bootstrap 下 PCC 与 R2 的 95% CI 均跨 0。
- external 逐通路 PCC 仅 `4/30` 改善、`26/30` 下降；internal 六个患者的 MAE 均未改善。
- 平均 raw MAE 有小幅改善，但不足以推翻整体 `candidate_effectiveness_fail`。
- 最终表述必须为：`engineering_pass / candidate_effectiveness_fail / no_go_formal`。

详细诊断见 `automation/logs/mpp2_paired_smoke_review_20260712.md`。

## 4. 禁止性约束

- 不得把 external XZY 用于 checkpoint 选择、rank/学习率选择或后验超参数优化。
- 不得建议自动启动相同 LoRA r=8 正式 seed 42、多 seed、rank sweep 或超参搜索。
- 不得把 historical contaminated 结果当作 repaired 结果的可替代证据。
- 不得重新生成或修改原始 ssGSEA、标准 split、train-only z-score、manifest 或 group 3/5 embargo 资产。
- 如果提出未来研究方向，只能写为“需要新的显式批准、且候选必须由 internal validation 决定的单因素 exploratory smoke”。

## 5. 推荐报告结构

1. 数据修复与可比性边界。
2. Repaired frozen formal MPP1-5 横向结果及多指标权衡。
3. MPP2 LoRA paired smoke 的工程结论与有效性门禁。
4. 局限性：单 seed smoke、空间相关性、通路异质性、external 不参与选择。
5. 当前决策：冻结训练，进入结果整理；未来实验需重新审批。
