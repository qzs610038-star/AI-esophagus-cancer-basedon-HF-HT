# 图表资产说明

> 状态：`pending_user_review`。量化图来自 `phase2_phase3_reanalysis_v002`；除固定 W004 AUC 差值外，均不得自动晋级为正式实验结论。

| 文件 | 用途 | 统计/证据边界 |
|---|---|---|
| `figure1_evidence_bridge.svg` | Phase 2→3 证据框架 | W007→W004 为 future integration（未来整合）虚线，不表示已完成端到端验证 |
| `figure3_w007_patient_balanced_metrics.png` | W007 患者平衡多指标 | internal 6 位患者等权；外部病例 E1 为单病例；四臂无总体赢家 |
| `figure4a_w004_fixed_auc_contrasts.png` | W004 固定 AUC 对比 | `COMPATIBILITY_ONLY`；重复切片级留出，患者非独立 |
| `figure4b_w004_test_diagnostics.png` | W004 AUROC/AUPRC/Brier 描述 | pooled repeated rows，仅作描述，不能视作独立患者 |

可复算源：

- 脚本：`scripts/explorations/phase2_phase3_reanalysis_v002/`
- 数据：`experiments/explorations/phase2_phase3_reanalysis_v002/`
- 图像生成器：`make_figures.py`
