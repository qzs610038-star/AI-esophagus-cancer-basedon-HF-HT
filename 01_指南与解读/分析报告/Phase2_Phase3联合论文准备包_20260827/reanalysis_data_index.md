# 复算数据索引

> 状态：`exploratory_reanalysis / pending_user_review`。本索引只路由已版本化数据，不复制或升级其证据等级。

## Phase 2：W007

| 数据 | 行数 | 口径 |
|---|---:|---|
| `experiments/explorations/phase2_phase3_reanalysis_v002/w007_patient_pathway_metrics.csv` | 840 | patient×pathway 明细 |
| `experiments/explorations/phase2_phase3_reanalysis_v002/w007_pathway_metrics.csv` | 240 | 先患者内计算、再患者等权的逐通路汇总 |
| `experiments/explorations/phase2_phase3_reanalysis_v002/w007_spatial_metrics.csv` | 840 | Moran’s I，病例内描述 |
| `experiments/explorations/phase2_phase3_reanalysis_v002/w007_hotspot_summary.csv` | 840 | 无显著性检验的热点象限计数 |

internal_val 为 6 位患者等权；公开稿中的外部病例 E1 为 1 位患者。其 spot 数不能替代独立患者样本量。

## Phase 3：W004

| 数据 | 行数 | 口径 |
|---|---:|---|
| `experiments/explorations/phase2_phase3_reanalysis_v002/w004_per_fit_metrics.csv` | 320 | task×arm×seed×fold×split 的 AUROC/AUPRC/Brier/校准 |
| `experiments/explorations/phase2_phase3_reanalysis_v002/w004_pooled_metrics.csv` | 16 | 重复行 pooled 描述，患者非独立 |
| `experiments/explorations/phase2_phase3_reanalysis_v002/w004_fixed_auc_differences.csv` | 4 | 与 Registry 接纳结论一致的固定配对差值 |
| `experiments/explorations/phase2_phase3_reanalysis_v002/w004_case_posthoc.csv` | 620 | `case_id` 事后描述，不是患者独立 OOF |
| `experiments/explorations/phase2_phase3_reanalysis_v002/w004_spatial_status.csv` | 8 | 均为 `not_computable`：prediction CSV 无坐标列 |

## 可复算入口

```powershell
& 'D:\miniconda\python.exe' `
  'scripts/explorations/phase2_phase3_reanalysis_v002/run_reanalysis.py' `
  --project-root '.' `
  --output-dir 'experiments/explorations/phase2_phase3_reanalysis_v002'
```

该命令只读取现有 W007/W004 prediction CSV，不连接服务器、不训练，也不写 Registry。
