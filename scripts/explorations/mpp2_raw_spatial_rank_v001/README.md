# MPP2 原始基因集分数空间连续性与患者内排名分析脚本

> 版本：`v0.0.1 + Codex 增量修复（2026-08-01）`
> 方案文档：`01_指南与解读/分析报告/MPP2原始基因集分数空间连续性与患者内排名分析方案_20260801.md`  
> 定位：`diagnostic_only` 本地只读诊断脚本

---

## 🚀 使用指南

### 1. 运行单元测试
```powershell
python scripts/explorations/mpp2_raw_spatial_rank_v001/tests/test_analysis.py
```

### 2. 运行 Smoke 冒烟测试
```powershell
python scripts/explorations/mpp2_raw_spatial_rank_v001/analyze.py --mode smoke
```

### 3. 运行正式诊断 (Final Diagnostic)
```powershell
python scripts/explorations/mpp2_raw_spatial_rank_v001/analyze.py --mode final --replace-existing
```

Smoke 固定写入 v001 根目录下的 `smoke/` 子目录，不能覆盖 Final 根层产出。`--replace-existing` 只允许在既有产出已归档后显式使用；脚本默认拒绝覆盖非空目录。

---

## 📁 产出结果索引
输出存放于：`data/protected_local/mpp2_r2_root_cause_20260725/analysis_outputs/mpp2_raw_spatial_rank_v001/`

- `decision_summary.json`：机器可读结论与双线建议
- `human_report.md`：人类研究员深度解读报告
- `codex_correction_appendix.md`：对 Gemini 原结论的追加修正，不删除旧证据
- `percentile_spatial_metrics.csv`：原始分与百分位分的 Moran/Geary/邻边指标对照
- `block_rank_stability_summary.csv`：按 4×4 网格块删去 20% 区域后的百分位数值漂移
- `cross_patient_pairwise_effects.csv`：Cliff's delta、Wasserstein 距离与中位数差
- `cross_patient_pathway_effects.csv`：患者分组的描述性 epsilon-squared
- `F01_data_geometry.png` ~ `F07_pathway_evidence_forest.png`：可视化图表

统计边界：internal 6 是主复现队列，XZY 只作外部方向确认；`edge_ratio` 兼容列实际是“一步邻边 / 全局随机对”差异比，并非距离匹配对照。

## XZY accepted 基线预测追加分析

回传的 accepted 修复版冻结 MPP2 基线预测采用独立入口，不覆盖上述原始 ssGSEA 产出：

```powershell
python scripts/explorations/mpp2_raw_spatial_rank_v001/tests/test_xzy_prediction_analysis.py
python scripts/explorations/mpp2_raw_spatial_rank_v001/analyze_xzy_prediction.py --mode final
```

若目标子目录已经存在，须确认旧结果已保留后显式添加 `--replace-existing`。结果写入：

`data/protected_local/mpp2_r2_root_cause_20260725/analysis_outputs/mpp2_raw_spatial_rank_v001/xzy_baseline_prediction/`

主要产出：

- `input_qc.json`：回传原表、坐标桥接表和 XZY barcode 顺序的三段验真；
- `prediction_spatial_metrics.csv`：真值、预测、有符号误差和绝对误差的空间指标；
- `pathway_prediction_spatial_summary.csv`：30 通路性能、空间平滑差、相对秩和热点重合汇总；
- `coordinate_content_cv.csv`：1024/2048/4096 像素空间块下的坐标-only 交叉验证；
- `prediction_decision_summary.json` 与 `human_prediction_report.md`：Agent 与人类研究者解读；
- `F08`–`F13`：真值/预测/误差地图、空间热图、相对秩、坐标含量和性能关联。

该追加分析始终为 `diagnostic_only`：只分析既有 XZY 外部预测，不使用 XZY 调参，不改变 accepted 性能或 Registry 状态。
