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
