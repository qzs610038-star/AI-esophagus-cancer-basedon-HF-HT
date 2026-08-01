# MPP2 原始基因集分数空间连续性与患者内排名分析脚本

> 版本：`v0.0.1`  
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
python scripts/explorations/mpp2_raw_spatial_rank_v001/analyze.py --mode final
```

---

## 📁 产出结果索引
输出存放于：`data/protected_local/mpp2_r2_root_cause_20260725/analysis_outputs/mpp2_raw_spatial_rank_v001/`

- `decision_summary.json`：机器可读结论与双线建议
- `human_report.md`：人类研究员深度解读报告
- `F01_data_geometry.png` ~ `F07_pathway_evidence_forest.png`：可视化图表
