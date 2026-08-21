from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
analysis = json.loads((OUT / "analysis_summary.json").read_text(encoding="utf-8"))
zdiag = analysis["zscore_diagnostics"]
ranges = analysis["external_metric_ranges"]
metrics = analysis["metrics"]
external_rank_matrix = []
for arm in ("FBR", "RCC", "HCR", "CPGCR"):
    row = {"arm": arm}
    for item in analysis["chart_data"]["external_ranks"]:
        if item["arm"] == arm:
            display_metric = "mean R²" if item["metric"] == "mean pathway R²" else item["metric"]
            row[display_metric] = item["rank"]
    external_rank_matrix.append(row)


def sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def values_sql(rows, columns):
    values = ",\n  ".join("(" + ", ".join(sql_literal(row.get(column)) for column in columns) + ")" for row in rows)
    names = ", ".join(f'"{column}"' for column in columns)
    return f"SELECT * FROM (VALUES\n  {values}\n) AS reviewed({names})"

sources = [
    {
        "id": "w007_registry",
        "label": "W007 accepted Experiment Registry record",
        "path": "experiments/experiment_registry.json",
        "description": "读取 mpp2_cpgcr_probe_v001_20260811 的 accepted 四臂指标并按每项指标排序。",
        "query": {
            "engine": "ANSI SQL VALUES",
            "sql": values_sql(external_rank_matrix, ["arm", "z-MSE", "z-MAE", "PCC", "raw MAE", "mean R²"]),
            "description": "用已复核的 accepted Registry 派生行重建图1数据。",
        },
    },
    {
        "id": "w007_training_histories",
        "label": "W007 RCC/HCR/CPGCR original training histories",
        "path": "project_state/inbox/W007/quarantine",
        "description": "读取 RCC、HCR、CPGCR 三份 training_history.csv；FBR 使用 accepted 冻结基线值作为参考线。",
        "query": {
            "engine": "ANSI SQL VALUES",
            "sql": values_sql(analysis["chart_data"]["training_history"], ["epoch", "RCC", "HCR", "CPGCR"]),
            "description": "用三份已复核 training_history.csv 派生行重建图2数据。",
        },
    },
    {
        "id": "w007_predictions",
        "label": "W007 original internal/XZY prediction tables",
        "path": "project_state/inbox/W007/quarantine",
        "description": "读取四臂 internal_val_predictions.csv 与 XZY_predictions.csv；复算指标、逐通路 z/raw R²、XZY truth 均值漂移、振幅比和回归斜率。",
        "query": {
            "engine": "ANSI SQL VALUES",
            "sql": values_sql(analysis["chart_data"]["domain_summary"], ["diagnostic", "percent", "numerator", "denominator"]),
            "description": "用原始 XZY_predictions.csv 复算的已复核汇总行重建图3数据。",
        },
    },
]

charts = [
    {
        "id": "external_rank_heatmap",
        "title": "External XZY 五项指标的四臂排名",
        "subtitle": "每项指标内排名；1 为数值最优。五项赢家不一致，且绝对范围很小。",
        "showDescription": True,
        "intent": "comparison",
        "question": "四臂是否存在跨指标一致的 external 优胜者？",
        "rationale": "热图同时保留 4×5 排名矩阵，避免把不同单位的指标强行放在同一数轴。",
        "type": "heatmap",
        "dataset": "external_rank_matrix",
        "sourceId": "w007_registry",
        "encodings": {
            "x": {"field": "arm", "type": "nominal", "label": "实验臂"},
            "y": {"fields": ["z-MSE", "z-MAE", "PCC", "raw MAE", "mean R²"], "type": "quantitative", "label": "指标内名次"},
        },
        "palette": {"kind": "sequential", "name": "blue"},
        "layout": "full",
    },
    {
        "id": "training_history_lines",
        "title": "Internal patient-balanced z-MSE 训练历史",
        "subtitle": "Seed 42；checkpoint 仅由 internal-val 选择；FBR 为冻结参考线，越低越好。",
        "showDescription": True,
        "intent": "trend",
        "question": "辅助目标或新增容量的训练进展是否稳定转化为 internal 回归改善？",
        "rationale": "15 个 epoch 足以显示最佳点后的验证反弹，以及 HCR 的早期停滞。",
        "type": "line",
        "dataset": "training_history",
        "sourceId": "w007_training_histories",
        "encodings": {
            "x": {"field": "epoch", "type": "ordinal", "label": "Epoch（训练轮次）"},
            "y": {"fields": ["RCC", "HCR", "CPGCR"], "type": "quantitative", "label": "patient-balanced z-MSE（越低越好）"},
        },
        "palette": {"kind": "categorical", "name": "w007-arms"},
        "labels": {"values": "endpoints"},
        "referenceLines": [{"axis": "y", "value": metrics["FBR"]["internal"]["patient_balanced_z_mse"], "label": "FBR 冻结基线", "color": "neutral", "lineStyle": "dashed"}],
        "settings": {"showPoints": "always"},
        "layout": "full",
    },
    {
        "id": "zscore_domain_bars",
        "title": "XZY 域偏移与预测振幅诊断",
        "subtitle": "FBR、30 条通路；100% 仅作真实振幅/斜率参照，不代表性能目标。",
        "showDescription": True,
        "intent": "comparison",
        "question": "外部负 R² 更符合逆 z-score 错误，还是患者域偏移与振幅压缩？",
        "rationale": "三个同尺度百分比直接展示 truth 均值漂移覆盖面与预测幅度不足。",
        "type": "bar",
        "dataset": "domain_summary",
        "sourceId": "w007_predictions",
        "encodings": {
            "x": {"field": "diagnostic", "type": "nominal", "label": "诊断量"},
            "y": {"field": "percent", "type": "quantitative", "label": "比例 / 相对理想值", "unit": "%"},
        },
        "valueFormat": "number",
        "unit": "%",
        "labels": {"values": "all"},
        "palette": {"kind": "sequential", "name": "blue"},
        "referenceLines": [{"axis": "y", "value": 100, "label": "真实振幅/理想斜率 100%", "color": "neutral", "lineStyle": "dashed"}],
        "settings": {"sort": "none", "orientation": "vertical", "categoryLabelPolicy": "wrap"},
        "layout": "full",
    },
]

summary = f"""## 技术摘要

- **总体审计结论：`CONDITIONAL GO`。** 可批准“无明确优胜者、z-score 不是已证实主因”的限定性结论；不得批准“CPGCR 显著优于 FBR”或“z-score 导致外部负 R²”的强因果表述。
- **四臂没有跨指标一致赢家。** External 指标范围仅为 z-MSE {ranges['z-MSE']:.6f}、z-MAE {ranges['z-MAE']:.6f}、PCC {ranges['PCC']:.6f}、raw MAE {ranges['raw MAE']:.3f}、mean pathway R² {ranges['mean pathway R²']:.6f}；单 seed 下不足以支持方法胜出。
- **z-score 不是负 raw R² 的直接计算主因。** 逐通路 z-space 与 raw-space R² 最大绝对差为 {zdiag['max_abs_z_vs_raw_pathway_r2_difference']:.2e}，符合相同正仿射逆变换下 R² 不变。
- **最强描述性解释是跨患者域偏移叠加振幅压缩。** XZY 有 {zdiag['pathways_abs_external_mean_gt_0_5z']}/30 条通路的 truth 均值偏离训练中心超过 0.5z；FBR 预测/真实振幅比中位数仅 {zdiag['median_amplitude_ratio']:.1%}，回归斜率中位数仅 {zdiag['median_regression_slope']:.1%}。这是诊断证据，不是随机化因果证明。
"""

blocks = [
    {"id": "title", "type": "markdown", "body": "# W007 四臂精简结论包（最终审核草案）"},
    {"id": "technical_summary", "type": "markdown", "body": summary},
    {
        "id": "finding_one",
        "type": "markdown",
        "sourceId": "w007_registry",
        "body": "## 三种残差臂均未形成稳定、全面的外部优势\n\nHCR 仅在 pooled PCC 数值最高；CPGCR 在 z-MSE、z-MAE、raw MAE 和 mean pathway R² 数值最好；RCC 只在 internal 汇总指标占优。排名方向混合，且绝对差异很小，因此图中“第 1 名”不等同于有统计或实际意义的胜出。",
    },
    {"id": "external_rank_chart_block", "type": "chart", "chartId": "external_rank_heatmap"},
    {
        "id": "finding_two",
        "type": "markdown",
        "sourceId": "w007_training_histories",
        "body": "## 训练目标的下降没有稳定转化为验证回归收益\n\nRCC 的协议最佳点在 epoch 5，之后训练绝对损失继续下降而 internal z-MSE 反弹；HCR 的协议最佳点在 epoch 1，对比损失后续下降并未带来验证改善；CPGCR 的协议最佳点在 epoch 5，随后同样出现反弹。该动态支持“新增容量过拟合/辅助目标与回归收益脱节”，但不能单凭训练曲线证明唯一因果机制。",
    },
    {"id": "training_chart_block", "type": "chart", "chartId": "training_history_lines"},
    {
        "id": "finding_three",
        "type": "markdown",
        "sourceId": "w007_predictions",
        "body": f"## z-score 不是已证实主因；域偏移与振幅压缩更贴近观测\n\n同一通路的 truth/prediction 从 z-space 到 raw-space 使用相同正线性逆变换，逐通路 R² 在两空间一致（最大差 {zdiag['max_abs_z_vs_raw_pathway_r2_difference']:.2e}），因此不能把负 raw R² 归因于 inverse z-score 计算本身。XZY 的 truth 均值广泛偏离训练中心，同时预测振幅和斜率明显偏小，更直接解释“PCC 尚可但 R² 为负”。z-score 仍可能通过训练目标尺度或患者间对齐方式间接影响学习，但 W007 没有归一化消融，不能称其为主因。",
    },
    {"id": "zscore_chart_block", "type": "chart", "chartId": "zscore_domain_bars"},
    {
        "id": "scope_definitions",
        "type": "markdown",
        "sourceId": "w007_registry",
        "body": "## 证据范围与指标定义\n\n- 实验：`mpp2_cpgcr_probe_v001_20260811`，`accepted`；科学分析状态仍为 `pending_user_review`。\n- 样本：internal-val 1,078 spots / 6 名患者；external XZY 1,039 spots / 1 名患者；30 条通路；seed 42。\n- checkpoint：只按 internal patient-balanced z-MSE 选择；XZY 仅在冻结后一次评估。\n- patient-balanced z-MSE：先在每名患者内计算 z-space MSE，再让患者等权。PCC 衡量共同变化方向；R² 同时惩罚均值偏差与振幅不匹配。",
    },
    {
        "id": "methodology",
        "type": "markdown",
        "body": "## 复算方法\n\n从四臂 8 份原始预测表和 3 份训练历史重新计算 pooled z-MSE、z-MAE、PCC、raw MAE、逐通路 R²、XZY truth 均值漂移、预测/真实标准差比及回归斜率；与 accepted JSON 的最大绝对差为 `%.2e`。没有重建 z-score、没有读取服务器、没有训练或重选 checkpoint。" % analysis["max_abs_metric_recompute_difference"],
    },
    {
        "id": "limitations",
        "type": "markdown",
        "body": "## 限制与不确定性\n\n- 只有一个 seed；外部只有一名患者，1,039 个空间 spots 不能视为 1,039 个独立患者样本。\n- 域偏移、振幅压缩和训练动态是描述性诊断；没有 z-score/归一化随机化消融，不能给出主因的因果排序。\n- 热图展示的是数值排名，不表达显著性；R² 的微小改善仍全部处于负值区间。",
    },
    {
        "id": "approval_wording",
        "type": "markdown",
        "body": "## 可直接批准或驳回的结论措辞\n\n### 建议批准\n\n> 基于已接受的 W007 单 seed 四臂结果，RCC、HCR 与 CPGCR 相对 FBR 仅产生幅度很小且跨指标方向不一致的变化，现有证据不支持任一残差臂明确优胜。逐通路 z-space 与 raw-space R² 在相同线性逆变换下保持一致，因此 inverse z-score 计算不是外部负 R² 的直接原因。现有描述性证据更支持 XZY 跨患者分布偏移与预测振幅压缩是外部失配的直接表现；z-score 可能是间接影响因素，但未经归一化消融，不能认定为主因。\n\n### 建议驳回\n\n> CPGCR 已显著且全面优于 FBR；W007 证明 z-score 是指标未改善和外部负 R² 的主因，只需更换归一化即可解决泛化问题。",
    },
    {
        "id": "next_steps",
        "type": "markdown",
        "body": "## 建议的审核动作\n\n1. 若接受限定性结论，直接批准上方“建议批准”段落。\n2. 若认为必须给出 z-score 的因果主因判断，应驳回当前强因果主张；现有 W007 证据不足，且本包不授权新增训练或消融。\n3. 不因本包启动训练、重选配置、修改 XZY 使用边界或改写正式结论文件。",
    },
    {
        "id": "further_questions",
        "type": "markdown",
        "body": "## 仍需用户决定的问题\n\n唯一决策点是：是否把“z-score 不是直接主因、域偏移与振幅压缩为更强描述性解释”升级为稳定结论。若不批准，本包保持待审草案，不改变任何正式事实源。",
    },
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "W007 四臂精简结论包（最终审核草案）",
        "generatedAt": "2026-08-21T00:00:00+08:00",
        "blocks": blocks,
        "charts": charts,
        "tables": [],
        "cards": [],
        "sources": sources,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": "2026-08-21T00:00:00+08:00",
        "status": "ready",
        "datasets": {
            "external_rank_matrix": external_rank_matrix,
            "training_history": analysis["chart_data"]["training_history"],
            "domain_summary": analysis["chart_data"]["domain_summary"],
        },
    },
    "sources": sources,
}

(OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

notes = """# W007 精简结论包：来源与图表映射

> 本文件是待审包的支持说明，不是正式项目结论。

| 图 | 分析问题 | 图形 | 数据 | 支持的结论 |
|---|---|---|---|---|
| 图1 | 是否有跨指标一致赢家 | 4×5 排名热图 | accepted Registry external metrics | 无一致赢家；名次不代表显著性 |
| 图2 | 训练目标是否转化为验证收益 | 多序列训练曲线 | 3 份 training_history.csv + FBR 参考 | 最佳点后反弹/辅助目标脱节 |
| 图3 | z-score 是否为外部负 R² 主因 | 横向诊断条形图 | FBR XZY_predictions.csv | 域偏移与振幅压缩更贴近观测 |

复算脚本：`analysis_w007.py`；完整派生数据：`analysis_summary.json`；报告源：`artifact.json`。

证据分级：W007 四臂结果为 `accepted`；本包的科学归因与措辞为 `pending_user_review`。
"""
(OUT / "source_notes.md").write_text(notes, encoding="utf-8")
print(OUT / "artifact.json")
