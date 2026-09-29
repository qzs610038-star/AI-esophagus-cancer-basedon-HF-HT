# 逐通路 Ridge 校准：失败记录与探索边界

**结论。** `mpp2_pathway_ridge_calibration_v001_20260717` 的正式记录为 `failed / rejected`；当前结果形成失败/拒绝记录，未产生可接纳的性能结论，也未进入主模型。

## 目的、对照与方法

- 目的：在冻结 MPP2 预测后，逐通路以 Ridge（岭回归）校准预测尺度，考察能否改善绝对预测。
- 对照：修复后的冻结基线。校准若拟合参数，拟合参数的范围限定为训练或内部验证合同允许的数据，外部 XZY 只作冻结后评价。
- 登记未提供可引用的内外 PCC 或种子汇总；随包工具可复算已存预测，供检查校准行为与失败原因。

## 状态、停止／转向与论文用途

- **状态：** current r003 的 result ID 已登记但 evidence 为 rejected。已知的具体语义错误属于 prior r002：fold-specific nested-LOPO OOF 预测被施加了“单一冻结校准器 PCC 不变性”检查；该次未读取 XZY。current r003 的失败原因仍未在 Registry 具体说明，prior r002 原因只归属于 r002。
- **解释边界：** 包内旧快照的 `next_action` 是当时建议；9 月 19 日现行登记已明确改为 `closed_failed_rejected_no_automatic_rerun_or_deployment`（失败关闭、不自动重跑或部署），原建议另存为历史字段，当前字段见[现行实验事实](../../06_证据摘录/现行实验事实_20260920.json)。本卡记录已有结果与失败关闭状态；线性校准的普遍有效性仍待独立评价。
- **论文用途：** 可在流程附录说明该路线没有产生可报告正式结果；主文不报告性能差值。
- **评审问题：** (1) 失败具体属于哪一项 gate／合同问题？(2) 替代任务的拟合数据边界如何提前固定？(3) 是否需要在论文中提及此未完成路线？
- **证据路径：** 当前状态见 [`../../06_证据摘录/现行实验事实_20260920.json`](../../06_证据摘录/现行实验事实_20260920.json)；历史证据见包内 [`../../06_证据摘录/实验登记摘录.json`](../../06_证据摘录/实验登记摘录.json)；原仓库 Registry。

## 登记定位

实验 ID `mpp2_pathway_ridge_calibration_v001_20260717`；current r003 result ID `mpp2-pathway-ridge-calibration-20260717-r003-result-20260718000516-8f133e77`；prior r002 result ID `mpp2-pathway-ridge-calibration-20260717-r002-result-20260717230507-c56518a2`。脚本为 `scripts/fit_mpp2_pathway_ridge_calibration.py`；预设选模字段为 `nested_lopo_patient_balanced_z_mse_with_per_pathway_raw_r2_guard`，lambda 网格为 0.1、1、10。Registry 未提供完整 `result_root`／报告路径。

## 关键流程与源码

```text
读取冻结基线在六名开发患者上的预测、真值与患者身份。
在内部数据上嵌套执行按患者留一；逐通路拟合正斜率岭回归校准。
从 0.1、1、10 中选择正则化强度，按患者均衡误差及既定检查项评估。
汇总内部折，拟合最终校准器：校准预测 = 斜率 × 基线预测 + 截距。
固定校准器与全部选择结果后，一次性评价外部 XZY。
保存校准前后预测及逐项检查结果；运行状态沿用该批失败记录。
```

[关键实现（GitHub 固定版本）](https://github.com/qzs610038-star/AI-esophagus-cancer-basedon-HF-HT/blob/8a9c0d08991deee08f6510e1344195568bf9a70a/scripts/fit_mpp2_pathway_ridge_calibration.py)。完整源码入口、历史版本与运行留档见[实验代码索引](../../06_证据摘录/实验代码索引.md)。

## 原始训练记录

本卡对应的原始训练记录、配置、训练历史与预测入口：[查看 05 号原始记录](../../07_原始训练记录/05/README.md)。
