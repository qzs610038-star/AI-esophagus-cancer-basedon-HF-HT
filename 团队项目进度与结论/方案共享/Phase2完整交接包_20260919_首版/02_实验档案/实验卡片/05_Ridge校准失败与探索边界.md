# 逐通路 Ridge 校准：失败记录与探索边界

**结论。** `mpp2_pathway_ridge_calibration_v001_20260717` 的正式记录为 `failed / rejected`；没有可接纳的性能结论，不能写为“校准完成”“校准无效”或“已用于主模型”。

## 目的、对照与方法

- 目的：在冻结 MPP2 预测后，逐通路以 Ridge（岭回归）校准预测尺度，考察能否改善绝对预测。
- 对照：修复后的冻结基线。校准若拟合参数，必须以训练或内部验证合同所允许的数据为界，不能使用外部 XZY 调参。
- 登记未提供可用的内外 PCC或种子汇总；不从原始文件补算。

## 状态、停止／转向与论文用途

- **状态：** current r003 的 result ID 已登记但 evidence 为 rejected。已知的具体语义错误属于 prior r002：把 fold-specific nested-LOPO OOF 预测错误地施加了“单一冻结校准器 PCC 不变性”检查；该次未读取 XZY。current r003 的失败原因没有在 Registry 具体写明，不能将 prior r002 的原因移植到 r003。
- **解释边界：** `next_action` 是当时登记的后续建议，不是当前授权待跑任务。本卡只记录已有结果，不能据此称线性校准无效或已获准重跑。
- **论文用途：** 可在流程附录说明该路线没有产生可报告正式结果；主文不报告性能差值。
- **评审问题：** (1) 失败具体属于哪一项 gate／合同问题？(2) 替代任务的拟合数据边界如何提前固定？(3) 是否需要在论文中提及此未完成路线？
- **证据路径：** 包内 [`../../06_证据摘录/实验登记摘录.json`](../../06_证据摘录/实验登记摘录.json)；原仓库 Registry。

## 登记定位

实验 ID `mpp2_pathway_ridge_calibration_v001_20260717`；current r003 result ID `mpp2-pathway-ridge-calibration-20260717-r003-result-20260718000516-8f133e77`；prior r002 result ID `mpp2-pathway-ridge-calibration-20260717-r002-result-20260717230507-c56518a2`。脚本为 `scripts/fit_mpp2_pathway_ridge_calibration.py`；预设选模字段为 `nested_lopo_patient_balanced_z_mse_with_per_pathway_raw_r2_guard`，lambda 网格为 0.1、1、10。Registry 未提供完整 `result_root`／报告路径。
