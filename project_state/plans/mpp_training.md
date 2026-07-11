# MPP 当前训练方案

> lifecycle: active
> state scope: `mpp_training`
> effective date: 2026-07-11

- `barcode-repair-20260711-d626ad8-v003` 已通过证据导入与显式门禁解除，后续训练必须绑定其 `data_manifest_id`，不得回退到旧标签资产。
- MPP2 是当前唯一后续训练主线；先重跑同批次 frozen baseline，再决定是否启动 LoRA r=8 smoke（最多 3 epoch）。
- MPP1、MPP3、MPP4、MPP5 的统一重跑结果继续保留，分别作为 PCC 上限、overlap/embargo 方法和 raw-scale 稳定性背景，不作为 active 分支。
- 下一项有效比较是同一批次、同一 split manifest、同一 train-only z-score 下的 MPP2 frozen baseline 与 MPP2 LoRA r=8。
- internal validation 用于 checkpoint 选择；external XZY 只能在 checkpoint 固定后评估，不能参与 z-score 拟合或早停。
- frozen baseline 与旧 MPP2 结果 `mpp2_std10val_xzy_ext_uni2h_mlp_20260706` 比较；若 external XZY PCC 绝对下降 `>=0.05`、raw MAE 相对增加 `>=10%` 或 raw R² 绝对下降 `>=0.10`，任一触发即暂停 LoRA，改为在修复数据上重跑 MPP1-5 frozen baselines 后再形成跨 MPP 结论。
- MPP2 新 frozen baseline 未成功导入、三项指标不完整或触发上述任一阈值时，统一调度入口必须阻断 LoRA；smoke 最多 3 epoch，只作为测试证据，不能进入最新正式结果列表。
- 详细本地参考文档：`01_指南与解读/分析报告/MPP2后续方案与LoRA新数据实验建议_20260709.md`。
- 实验事实以 `experiments/experiment_registry.json` 为准。
