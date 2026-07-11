# MPP 当前训练方案

> lifecycle: active
> state scope: `mpp_training`
> effective date: 2026-07-09

- MPP2 是当前唯一后续训练主线。
- MPP1、MPP3、MPP4、MPP5 的统一重跑结果继续保留，分别作为 PCC 上限、overlap/embargo 方法和 raw-scale 稳定性背景，不作为 active 分支。
- 下一项有效比较是同一批次、同一 split manifest、同一 train-only z-score 下的 MPP2 frozen baseline 与 MPP2 LoRA r=8。
- internal validation 用于 checkpoint 选择；external XZY 只能在 checkpoint 固定后评估，不能参与 z-score 拟合或早停。
- `mpp_standard_splits/path_index.json` 当前记录到五组 train 标签存在冲突重复 barcode；在另行批准并验证重建规则前，不得启动新的 MPP 训练。
- 该门禁同时存在于统一调度入口和 `train_mpp_uni2h_mlp.py` 内，直接运行脚本也不能绕过；smoke 最多 3 epoch，只作为测试证据，不能进入最新正式结果列表。
- 详细本地参考文档：`01_指南与解读/分析报告/MPP2后续方案与LoRA新数据实验建议_20260709.md`。
- 实验事实以 `experiments/experiment_registry.json` 为准。
