# LoRA dropout=0.10：正则化尝试

**结论。** 将 LoRA rank=8 的 dropout 设为 0.10 后，单种子内部整体 PCC未改善，登记明确关闭该路线，不再重试或调参。

## 目的、对照与方法

- 目的：检查 LoRA 过拟合或训练扰动是否能用 dropout 缓解。
- 对照：前一张卡的 LoRA r=8；数据为 MPP2、barcode-repair-v003。外部 XZY 仅作评估，未用于模型选择。
- 这是种子 42、最多 3 epoch 的单种子 smoke；内部选模为 `val_loss_min_internal_val_manifest`，外部只在内部 checkpoint 选定后评估。不报告患者—通路等权 PCC，也没有种子均值／SD。

## 已登记结果与状态

内部整体 PCC 为 **0.792907**，XZY 整体 PCC 为 **0.643799**；均为 accepted 运行事实。与无 dropout 的 LoRA 相比，内部没有改善，外部也低于 0.646439。不同运行并非三种子正式比较，差值只作路线停止依据。

## 停止／转向、论文用途与评审

- **停止原因：** Registry 后续动作为 `close_dropout10_route_no_retry_or_tuning_after_internal_first_failure`。
- **论文用途：** 负向优化记录，适合补充材料或方法演进文字；不进入主结果表。
- **评审问题：** (1) dropout 是唯一改变因素的证据是否完整？(2) 单种子内部失败作为停止规则是否预先约定？(3) 是否需要将此实验与后续 contrastive v3 分开叙述？
- **证据路径：** 包内 [`../../06_证据摘录/实验登记摘录.json`](../../06_证据摘录/实验登记摘录.json)；原仓库 Registry。

## 登记定位

实验 ID `mpp2_lora_r8_dropout10_smoke_20260714`；result ID `mpp2-lora-r8-dropout10-smoke-20260714-r001-result-20260714182311-7971c360`；脚本 `train_mpp_uni2h_lora.py`。与 S0 配对参考的 result ID 为 `mpp2-paired-s0-frozen-continue-smoke-20260712-r002-result-20260712173215-1393b29c`。Registry 未提供完整 `result_root`／报告路径。
