# LoRA dropout=0.10：正则化尝试

**结论。** 将 LoRA rank=8 的 dropout 设为 0.10 后，单种子内部整体 PCC未改善，登记明确关闭该路线，不再重试或调参。

## 目的、对照与方法

- 目的：检查 LoRA 过拟合或训练扰动是否能用 dropout 缓解。
- 对照为前一张卡的 LoRA r=8；数据为 MPP2、barcode-repair-v003。外部 XZY 只在内部 checkpoint 冻结后评价。
- 这是种子 42、最多 3 epoch 的单种子 smoke；内部选模为 `val_loss_min_internal_val_manifest`，外部只在内部 checkpoint 选定后评估。当前登记提供整体 PCC，患者—通路等权 PCC与种子均值/SD未登记。

## 已登记结果与状态

内部整体 PCC 为 **0.792907**，XZY 整体 PCC 为 **0.643799**；均为 accepted 运行事实。与无 dropout 的 LoRA 相比，内部没有改善，外部也低于 0.646439。不同运行并非三种子正式比较，差值只作路线停止依据。

## 停止／转向、论文用途与评审

- **停止原因：** Registry 后续动作为 `close_dropout10_route_no_retry_or_tuning_after_internal_first_failure`。
- **论文用途：** 负向优化记录，适合补充材料或方法演进文字；不进入主结果表。
- **评审问题：** (1) dropout 是唯一改变因素的证据是否完整？(2) 单种子内部失败作为停止规则是否预先约定？(3) 是否需要将此实验与后续 contrastive v3 分开叙述？
- **证据路径：** 当前状态见 [`../../06_证据摘录/现行实验事实_20260920.json`](../../06_证据摘录/现行实验事实_20260920.json)；历史证据见包内 [`../../06_证据摘录/实验登记摘录.json`](../../06_证据摘录/实验登记摘录.json)；原仓库 Registry。

## 登记定位

实验 ID `mpp2_lora_r8_dropout10_smoke_20260714`；result ID `mpp2-lora-r8-dropout10-smoke-20260714-r001-result-20260714182311-7971c360`；脚本 `train_mpp_uni2h_lora.py`。与 S0 配对参考的 result ID 为 `mpp2-paired-s0-frozen-continue-smoke-20260712-r002-result-20260712173215-1393b29c`。Registry 未提供完整 `result_root`／报告路径。

## 关键流程与源码

```text
沿用 S1 的数据划分、种子 42、编码器与已接纳回归头。
冻结编码器原参数，注入 rank=8、alpha=16 的 LoRA 适配器。
将适配器 dropout（随机失活比例）设为 0.10，其余参数按本次配置。
联合更新适配器和回归头，最多训练 3 轮。
按内部验证损失保留最佳检查点。
固定检查点后计算内部与 XZY 预测，与既有 LoRA 试跑对照。
```

[关键实现（GitHub 固定版本）](https://github.com/qzs610038-star/AI-esophagus-cancer-basedon-HF-HT/blob/8a9c0d08991deee08f6510e1344195568bf9a70a/train_mpp_uni2h_lora.py)。完整源码入口、历史版本与运行留档见[实验代码索引](../../06_证据摘录/实验代码索引.md)。

## 原始训练记录

本卡对应的原始训练记录、配置、训练历史与预测入口：[查看 04 号原始记录](../../07_原始训练记录/04/README.md)。
