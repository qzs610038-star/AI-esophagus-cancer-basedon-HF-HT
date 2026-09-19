# S0/S1 LoRA 配对：首次轻量微调

**结论。** 修复 MPP2 基线后，冻结继续训练（S0）与 rank=8 LoRA（低秩适配，S1）完成单种子配对；S1 未提供足以继续扩大此配置的增量依据。

## 目的、对照与方法

- 目的：测试对编码器作轻量适配是否优于冻结继续训练。
- 对照：S0 为冻结继续；S1 为 LoRA rank=8。二者均使用 MPP2、barcode-repair-v003、同一内部与 XZY 评估框架。
- 选模：种子 42、最多 3 epoch，内部 `val_loss_min_internal_val_manifest` 选模；外部 XZY 仅在内部 checkpoint 选定后评估。两项均为 smoke（冒烟）规模、单种子，不能当作随机性稳定性证据。

## 已登记结果与状态

| 臂 | 内部整体 PCC | XZY 整体 PCC | 状态 |
|---|---:|---:|---|
| S0 冻结继续 | 0.794792 | 0.644546 | accepted |
| S1 LoRA r=8 | 0.792969 | 0.646439 | accepted |

患者—通路等权 PCC、种子均值和 SD 未提供。登记的外部整体 PCC差为约 +0.0019，且外部仅 4/30 通路出现改善；内部 PCC 未改善。两项事实用于停止判断，不代表以外部 XZY 调参。

## 停止／转向、论文用途与评审

- **停止原因：** 登记对 S1 的后续动作为在预测诊断后关闭当前 LoRA r=8 配置。
- **论文用途：** 可在方法演进或补充材料报告为已接纳的未改善探索；不用于支持主模型。
- **评审问题：** (1) 单种子、内部未改善时，关闭该配置是否合理？(2) 这组配对是否足够“同合同”以解释外部微小差异？(3) 是否需要展示每通路差异而非只给总体 PCC？
- **证据路径：** 包内 [`../../06_证据摘录/实验登记摘录.json`](../../06_证据摘录/实验登记摘录.json)；原仓库 Registry。

## 登记定位

实验 ID／result ID：S0 为 `mpp2_paired_s0_frozen_continue_smoke_20260712` / `mpp2-paired-s0-frozen-continue-smoke-20260712-r002-result-20260712173215-1393b29c`；S1 为 `mpp2_paired_s1_lora_r8_smoke_20260712` / `mpp2-paired-s1-lora-r8-smoke-20260712-r002-result-20260712173316-163abad4`。两项脚本均为 `train_mpp_uni2h_lora.py`，配对合同为相同修复 manifest、head checkpoint、种子、样本顺序和验证划分；Registry 未提供完整 `result_root` 或报告路径。
