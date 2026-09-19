# MPP 训练方案（历史合同）

> **历史训练合同（2026-09-19）**：本文保留 MPP 修复、LoRA、Ridge 和 accepted 基线的历史训练边界；不再作为当前 Phase 3 模型入口或活跃执行计划，也不授权新训练。当前方法选择见 [`Phase2组会决策与论文后续任务`](../governance/Phase2组会决策与论文后续任务_20260912.md)，当前交付模型见 [`Phase2最终模型_Phase3交接包`](../../团队项目进度与结论/qzs/Phase2最终模型_Phase3交接包_20260918/README.md)。
>
> 本文列出的旧 accepted 结果仍在各自声明范围内有效；“历史”只描述本页作为当前路由的生命周期，不得把旧 accepted 基线写成无效。
>
> 下列“必须/禁止”均属于这份历史合同的证据与复现边界；它们不会单独授权当前训练、服务器操作或 Phase 3 部署。
>
> lifecycle: historical
> state scope: `mpp_training`
> effective date: 2026-07-13

- `barcode-repair-20260711-d626ad8-v003` 已通过证据导入与显式门禁解除，后续训练必须绑定其 `data_manifest_id`，不得回退到旧标签资产。
- MPP1、MPP2、MPP3、MPP4、MPP5 repaired frozen formal 结果均已通过当时的 Gitee 回传包校验并导入；MPP3、MPP5 保持 bbox embargo。它们在各自登记范围内仍是 `evidence_status=accepted` 且 `provenance_complete=true` 的历史结果，可用于正式报告；当前交付路径不再使用 Gitee。
- MPP2 online-vs-cache parity 已通过；paired S0 frozen-continue / S1 LoRA r=8 smoke 已完成工程与安全校验。
- LoRA r=8 当前结论固定为 `engineering_pass / candidate_effectiveness_fail / no_go_formal`：PCC 增益未达到预注册门限，prediction supplement 也未证明患者或通路层面的稳定收益。
- 不启动相同 LoRA r=8 正式 seed 42、多 seed、rank sweep 或超参数搜索。任何后续训练必须获得新的显式用户批准，并且候选只能由 internal validation 决定。
- `DIR-20260714-001` 定义、`DIR-20260714-003` 批准的单次 `lora_dropout=0.10` smoke 已完成并通过结果包导入（`mpp2-lora-r8-dropout10-smoke-20260714-r001-result-20260714182311-7971c360`）。其 best internal val loss `0.38228295` 高于 S0 的 `0.37892210`，best internal PCC `0.79290723` 低于 S0 的 `0.79479160`；故未通过预注册 internal-first 门槛，dropout=0.10 路线关闭，禁止重试、正式训练或 sweep。
- `mpp2_pathway_ridge_calibration_v001_20260717` 已完成正式 r003 并以 `status=failed / evidence_status=rejected` 关闭：校准后 XZY mean raw R² 从 `-0.08801664` 变为 `-0.08803688`，mean raw MAE 从 `1176.21137` 变为 `1176.22480`，外部门失败。`CalibratedMPP2` 不得进入 Phase 3 或部署，也不得依据 XZY 重试或调参。后续关于 H&E 与 accepted 冻结 MPP2 原始输出的联合输入是本页记录的历史决策；当前 Phase 3 交付模型和复现入口以顶部交接包为准，不得把该历史 accepted 基线误写为无效。
- internal validation 用于 checkpoint 选择；external XZY 只能在 checkpoint 固定后评估，不能参与 z-score 拟合或早停。
- historical contaminated 结果只能作明确标注的背景参照，不得替代 repaired-data 证据或与其混写为同一批次结论。
- 报告生成入口：`automation/report_inputs/gemini_mpp_report_handoff_20260713.md`；详细 LoRA 诊断：`automation/logs/mpp2_paired_smoke_review_20260712.md`。
- 实验事实以 `experiments/experiment_registry.json` 为准。
