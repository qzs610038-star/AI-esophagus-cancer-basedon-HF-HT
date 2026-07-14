# MPP 当前训练方案

> lifecycle: active
> state scope: `mpp_training`
> effective date: 2026-07-13

- `barcode-repair-20260711-d626ad8-v003` 已通过证据导入与显式门禁解除，后续训练必须绑定其 `data_manifest_id`，不得回退到旧标签资产。
- MPP1、MPP2、MPP3、MPP4、MPP5 repaired frozen formal 结果均已通过 Gitee 回传包校验并导入；MPP3、MPP5 保持 bbox embargo。当前只允许将这些 `evidence_status=accepted` 且 `provenance_complete=true` 的结果用于正式报告。
- MPP2 online-vs-cache parity 已通过；paired S0 frozen-continue / S1 LoRA r=8 smoke 已完成工程与安全校验。
- LoRA r=8 当前结论固定为 `engineering_pass / candidate_effectiveness_fail / no_go_formal`：PCC 增益未达到预注册门限，prediction supplement 也未证明患者或通路层面的稳定收益。
- 不启动相同 LoRA r=8 正式 seed 42、多 seed、rank sweep 或超参数搜索。任何后续训练必须获得新的显式用户批准，并且候选只能由 internal validation 决定。
- `DIR-20260714-001` 定义、`DIR-20260714-003` 批准的单次 `lora_dropout=0.10` smoke 已完成并通过结果包导入（`mpp2-lora-r8-dropout10-smoke-20260714-r001-result-20260714182311-7971c360`）。其 best internal val loss `0.38228295` 高于 S0 的 `0.37892210`，best internal PCC `0.79290723` 低于 S0 的 `0.79479160`；故未通过预注册 internal-first 门槛，dropout=0.10 路线关闭，禁止重试、正式训练或 sweep。
- internal validation 用于 checkpoint 选择；external XZY 只能在 checkpoint 固定后评估，不能参与 z-score 拟合或早停。
- historical contaminated 结果只能作明确标注的背景参照，不得替代 repaired-data 证据或与其混写为同一批次结论。
- 报告生成入口：`automation/report_inputs/gemini_mpp_report_handoff_20260713.md`；详细 LoRA 诊断：`automation/logs/mpp2_paired_smoke_review_20260712.md`。
- 实验事实以 `experiments/experiment_registry.json` 为准。
