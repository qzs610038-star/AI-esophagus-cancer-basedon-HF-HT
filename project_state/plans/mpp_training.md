# MPP 当前训练方案

> lifecycle: active
> state scope: `mpp_training`
> effective date: 2026-07-13

- `barcode-repair-20260711-d626ad8-v003` 已通过证据导入与显式门禁解除，后续训练必须绑定其 `data_manifest_id`，不得回退到旧标签资产。
- MPP1、MPP2、MPP3、MPP4、MPP5 repaired frozen formal 结果均已通过 Gitee 回传包校验并导入；MPP3、MPP5 保持 bbox embargo。当前只允许将这些 `evidence_status=accepted` 且 `provenance_complete=true` 的结果用于正式报告。
- MPP2 online-vs-cache parity 已通过；paired S0 frozen-continue / S1 LoRA r=8 smoke 已完成工程与安全校验。
- LoRA r=8 当前结论固定为 `engineering_pass / candidate_effectiveness_fail / no_go_formal`：PCC 增益未达到预注册门限，prediction supplement 也未证明患者或通路层面的稳定收益。
- 不启动相同 LoRA r=8 正式 seed 42、多 seed、rank sweep 或超参数搜索。除下述经显式批准的单次 dropout smoke 外，任何后续训练必须获得新的显式用户批准，并且候选只能由 internal validation 决定。
- `DIR-20260714-001` 定义、`DIR-20260714-003` 明确批准一次低配置成本探索：在既有 MPP2 LoRA 设置上唯一改变 `lora_dropout=0.10`，其余数据、checkpoint、rank、alpha、学习率、weight decay、seed、最多 3 epoch 与 checkpoint 规则保持不变。执行前必须完成 experiment registry 登记、job manifest 本地验证、Gitee 同步和服务器 `job run --dry-run`；仅在 dry-run 全部 PASS 后执行一次 run，禁止重试、正式训练或 sweep。
- internal validation 用于 checkpoint 选择；external XZY 只能在 checkpoint 固定后评估，不能参与 z-score 拟合或早停。
- historical contaminated 结果只能作明确标注的背景参照，不得替代 repaired-data 证据或与其混写为同一批次结论。
- 报告生成入口：`automation/report_inputs/gemini_mpp_report_handoff_20260713.md`；详细 LoRA 诊断：`automation/logs/mpp2_paired_smoke_review_20260712.md`。
- 实验事实以 `experiments/experiment_registry.json` 为准。
