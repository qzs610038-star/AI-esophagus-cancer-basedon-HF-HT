# Experiment Dashboard

> 当前状态版本 `265`，核对时间 `2026-09-19T18:15:10+08:00`。本页为派生导航，接纳状态以 [Registry（实验注册器）](experiment_registry.json) 为准。

## 当前进展与新补登记（2026-09-19）

| 对象 | 状态及下一步 |
|---|---|
| 全视野三编码器批次 | accepted（已接纳）；保留单 XZY 和物理支持域未确认的局限 |
| [当前模型包](../团队项目进度与结论/qzs/Phase2最终模型_Phase3交接包_20260918/README.md) / [首版交接包](../团队项目进度与结论/方案共享/Phase2完整交接包_20260919_首版/00_从这里开始.md) | 本地材料已生成；实际发送、队友验收未确认 |
| [任务3全视野密度对照](phase2_task3_fullfov_density_ablation_v1/README.md) | 1.0.1 包已登记，本地验证完成；无正式结果登记 |
| [第二批空间头基础模型替换](phase2_backbone_spatial_ablation_batch2_20260919/README.md) | v002 代码补登；planned，不代表训练启动或完成 |
| [MPP2 独立复用包](mpp2_uni2h_mlp_baseline_20260906/README.md) | v001 代码补登，不替代旧 accepted 基线 |
| LJQ 任务2/3旧协议回传 | full 与 smoke 分开登记；full 待审，smoke 仅诊断；不替代中央任务2/3新计划 |
| Ridge / 旧 MPP 训练合同 | 已退出活跃计划；Ridge 原 rejected 状态保留 |

[当前机器状态](../project_state/current_state.json) · [待决事项与建议风险](../maintenance_logs/项目事实与规则治理完成与待决事项_20260919.md)。下方保留原数值表和旧记录，旧 next_action、W### 或门控文字仅供追溯。


## Phase2 全视野修复与调参结果已复核接纳（2026-09-18）

- `phase2_fullfov_hpo_v1` / `20260916_231813_101_7b1b9a79`：111 次训练、51 份 XZY 外部预测已完成；`registered`、`accepted`，接纳范围限本批与单外部患者。
- 最终空间臂 XZY 双 PCC（患者—通路等权平均 / 整体展平）：UNI `0.559422 / 0.656756`，UNI2-h `0.571073 / 0.683748`，Virchow2 `0.577756 / 0.678311`；均为三种子均值，其他臂及标准差见[审核前结果汇报](results/phase2_fullfov_hpo_v1/20260916_231813_101_7b1b9a79/analysis/审核前结果汇报_20260918.md)。
- [复核接纳记录](results/phase2_fullfov_hpo_v1/20260916_231813_101_7b1b9a79/analysis/复核接纳记录_20260918.md)；原批次 Phase3 合同未回传；后续独立模型交接包已生成，单一 XZY 患者的结果不用于重新选择已冻结的配方。

## 空间热启动 v1 已审核接纳（2026-09-10）

- `phase2_spatial_warmstart_v1` / `20260910_190620_252_10e4d946`：`registered`、`accepted`，单种子42。
- 已选`spatial_joint`内部双PCC=`0.6734 / 0.8090`、zMSE=`0.3552`；XZY双PCC=`0.5582 / 0.6720`、zMSE=`0.6418`。
- XZY相对来源step0双PCC提高`0.0157 / 0.0088`，zMSE降低`0.0221`；固定β=1仅为不参与选择的诊断。
- 边界：外部仅一名患者，单种子无样本标准差，不能声称广泛跨患者泛化；原始尺度指标不可审计。
- [完整审核结果](results/phase2_spatial_warmstart_v1/warmstart_no_lora_v1/20260910_190620_252_10e4d946/analysis/external_xzy_review.md)

## 本轮软连接实验已完整登记（2026-09-08）

- 实验：`phase2_softlink_local_v2`；批次：`20260908_005325_810_8d306c10`；12次正式训练及12次XZY预测完成。
- 登记状态：`registered`；证据状态：`accepted`（按当前实验登记核对；本次未改变接纳状态）。
- 注册表已录入23种明确口径指标的汇总与逐种子值，并保留逐患者、逐通路、同轮15/25、固定平滑及历史参考；未计算项写明原因。
- PCC依次为“逐通路平均 PCC（即患者—通路等权平均 PCC） / 平均池化 pooled PCC（整体展平 PCC）”，两种口径同时保留（下表依此次序）：

| 实验臂 | 内部两种PCC | 外部两种PCC |
|---|---:|---:|
| point | 0.6533 / 0.7985 | 0.5131 / 0.6501 |
| relation | 0.6530 / 0.7986 | 0.5164 / 0.6496 |
| spatial | 0.6663 / 0.8056 | 0.5580 / 0.6783 |
| joint | 0.6660 / 0.8056 | 0.5582 / 0.6759 |

- 空间臂主要指标改善；内部R²及残差Moran未同步改善。不同PCC回答不同问题，不能按绝对数值高低互证优越。
- 外部只有XZY一名患者；三种子标准差不是泛化置信区间。内部为同患者留出点位。
- [完整指标与验证](results/phase2_softlink_local_v2/20260908_005325_810_8d306c10/analysis/registration_record.json) · [分析报告](../01_指南与解读/分析报告/Phase2软连接v2_1_实验结果与机制分析_20260908.md)

> 依据实验登记维护；会议决策更新：2026-09-05T19:25:48+08:00。
> Source of truth: `experiments/experiment_registry.json`
>
> 🟢 **JFX0729 数据更正状态**：更正数据已同步、本地处理并通过 7/7 checks passed；旧 JFX 相关结果仅作历史参考。下一步：defer JFX token-cache rebuild; a new data transformation is planned and caches/retraining should be rebuilt only after that transform is finalized and applied consistently。
>
> 当前主线：固定经典空间残差；后续准备任务交接、单点基线模型消融与论文材料。
>
> ⚠️ **旧三患者实验结论**：Old 3-patient LoRA/Token/frequency/spatial-repair conclusions must not be used as formal evidence. They may only seed hyperparameters and risk checks for new MPP2 experiments.

## 9月12日决策及后续单点消融记录（历史时点）

- 方法已固定：第一轮 `phase2_softlink_local_v2` 的 `spatial`（经典空间残差）臂；暂停进一步改进。
- 任务2（基因预测→通路重建）、任务3（隔点训练→稠密测试）：核对已有材料，准确对齐输入、输出及操作后交另一位团队成员实践，当前待交接对齐。
- 基础模型消融：UNI2-h、UNI 与 Virchow2-CLS 已在统一单点预测协议下完成三随机种子比较，结果已复核并登记为 accepted（已接纳）。空间残差方案保持固定。
- 登记结论：在固定历史输入与统一单点预测协议下，Virchow2-CLS 在三种子内部验证的预定选模指标上稳定、小幅优于 UNI2-h；外部 XZY 的逐通路等权 PCC 较高，但整体展平 PCC和误差指标未优于 UNI2-h。UNI 在内部与外部均整体落后
- 后续重心：论文结构与指标选择、代码整理、关键架构图。全部指标继续保留，两种PCC并列，论文主副指标待定。
- 本次完成基础模型消融结果登记；其余组会任务和既有实验接纳状态不变。

- [完整组会决策与后续任务](../project_state/governance/Phase2组会决策与论文后续任务_20260912.md)
## Status Overview

| ID | Family | Status | Fold | Encoder | Tokens | Best Val PCC | Best Val Loss | Train-Val Gap | Epoch |
|----|--------|--------|:----:|---------|:------:|:------------:|:-------------:|:-------------:|:-----:|
| mpp2_std10val_xzy_ext_uni2h_mlp_20260706 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.7827 | 0.3984 | 0.0217 | 14 | ext_XZY_PCC=0.6489 |
| mpp2_barcode_repair_v003_frozen_baseline_20260711 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.7971 | 0.3749 | — | 15 | ext_XZY_PCC=0.6549 |
| mpp1_barcode_repair_v003_frozen_recheck_20260711 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.7597 | 0.4323 | — | 16 | ext_XZY_PCC=0.6900 |
| mpp3_barcode_repair_v003_frozen_recheck_20260711 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.8225 | 0.3488 | — | 11 | ext_XZY_PCC=0.6436 |
| mpp4_barcode_repair_v003_frozen_recheck_20260711 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.8209 | 0.3198 | — | 12 | ext_XZY_PCC=0.6151 |
| mpp5_barcode_repair_v003_frozen_recheck_20260711 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.8347 | 0.3166 | — | 10 | ext_XZY_PCC=0.6072 |
| mpp2_online_cache_parity_v003_20260711 | mpp_preflight | :white_check_mark: done | — |  | — | — | — | — | — |
| mpp2_paired_s0_frozen_continue_smoke_20260712 | mpp_uni2h_lora | :white_check_mark: done | — |  | — | 0.7948 | 0.3789 | — | 3 | ext_XZY_PCC=0.6445 |
| mpp2_paired_s1_lora_r8_smoke_20260712 | mpp_uni2h_lora | :white_check_mark: done | — |  | — | 0.7930 | 0.3823 | — | 1 | ext_XZY_PCC=0.6464 |
| mpp2_lora_r8_dropout10_smoke_20260714 | mpp_uni2h_lora | :white_check_mark: done | — |  | — | 0.7929 | 0.3823 | — | 1 | ext_XZY_PCC=0.6438 |
| mpp2_pathway_ridge_calibration_v001_20260717 | mpp_posthoc_pathway_calibration | :red_circle: failed | — |  | — | — | — | — | — |
| phase2_gene_reconstruction_comparison_v001_20260904 | phase2_supplemental | :large_blue_circle: planned | — |  | — | — | — | — | — |
| phase2_spatial_density_comparison_v001_20260904 | phase2_supplemental | :large_blue_circle: planned | — |  | — | — | — | — | — |
| phase2_pathology_model_benchmark_v001_20260904 | phase2_supplemental | :white_check_mark: done | — |  | — | — | — | — | — |
| online_tokens_gfnet_fold1_65t_legacy | online_tokens | :white_check_mark: done | 1 | gfnet | 65 | 0.3914 | 0.3337 | 0.1683 | 2 |
| smoke_gfnet_65t | online_tokens | :white_check_mark: done | 1 | gfnet | 65 | 0.3933 | 0.3314 | 0.0935 | 1 |
| online_tokens_transformer_fold1_65t | online_tokens | :warning: done_incomplete_data | 1 | transformer | 65 | 0.3821 | — | — | 4 |
| gitee_roundtrip_pilot_w006_v001_20260810 | governed_v3 | :question: closed | — |  | — | — | — | — | — |
| A0b_gfnet_clean_token_65t_fold1 | online_tokens | :white_check_mark: done | 1 | gfnet | 65 | 0.3967 | 0.3320 | 0.1600 | 2 |
| A1_gfnet_modrelu_65t_fold1 | online_tokens | :white_check_mark: done | 1 | gfnet | 65 | 0.3724 | 0.3394 | 0.2412 | 4 |
| B1a_cls_freq_gfnet_fold1 | online_cls | :white_check_mark: done | 1 |  | — | 0.3881 | 0.3297 | 0.1959 | 2 |
| B1b_cls_freq_mean_fold1 | online_cls | :white_check_mark: done | 1 |  | — | 0.3955 | 0.3313 | 0.1985 | 2 |
| C1_gfnet_lora_65t_fold1 | online_tokens | :white_check_mark: done | 1 | gfnet | 65 | 0.4169 | 0.3321 | 0.1937 | 3 |
| S1a_gfnet_frozen_cls_pool8x8_fold1 | online_tokens | :white_check_mark: done | 1 | gfnet | 65 | 0.3785 | 0.3366 | 0.2603 | 5 |
| mpp_v3_xzy_external_uni2h_mlp_20260701 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | — | — | — | 50 | ext_XZY_PCC=0.6020 |
| mpp_v3bis_valZHZ_xzy_external_uni2h_mlp_20260704 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.7146 | 1.0574 | 0.0188 | 1 | ext_XZY_PCC=0.6665 |
| mpp1_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.5309 | 0.3071 | 0.2340 | 6 | ext_XZY_PCC=0.7056 |
| mpp4_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.6844 | 0.5382 | 0.1597 | 10 | ext_XZY_PCC=0.5871 |
| mpp2_partnerwzk_xzy_external_uni2h_mlp_20260704 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.6625 | 0.2360 | 0.1349 | 13 | ext_XZY_PCC=0.4555 |
| mpp5_partnerwzk_xzy_external_uni2h_mlp_20260704 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.8459 | 0.1300 | 0.1350 | 112 | ext_XZY_PCC=0.4472 |
| mpp1_std10val_xzy_ext_uni2h_mlp_20260706 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.6473 | 0.5965 | -0.0223 | 6 | ext_XZY_PCC=0.7103 |
| mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.8016 | 0.3877 | -0.0522 | 4 | ext_XZY_PCC=0.6462 |
| mpp4_std10val_xzy_ext_uni2h_mlp_20260706 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.8075 | 0.3431 | 0.0075 | 9 | ext_XZY_PCC=0.6064 |
| mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | mpp_uni2h_mlp | :white_check_mark: done | — |  | — | 0.8069 | 0.3641 | -0.0082 | 7 | ext_XZY_PCC=0.5759 |
| mpp2_huber_loss_paired_v001_20260728 | governed_v3 | :white_check_mark: done | — |  | — | — | — | — | — | pair_ΔPCC=-0.0022 |
| mpp2_cpgcr_probe_v001_20260811 | governed_v3 | :white_check_mark: done | — |  | — | — | — | — | — |
| phase3_dual_baseline_spatial_pathway_transfer_v001_20260804 | governed_v3 | :white_check_mark: done | — |  | — | — | — | — | — | pair_ΔPCC=+0.0232 |
| S1b_gfnet_lora_r8_cls_pool8x8_fold1 | online_tokens | :pause_button: paused | 1 | gfnet | 65 | — | — | — | — |

## 历史优先队列（仅追溯，不授权后续运行）

| Priority | ID | Status | Next Action |
|:--------:|----|--------|-------------|
| P0 | mpp2_std10val_xzy_ext_uni2h_mlp_20260706 | done | selected_for_mpp2_newdata_lora |
| P0 | mpp2_barcode_repair_v003_frozen_baseline_20260711 | done | closed_use_accepted_repaired_result_for_reporting |
| P0 | mpp1_barcode_repair_v003_frozen_recheck_20260711 | done | closed_use_accepted_repaired_result_for_reporting |
| P0 | mpp3_barcode_repair_v003_frozen_recheck_20260711 | done | closed_use_accepted_repaired_result_for_reporting |
| P0 | mpp4_barcode_repair_v003_frozen_recheck_20260711 | done | closed_use_accepted_repaired_result_for_reporting |
| P0 | mpp5_barcode_repair_v003_frozen_recheck_20260711 | done | closed_use_accepted_repaired_result_for_reporting |
| P0 | mpp2_online_cache_parity_v003_20260711 | done | prepare_paired_smoke |
| P0 | mpp2_paired_s0_frozen_continue_smoke_20260712 | done | paired_smoke_completed_reviewed_no_redispatch |
| P0 | mpp2_paired_s1_lora_r8_smoke_20260712 | done | close_current_lora_r8_configuration_after_prediction_diagnostics |
| P0 | mpp2_lora_r8_dropout10_smoke_20260714 | done | close_dropout10_route_no_retry_or_tuning_after_internal_first_failure |
| P0 | mpp2_pathway_ridge_calibration_v001_20260717 | failed | closed_failed_rejected_no_automatic_rerun_or_deployment |
| P2 | phase2_gene_reconstruction_comparison_v001_20260904 | planned | 准备输入输出与操作合同；模型替换限单点基线 |
| P2 | phase2_spatial_density_comparison_v001_20260904 | planned | 准备输入输出与操作合同；模型替换限单点基线 |
| P2 | phase2_pathology_model_benchmark_v001_20260904 | done | closed_registered_accepted_no_automatic_training |
| baseline | online_tokens_gfnet_fold1_65t_legacy | done | archived_as_legacy_baseline |
| baseline | smoke_gfnet_65t | done | archived |
| baseline | online_tokens_transformer_fold1_65t | done_incomplete_data | archived |
| — | gitee_roundtrip_pilot_w006_v001_20260810 | closed | closed_user_approved_retained_worktree |
| historical | A0b_gfnet_clean_token_65t_fold1 | done | historical_tuning_reference |
| historical | A1_gfnet_modrelu_65t_fold1 | done | historical_tuning_reference |
| historical | B1a_cls_freq_gfnet_fold1 | done | historical_tuning_reference |
| historical | B1b_cls_freq_mean_fold1 | done | historical_tuning_reference |
| historical | C1_gfnet_lora_65t_fold1 | done | historical_tuning_reference_mpp2_lora_first |
| historical | S1a_gfnet_frozen_cls_pool8x8_fold1 | done | historical_tuning_reference |
| historical | mpp_v3_xzy_external_uni2h_mlp_20260701 | done | historical_superseded_by_mpp2_policy |
| historical | mpp_v3bis_valZHZ_xzy_external_uni2h_mlp_20260704 | done | historical_superseded_by_mpp2_policy |
| historical | mpp1_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | done | historical_superseded_by_mpp2_policy |
| historical | mpp4_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | done | historical_superseded_by_mpp2_policy |
| historical | mpp2_partnerwzk_xzy_external_uni2h_mlp_20260704 | done | historical_superseded_by_mpp2_policy |
| historical | mpp5_partnerwzk_xzy_external_uni2h_mlp_20260704 | done | historical_superseded_by_mpp2_policy |
| historical | mpp1_std10val_xzy_ext_uni2h_mlp_20260706 | done | archived_mpp2_selected |
| historical | mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | done | archived_mpp2_selected |
| historical | mpp4_std10val_xzy_ext_uni2h_mlp_20260706 | done | archived_mpp2_selected |
| historical | mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | done | archived_mpp2_selected |
| — | mpp2_huber_loss_paired_v001_20260728 | done | closed_no_retry |
| — | mpp2_cpgcr_probe_v001_20260811 | done | review_four_arm_analysis_no_redispatch |
| — | phase3_dual_baseline_spatial_pathway_transfer_v001_20260804 | done | open_new_patient_independent_protocol |
| historical | S1b_gfnet_lora_r8_cls_pool8x8_fold1 | paused | paused_mpp2_lora_first |

## 历史决策记录（按各实验后续状态限定）

| Experiment | Gate |
|------------|------|
| mpp2_std10val_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: selected for follow-up because MPP2 matches the cohort/external-test sampling protocol. Next: MPP2 new-data LoRA r=8 vs same-batch frozen baseline. |
| mpp2_barcode_repair_v003_frozen_baseline_20260711 | Safety guard was evaluated and led to the completed repaired MPP1/3/4/5 rechecks; use this accepted repaired baseline for reporting and do not redispatch automatically. |
| mpp1_barcode_repair_v003_frozen_recheck_20260711 | DIR-20260711-006 satisfied by validated result import; use this repaired formal result for reporting and do not redispatch automatically. |
| mpp3_barcode_repair_v003_frozen_recheck_20260711 | DIR-20260711-006 satisfied by validated result import with bbox embargo preserved; use this repaired formal result for reporting and do not redispatch automatically. |
| mpp4_barcode_repair_v003_frozen_recheck_20260711 | DIR-20260711-006 satisfied by validated result import; use this repaired formal result for reporting and do not redispatch automatically. |
| mpp5_barcode_repair_v003_frozen_recheck_20260711 | DIR-20260711-006 satisfied by validated result import with bbox embargo preserved; use this repaired formal result for reporting and do not redispatch automatically. |
| mpp2_online_cache_parity_v003_20260711 | Parity passed 48/48 samples; prepare paired S0 frozen-continue and S1 LoRA r=8 smoke, then evaluate incremental benefit before any formal training. |
| mpp2_paired_s0_frozen_continue_smoke_20260712 | Run paired with S1 using identical repaired manifest, head checkpoint, seed, sample order and validation split; smoke only. |
| mpp2_paired_s1_lora_r8_smoke_20260712 | Run paired with S0 using identical repaired manifest, head checkpoint, seed, sample order and validation split; smoke only. |
| mpp2_lora_r8_dropout10_smoke_20260714 | Single-factor dropout=0.10 only; compare against accepted S0 after internal checkpoint selection; do not retry or tune from XZY. Observed best_val_loss=0.38228295 exceeds S0=0.37892210 and best_val_pcc=0.79290723 is below S0=0.79479160, so the pre-registered internal-first gate failed. |
| mpp2_pathway_ridge_calibration_v001_20260717 | DIR-20260717-003 authorizes exactly one replacement execution after r002's gate-semantic failure. Internal nested-LOPO must pass before calibrator freeze and external evaluation; fold-specific OOF PCC differences are diagnostic only, while PCC invariance is checked on the final single frozen calibrator. XZY must remain unread until that freeze is recorded. |
| phase2_gene_reconstruction_comparison_v001_20260904 | Do not claim an executable protocol until the gene-expression and pathway-scoring inputs are confirmed. |
| phase2_spatial_density_comparison_v001_20260904 | Apply sparsification only after the leakage-safe train/test split; keep the test set dense and unchanged. |
| phase2_pathology_model_benchmark_v001_20260904 | 在固定历史输入与统一单点预测协议下，Virchow2-CLS 在三种子内部验证的预定选模指标上稳定、小幅优于 UNI2-h；外部 XZY 的逐通路等权 PCC 较高，但整体展平 PCC和误差指标未优于 UNI2-h。UNI 在内部与外部均整体落后 |
| C1_gfnet_lora_65t_fold1 | 2026-07-09 superseded: old 3-patient/JFX-affected Token+LoRA result is tuning reference only. Do not expand old Fold2/3 as evidence; rerun LoRA claims under MPP2/new-data protocol. |
| mpp1_std10val_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| mpp4_std10val_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| S1b_gfnet_lora_r8_cls_pool8x8_fold1 | 2026-07-09 superseded: old 3-patient/JFX-affected spatial Token+LoRA gate is tuning reference only. Do not launch before MPP2 new-data LoRA baseline is established. |
