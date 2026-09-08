# 实验进度

## 本轮软连接实验已完整登记（2026-09-08）

- 实验：`phase2_softlink_local_v2`；批次：`20260908_005325_810_8d306c10`；12次正式训练及12次XZY预测完成。
- 登记状态：`registered`；证据状态：`pending`（已登记、待复核，未标为accepted）。
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

> 依据实验登记生成；本次按用户会议决定更新相关进度。
> 本次按会议决策更新相关导航；不生成普通文档哈希。
> state_revision: `256`；updated_at: `2026-09-05T19:25:48+08:00`

## 当前 Phase 2 会议决策（2026-09-05）

- 当前主线：软对比联合学习，优先针对本数据改进；冻结UNI2-h加两层MLP保留为对照。
- 全部指标暂时待定且继续保留；逐通路平均PCC与展平/整体PCC均记录，最终发表指标后续再定。
- 比较范围：本项目内部前后及方案对照；暂不进行跨论文实验数值优劣比较。
- 排序：先改进软对比指标；六折留一患者暂缓；基因重建、密度、多基础模型及Phase 2/3衔接补充实验后置。
- 用户倾向后续选取有优势的指标发表；两种PCC的绝对高低不能互证优越，需保留完整结果和明确计算定义。
- 本轮只登记决定，未训练、未产生或接纳新结果。详细记录：[project_state/governance/Phase2会议决策_20260905.md](../project_state/governance/Phase2会议决策_20260905.md)。


## 当前与待处理

| 可读名称 | experiment ID | result ID | W### | 目的/比较 | 阶段 | 状态 | 证据等级 | 关键结果 | 当前结论 | 下一步 | 更新时间 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpp2_online_cache_parity_v003_20260711 | mpp2_online_cache_parity_v003_20260711 | legacy-import-mpp2_online_cache_parity_v003_20260711 | — | Parity passed 48/48 samples; prepare paired S0 frozen-continue and S1 LoRA r=8 smoke, then evaluate incremental benefit before any formal training. | preflight | done | planned | 暂无 | 待审查 | prepare_paired_smoke | — |
| Phase 2 基因预测后通路重建对照 | phase2_gene_reconstruction_comparison_v001_20260904 | — | — | Do not claim an executable protocol until the gene-expression and pathway-scoring inputs are confirmed. | formal | planned | planned | 暂无 | 待审查 | 暂缓：先改进软对比，再补协议与实验 | — |
| Phase 2 多病理模型公平比较 | phase2_pathology_model_benchmark_v001_20260904 | — | — | Historical model numbers are references only until rerun under the common protocol. | formal | planned | planned | 暂无 | 待审查 | 暂缓：先改进软对比，再补协议与实验 | — |
| Phase 2 稀疏训练与稠密测试对照 | phase2_spatial_density_comparison_v001_20260904 | — | — | Apply sparsification only after the leakage-safe train/test split; keep the test set dense and unchanged. | formal | planned | planned | 暂无 | 待审查 | 暂缓：先改进软对比，再补协议与实验 | — |
| S1b_gfnet_lora_r8_cls_pool8x8_fold1 | S1b_gfnet_lora_r8_cls_pool8x8_fold1 | — | — | 2026-07-09 superseded: old 3-patient/JFX-affected spatial Token+LoRA gate is tuning reference only. Do not launch before MPP2 new-data LoRA baseline is established. | — | paused | pending | 暂无 | 待审查 | paused_mpp2_lora_first | — |

## 已接纳结果

| 可读名称 | experiment ID | result ID | W### | 目的/比较 | 阶段 | 状态 | 证据等级 | 关键结果 | 当前结论 | 下一步 | 更新时间 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpp1_barcode_repair_v003_frozen_recheck_20260711 | mpp1_barcode_repair_v003_frozen_recheck_20260711 | mpp1-repair-v003-recheck-20260711-result-20260711203346-1a513ba4 | — | DIR-20260711-006 satisfied by validated result import; use this repaired formal result for reporting and do not redispatch automatically. | formal | done | accepted | Val PCC=0.7597；Val loss=0.4323；XZY PCC=0.6900；XZY MAE=0.8597；Test loss=1.2151 | 已接纳 | closed_use_accepted_repaired_result_for_reporting | 2026-07-11T12:50:59+00:00 |
| mpp2_barcode_repair_v003_frozen_baseline_20260711 | mpp2_barcode_repair_v003_frozen_baseline_20260711 | mpp2-repair-v003-frozen-20260711-r002-result-20260711164140-f2b1b51d | — | Safety guard was evaluated and led to the completed repaired MPP1/3/4/5 rechecks; use this accepted repaired baseline for reporting and do not redispatch automatically. | formal | done | accepted | Val PCC=0.7971；Val loss=0.3749；XZY PCC=0.6549；XZY MAE=0.6191；Test loss=0.6621 | 已接纳 | closed_use_accepted_repaired_result_for_reporting | 2026-07-11T08:42:23+00:00 |
| mpp2_lora_r8_dropout10_smoke_20260714 | mpp2_lora_r8_dropout10_smoke_20260714 | mpp2-lora-r8-dropout10-smoke-20260714-r001-result-20260714182311-7971c360 | — | Single-factor dropout=0.10 only; compare against accepted S0 after internal checkpoint selection; do not retry or tune from XZY. Observed best_val_loss=0.38228295 exceeds S0=0.37892210 and best_val_pcc=0.79290723 is below S0=0.79479160, so the pre-registered internal-first gate failed. | smoke | done | accepted | Val PCC=0.7929；Val loss=0.3823；XZY PCC=0.6438；XZY MAE=0.6349；Test loss=0.6843 | 已接纳 | close_dropout10_route_no_retry_or_tuning_after_internal_first_failure | 2026-07-14T10:25:16+00:00 |
| mpp2_paired_s0_frozen_continue_smoke_20260712 | mpp2_paired_s0_frozen_continue_smoke_20260712 | mpp2-paired-s0-frozen-continue-smoke-20260712-r002-result-20260712173215-1393b29c | — | Run paired with S1 using identical repaired manifest, head checkpoint, seed, sample order and validation split; smoke only. | smoke | done | accepted | Val PCC=0.7948；Val loss=0.3789；XZY PCC=0.6445；XZY MAE=0.6385；Test loss=0.6942 | 已接纳 | paired_smoke_completed_reviewed_no_redispatch | 2026-07-12T09:36:00+00:00 |
| mpp2_paired_s1_lora_r8_smoke_20260712 | mpp2_paired_s1_lora_r8_smoke_20260712 | mpp2-paired-s1-lora-r8-smoke-20260712-r002-result-20260712173316-163abad4 | — | Run paired with S0 using identical repaired manifest, head checkpoint, seed, sample order and validation split; smoke only. | smoke | done | accepted | Val PCC=0.7930；Val loss=0.3823；XZY PCC=0.6464；XZY MAE=0.6339；Test loss=0.6846 | 已接纳 | close_current_lora_r8_configuration_after_prediction_diagnostics | 2026-07-12T09:36:09+00:00 |
| mpp2_std10val_xzy_ext_uni2h_mlp_20260706 | mpp2_std10val_xzy_ext_uni2h_mlp_20260706 | legacy-import-mpp2_std10val_xzy_ext_uni2h_mlp_20260706 | — | 2026-07-09: selected for follow-up because MPP2 matches the cohort/external-test sampling protocol. Next: MPP2 new-data LoRA r=8 vs same-batch frozen baseline. | — | done | accepted | Val PCC=0.7827；Val loss=0.3984；XZY PCC=0.6489；XZY MAE=0.6357；Test loss=0.6928 | 已接纳 | selected_for_mpp2_newdata_lora | 2026-07-08T05:56:05 |
| mpp3_barcode_repair_v003_frozen_recheck_20260711 | mpp3_barcode_repair_v003_frozen_recheck_20260711 | mpp3-repair-v003-recheck-20260711-result-20260711204217-fb8a85ac | — | DIR-20260711-006 satisfied by validated result import with bbox embargo preserved; use this repaired formal result for reporting and do not redispatch automatically. | formal | done | accepted | Val PCC=0.8225；Val loss=0.3488；XZY PCC=0.6436；XZY MAE=0.6383；Test loss=0.6880 | 已接纳 | closed_use_accepted_repaired_result_for_reporting | 2026-07-11T12:51:00+00:00 |
| mpp4_barcode_repair_v003_frozen_recheck_20260711 | mpp4_barcode_repair_v003_frozen_recheck_20260711 | mpp4-repair-v003-recheck-20260711-result-20260711204236-49f9726f | — | DIR-20260711-006 satisfied by validated result import; use this repaired formal result for reporting and do not redispatch automatically. | formal | done | accepted | Val PCC=0.8209；Val loss=0.3198；XZY PCC=0.6151；XZY MAE=0.5094；Test loss=0.4635 | 已接纳 | closed_use_accepted_repaired_result_for_reporting | 2026-07-11T12:51:01+00:00 |
| mpp5_barcode_repair_v003_frozen_recheck_20260711 | mpp5_barcode_repair_v003_frozen_recheck_20260711 | mpp5-repair-v003-recheck-20260711-result-20260711204418-eef816b8 | — | DIR-20260711-006 satisfied by validated result import with bbox embargo preserved; use this repaired formal result for reporting and do not redispatch automatically. | formal | done | accepted | Val PCC=0.8347；Val loss=0.3166；XZY PCC=0.6072；XZY MAE=0.5082；Test loss=0.4723 | 已接纳 | closed_use_accepted_repaired_result_for_reporting | 2026-07-11T12:51:02+00:00 |
| MPP2连续通路几何软对比残差四臂探针 | mpp2_cpgcr_probe_v001_20260811 | W007-four-arm-seed42-result-R001 | W007 | 未登记 | formal | done | accepted | 暂无 | four-arm verification accepted on 2026-08-16: single-seed differences are small and mixed; no residual arm is a clear winner. Deep-attribution analysis and interpretation reports remain pending_user_review. | review_four_arm_analysis_no_redispatch | 2026-08-15T07:16:45+00:00 |
| MPP2 paired MSE vs Huber delta1 | mpp2_huber_loss_paired_v001_20260728 | PAIR-W001-A003-A004 | W001 | 未登记 | formal | done | accepted | Control PCC=0.6549；Treatment PCC=0.6527；ΔPCC=-0.0022 | Huber(delta=1)未提升external XZY pooled PCC；相同合同不得重试。 | closed_no_retry | 2026-07-29T13:20:03+00:00 |
| Phase 3 双线基线与跨队列空间通路迁移 | phase3_dual_baseline_spatial_pathway_transfer_v001_20260804 | — | W004 | 未登记 | formal | done | accepted | Control PCC=0.8386；Treatment PCC=0.8618；ΔPCC=+0.0232 | A002 minus A001 improves mean per-fit pCR AUC by 0.023182 and supports further patient-independent validation; the MPR difference is weaker. | open_new_patient_independent_protocol | 2026-08-27T15:29:11+00:00 |
| mpp1_std10val_xzy_ext_uni2h_mlp_20260706 | mpp1_std10val_xzy_ext_uni2h_mlp_20260706 | legacy-import-mpp1_std10val_xzy_ext_uni2h_mlp_20260706 | — | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. | — | done | accepted | Val PCC=0.6473；Val loss=0.5965；XZY PCC=0.7103；XZY MAE=0.8652；Test loss=1.2223 | 已接纳 | archived_mpp2_selected | 2026-07-08T05:56:05 |
| mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | legacy-import-mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | — | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. | — | done | accepted | Val PCC=0.8016；Val loss=0.3877；XZY PCC=0.6462；XZY MAE=0.6211；Test loss=0.6745 | 已接纳 | archived_mpp2_selected | 2026-07-08T05:56:06 |
| mpp4_std10val_xzy_ext_uni2h_mlp_20260706 | mpp4_std10val_xzy_ext_uni2h_mlp_20260706 | legacy-import-mpp4_std10val_xzy_ext_uni2h_mlp_20260706 | — | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. | — | done | accepted | Val PCC=0.8075；Val loss=0.3431；XZY PCC=0.6064；XZY MAE=0.5213；Test loss=0.4764 | 已接纳 | archived_mpp2_selected | 2026-07-08T05:56:06 |
| mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | legacy-import-mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | — | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. | — | done | accepted | Val PCC=0.8069；Val loss=0.3641；XZY PCC=0.5759；XZY MAE=0.5444；Test loss=0.5306 | 已接纳 | archived_mpp2_selected | 2026-07-08T05:56:06 |

## 历史记录

| 可读名称 | experiment ID | result ID | W### | 目的/比较 | 阶段 | 状态 | 证据等级 | 关键结果 | 当前结论 | 下一步 | 更新时间 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpp2_pathway_ridge_calibration_v001_20260717 | mpp2_pathway_ridge_calibration_v001_20260717 | mpp2-pathway-ridge-calibration-20260717-r003-result-20260718000516-8f133e77 | — | DIR-20260717-003 authorizes exactly one replacement execution after r002's gate-semantic failure. Internal nested-LOPO must pass before calibrator freeze and external evaluation; fold-specific OOF PCC differences are diagnostic only, while PCC invariance is checked on the final single frozen calibrator. XZY must remain unread until that freeze is recorded. | formal | failed | rejected | 暂无 | 已拒绝 | validate gate-correction regression then create one bound replacement formal job manifest | 2026-07-17T16:07:20+00:00 |
| 零训练Gitee往返试点W006 | gitee_roundtrip_pilot_w006_v001_20260810 | — | W006 | 未登记 | preflight | closed | pending | 暂无 | 待审查 | closed_user_approved_retained_worktree | 2026-08-11T00:00:00+00:00 |
| online_tokens_gfnet_fold1_65t_legacy | online_tokens_gfnet_fold1_65t_legacy | legacy-import-online_tokens_gfnet_fold1_65t_legacy | — | 未登记 | — | done | historical | Val PCC=0.3914；Val loss=0.3337 | 仅历史参考 | archived_as_legacy_baseline | 2026-06-10T02:30:59 |
| online_tokens_transformer_fold1_65t | online_tokens_transformer_fold1_65t | legacy-import-online_tokens_transformer_fold1_65t | — | 未登记 | — | done_incomplete_data | historical | Val PCC=0.3821 | 仅历史参考 | archived | 2026-06-09 |
| smoke_gfnet_65t | smoke_gfnet_65t | legacy-import-smoke_gfnet_65t | — | 未登记 | — | done | historical | Val PCC=0.3933；Val loss=0.3314 | 仅历史参考 | archived | 2026-06-09T17:06:36 |
| A0b_gfnet_clean_token_65t_fold1 | A0b_gfnet_clean_token_65t_fold1 | legacy-import-A0b_gfnet_clean_token_65t_fold1 | — | A0b serves as the clean-token (cls_patch64) baseline for A1 comparison. | — | done | historical | Val PCC=0.3967；Val loss=0.3320 | 仅历史参考 | historical_tuning_reference | 2026-06-11 |
| A1_gfnet_modrelu_65t_fold1 | A1_gfnet_modrelu_65t_fold1 | legacy-import-A1_gfnet_modrelu_65t_fold1 | — | A1 vs A0b: modReLU degraded val_loss (-2.2%), val_pcc (-6.1%), Train-Val Gap widened (+50%). Gate: FAIL → frequency sparsification hypothesis disproven. A2 (frequency augmentation) and A3 (AFNO) cancelled. | — | done | historical | Val PCC=0.3724；Val loss=0.3394 | 仅历史参考 | historical_tuning_reference | 2026-06-11 |
| B1a_cls_freq_gfnet_fold1 | B1a_cls_freq_gfnet_fold1 | legacy-import-B1a_cls_freq_gfnet_fold1 | — | B1a vs B1b: peak PCC 0.3956 vs 0.3955 — nearly identical. Frequency branch adds no incremental value over mean pooling. Both below CLS Frozen baseline (0.4113). Gate: FAIL → frequency side-branch disproven for CLS mode. | — | done | historical | Val PCC=0.3881；Val loss=0.3297 | 仅历史参考 | historical_tuning_reference | 2026-06-11 |
| B1b_cls_freq_mean_fold1 | B1b_cls_freq_mean_fold1 | legacy-import-B1b_cls_freq_mean_fold1 | — | B1b (mean pool) same as B1a (gfnet) → frequency branch has no mechanism advantage. Both < CLS Frozen 0.4113. The CLS+patching side-branch architecture itself may not help. | — | done | historical | Val PCC=0.3955；Val loss=0.3313 | 仅历史参考 | historical_tuning_reference | 2026-06-11 |
| C1_gfnet_lora_65t_fold1 | C1_gfnet_lora_65t_fold1 | legacy-import-C1_gfnet_lora_65t_fold1 | — | 2026-07-09 superseded: old 3-patient/JFX-affected Token+LoRA result is tuning reference only. Do not expand old Fold2/3 as evidence; rerun LoRA claims under MPP2/new-data protocol. | — | done | historical | Val PCC=0.4169；Val loss=0.3321 | 仅历史参考 | historical_tuning_reference_mpp2_lora_first | 2026-06-11 |
| S1a_gfnet_frozen_cls_pool8x8_fold1 | S1a_gfnet_frozen_cls_pool8x8_fold1 | legacy-import-S1a_gfnet_frozen_cls_pool8x8_fold1 | — | S1a >= A0b+0.01 (>=0.4067) -> cls_pool8x8 replaces cls_patch64 as Token route default. < A0b+0.01 -> spatial coverage not bottleneck; proceed to Step 2. | — | done | historical | Val PCC=0.3785；Val loss=0.3366 | 仅历史参考 | historical_tuning_reference | 2026-06-13T06:05:39 |
| mpp1_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | mpp1_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | legacy-import-mpp1_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | — | 未登记 | — | done | historical | Val PCC=0.5309；Val loss=0.3071；XZY PCC=0.7056 | 仅历史参考 | historical_superseded_by_mpp2_policy | 2026-07-04T07:38:44 |
| mpp2_partnerwzk_xzy_external_uni2h_mlp_20260704 | mpp2_partnerwzk_xzy_external_uni2h_mlp_20260704 | legacy-import-mpp2_partnerwzk_xzy_external_uni2h_mlp_20260704 | — | 未登记 | — | done | historical | Val PCC=0.6625；Val loss=0.2360；XZY PCC=0.4555；XZY MAE=0.6881 | 仅历史参考 | historical_superseded_by_mpp2_policy | 2026-07-04T08:02:59 |
| mpp4_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | mpp4_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | legacy-import-mpp4_partner_valHYZ_xzy_ext_uni2h_mlp_20260704 | — | 未登记 | — | done | historical | Val PCC=0.6844；Val loss=0.5382；XZY PCC=0.5871 | 仅历史参考 | historical_superseded_by_mpp2_policy | 2026-07-04T07:38:56 |
| mpp5_partnerwzk_xzy_external_uni2h_mlp_20260704 | mpp5_partnerwzk_xzy_external_uni2h_mlp_20260704 | legacy-import-mpp5_partnerwzk_xzy_external_uni2h_mlp_20260704 | — | 未登记 | — | done | historical | Val PCC=0.8459；Val loss=0.1300；XZY PCC=0.4472；XZY MAE=0.6615 | 仅历史参考 | historical_superseded_by_mpp2_policy | 2026-07-04T08:02:59 |
| mpp_v3_xzy_external_uni2h_mlp_20260701 | mpp_v3_xzy_external_uni2h_mlp_20260701 | legacy-import-mpp_v3_xzy_external_uni2h_mlp_20260701 | — | 未登记 | — | done | historical | XZY PCC=0.6020 | 仅历史参考 | historical_superseded_by_mpp2_policy | 2026-07-03T18:37:24 |
| mpp_v3bis_valZHZ_xzy_external_uni2h_mlp_20260704 | mpp_v3bis_valZHZ_xzy_external_uni2h_mlp_20260704 | legacy-import-mpp_v3bis_valZHZ_xzy_external_uni2h_mlp_20260704 | — | 未登记 | — | done | historical | Val PCC=0.7146；Val loss=1.0574；XZY PCC=0.6665 | 仅历史参考 | historical_superseded_by_mpp2_policy | 2026-07-04T05:32:11 |
