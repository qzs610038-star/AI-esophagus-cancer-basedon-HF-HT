# Experiment Dashboard

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

> 依据实验登记维护；会议决策更新：2026-09-05T19:25:48+08:00。
> Source of truth: `experiments/experiment_registry.json`
>
> 🟢 **JFX0729 数据更正状态**：更正数据已同步、本地处理并通过 7/7 checks passed；旧 JFX 相关结果仅作历史参考。下一步：defer JFX token-cache rebuild; a new data transformation is planned and caches/retraining should be rebuilt only after that transform is finalized and applied consistently。
>
> ✅ **当前主线（2026-09-05）**：软对比联合学习；冻结UNI2-h加两层MLP保留为固定对照。
> 下一步：优先针对本数据讨论软对比改进，指标提升后再开展后续补充；六折继续暂缓。
>
> ⚠️ **旧三患者实验结论**：Old 3-patient LoRA/Token/frequency/spatial-repair conclusions must not be used as formal evidence. They may only seed hyperparameters and risk checks for new MPP2 experiments.

## 当前 Phase 2 会议决策（2026-09-05）

- 当前主线：软对比联合学习，优先针对本数据改进；冻结UNI2-h加两层MLP保留为对照。
- 全部指标暂时待定且继续保留；逐通路平均PCC与展平/整体PCC均记录，最终发表指标后续再定。
- 比较范围：本项目内部前后及方案对照；暂不进行跨论文实验数值优劣比较。
- 排序：先改进软对比指标；六折留一患者暂缓；基因重建、密度、多基础模型及Phase 2/3衔接补充实验后置。
- 用户倾向后续选取有优势的指标发表；两种PCC的绝对高低不能互证优越，需保留完整结果和明确计算定义。
- 本轮只登记决定，未训练、未产生或接纳新结果。详细记录：[project_state/governance/Phase2会议决策_20260905.md](../project_state/governance/Phase2会议决策_20260905.md)。


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
| phase2_pathology_model_benchmark_v001_20260904 | phase2_supplemental | :large_blue_circle: planned | — |  | — | — | — | — | — |
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

## Priority Queue

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
| P0 | mpp2_pathway_ridge_calibration_v001_20260717 | failed | validate gate-correction regression then create one bound replacement formal job manifest |
| P2 | phase2_gene_reconstruction_comparison_v001_20260904 | planned | 暂缓：先改进软对比，再补协议与实验 |
| P2 | phase2_spatial_density_comparison_v001_20260904 | planned | 暂缓：先改进软对比，再补协议与实验 |
| P2 | phase2_pathology_model_benchmark_v001_20260904 | planned | 暂缓：先改进软对比，再补协议与实验 |
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

## Decision Gates

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
| phase2_pathology_model_benchmark_v001_20260904 | Historical model numbers are references only until rerun under the common protocol. |
| C1_gfnet_lora_65t_fold1 | 2026-07-09 superseded: old 3-patient/JFX-affected Token+LoRA result is tuning reference only. Do not expand old Fold2/3 as evidence; rerun LoRA claims under MPP2/new-data protocol. |
| mpp1_std10val_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| mpp4_std10val_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706 | 2026-07-09: archived because team selected MPP2 as the only follow-up MPP scheme. |
| S1b_gfnet_lora_r8_cls_pool8x8_fold1 | 2026-07-09 superseded: old 3-patient/JFX-affected spatial Token+LoRA gate is tuning reference only. Do not launch before MPP2 new-data LoRA baseline is established. |
