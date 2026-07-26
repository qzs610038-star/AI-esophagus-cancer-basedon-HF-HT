# 维护机制改造前实验成果保留快照

> 记录时间：2026-07-14T16:47:35+08:00
> 目的：在实施维护机制优化前，固化当前已验收实验、受保护资产和状态包的可核验元数据。
> 范围：仅记录元数据与 SHA-256；本记录不复制、修改、删除或重新生成任何实验数据、标签、缓存、checkpoint 或结果包。

## 快照边界

- 状态版本：`63`
- 状态来源提交：`27e538136dc94796d803f9215d9bfa8756e35337`
- 活跃修复数据 manifest：`barcode-repair-20260711-d626ad8-v003:1204018178a4d355`
- 当前硬阻断保持不变：`direct_server_connection`、`formal_training_without_user_approval`、`implicit_mpp_asset_deletion_or_regeneration`、`promote_pending_result_to_current_conclusion`
- 受保护资产仍不在本次维护改造范围：MPP 原始 ssGSEA、标准划分、train-only z-score 参数、manifest、group 3/5 embargo 审计，以及所有 checkpoints/缓存。

## 快照校验值

| 文件 | SHA-256 |
|---|---|
| `experiments/experiment_registry.json` | `4a668844c52f5ff722d0b7020c4628ce83c44ac3500117278e2a29ca686bf975` |
| `project_state/current_state.json` | `60442880f9d7f1c9fb33c21815351337e9071e0c18341d5dbc84dcef4041bad8` |
| `project_state/mpp_repair_registry.json` | `cb9565694282aee5cc56a0d02386a90b1e9db93e3f1bbe361ef055acfdc1cf18` |
| `mpp_standard_splits/path_index.json` | `e5a7978618f9c62457aa9a1940d64a11078b45fa649f037a979cdc7fb7bdc3ad` |

维护完成后，应重新计算以上四项哈希；任何差异都必须能由批准的状态、注册表或受保护资产维护操作解释。维护机制代码、文档和非证据性诊断日志的变化本身不构成实验结果变化。

## 已验收实验成果清单

| 实验 ID | 阶段 | 证据完整性 | 数据 manifest | external XZY PCC | raw MAE | raw R2 |
|---|---|---:|---|---:|---:|---:|
| `mpp1_std10val_xzy_ext_uni2h_mlp_20260706` | legacy | 否 | `mpp-standard-splits-v1` | 0.7103 | 1384.9762 | -0.4192 |
| `mpp2_std10val_xzy_ext_uni2h_mlp_20260706` | legacy | 否 | `mpp-standard-splits-v1` | 0.6489 | 1209.9316 | -0.1554 |
| `mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706` | legacy | 否 | `mpp-standard-splits-v1` | 0.6462 | 1175.6385 | -0.1157 |
| `mpp4_std10val_xzy_ext_uni2h_mlp_20260706` | legacy | 否 | `mpp-standard-splits-v1` | 0.6064 | 1083.0882 | 0.0721 |
| `mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706` | legacy | 否 | `mpp-standard-splits-v1` | 0.5759 | 1125.1573 | -0.0479 |
| `mpp2_barcode_repair_v003_frozen_baseline_20260711` | formal | 是 | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` | 0.6549 | 1176.2114 | -0.0880 |
| `mpp1_barcode_repair_v003_frozen_recheck_20260711` | formal | 是 | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` | 0.6900 | 1360.9619 | -0.3719 |
| `mpp3_barcode_repair_v003_frozen_recheck_20260711` | formal | 是 | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` | 0.6436 | 1209.3894 | -0.1265 |
| `mpp4_barcode_repair_v003_frozen_recheck_20260711` | formal | 是 | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` | 0.6151 | 1053.1715 | 0.1090 |
| `mpp5_barcode_repair_v003_frozen_recheck_20260711` | formal | 是 | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` | 0.6072 | 1045.6602 | 0.0811 |
| `mpp2_paired_s0_frozen_continue_smoke_20260712` | smoke | 是 | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` | 0.6445 | 1215.2839 | -0.1449 |
| `mpp2_paired_s1_lora_r8_smoke_20260712` | smoke | 是 | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` | 0.6464 | 1206.8263 | -0.1359 |

## 已知维护前状态

- 严格状态检查在快照时为 `PASS=5 WARN=6 FAIL=2`。
- 两项失败为：document registry 落后当前状态超过一个 revision；`project_state/plans/mpp_training.md` 的 active normative hash 过期。
- 这两项为维护前既有的状态/文档同步问题，尚未对上述 accepted 实验记录、修复证据或受保护数据做任何改写。
- 当前工作区已有状态和文档变更，以及未跟踪的 `scripts/generate_lora_plots.py`；后续维护改造必须将其与本次机制代码变更分开审查。

## 维护改造限制

1. 本次改造只能修改维护机制代码、schema、测试、文档和新建的非证据性审计目录。
2. 不得修改 `histogene/`、`egnv1/`、`egnv2/`，不得改动任何 MPP 数据资产或训练产物。
3. 不得重新生成标准划分、z-score、manifest 或 embargo 审计；若未来确有需要，必须单独获得用户批准并建立新的比较任务。
4. 新增的 diagnostic/explore 机制不得将输出写入 `experiments/experiment_registry.json` 或替换任何 accepted result。
