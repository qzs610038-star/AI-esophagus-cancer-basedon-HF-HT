# G12 独立审核回执：W001 MPP2 paired MSE vs Huber δ=1 运行与回传

审核时间：2026-07-29

## 结论

**PASS（仅 G12 运行与回传边界）**。A003（MSE）与 A004（Huber `delta=1.0`）均已在服务器完成，并以各自 Gitee server-owned result ref 回传。两臂均有可审计的 `EXPERIMENT_STARTED` 与 `EXPERIMENT_TERMINAL(status=completed, returncode=0)` 事件；两次正式拟合合计消耗 `2` run units，按 approval `run_limit=4` 计算剩余 `2` units，仅保留给用户明确批准的工程故障重试。

本回执不执行 `result import`、accept、科学结论或 G13；本地 `project_state/attempt_events.jsonl` 因而仍仅保留 prepared 事件，不能被误读为结果已接纳。

## 绑定与远端 ref

| attempt | loss | result id | Gitee result ref | remote commit | parent governance commit |
| --- | --- | --- | --- | --- | --- |
| A003 / W001-A003 | `nn.MSELoss()` | `W001-A003-result-R003` | `automation/server/W001/A003` | `d59221d650821796a561f65fc98561334f914757` | `6c0f9cad173068ef194f3c1558a78e7fe8ceb313` |
| A004 / W001-A004 | `nn.HuberLoss(delta=1.0, reduction="mean")` | `W001-A004-result-R001` | `automation/server/W001/A004` | `445f72bd12c4409d775b8e147ff6d218c9711244` | `6c0f9cad173068ef194f3c1558a78e7fe8ceb313` |

两包均绑定：

- experiment：`mpp2_huber_loss_paired_v001_20260728`
- source commit：`a04319a1d6aeafcd35d6adc7e2893f1d7b683bda`
- approval：`APR-mpp2-huber-loss-paired-v001-20260728-r002`
- critical contract：`df0c5df369a3049125d06b985683c7bf7c4027664fc48d8b86f103e0ad8044e5`
- phase：`formal`；each `run_units=1`；result status：`success`

## 运行终态与预算

| attempt | terminal recorded_at | terminal status | returncode | run units |
| --- | --- | --- | ---: | ---: |
| A003 | `2026-07-29T04:19:18Z` | `completed` | 0 | 1 |
| A004 | `2026-07-29T04:50:40Z` | `completed` | 0 | 1 |

预算账本（运行证据口径）：`run_limit=4`、`run_consumed=2`、`remaining=2`。未发生隐式重试，也未运行 A005 或其他新 attempt。

## 回传结果包清单

| attempt | bundle root | critical artifacts | supporting artifacts | registered large artifact |
| --- | --- | --- | --- | --- |
| A003 | `automation/returns/W001/A003/R003` | `job.json`、`attempt_started.json`、`attempt_terminal.json`、`training_summary.txt`、`best_epoch.txt`、`training_history.csv`、`metrics.json` | `predictions_external_xzy.csv`、`predictions.csv`、`per_pathway_pcc_external_xzy.csv`、`per_pathway_pcc_external_xzy_rawscale.csv` | `best_checkpoint.pth`：6,420,688 B；SHA-256 `c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98` |
| A004 | `automation/returns/W001/A004/R001` | `job.json`、`attempt_started.json`、`attempt_terminal.json`、`training_summary.txt`、`best_epoch.txt`、`training_history.csv`、`metrics.json` | `predictions_internal_val.csv`、`predictions_external_xzy.csv`、`predictions.csv`、`per_pathway_pcc_external_xzy.csv`、`per_pathway_pcc_external_xzy_rawscale.csv` | `best_checkpoint.pth`：6,420,688 B；SHA-256 `6fe11dbc244952eb829d8e0b1a5bc730cb1179373ae86072b162f34ba8911db5` |

每个 artifact 的 `size_bytes` 与 SHA-256 已置入相应 `result.json`；两份 result-v2 信封均通过 `governance result-v2 validate`。

## 完整性核验

- A003：11/11 返回 artifact 已核验；A004：12/12 返回 artifact 已核验。
- `job.json`、`attempt_started.json`、`attempt_terminal.json` 共 6 个 JSON 文件与远端 Git blob 的 size/SHA-256 逐字节一致。
- 其余 17 个文本/CSV artifact 的 envelope 记录为服务器 Windows 文件的 CRLF 字节流；Gitee blob 为 Git 规范化后的 LF 字节流。逐一将 LF blob 唯一物化为 CRLF 后，size 和 SHA-256 均与 envelope 完全一致；未接受任何其他差异。
- 两个 return commit 的唯一父提交均为治理包 `6c0f9cad173068ef194f3c1558a78e7fe8ceb313`。
- 本地严格门禁最近一次结果：`PASS=7`、`WARN=6`、`FAIL=0`；WARN 均为既有 legacy/protected-data 提示，非本次 G12 失败。

## 运行结果记录（仅事实摘录，不作比较或科学结论）

- A003 returned metrics：external XZY `PCC=0.6549`、`MAE=0.6191`、`R²=-0.0880`、`MAE_raw=1176.2114`、`R²_raw=-0.0880`。
- A004 returned metrics：internal validation `PCC=0.7975`、`MAE=0.4504`、`R²=0.6358`、`loss=0.1723`；external XZY `PCC=0.6527`、`MAE=0.6184`、`R²=-0.0901`、`MAE_raw=1174.2159`、`R²_raw=-0.0901`。
- 两臂均声明 checkpoint 仅由内部 validation loss 冻结；XZY 均在冻结后评估一次，未用于选择或调参。

## 边界与后续状态

- G12 已完成的范围：正式运行、终态记录、Gitee 回传、远端 SHA、result-v2 信封及 artifact 完整性核验。
- 未完成且本回执未触发的范围：result import、accept、实验 Registry 状态提升、科学比较/结论、G13。
- 阻塞：无 G12 运行或回传阻塞。后续若要处理结果，必须由新的明确授权进入 quarantine/import 流程；不得以本回执替代接纳。
