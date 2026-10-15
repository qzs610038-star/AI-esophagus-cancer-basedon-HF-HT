# G13 审核回执：W001 MSE / Huber 双包验证后治理导入

审核日期：2026-07-29

## 范围与结论边界

- 目标：对 W001 的 A003（MSE）与 A004（Huber delta=1）修复封装修订执行双包 aggregate preflight，随后只经治理 v3 `result-bundle-v1 record-import` 写入验证事件。
- 非目标：不比较任何指标、不形成科学结论、不接受 evidence、不启动训练、不使用剩余 run units，也不启动 G14。
- 结论：**PASS（仅验证后导入与 pending-review 边界）**。两个包均形成唯一的 `RESULT_IMPORT_VERIFIED` 事件；事件只是完整包验证的留痕，未触发 `accepted`、`done` 或科学结论。

## 前置条件与工作树

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| W001 locus | PASS | 路径 `D:\AI空间转录病理研究\PFMval_new_w001_mpp2_huber_loss_20260728`；branch `codex/w001-mpp2-huber-loss-20260728`；HEAD `31e9993fb0deef64ab2baa60da985d73bf97ca32` |
| 严格门禁 | PASS | `python deploy/pfmval_ops.py agent start-check --strict`：PASS=7、WARN=6、FAIL=0；WARN 均为既有 legacy envelope/protected-label 债务 |
| G12-R 前置 | PASS | `project_state/governance/G12_R_W001_result_bundle_final_audit_receipt_20260729.md` 明确确认 A003/R004、A004/R002 通过完整 `result_bundle_v1` 校验，且未 import/accept |
| 旧 revision 隔离 | PASS | 只接收 A003/R004 与 A004/R002；未导入 A003/R003 或 A004/R001 |

## Gitee refs 与隔离物化

仅通过 remote `gitee` 重新查询并 fetch 以下 refs，之后使用 `git archive` 物化到隔离目录 `D:\AI空间转录病理研究\PFMval_new_w001_mpp2_huber_loss_20260728_quarantine\G13_20260729T210212`。

| Attempt / revision | Gitee commit | Parent | bundle SHA-256 |
| --- | --- | --- | --- |
| A003 / R004 | `83a7362e30cdcae871c638f03733815db501d963` | `d59221d650821796a561f65fc98561334f914757` | `f962f9a845e500648d147be6e2ea365be7de266fd0adb10dcedd20ad760c45e1` |
| A004 / R002 | `a34f257f50bf1bf68a93571a37bb7dfc11b6236b` | `445f72bd12c4409d775b8e147ff6d218c9711244` | `a2bfca4b0140e90c3c8ce7c90a34b0ad3d3266c7e1f5b9d885c4876494dc2c78` |

## 双包 aggregate preflight

| 检查 | A003/R004 | A004/R002 |
| --- | --- | --- |
| JSON Schema、`result_bundle_v1` 语义与闭包 | PASS | PASS |
| artifact 大小、SHA-256、LF/CRLF 规范化 | PASS，12 artifacts | PASS，13 artifacts |
| experiment / workspace / protocol | `mpp2_huber_loss_paired_v001_20260728` / W001 / rev 2 | 同左 |
| approval / contract / source commit | `APR-mpp2-huber-loss-paired-v001-20260728-r002` / `df0c5d…044e5` / `a04319…83bda` | 同左 |
| started / terminal / returncode | `EXPERIMENT_STARTED` / `EXPERIMENT_TERMINAL`，`completed`，0 | `EXPERIMENT_STARTED` / `EXPERIMENT_TERMINAL`，`completed`，0 |
| run units | A003 唯一 attempt，1 | A004 唯一 attempt，1 |

聚合去重后 `run_consumed=2`、`run_limit=4`、`remaining=2`。两包均已通过后才发生任何 import 写入。

## v3 验证后导入与幂等性

使用的唯一写入口为：

```powershell
python deploy/pfmval_ops.py governance result-bundle-v1 record-import --bundle <quarantine-bundle> --bundle-sha256 <validated-sha>
```

未使用旧 `result import`，也未直接调用绕过 bundle 校验的 `result-v2 record-import`。

| result id | event id | 初次执行 | 重复执行 |
| --- | --- | --- | --- |
| `W001-A003-result-R004` | `import-W001-A003-result-R004` | `recorded` | `already_recorded` |
| `W001-A004-result-R002` | `import-W001-A004-result-R002` | `recorded` | `already_recorded` |

`project_state/result_import_events.jsonl` 中恰有上述两条 `RESULT_IMPORT_VERIFIED` 记录；它们保留 result `status=success` 的运行终态，却不改变 registry 的 `planned` / `evidence_status=pending`。因此导入证据仍处于 **pending_review**，并非 accepted scientific evidence。

## 后续验证要求

- 执行项目 CLI 的 `state sync` 与 `views refresh`，再执行全量 pytest 和严格门禁。
- G14 未启动；本回执不构成任何后续训练、比较或 evidence acceptance 授权。
