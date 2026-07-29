# G12-R 最终独立审核回执：W001 result bundle 仅封装修订

审核日期：2026-07-29

## Claim

- Testable claim：`result_bundle_v1` 已形成 build、validate、result-v2 validate 与 record-import 共用的完整校验入口；A003/R004 与 A004/R002 是仅修正封装元数据和包结构的不可变新 revision，并已仅经 Gitee fast-forward 回传。
- Requested decision：是否可关闭 G12-R 的实现、封装、校验与回传范围。
- Scope：W001；A003/R003 -> R004；A004/R001 -> R002；不包括 result import/accept、G13 或科学结论。

## Sources inspected

| Source | Lifecycle | What it proves |
| --- | --- | --- |
| `scripts/pfmval_result_bundle.py`、`scripts/pfmval_state.py`，实现提交 `50fb973e4ce3092a408398c94310214a87ecdba0` | current | 单一入口、完整 Schema 后端、语义/闭包/大小/SHA-256/LF-CRLF 校验 |
| Gitee `automation/server/W001/A003` -> `83a7362e30cdcae871c638f03733815db501d963` | pending_review | A003/R004 远端实际 commit 与 bundle |
| Gitee `automation/server/W001/A004` -> `a34f257f50bf1bf68a93571a37bb7dfc11b6236b` | pending_review | A004/R002 远端实际 commit 与 bundle |
| A003/R003 tree `8e12511f6293ecd1f0b5cb486370acb2c91dc49d`；A004/R001 tree `e49e5d1e2a9afbd267f587f269399ac88347e959` | historical | 原 revision 在新 commit 中保持原树身份 |
| `G12_W001_mpp2_huber_loss_execution_audit_receipt_20260729.md`、`G12_W001_mpp2_huber_loss_run_return_receipt_20260729.md` | historical/current audit chain | G12 前置阻塞与后续运行/回传审计沿革；不替代本次远端复核 |
| `experiments/experiment_registry.json`、`project_state/attempt_events.jsonl` | current | experiment 仍为 `planned/pending`；未登记 R004/R002；预算上限与 attempt 绑定 |

## Checks

| Result | Check | Fresh evidence | Boundary |
| --- | --- | --- | --- |
| PASS | W001 locus | branch `codex/w001-mpp2-huber-loss-20260728`；审核前 HEAD `8a44cbe3419a67dc8588837344707788e439c3e7`；clean | current implementation worktree |
| PASS | Gitee-only ref | `git ls-remote` 仅列出 W001/A003 与 W001/A004 result refs，SHA 与回传一致 | 未使用 SSH/SCP/远程 shell |
| PASS | A003 commit/parent | `83a7362...` 的唯一父提交为 `d59221d...` | fast-forward from immutable R003 |
| PASS | A004 commit/parent | `a34f257...` 的唯一父提交为 `445f72b...` | fast-forward from immutable R001 |
| PASS | revision 唯一新增 | A003 commit 只新增 `automation/returns/W001/A003/R004/**`；A004 commit 只新增 `automation/returns/W001/A004/R002/**` | 无历史路径修改 |
| PASS | 历史 revision 不变 | R003 与 R001 在各自父/子 commit 中 tree SHA 完全相同 | immutable history |
| PASS | artifact 内容不变 | A003 旧 11 个 artifact 与 R004 对应 Git blob 11/11 相同；A004 为 12/12；新包仅另增 `bundle_integrity.json` | 内容身份；不把封装元数据修正误判为内容变化 |
| PASS | 绑定、指标与 large artifact 保留 | job、attempt、experiment、protocol、approval、contract、source commit、phase、run units、metrics、metric artifact ids 与 large artifact 原字段均无差异 | 未修改模型指标或科学内容 |
| PASS | completed -> success | 两个 terminal artifact 均为 `completed/returncode=0`；两个新 result envelope 均为 `success` | 固定终态映射 |
| PASS | A003 完整校验 | 固定实现强制使用 PowerShell `Test-Json -SchemaFile` 后返回 `valid`、artifact count `12`、bundle SHA-256 `f962f9a845e500648d147be6e2ea365be7de266fd0adb10dcedd20ad760c45e1` | 完整 Draft 2020-12 Schema + 语义/闭包/完整性 |
| PASS | A004 完整校验 | 同一入口返回 `valid`、artifact count `13`、bundle SHA-256 `a2bfca4b0140e90c3c8ce7c90a34b0ad3d3266c7e1f5b9d885c4876494dc2c78` | 与 A003 相同校验路径 |
| PASS | 数组、ID、retention、路径与 LF/CRLF | 两包均通过 result_bundle_v1 完整 validator；artifact 闭包、唯一相对路径、artifact_id、retention、原始/Git 规范化完整性均无失败 | Schema 未修改、未放宽 |
| PASS | 幂等/FF/远端 SHA | publish 回传为 `published`、`remote_verified=true`；本地重新 fetch 后 ref SHA 一致 | Gitee transport |
| PASS | 预算未增加 | 两个原正式结果各 `run_units=1`；新 revision 复用相同 terminal/artifact blob；无 A005 或其他新 result ref | `run_consumed=2/4` |
| PASS | 未 import/accept/G13 | R004/R002 在本地 state/registry JSON/JSONL 中命中数均为 0；experiment 仍为 `planned`、`evidence_status=pending`、`result_id=null` | revisions 保持 `pending_review` |
| PASS | 回归测试 | `python -m pytest -q` -> `190 passed, 2 warnings` | warnings 为既有 R² 小样本提示 |
| PASS | 严格门禁 | `PASS=7 WARN=6 FAIL=0` | 6 WARN 为既有 legacy envelope/protected-label 债务 |

## Verdict

- Overall：**GO（仅 G12-R 实现、封装、完整校验与 Gitee 回传范围）**。
- Reason：所有完成条件均有新鲜本地与远端证据；两个 revision 通过同一完整校验入口，远端 commit/parent/bundle SHA 可复核，历史 revision 与 artifact 内容保持不变，预算仍为 `2/4`。
- Evidence scope：R004/R002 为 `pending_review` 封装修订，不是 accepted experiment result；本回执不授权 import/accept、G13 或科学结论。

## Remaining

- Blocking：无 G12-R 阻塞。
- Non-blocking：严格门禁的 6 个既有 WARN 不属于本任务修复范围。
- Next authorized action：无。若后续需要 G13 import/accept，必须由新的明确用户授权启动，并再次验证当时的远端 ref 与本地状态。
