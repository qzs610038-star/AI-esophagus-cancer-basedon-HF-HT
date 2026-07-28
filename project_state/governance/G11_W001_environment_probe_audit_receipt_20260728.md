# G11 独立审核回执：W001 environment probe

- 审核时间：2026-07-28（request→return→local record 完整往返）
- 结论：**PASS**
- G12：**NOT RUN**

## 身份与请求

| 项目 | 已核验值 |
| --- | --- |
| diagnostic ID | `diagnostic-20260728-w001-g11-environment-probe` |
| command ID | `environment_probe`（仅解释器、PyTorch/CUDA、磁盘与基础运行环境元数据） |
| source branch / commit | `codex/w001-mpp2-huber-loss-20260728-bound` / `a04319a1d6aeafcd35d6adc7e2893f1d7b683bda` |
| source remote ref / SHA | `refs/heads/codex/w001-mpp2-huber-loss-20260728-bound` / `a04319a1d6aeafcd35d6adc7e2893f1d7b683bda` |
| request branch / commit | `codex/w001-mpp2-huber-loss-20260728` / `04ba4b58964ea6befb137bb16664d7af516b3996` |
| request file SHA-256 | `a23f2d9849537757b61750f913ae83228e0a2330ad4d49e571f9aecb1059e49d` |
| return ref / SHA | `refs/heads/automation/diagnostics/w001-g11-environment-probe` / `486a9af5c76df81cdb2b7b7c14be8d85f54ee123` |
| return parent | `2c0d8ad98b0bbd44d6c47c3c5232e20dcd5d51cd`（首次回传的快进规范化修正） |
| server completed record | `recorded_at=2026-07-28T15:51:39Z`；`source_commit=a04319a...`；`command_id=environment_probe` |
| executed probe | `executed_at=2026-07-28T15:40:37.1892593Z`；detached HEAD=`true`；`exit_code=0` |
| returned output | `environment_probe.json`；1316 bytes；SHA-256=`86c836713d66e8f82d54ae3c145084e9b881db8e253cb8bb44da485bfeb397cd` |
| local verification hash | `f5fd00f46afed0ac1f5e4657ecf7f48954a0c8d212faf3580ac1135a28851096` |

`local verification hash` 是以下固定字段（换行 UTF-8）的 SHA-256：diagnostic/command/source/request/return SHA、output size/hash、`executed_at`/exit code、`run_limit=4`、`run_consumed=0`、A001/A002 状态、`EXPERIMENT_STARTED=0` 与本轮 `new_result_paths=0`。

## 通道与安全边界

- 仅使用 `git push`、`git ls-remote`、`git fetch` 访问命名为 `gitee` 的远端；未使用 SSH/SCP/HTTP remote command/Tunnel 作为服务器执行通道。
- request 内容绑定 source commit、source branch、return branch 与 allowlisted `environment_probe`；其合同明确为 `arbitrary_shell=false`、`training=false`、`experiment_registry_write=false`、`current_state_write=false`、`protected_asset_write=false`、`result_import=false`。
- 已验证 source 工作树 `W001` 为 clean、HEAD=`a04319a...`；治理工作树为 clean、HEAD=`5d340bc...`（request 前），且 source commit 是该治理提交的祖先。

## 不变性复核

| 检查 | 前后值 / 证据 |
| --- | --- |
| run budget | `run_limit=4`，`run_consumed=0`；未调用 job run/dispatch |
| A001 / A002 | 均保持 `ATTEMPT_PREPARED`；两份 prepared job manifest 都通过 `job validate` |
| Experiment Result | 仅新增 diagnostic output 与 append-only diagnostic record；无 G11 result envelope、未运行 import/accept |
| experiment status | 仍为 planned；无 `EXPERIMENT_STARTED` event |
| 受保护范围 | 未修改 `histogene/`、`egnv1/`、`egnv2/` 或受保护 MPP 数据 |

## 门禁与测试

- `python deploy/pfmval_ops.py agent start-check --task diagnostic`：`PASS=5 WARN=0 FAIL=0`。
- `python deploy/pfmval_ops.py agent start-check --strict`：exit 0，`PASS=7 WARN=6 FAIL=0`；六项均为既有 legacy result-envelope/MPP 条码债务，G11 的两次提交未触及其关联文件，故本 Goal 无新增 WARN/FAIL。
- `workspace check-locus --workspace-id W001 --cwd <W001 source>`：PASS。
- `job validate --manifest project_state/governance/g10_A0_job_prepared.json`：PASS。
- `job validate --manifest project_state/governance/g10_A1_job_prepared.json`：PASS。
- request 创建命令：PASS；提交前审查仅含诊断 request 与 append-only request event。
- `git fetch gitee <return-ref>`：PASS；仅建立远端跟踪 ref，未 merge。
- return tree、canonical LF `diagnostic_completed` event 与本地文件三者的 size/hash 均为 `1316` / `86c836...`；旧的错误 hash event 按 append-only 规则保留但未被采用。
- `diagnostic record --diagnostic-id ... --output .../environment_probe.json`：PASS；本地 record 产出同一 SHA-256。
- `git ls-remote gitee`：source、request 与 return ref SHA 均已核验。

## 完成界限

服务器仅通过 Gitee 获取 exact source commit 并返回 allowlisted、非证据性的环境探针；本地仅 fetch、只读核验后记录 output hash。该诊断没有进入 Experiment Result，未改变训练预算或尝试状态。G11 到此结束，G12 不得自动开始。
