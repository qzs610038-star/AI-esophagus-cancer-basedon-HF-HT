# G11 独立审核回执：W001 environment probe

- 审核时间：2026-07-28（本地发起与推送阶段）
- 结论：**WARN — awaiting_server_return**
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
| return ref | `refs/heads/automation/diagnostics/w001-g11-environment-probe`（尚不存在） |
| local verification hash | `65d1d7f39652dd1b7f027590bf6bac3f0446c89a53928a105bab104ad4c7bb85` |

`local verification hash` 是以下固定字段（换行 UTF-8）的 SHA-256：diagnostic/command/source/request/request-file hash、`run_limit=4`、`run_consumed=0`、A001/A002 状态、`EXPERIMENT_STARTED=0` 与本轮 `new_result_paths=0`。

## 通道与安全边界

- 仅使用 `git push`、`git ls-remote` 访问命名为 `gitee` 的远端；未使用 SSH/SCP/HTTP remote command/Tunnel 作为服务器执行通道。
- request 内容绑定 source commit、source branch、return branch 与 allowlisted `environment_probe`；其合同明确为 `arbitrary_shell=false`、`training=false`、`experiment_registry_write=false`、`current_state_write=false`、`protected_asset_write=false`、`result_import=false`。
- 已验证 source 工作树 `W001` 为 clean、HEAD=`a04319a...`；治理工作树为 clean、HEAD=`5d340bc...`（request 前），且 source commit 是该治理提交的祖先。

## 不变性复核

| 检查 | 前后值 / 证据 |
| --- | --- |
| run budget | `run_limit=4`，`run_consumed=0`；未调用 job run/dispatch |
| A001 / A002 | 均保持 `ATTEMPT_PREPARED`；两份 prepared job manifest 都通过 `job validate` |
| Experiment Result | 本轮 Git 变更仅为 `request.json` 与 `project_state/diagnostics.jsonl`；未新增 result envelope、未运行 import/accept |
| experiment status | 仍为 planned；无 `EXPERIMENT_STARTED` event |
| 受保护范围 | 未修改 `histogene/`、`egnv1/`、`egnv2/` 或受保护 MPP 数据 |

## 门禁与测试

- `python deploy/pfmval_ops.py agent start-check --task diagnostic`：`PASS=5 WARN=0 FAIL=0`。
- `python deploy/pfmval_ops.py agent start-check --strict`：exit 0，`PASS=7 WARN=6 FAIL=0`；六项均为既有 legacy result-envelope/MPP 条码债务，G11 的两次提交未触及其关联文件，故本 Goal 无新增 WARN/FAIL。
- `workspace check-locus --workspace-id W001 --cwd <W001 source>`：PASS。
- `job validate --manifest project_state/governance/g10_A0_job_prepared.json`：PASS。
- `job validate --manifest project_state/governance/g10_A1_job_prepared.json`：PASS。
- request 创建命令：PASS；提交前审查仅含诊断 request 与 append-only request event。
- `git ls-remote gitee`：source 和 request ref SHA 如上；return ref 未出现。

## 暂停点

Gitee request 已可被服务器侧获取，但没有 return ref，因此尚无 `executed_at`、exit code 或 output SHA-256，也未运行本地 `diagnostic record`。这不是实验结果，不能宣称 G11 PASS。请在服务器侧通过 Gitee 获取 exact source commit、执行 allowlisted probe 并将记录推送至指定 return branch；本地后续只能 fetch 与只读核验，再调用 `diagnostic record`。
