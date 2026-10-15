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

## 复盘与后续改善（2026-07-28；非规范性提案）

### 用户体验记录

1. 每次服务器同步命令过于繁琐；SHA-256 校验已多次失败并导致重试。期望定位根因，标准化服务器拉取、检测和回传命令，使常规往返可一次通过，或至少显著简化。
2. 对话不应等到用户要求后才读取服务器配置并给出可用命令。对已授权的 Gitee 服务器往返，Agent 应主动完成配置读取、精确 ref 核验与可复制操作卡生成。

### 本次可证实根因

- **命令重复的系统性原因**：`sync-server` 当前是 retired compatibility router，编号工作树/Gitee 协议仍为 `code-pending`；仓库只有 `diagnostic request` 与 `diagnostic record`，没有可执行的服务器 `environment_probe` runner 或单一 roundtrip 命令。因此每次都需人工拼接路径、ref、工作树、解释器与回传步骤。
- **首次 SHA-256 失败的直接原因**：服务器使用 Windows 默认 CRLF 写入 `environment_probe.json` 后立即计算了 `1369` bytes / `e7652f...`；Git 属性 `* text=auto eol=lf` 将同一文本规范化为 `1316` bytes / `86c836...` 后才写入 Gitee。文件有 53 个换行，字节差也恰为 53，证明每行发生一次 CRLF→LF 转换。修正回传 `486a9af...` 将 JSON 明确写为 UTF-8 无 BOM + LF，并追加规范化的 output hash record。

### 建议固化的最小标准

以下为**待单独批准实现**的改进，不改变本次 G11 的已验收边界。

1. 新增一个受测试的 `environment_probe` server runner：只接受 request 中的 allowlisted `command_id`；固定生成 schema JSON，不接收任意 shell 文本、不读取训练结果、不触发模型或 checkpoint。
2. 所有需回传并被哈希的文本产物必须使用 UTF-8 无 BOM + LF 写入；runner 在写入后、record 前计算 SHA-256，并对输出路径执行一次字节级 LF 断言。禁止在 `Set-Content` 的 Windows 默认换行上计算可回传哈希。
3. 将 Gitee 往返收敛为三张固定操作卡：
   - **server fetch card**：从 `configs/server_paths.yaml` 解析 `server_repo_worktree` / `server_automation_worktrees`，fetch exact ref、核对 SHA、创建 detached worktree；禁止 `pull/reset/clean`。
   - **server diagnostic-return card**：运行 allowlisted runner、校验 canonical output SHA、append-only record、仅 fast-forward push `automation/diagnostics/*`。
   - **local verify card**：fetch 到远端跟踪 ref、不 merge；核对父提交、文件白名单、output 字节/哈希、diagnostic event 与 `run_consumed=0`。
4. 对话交互规则：用户一旦明确授权 Gitee-only server 往返，Agent 的第一轮操作卡必须主动读取服务器路径配置、显示已解析的路径/branch/SHA、列出预期输出和停止条件；不得要求用户额外提示“先自查服务器配置”。
5. 在实现前维持当前安全边界：命令卡仍需逐次由实际 config、remote SHA 和 request 内容填充；不得把本文提案误当作已部署自动化，也不得扩大到训练、result import 或受保护资产操作。
