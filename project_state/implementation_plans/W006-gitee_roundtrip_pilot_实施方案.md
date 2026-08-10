# W006-gitee_roundtrip_pilot 实施方案

> 状态：`pending_user_review`（起草待用户二次批准）
> 创建日期：2026-08-10

## 1. 来源与目标

| 项 | 值 |
|---|---|
| 来源部署方案 | `01_指南与解读/部署方案/服务器零训练Gitee往返试点检查方案_20260810.md` |
| 来源方案 revision | 2026-08-10 版（用户已于 2026-08-10 批准，方案状态转为 approved） |
| 目标 experiment | 拟登记 `gitee_roundtrip_pilot_w006_v001_20260810`（零训练试点，非正式实验） |
| 拟绑定工作树 | **W006**（永久编号，不得复用；W005 为 `abandoned_reserved`） |
| 工作树短名 | `gitee_roundtrip_pilot` |
| 试点性质 | 零训练（no training）：不训练、不导入结果、不接纳证据、不更新 accepted 指标 |
| 传输通道 | Gitee-only（remote: `gitee`）；禁止 SSH/SCP/HTTP remote command/Tunnel |

## 2. 修改前配置清单（baseline snapshot）

执行任何改动前逐项记录并核对：

- [ ] 当前 `git HEAD`（main 分支）与工作区 dirty 状态快照
- [ ] `configs/server_paths.yaml` 全文快照（唯一机器配置事实源）
- [ ] `project_state/workspace_registry.json` 的 `next_workspace_number`（当前应为 6）与 W001-W005 记录
- [ ] `experiments/experiment_registry.json` 现有 experiments 列表快照
- [ ] `project_state/schemas/return_profile_v1.schema.json` 存在且受管
- [ ] `project_state/plans/server_maintenance.md`、`AGENTS.md` 复核
- [ ] start-check 基线输出（PASS=7 WARN=7 FAIL=0，2026-08-10）

## 3. 分步任务

### T1 本地基线提交与 source_commit 冻结
1. 将当前非忽略工作区完整提交到本地 Git（基线提交，不推送任何 remote）。
2. 冻结 `source_commit` = 该基线提交 SHA。
3. 记录 `workspace_id=W006`、拟用 `diagnostic_id`、`return_revision`。

### T2 工作树绑定（仅在本方案获批后执行）
1. `python deploy/pfmval_ops.py workspace init` 创建/登记 W006 本地工作树。
2. `python deploy/pfmval_ops.py experiment register --experiment-id gitee_roundtrip_pilot_w006_v001_20260810 --display-name "零训练Gitee往返试点W006" --phase preflight --run-limit 1 --critical-contract <contract.json> --workspace-path <W006路径>`
3. 建立本次改动代码目录（位于 W006 内）与修改前配置备份。

### T3 Gitee 诊断请求（allowlisted）
1. 构造 diagnostic request：绑定精确 `source_commit`、固定 source/return branch、allowlisted command id。
2. 诊断分支位于 `automation/diagnostics/*`。
3. 请求中不得含任意 shell 文本、训练参数、result import 指令。
4. 经 Gitee 推送，等待服务器 allowlisted diagnostic 执行。

### T4 服务器 allowlisted diagnostic 核验
服务器侧至少核验以下对象存在性/类型/边界：
- `server_governance_checkout`（`D:\AIPatho\qzs\pfmval_governance`，git_worktree）
- `server_experiment_worktrees`（`D:\AIPatho\qzs\pfmval_automation`，directory）
- `server_experiment_runs`（`D:\AIPatho\qzs\pfmval_experiment_runs`，directory）
- `server_result_returns`（`D:\AIPatho\qzs\pfmval_result_returns`，directory）
- `server_diagnostics`（`D:\AIPatho\qzs\pfmval_diagnostics`，directory）
- `server_runtime_bundles`（`D:\AIPatho\qzs\pfmval_runtime_bundles`，directory）
- 解释器 `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe`
- 确认 retired pointer（server_config / server_governance_config）未被当作当前事实源

### T5 最小 return 闭包
构造并回传（仍经 Gitee）：
- terminal receipt + started receipt（成对）
- resolved argv/config
- 至少 1 个原始训练 CSV（被 `.gitignore` 命中则精确 force-add）
- 至少 1 个原始训练 TXT（同上）
- remote tree 闭包校验结果（与 manifest 对账）
- 用 `return_profile_v1` 约束核对 success/failed/incomplete 终态矩阵
- 不生成训练结果结论，不导入 Registry

### T6 本地回收核验
逐项核对方案 §2.5 的 7 项：
1. `source_commit` 与冻结值一致
2. `workspace_id` 为 W006 或更高
3. terminal/started receipt 成对
4. CSV/TXT 进入 staged/remote tree
5. remote tree 闭包与 manifest 对账一致
6. 无新增 FAIL
7. 未误触训练/结果导入/evidence accept

### T7 失败分流（若失败）
按方案 §2.6 记录：路径失败 / 解释器失败 / 传输失败 / return 闭包失败 / 越界失败；只形成补证或修订建议，不转入训练。

### T8 证据回填
按方案 §3 模板回填：`source_commit`、`workspace_id`、`diagnostic_id`、`return_revision`、`checked_paths`、`python_interpreter_check`、`force_added_artifacts`、`remote_tree_closure`、`new_warns`、`new_fails`、`final_verdict`（GO | CONDITIONAL GO | NO-GO）、`notes`。

## 4. 验证命令

```powershell
python deploy/pfmval_ops.py agent start-check --strict
python deploy/pfmval_ops.py workspace status
python deploy/pfmval_ops.py workspace scan
python deploy/pfmval_ops.py paths validate
```

## 5. 证据路径

- 本实施方案：`project_state/implementation_plans/W006-gitee_roundtrip_pilot_实施方案.md`
- 诊断/return 请求与回执：W006 工作树内 `diagnostics/` 与 `returns/` 目录（待 T3-T5 建立）
- 服务器核验输出：W006 工作树内 `diagnostics/<diagnostic_id>/`
- 终态记录与证据回填：本文件 T8 追加

## 6. 进度状态

| 任务 | 状态 | commit | 证据 |
|---|---|---|---|
| T1 本地基线提交 + source_commit 冻结 | `completed_pending_user_review` | 42b2643（用户批准实施方案的提交即基线） | 本文件 §2；branch/HEAD/clean 核验通过 |
| T2 工作树绑定 | `completed_pending_user_review` | b452e00 | pilot/critical_contract.json、experiment_registry.json、workspace_registry.json |
| T3 Gitee 诊断请求 | `completed_pending_user_review`（含缺失登记，见下方补充说明） | 418196d | automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-pathprobe-r001/ |
| T4 服务器 allowlisted diagnostic | in_progress | - | 改用 environment_probe（见下方补充说明） |
| T5 最小 return 闭包 | pending | - | - |
| T6 本地回收核验 | pending | - | - |
| T7 失败分流 | pending | - | - |
| T8 证据回填 | pending | - | - |

### 进度补充说明（2026-08-10，已获用户审核/决策）

- T2 已登记实验 `gitee_roundtrip_pilot_w006_v001_20260810`（preflight，run_limit=1），critical contract canonical SHA-256 `6bb3f473d67f2bb727394262fabbed15ba3d4bcca1f25f4bc2efa5a96bcdaa2f`；W006 已登记（next_workspace_number=7）。T2 协议批准 commit `7faea06`（approval-w006-gitee-rt-pilot-protocol-v1-20260810）。
- **缺失登记（用户要求，供后续考虑是否修复）**：`path_probe`（以及 `cache_probe`、`dry_run`、`single_batch_forward`）虽在 diagnostic command allowlist 中，但 `scripts/pfmval_state.py` 的 `DIAGNOSTIC_RUNNER_COMMANDS = {"environment_probe"}` 显示服务器侧固定 runner 目前仅实现 `environment_probe` 一个收集器。`path_probe` 请求已生成并推送（diagnostic-20260810-gitee-rt-pilot-pathprobe-r001），但服务器自动通道无法执行该 command_id；本次试点已按用户决策改用 `environment_probe` 完成可自动执行的部分，T4 的路径存在性/类型/边界核验转由用户服务器侧现场确认后按清单回传。是否补齐 `path_probe` 固定收集器由用户后续另行决策，不纳入本试点交付。
- 传输已推送：source 分支 `codex/w006-gitee-roundtrip-pilot-20260810-bound` 已推送 Gitee（remote: gitee），含 T2/T3/协议批准共 3 个 commit（b452e00、418196d、7faea06）。
- **服务器命令操作卡（2026-08-10 追加，用户要求）**：`pilot/diagnostics/server_operation_card_20260810.md` 为服务器侧执行总卡，包含身份绑定（source_commit/分支/诊断 id）、Gitee fetch、environment_probe 固定 runner 执行、7 路径+解释器+retired pointer 现场只读核验、最小 return 闭包构造与回传命令及边界约束。本卡当前**仅本地保存供用户阅读，暂不推送**；后续随故障调试需要再更新或做必要的最小推送。

> 门禁：本文件须经用户明确二次批准后，方可绑定/创建 W006 工作树并开始 T2-T8。（用户已于 2026-08-10 明确批准本方案并绑定 W006；本行保留为历史门禁说明）
