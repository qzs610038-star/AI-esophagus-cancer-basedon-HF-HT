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
| T2 工作树绑定 | `completed_pending_user_review` | b452e00、7faea06 | pilot/critical_contract.json、experiment_registry.json、workspace_registry.json |
| T3 Gitee 诊断请求 | `completed_pending_user_review`（含缺失登记） | 418196d、217d74e | automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-pathprobe-r001/、...envprobe-r001/ |
| T4 服务器 allowlisted diagnostic | `completed_pending_user_review` | 0ca8135（服务器 return 首 commit） | environment_probe.json（926B，source_commit 匹配冻结值） |
| T5 最小 return 闭包 | `completed_pending_user_review` | 0cf0f09（服务器 return 追加 commit） | started/terminal/resolved_config/CSV/TXT/remote_tree_closure 全量回传 |
| T6 本地回收核验 | `completed_pending_user_review` | 161ee6b、24d1eb4 | 七项核验全过（见 §6 补充说明） |
| T7 失败分流 | `completed_pending_user_review` | 见 000c55a（r3 修正） | 无试点失败项；路径/传输/闭包报错均已定位修复 |
| T8 证据回填 | `completed_pending_user_review` | 本次追加 | 本文件 §6 T8 回填块 |

## T8 证据回填（2026-08-10，final_verdict 供用户审核）

- `source_commit`: `42b26431262efdfe93766f2e3bd3d54d26999472`（冻结值；environment_probe.json、started.json、terminal.json 三处均一致）
- `workspace_id`: W006
- `diagnostic_id`: `diagnostic-20260810-gitee-rt-pilot-envprobe-r001`（自动通道）；`diagnostic-20260810-gitee-rt-pilot-pathprobe-r001`（已登记无 runner，未执行）
- `return_revision`: R001（return 分支 `automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/return`，HEAD=0cf0f09，基于 3495d94 fast-forward）
- `checked_paths`: 7/7 — server_governance_checkout(git_worktree ✅)、server_experiment_worktrees ✅、server_experiment_runs ✅、server_result_returns ✅、server_diagnostics ✅（初检缺失，runner 自动创建后存在）、server_runtime_bundles ✅、python 解释器 ✅；retired pointer server_config/server_governance_config 确认存在且不作事实源
- `python_interpreter_check`: `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe` = Python 3.13.5，torch 2.6.0+cu124，CUDA 12.4，RTX 4080，exit_code=0
- `force_added_artifacts`: 无（check-ignore 确认诊断目录 CSV/TXT/JSON 均不被 .gitignore 命中，普通 add 已入 remote tree；服务器侧 CRLF→LF 警告为行尾规范化提示，非错误）
- `remote_tree_closure`: return 分支含 9 文件（request.json、operation_cards.md、environment_probe.json、started.json、terminal.json、resolved_config.json、raw_training_sample.csv、raw_training_sample.txt、remote_tree_closure.txt），与回传 manifest 对账一致
- `new_warns`: 无试点新增 WARN（W006 内 start-check 的 7 个 WARN 均为基线既有；3 个 FAIL 均为工作树环境差异，处置 A 已获用户批准）
- `new_fails`: 0（试点动作未引入任何 FAIL；主仓同 HEAD 基线 PASS=7 WARN=7 FAIL=0）
- `final_verdict`: **CONDITIONAL GO** — 理由：①路径 7/7 + 解释器核验通过；②allowlisted environment_probe 成功返回且 source_commit 匹配；③最小 return 闭包完整（receipt 成对、CSV/TXT、resolved config、remote tree 闭包对账全过）；④无新增 FAIL。保留条件（需用户知情）：a) `path_probe` 固定 runner 缺失（已登记，待用户决定是否另开治理任务补齐）；b) `server_diagnostics` 目录由 runner 自动创建，未走独立"现场创建批准"流程（server_maintenance.md 要求），建议用户知悉并认可该创建行为；c) 操作卡 r2-r4 涉及的现场排障（运行目录/checkout/refspec）为既有工具使用要点，不构成试点失败。
- `notes`: ①全程零训练：未执行 smoke/single-batch forward 之外任何训练性动作；②未导入 Registry 结果、未 accept 证据、未更新 accepted 指标（experiment 状态仍为 planned/preflight）；③传输全程仅 Gitee（remote: gitee），无 SSH/SCP/HTTP/Tunnel；④W006 分支本地 HEAD 已推进至最终 pilot 提交（0d11930），workspace_registry `current_source_commit` 已同步刷新（该字段语义为工作树 HEAD 快照）；⑤`diagnostic record` 已登记 environment_probe.json（926B，UTF-8/LF 校验通过）；⑥W005 未复用。
- **重要发现（用户要求登记，供 main 工作分支回收后统一治理）**：`validate_state` 对 active 工作树强制"git HEAD == registry.current_source_commit 严格相等"，与"登记后持续提交"的 W006 流程不兼容（SHA 自引用环，不可自行闭合）；W001/W004/W005 因登记后未再提交故未触发。已按用户裁决（方案 A）接受现状、不改治理代码，并登记为治理改进项，详见 `project_state/governance/W006_pilot_finding_active_workspace_head_check_20260810.md`（含影响面、建议治理方向与试点内处置）。

### 进度补充说明（2026-08-10，已获用户审核/决策）

- T2 已登记实验 `gitee_roundtrip_pilot_w006_v001_20260810`（preflight，run_limit=1），critical contract canonical SHA-256 `6bb3f473d67f2bb727394262fabbed15ba3d4bcca1f25f4bc2efa5a96bcdaa2f`；W006 已登记（next_workspace_number=7）。T2 协议批准 commit `7faea06`（approval-w006-gitee-rt-pilot-protocol-v1-20260810）。
- **缺失登记（用户要求，供后续考虑是否修复）**：`path_probe`（以及 `cache_probe`、`dry_run`、`single_batch_forward`）虽在 diagnostic command allowlist 中，但 `scripts/pfmval_state.py` 的 `DIAGNOSTIC_RUNNER_COMMANDS = {"environment_probe"}` 显示服务器侧固定 runner 目前仅实现 `environment_probe` 一个收集器。`path_probe` 请求已生成并推送（diagnostic-20260810-gitee-rt-pilot-pathprobe-r001），但服务器自动通道无法执行该 command_id；本次试点已按用户决策改用 `environment_probe` 完成可自动执行的部分，T4 的路径存在性/类型/边界核验转由用户服务器侧现场确认后按清单回传。是否补齐 `path_probe` 固定收集器由用户后续另行决策，不纳入本试点交付。
- 传输已推送：source 分支 `codex/w006-gitee-roundtrip-pilot-20260810-bound` 已推送 Gitee（remote: gitee），含 T2/T3/协议批准共 3 个 commit（b452e00、418196d、7faea06）。
- **服务器命令操作卡（2026-08-10 追加，用户要求）**：`pilot/diagnostics/server_operation_card_20260810.md` 为服务器侧执行总卡，包含身份绑定（source_commit/分支/诊断 id）、Gitee fetch、environment_probe 固定 runner 执行、7 路径+解释器+retired pointer 现场只读核验、最小 return 闭包构造与回传命令及边界约束。本卡当前**仅本地保存供用户阅读，暂不推送**；后续随故障调试需要再更新或做必要的最小推送。
- **T4 服务器侧核验结果登记（2026-08-10 r2，用户现场核验回传）**：详见 `pilot/diagnostics/t4_server_verification_result_20260810.md`。fetch 成功（FETCH_HEAD=3495d94）；路径核验 **6/7 通过，`server_diagnostics`（D:\AIPatho\qzs\pfmval_diagnostics）缺失**（该 path id 带 `live_verification_required: true`，server_maintenance.md 要求现场创建批准，路径登记本身不代表服务器变更）；解释器 `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe` 可用（Python 3.13.5）；`server_config`/`server_governance_config` 确认为 retired pointer 且文件存在（仅兼容核对，不作事实源）。按部署方案 §1.5 口径，`server_diagnostics` 缺失对应"诊断根目录需要后续现场创建批准"情形，初步倾向 **CONDITIONAL GO 候选**，最终 verdict 待 T5/T6 完成后确定。
- **T4 排障进展登记（2026-08-10 r3，用户二次回传）**：runner 首轮"无输出"根因为运行目录不在治理 checkout；改用 Set-Location + 绝对路径解释器后报 `diagnostic request is missing` —— 根因系治理 checkout 仅 fetch 未 checkout（HEAD detached 于旧 commit 40fde93d），工作树无 `automation/diagnostics/<id>/request.json`。操作卡 r3 已增加 `git checkout --detach FETCH_HEAD` + request.json 自检。另修正 T5 回传命令两处报错：`git add` 必须用仓库相对路径（绝对路径报 Invalid path）、push 须完整 refspec `HEAD:refs/heads/automation/diagnostics/<id>/return`（否则 not a full refname）；并确认 environment_probe 输出落在治理仓库内（不依赖缺失的 server_diagnostics 外部目录）。待服务器按操作卡 r3 重跑确认。

> 门禁：本文件须经用户明确二次批准后，方可绑定/创建 W006 工作树并开始 T2-T8。（用户已于 2026-08-10 明确批准本方案并绑定 W006；本行保留为历史门禁说明）

### Closeout update（2026-08-11，用户已批准）

- 用户明确批准关闭 W006。`workspace close --preview` 对 `D:\AI空间转录病理研究\PFMval_new_governed_workspaces\W006` 返回 `close_ready`（无 blocker、无 attempt、无保留资产）；Registry 已将 W006 标记为 `close_ready`、试点 experiment 标记为 `closed`，保留分支与 Git 历史，只读且不可再派发 job。
- 本次不移除物理 worktree、分支或资产。未来若需物理移除，仍须针对该精确绝对路径另行获得最终批准。
