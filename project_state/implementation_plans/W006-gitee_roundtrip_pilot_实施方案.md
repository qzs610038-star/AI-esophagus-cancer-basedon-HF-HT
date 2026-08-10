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
| T1 本地基线提交 + source_commit 冻结 | `pending_user_review` | - | - |
| T2 工作树绑定 | `pending_user_review` | - | - |
| T3 Gitee 诊断请求 | pending | - | - |
| T4 服务器 allowlisted diagnostic | pending | - | - |
| T5 最小 return 闭包 | pending | - | - |
| T6 本地回收核验 | pending | - | - |
| T7 失败分流 | pending | - | - |
| T8 证据回填 | pending | - | - |

> 门禁：本文件须经用户明确二次批准后，方可绑定/创建 W006 工作树并开始 T2-T8。
