# W006 T4 服务器侧核验结果登记（2026-08-10）

> 核验方式：用户服务器现场只读核验（按 t4_server_path_checklist_20260810.md）
> 传输：Gitee-only；本登记为非证据诊断记录，不构成 accepted 证据。
> 来源：configs/server_paths.yaml（唯一机器配置事实源）

## 1. 服务器 fetch 状态

- 服务器治理 checkout：`D:\AIPatho\qzs\pfmval_governance`
- `git fetch gitee codex/w006-gitee-roundtrip-pilot-20260810-bound`：成功
- FETCH_HEAD：`3495d94a61ce7c79d2bcb14f806948f33fd587a7`
- 说明：分支 HEAD 已推进至 3495d94（本地最新 5891389 为操作卡本地 commit，按用户要求未推送）；诊断请求 `diagnostic-20260810-gitee-rt-pilot-envprobe-r001` 与 `pathprobe-r001` 均位于 3495d94 之前，服务器对象库已含全部所需 commit。

## 2. 路径对象核验结果

| # | path_id | 期望路径 | 期望类型 | 核验结果 | 结论 |
|---|---|---|---|---|---|
| 1 | server_governance_checkout | D:\AIPatho\qzs\pfmval_governance | git_worktree | True / Directory | ✅ 存在且类型正确 |
| 2 | server_experiment_worktrees | D:\AIPatho\qzs\pfmval_automation | directory | True | ✅ 存在 |
| 3 | server_experiment_runs | D:\AIPatho\qzs\pfmval_experiment_runs | directory | True | ✅ 存在 |
| 4 | server_result_returns | D:\AIPatho\qzs\pfmval_result_returns | directory | True | ✅ 存在 |
| 5 | server_diagnostics | D:\AIPatho\qzs\pfmval_diagnostics | directory | **False** | ❌ **缺失（需现场创建批准）** |
| 6 | server_runtime_bundles | D:\AIPatho\qzs\pfmval_runtime_bundles | directory | True | ✅ 存在 |
| 7 | python 解释器 | C:\Users\AIPatho1\pfmval_env\Scripts\python.exe | file(可执行) | 3.13.5 Anaconda | ✅ 可执行，版本 Python 3.13.5 |

## 3. retired pointer 确认

| # | path_id | 路径 | 存在性 | 确认结论 |
|---|---|---|---|---|
| 8 | server_config | D:\AIPatho\qzs\pfmval_deploy_git\configs\config.server.yaml | True | 仅 retired pointer，不作为机器配置事实源 |
| 9 | server_governance_config | D:\AIPatho\qzs\pfmval_governance\configs\config.server.yaml | True | 仅 retired pointer，不作为机器配置事实源 |

## 4. 待排障事项

### 4.1 `diagnostic run-allowlisted` 无输出 / request missing（已定位根因）
- 现象 1：裸 `python deploy/pfmval_ops.py ...` 在 `C:\Users\AIPatho1>` 运行无输出 → 运行目录不在治理 checkout、相对路径脚本不存在。
- 现象 2：Set-Location + 绝对路径解释器重跑后报 `[FAIL] diagnostic request is missing: D:\AIPatho\qzs\pfmval_governance\automation\diagnostics\diagnostic-20260810-gitee-rt-pilot-envprobe-r001\request.json`。
- 根因：治理 checkout 仅 `git fetch`，未 `checkout` 到 FETCH_HEAD；其 HEAD detached 于旧 commit `40fde93d`，工作树内不存在 `automation/diagnostics/<id>/`，runner 找不到 request.json。
- 修复（操作卡 r3）：§1 增加 `Set-Location` + `git checkout --detach FETCH_HEAD` + `Test-Path .\automation\diagnostics\<id>\request.json` 自检；§2 用绝对路径解释器运行。待服务器重跑确认获得 environment_probe.json。

### 4.2 操作卡 §4 回传命令两处报错（已修正）
- `git add -f "D:\AIPatho\qzs\pfmval_diagnostics\..."` → `fatal: Invalid path 'D:/AIPatho/qzs/pfmval_diagnostics'`：git add 不支持绝对路径，且 server_diagnostics 目录本身缺失；改为治理仓库内相对路径 `automation/diagnostics/<id>`。
- `git push gitee HEAD:automation/diagnostics/...` → `not a full refname`：refspec 需完整 `HEAD:refs/heads/automation/diagnostics/<id>/return`。
- 说明：environment_probe runner 输出实际落在治理仓库内 `automation/diagnostics/<id>/environment_probe.json`，不依赖缺失的外部 server_diagnostics 目录；T5 最小 return 闭包同样在仓库内相对目录构造。

## 5. 对试点判定口径的影响

- 按部署方案 §1.5：`server_diagnostics` 缺失属于**路径核验未完全通过**；该目录带 `live_verification_required: true`，server_maintenance.md 亦注明"诊断根目录需要后续现场创建批准，路径登记本身不代表已执行服务器变更"。
- 初步倾向：**CONDITIONAL GO 候选**（诊断根目录需现场创建批准 + run-allowlisted 输出待重跑确认），最终 verdict 待 T5/T6 完成后再定。
