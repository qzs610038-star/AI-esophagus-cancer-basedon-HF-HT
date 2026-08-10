# W006 T4 服务器侧路径核验清单

> 用途：服务器现场只读核验（allowlisted、非证据性），结果回传后用于本地对账。
> 生成：2026-08-10 | 来源：configs/server_paths.yaml（唯一机器配置事实源）
> 方式：由用户在服务器侧逐项现场确认后填写"核验结果"列并回传，不执行任何修改。

## 核验对象（7 个路径对象 + 解释器）

| # | path_id | 期望路径 | 期望类型 | 期望状态 | 边界角色 | 核验结果（存在性/类型/边界） |
|---|---|---|---|---|---|---|
| 1 | server_governance_checkout | D:\AIPatho\qzs\pfmval_governance | git_worktree | active | governance_dispatch_checkout | 待填 |
| 2 | server_experiment_worktrees | D:\AIPatho\qzs\pfmval_automation | directory | active | persistent_numbered_workspace_root | 待填 |
| 3 | server_experiment_runs | D:\AIPatho\qzs\pfmval_experiment_runs | directory | active | attempt_output_root | 待填 |
| 4 | server_result_returns | D:\AIPatho\qzs\pfmval_result_returns | directory | active | result_bundle_staging_root | 待填 |
| 5 | server_diagnostics | D:\AIPatho\qzs\pfmval_diagnostics | directory | active | non_evidence_diagnostic_root（live_verification_required） | 待填 |
| 6 | server_runtime_bundles | D:\AIPatho\qzs\pfmval_runtime_bundles | directory | active | immutable_runtime_bundle_root | 待填 |
| 7 | python 解释器 | C:\Users\AIPatho1\pfmval_env\Scripts\python.exe | file(可执行) | active | runtime python_interpreter | 待填 |

## retired pointer 确认（不得作为当前事实源）

| # | path_id | 路径 | status | role | 确认结果 |
|---|---|---|---|---|---|
| 8 | server_config | D:\AIPatho\qzs\pfmval_deploy_git\configs\config.server.yaml | deprecated | retired_server_config_pointer | 待确认：仅作兼容核对/fail-closed 历史指针 |
| 9 | server_governance_config | D:\AIPatho\qzs\pfmval_governance\configs\config.server.yaml | deprecated | retired_server_config_pointer | 待确认：仅作兼容核对/fail-closed 历史指针 |

## 服务器建议核验命令（只读）

```powershell
# 1-6：目录/工作树存在性与类型
Test-Path 'D:\AIPatho\qzs\pfmval_governance'; (Get-Item 'D:\AIPatho\qzs\pfmval_governance').Attributes
Test-Path 'D:\AIPatho\qzs\pfmval_automation'; Test-Path 'D:\AIPatho\qzs\pfmval_experiment_runs'
Test-Path 'D:\AIPatho\qzs\pfmval_result_returns'; Test-Path 'D:\AIPatho\qzs\pfmval_diagnostics'
Test-Path 'D:\AIPatho\qzs\pfmval_runtime_bundles'
# 7：解释器
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' -c "import sys; print(sys.version)"
# 8-9：retired pointer 只读确认（存在即可，不读取内容为事实源）
Test-Path 'D:\AIPatho\qzs\pfmval_deploy_git\configs\config.server.yaml'
Test-Path 'D:\AIPatho\qzs\pfmval_governance\configs\config.server.yaml'
# 附加：environment_probe 固定 runner 执行（仅 environment_probe 有 runner）
python deploy/pfmval_ops.py diagnostic run-allowlisted --diagnostic-id diagnostic-20260810-gitee-rt-pilot-envprobe-r001
```

## 回传方式

核验结果请按本清单格式回填，作为诊断回传包一部分经 Gitee `automation/diagnostics/*` 回传；本地将用 return_profile_v1 与 Git tree 闭包核对终态矩阵。
