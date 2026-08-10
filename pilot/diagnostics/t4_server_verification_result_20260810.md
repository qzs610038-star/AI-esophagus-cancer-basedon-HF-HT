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

### 4.1 `diagnostic run-allowlisted` 无输出（未获得 environment_probe.json）
- 现象：服务器上运行 `python deploy/pfmval_ops.py diagnostic run-allowlisted --diagnostic-id diagnostic-20260810-gitee-rt-pilot-envprobe-r001` 无输出。
- 疑因（按可能性排序）：
  1. 运行目录不在治理 checkout（PS 提示符显示 C:\Users\AIPatho1）→ `deploy\pfmval_ops.py` 相对路径不存在；
  2. 裸 `python` 未解析到 pfmval_env 解释器（核验已确认绝对路径解释器可用）；
  3. 治理 checkout 工作树 HEAD 未切换到含 request.json 的分支 → `diagnostic request is missing`（该情形应有 traceback 输出，与"无输出"不符）。
- 待服务器重跑确认（命令见 server_operation_card 更新版 §2）。

### 4.2 操作卡占位符导致 ParserError
- 现象：`git add -f <return_dir>/...` 中的 `<` 为 PowerShell 保留符 → ParserError。
- 处置：操作卡已修正（见 server_operation_card_20260810.md 更新版 §4），占位符改为真实路径示例，避免尖括号。

## 5. 对试点判定口径的影响

- 按部署方案 §1.5：`server_diagnostics` 缺失属于**路径核验未完全通过**；该目录带 `live_verification_required: true`，server_maintenance.md 亦注明"诊断根目录需要后续现场创建批准，路径登记本身不代表已执行服务器变更"。
- 初步倾向：**CONDITIONAL GO 候选**（诊断根目录需现场创建批准 + run-allowlisted 输出待重跑确认），最终 verdict 待 T5/T6 完成后再定。
