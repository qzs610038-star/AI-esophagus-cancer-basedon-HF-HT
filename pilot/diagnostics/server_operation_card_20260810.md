# W006 零训练试点 服务器命令操作卡

> 编号：server_operation_card_20260810 | 适用：服务器侧（D:\AIPatho\qzs）
> 试点：gitee_roundtrip_pilot_w006_v001_20260810 | 性质：零训练、allowlisted 诊断、非证据
> 传输：仅 Gitee（remote: gitee）| 禁止 SSH/SCP/HTTP remote command/Tunnel
> 本卡随故障调试可更新；暂不要求推送，用户本地查看。
> 更新记录：2026-08-10 r4 — §4 扩展为完整 T5 最小闭包构造命令（started/terminal receipt、resolved_config、占位 CSV/TXT、remote_tree_closure），已确认诊断目录文件不被 .gitignore 命中；return 分支已有 environment_probe.json（0ca8135），本步追加 commit 后 fast-forward push。

## 0. 身份绑定（每次执行前核对）

| 项 | 值 |
|---|---|
| 工作树 | W006（永久编号，不复用；W005 为 abandoned_reserved） |
| source_commit | `42b26431262efdfe93766f2e3bd3d54d26999472` |
| source 分支 | `codex/w006-gitee-roundtrip-pilot-20260810-bound` |
| 诊断请求 | `diagnostic-20260810-gitee-rt-pilot-envprobe-r001`（command_id=`environment_probe`） |
| 路径核验清单 | `pilot/diagnostics/t4_server_path_checklist_20260810.md` |
| 机器配置事实源 | 治理仓库 `configs/server_paths.yaml`（server_config / server_governance_config 仅为 retired pointer） |

## 1. 拉取请求（Gitee-only）

> 必须 fetch **并 checkout 到 FETCH_HEAD**。根因：治理 checkout 此前 detached 于旧 commit（40fde93d），只 fetch 未 checkout → 工作树无请求文件 → runner 报 `diagnostic request is missing`。执行前确认治理 checkout 工作树干净（dirty 时先回执，勿强切）。source_commit=42b2643 是契约冻结值（request.json 内部绑定），分支 HEAD 领先于它属正常。

```powershell
git -C D:\AIPatho\qzs\pfmval_governance fetch gitee codex/w006-gitee-roundtrip-pilot-20260810-bound
Set-Location D:\AIPatho\qzs\pfmval_governance
git checkout --detach FETCH_HEAD
Test-Path .\automation\diagnostics\diagnostic-20260810-gitee-rt-pilot-envprobe-r001\request.json
```

## 2. 执行固定 runner（仅 environment_probe 有固定收集器）

> 用绝对路径解释器运行（避免裸 python 解析到错误环境）。输出落在仓库内 `automation/diagnostics/<id>/environment_probe.json`，与缺失的外部 server_diagnostics 目录无关。

```powershell
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' D:\AIPatho\qzs\pfmval_governance\deploy\pfmval_ops.py diagnostic run-allowlisted --diagnostic-id diagnostic-20260810-gitee-rt-pilot-envprobe-r001
```

- 预期产物：`automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/environment_probe.json`
- 无输出排查顺序：① `Get-Location` 是否为治理 checkout；② `Test-Path .\deploy\pfmval_ops.py`；③ 换绝对路径解释器（上命令已给出）；④ 若仍无输出，把完整终端粘贴回传分析。
- 说明：`path_probe`（及 cache_probe/dry_run/single_batch_forward）当前**无固定 runner**（`DIAGNOSTIC_RUNNER_COMMANDS={"environment_probe"}`）；本次试点已登记该缺失，不手写命令替代。
- 若该命令在治理 checkout 主工作树不可直接运行，请检出 W006 分支的 detached worktree 后运行；不得在源分支上直接修改。

## 3. 路径现场只读核验（T4）——已完成，见 t4_server_verification_result_20260810.md

已核验：6/7 存在（server_diagnostics 缺失需现场创建批准）、解释器 3.13.5 可用、retired pointer 确认。复查只需跑缺失项：

```powershell
Test-Path 'D:\AIPatho\qzs\pfmval_diagnostics'
```

## 4. 最小 return 闭包（T5）

服务器侧需构造并回传（零训练试点最小集合，全部位于治理仓库内 `automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/`）：
- started receipt：`started.json`
- terminal receipt：`terminal.json`（终态矩阵按 return_profile_v1：零训练无预测 → 状态 `failed` = terminal_json_plus_raw_csv_txt 最贴合；notes 注明零训练预期无预测）
- resolved argv/config：`resolved_config.json`
- ≥1 个原始训练 CSV：`raw_training_sample.csv`（零训练试点占位样例，非真实训练产物）
- ≥1 个原始训练 TXT：`raw_training_sample.txt`（同上）
- remote tree 闭包校验：`remote_tree_closure.txt`（git ls-files 清单 + 与返回 manifest 对账）

> 注意：已确认 `automation/diagnostics/<id>/` 下 CSV/TXT/JSON 均不被 .gitignore 命中（本地 check-ignore 验证 exit=1），普通 `git add` 即可；如服务器侧 .gitignore 有差异导致忽略，则用 `git add -f`。return 分支已存在（0ca8135），本步为追加新 commit 后 fast-forward push。git add 必须用仓库相对路径；push 必须用完整 refspec。

```powershell
$D = "D:\AIPatho\qzs\pfmval_governance\automation\diagnostics\diagnostic-20260810-gitee-rt-pilot-envprobe-r001"
$S = "42b26431262efdfe93766f2e3bd3d54d26999472"
$T = Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ'
@(
  @{file='started.json'; content=(@{schema_version='1.0'; receipt_type='started'; diagnostic_id='diagnostic-20260810-gitee-rt-pilot-envprobe-r001'; source_commit=$S; source_branch='codex/w006-gitee-roundtrip-pilot-20260810-bound'; command_id='environment_probe'; started_at=$T} | ConvertTo-Json -Depth 4)},
  @{file='terminal.json'; content=(@{schema_version='1.0'; receipt_type='terminal'; diagnostic_id='diagnostic-20260810-gitee-rt-pilot-envprobe-r001'; source_commit=$S; status='failed'; terminal_matrix='terminal_json_plus_raw_csv_txt'; note='zero-training pilot: no predictions by design'; completed_at=$T} | ConvertTo-Json -Depth 4)},
  @{file='resolved_config.json'; content=(@{schema_version='1.0'; diagnostic_id='diagnostic-20260810-gitee-rt-pilot-envprobe-r001'; resolved_argv=@('--diagnostic-id','diagnostic-20260810-gitee-rt-pilot-envprobe-r001'); config_source='configs/server_paths.yaml'; python_interpreter='C:\Users\AIPatho1\pfmval_env\Scripts\python.exe'} | ConvertTo-Json -Depth 4)},
  @{file='raw_training_sample.csv'; content=('sample_id,pathway,value,note' + "`n" + 'ZT-0001,PLACEHOLDER,0.0,zero-training-pilot-sample')},
  @{file='raw_training_sample.txt'; content=('zero-training pilot raw TXT placeholder' + "`n" + 'diagnostic-20260810-gitee-rt-pilot-envprobe-r001')}
) | ForEach-Object { Set-Content -LiteralPath (Join-Path $D $_.file) -Value $_.content -Encoding UTF8 }
git -C D:\AIPatho\qzs\pfmval_governance ls-files "automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001" > (Join-Path $D 'remote_tree_closure.txt')
Set-Location D:\AIPatho\qzs\pfmval_governance
git add automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001
git commit -m "diagnostic: return W006 env probe + minimal closure (receipts, csv, txt)"
git push gitee HEAD:refs/heads/automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/return
```

约束：
- 回传分支固定在 `automation/diagnostics/*` 下；不与 smoke/formal 结果流混用。
- 不得 `git clean -fd`；不得删除 checkpoints/MPP 数据/缓存。
- 不生成训练结论、不导入 Registry、不更新 accepted 指标。

## 5. 回传后本地对账（本地侧执行，供参考）

- `git fetch gitee` return 分支 → 核对 environment_probe.json 的 source_commit
- 用 `project_state/schemas/return_profile_v1.schema.json` 核对 success/failed/incomplete 终态矩阵
- CSV/TXT 是否进入 staged/remote tree；remote tree 闭包与 manifest 对账

## 边界与禁止

- 本卡所有命令均允许只读/allowlisted 诊断；禁止任何训练性执行（含 smoke/single-batch forward 之外的一切）。
- 禁止 result import、result accept、accepted metrics 更新。
- 禁止非 Gitee 传输通道；推送前如涉及新命令形态，先向用户回执。
