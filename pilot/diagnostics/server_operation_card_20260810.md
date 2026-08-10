# W006 零训练试点 服务器命令操作卡

> 编号：server_operation_card_20260810 | 适用：服务器侧（D:\AIPatho\qzs）
> 试点：gitee_roundtrip_pilot_w006_v001_20260810 | 性质：零训练、allowlisted 诊断、非证据
> 传输：仅 Gitee（remote: gitee）| 禁止 SSH/SCP/HTTP remote command/Tunnel
> 本卡随故障调试可更新；暂不要求推送，用户本地查看。
> 更新记录：2026-08-10 r3 — 修复"request missing"根因（治理 checkout 此前 detached 于旧 commit 40fde93d，仅 fetch 未 checkout；§1 增加 `git checkout --detach FETCH_HEAD` 与 request.json 自检）；§2 补充输出落点（仓库内 automation/diagnostics/<id>/，不依赖缺失的 server_diagnostics 外部目录）；§4 修正 git add 为仓库相对路径、push 使用完整 refs/heads/ refspec（修复 Invalid path / not a full refname 两处报错）。

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
- terminal receipt（终态 JSON，如 terminal.json）
- started receipt（启动回执，如 started.json）
- resolved argv/config（实际生效参数/配置，如 resolved_config.json）
- ≥1 个原始训练 CSV（被 .gitignore 命中则精确 force-add）
- ≥1 个原始训练 TXT（同上）
- remote tree 闭包校验结果（git ls-files / index 清单与 return manifest 对账）

> 在治理 checkout 内（已 checkout FETCH_HEAD 的工作树）、对仓库相对目录 add/commit/push。注意：git add 必须用**仓库相对路径**（绝对路径报 Invalid path）；push 必须用完整 refspec `HEAD:refs/heads/<branch>`（否则报 not a full refname）。若 CSV/TXT 被 .gitignore 命中，`git add -f` 强制加入。

```powershell
git add automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001
git commit -m "diagnostic: return W006 env probe + minimal closure"
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
