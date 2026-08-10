# diagnostic-20260810-gitee-rt-pilot-envprobe-r001 操作卡

> 仅适用于 allowlist 中的 `environment_probe` 诊断；不是训练、结果导入或实验结论。
> 固定传输通道：Gitee。输出必须为 UTF-8（无 BOM）且使用 LF 换行。

## 已解析参数

- 治理请求分支：`codex/w006-gitee-roundtrip-pilot-20260810-bound`
- 源码分支：`codex/w006-gitee-roundtrip-pilot-20260810-bound`
- 源码提交：`42b26431262efdfe93766f2e3bd3d54d26999472`
- 回传分支：`automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/return`
- 服务器仓库：`D:\AIPatho\qzs\pfmval_governance`
- 诊断工作树根：`D:\AIPatho\qzs\pfmval_diagnostics`

## 卡 1：本地发布请求

```powershell
git push gitee HEAD:codex/w006-gitee-roundtrip-pilot-20260810-bound
git push gitee codex/w006-gitee-roundtrip-pilot-20260810-bound:codex/w006-gitee-roundtrip-pilot-20260810-bound
```

停止条件：任一 push 失败，或远端源码分支未包含 `42b26431262efdfe93766f2e3bd3d54d26999472`。

## 卡 2：服务器受限执行与回传

```powershell
$repo = 'D:\AIPatho\qzs\pfmval_governance'
$automationRoot = 'D:\AIPatho\qzs\pfmval_diagnostics'
$diagnosticId = 'diagnostic-20260810-gitee-rt-pilot-envprobe-r001'
$governanceBranch = 'codex/w006-gitee-roundtrip-pilot-20260810-bound'
$sourceBranch = 'codex/w006-gitee-roundtrip-pilot-20260810-bound'
$sourceCommit = '42b26431262efdfe93766f2e3bd3d54d26999472'
$returnBranch = 'automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/return'
$requestWorktree = Join-Path $automationRoot ($diagnosticId + '-request')

git -C $repo fetch gitee `
  ('+refs/heads/' + $governanceBranch + ':refs/remotes/gitee/' + $governanceBranch) `
  ('+refs/heads/' + $sourceBranch + ':refs/remotes/gitee/' + $sourceBranch)
if ((git -C $repo rev-parse ('gitee/' + $sourceBranch)) -ne $sourceCommit) { throw 'source SHA mismatch' }
git -C $repo worktree add --detach $requestWorktree ('gitee/' + $governanceBranch)
python (Join-Path $requestWorktree 'deploy/pfmval_ops.py') agent start-check --task diagnostic --host-scope server
python (Join-Path $requestWorktree 'deploy/pfmval_ops.py') diagnostic run-allowlisted --diagnostic-id $diagnosticId
git -C $requestWorktree switch -C $returnBranch
git -C $requestWorktree add -- automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/environment_probe.json
git -C $requestWorktree commit -m ('diagnostic: return ' + $diagnosticId)
git -C $requestWorktree push gitee ('HEAD:' + $returnBranch)
```

停止条件：诊断门禁失败、源码 SHA 不一致、工作树非干净状态、输出字节契约失败，或 push 失败。

## 卡 3：本地取回与验证

```powershell
git fetch gitee +refs/heads/automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/return:refs/remotes/gitee/automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/return
git restore --source gitee/automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/return -- automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/environment_probe.json
python deploy/pfmval_ops.py diagnostic record --diagnostic-id diagnostic-20260810-gitee-rt-pilot-envprobe-r001 --output automation/diagnostics/diagnostic-20260810-gitee-rt-pilot-envprobe-r001/environment_probe.json
python deploy/pfmval_ops.py agent start-check --strict
```

验收条件：`diagnostic record` 返回路径、大小和 Git 闭包校验且严格门禁 `FAIL=0`；不得创建 result envelope，不得消耗 run unit。
