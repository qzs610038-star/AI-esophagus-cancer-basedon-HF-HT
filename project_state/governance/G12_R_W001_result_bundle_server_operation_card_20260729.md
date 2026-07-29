# G12-R 最短服务器操作卡：W001 仅封装修订回传

> 作用：从不可变 A003/R003、A004/R001 构建 A003/R004、A004/R002，并仅经 Gitee fast-forward 回传。  
> 禁止：训练、重试、创建新 attempt、`record-import`、result import/accept、修改指标或科学结论。  
> 实现提交：`9840efe48003b66e3195eb0cafab4ec0752a3207`

在服务器 PowerShell 7 中一次性执行：

```powershell
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = 'D:\AIPatho\qzs\pfmval_deploy_git'
$RootBase = 'D:\AIPatho\qzs\pfmval_automation\g12r_result_bundle_20260729'
$RunId = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Root = Join-Path $RootBase $RunId
$Python = 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe'
$ImplSha = '9840efe48003b66e3195eb0cafab4ec0752a3207'
$A003Parent = 'd59221d650821796a561f65fc98561334f914757'
$A004Parent = '445f72bd12c4409d775b8e147ff6d218c9711244'
$ImplRef = 'refs/remotes/gitee/codex/w001-mpp2-huber-loss-20260728'
$A003Ref = 'refs/remotes/gitee/automation/server/W001/A003'
$A004Ref = 'refs/remotes/gitee/automation/server/W001/A004'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "registered server Python is missing: $Python"
}
& $Python -c 'import jsonschema, sys; print(sys.executable)'
if ($LASTEXITCODE -ne 0) { throw 'registered server Python/jsonschema probe failed' }

New-Item -ItemType Directory -Path $RootBase -Force | Out-Null
if (Test-Path -LiteralPath $Root) { throw "fresh run root collision: $Root" }
New-Item -ItemType Directory -Path $Root | Out-Null

git -C $Repo fetch --no-tags gitee `
  'refs/heads/codex/w001-mpp2-huber-loss-20260728:refs/remotes/gitee/codex/w001-mpp2-huber-loss-20260728' `
  'refs/heads/automation/server/W001/A003:refs/remotes/gitee/automation/server/W001/A003' `
  'refs/heads/automation/server/W001/A004:refs/remotes/gitee/automation/server/W001/A004'
if ($LASTEXITCODE -ne 0) { throw 'Gitee fetch failed' }

$ImplTip = (git -C $Repo rev-parse $ImplRef).Trim()
git -C $Repo cat-file -e "${ImplSha}^{commit}"
if ($LASTEXITCODE -ne 0) { throw 'implementation commit is unavailable' }
git -C $Repo merge-base --is-ancestor $ImplSha $ImplTip
if ($LASTEXITCODE -ne 0) {
    throw "implementation SHA is not an ancestor of fetched branch tip: $ImplTip"
}
if ((git -C $Repo rev-parse $A003Ref).Trim() -ne $A003Parent) {
    throw 'A003 remote ref changed; stop without choosing or overwriting a revision'
}
if ((git -C $Repo rev-parse $A004Ref).Trim() -ne $A004Parent) {
    throw 'A004 remote ref changed; stop without choosing or overwriting a revision'
}

git -C $Repo cat-file -e "${A003Parent}:automation/returns/W001/A003/R004" 2>$null
if ($LASTEXITCODE -eq 0) { throw 'A003/R004 already exists; stop' }
git -C $Repo cat-file -e "${A004Parent}:automation/returns/W001/A004/R002" 2>$null
if ($LASTEXITCODE -eq 0) { throw 'A004/R002 already exists; stop' }

$Impl = Join-Path $Root 'impl'
$Src3 = Join-Path $Root 'source-a003'
$Src4 = Join-Path $Root 'source-a004'
$Bundle3 = Join-Path $Root 'bundle-a003-r004'
$Bundle4 = Join-Path $Root 'bundle-a004-r002'
$Build3Log = Join-Path $Root 'a003-r004-build.log'
$Build4Log = Join-Path $Root 'a004-r002-build.log'

try {
    git -C $Repo worktree add --detach $Impl $ImplSha
    if ($LASTEXITCODE -ne 0) { throw 'implementation worktree creation failed' }
    git -C $Repo worktree add --detach $Src3 $A003Parent
    if ($LASTEXITCODE -ne 0) { throw 'A003 source worktree creation failed' }
    git -C $Repo worktree add --detach $Src4 $A004Parent
    if ($LASTEXITCODE -ne 0) { throw 'A004 source worktree creation failed' }

    & $Python (Join-Path $Impl 'deploy\pfmval_ops.py') governance result-bundle-v1 build `
      --source-bundle (Join-Path $Src3 'automation\returns\W001\A003\R003') `
      --staging $Bundle3 `
      --result-id 'W001-A003-result-R004' `
      --artifact-retention 'retain_in_immutable_result_bundle' *>&1 | Tee-Object -FilePath $Build3Log
    $Build3Exit = $LASTEXITCODE
    if ($Build3Exit -ne 0) {
        throw "A003/R004 build failed with exit $Build3Exit; log: $Build3Log"
    }

    & $Python (Join-Path $Impl 'deploy\pfmval_ops.py') governance result-bundle-v1 build `
      --source-bundle (Join-Path $Src4 'automation\returns\W001\A004\R001') `
      --staging $Bundle4 `
      --result-id 'W001-A004-result-R002' `
      --artifact-retention 'retain_in_immutable_result_bundle' *>&1 | Tee-Object -FilePath $Build4Log
    $Build4Exit = $LASTEXITCODE
    if ($Build4Exit -ne 0) {
        throw "A004/R002 build failed with exit $Build4Exit; log: $Build4Log"
    }

    $P3 = & $Python (Join-Path $Impl 'deploy\pfmval_ops.py') governance result-bundle-v1 publish `
      --bundle $Bundle3 `
      --remote gitee `
      --ref 'automation/server/W001/A003' `
      --revision-path 'automation/returns/W001/A003/R004' `
      --expected-parent $A003Parent `
      --commit-message 'G12-R: return W001 A003 packaging-only R004' |
      Out-String | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'A003/R004 publish failed' }

    $P4 = & $Python (Join-Path $Impl 'deploy\pfmval_ops.py') governance result-bundle-v1 publish `
      --bundle $Bundle4 `
      --remote gitee `
      --ref 'automation/server/W001/A004' `
      --revision-path 'automation/returns/W001/A004/R002' `
      --expected-parent $A004Parent `
      --commit-message 'G12-R: return W001 A004 packaging-only R002' |
      Out-String | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'A004/R002 publish failed' }

    [ordered]@{
        operation = 'G12-R packaging-only'
        implementation_sha = $ImplSha
        training_ran = $false
        result_import_ran = $false
        run_consumed = '2/4'
        A003 = $P3
        A004 = $P4
    } | ConvertTo-Json -Depth 6
}
finally {
    foreach ($Path in @($Impl, $Src3, $Src4)) {
        if (Test-Path -LiteralPath $Path) {
            git -C $Repo worktree remove --force $Path
        }
    }
}
```

将最后输出的 JSON 原样回传。不要执行其他命令；本地审核将随后核验远端 ref、commit、父提交、bundle SHA、完整 Schema 和 `run_consumed=2/4`。
