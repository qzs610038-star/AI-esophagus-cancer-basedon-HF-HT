# MPP2 paired smoke prediction supplement

本任务只回传 S0/S1 已完成 smoke 的 internal/external prediction CSV，用于本地诊断。不得重新训练、覆盖原结果包或提交 checkpoint。

## 服务器 PowerShell

```powershell
$Repo = 'D:\AIPatho\qzs\pfmval_deploy_git'
$Worktree = 'D:\AIPatho\qzs\pfmval_prediction_supplement_20260712'
$Dispatch = 'automation/local/mpp2-paired-smoke-prediction-supplement-20260712'
$ResultBranch = 'automation/server/mpp2-paired-smoke-prediction-supplement-20260712-r001'
$Out = "$Worktree\automation\results\mpp2-paired-smoke-prediction-supplement-20260712-r001"

$S0 = "$Repo\checkpoints\mpp_uni2h_mlp\mpp2_repaired_s0_frozen_continue_smoke_20260712_r002"
$S1 = "$Repo\checkpoints\mpp_uni2h_mlp\mpp2_repaired_s1_lora_r8_smoke_20260712_r002"

Set-Location $Repo
git fetch gitee $Dispatch

if (Test-Path -LiteralPath $Worktree) {
    throw "补充证据 worktree 已存在，请先人工检查：$Worktree"
}

git worktree add -b $ResultBranch $Worktree "gitee/$Dispatch"
if ($LASTEXITCODE -ne 0) { throw '创建补充证据 worktree 失败' }

foreach ($path in @(
    "$S0\predictions_internal_val.csv",
    "$S0\predictions_external_xzy.csv",
    "$S1\predictions_internal_val.csv",
    "$S1\predictions_external_xzy.csv"
)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "缺少预测文件：$path" }
}

New-Item -ItemType Directory -Path $Out | Out-Null
Copy-Item -LiteralPath "$S0\predictions_internal_val.csv" -Destination "$Out\s0_predictions_internal_val.csv"
Copy-Item -LiteralPath "$S0\predictions_external_xzy.csv" -Destination "$Out\s0_predictions_external_xzy.csv"
Copy-Item -LiteralPath "$S1\predictions_internal_val.csv" -Destination "$Out\s1_predictions_internal_val.csv"
Copy-Item -LiteralPath "$S1\predictions_external_xzy.csv" -Destination "$Out\s1_predictions_external_xzy.csv"

$hashes = Get-ChildItem -LiteralPath $Out -File | Sort-Object Name | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    [pscustomobject]@{ file = $_.Name; size_bytes = $_.Length; sha256 = $hash.Hash.ToLowerInvariant() }
}
$hashes | ConvertTo-Json | Set-Content -LiteralPath "$Out\sha256_manifest.json" -Encoding utf8NoBOM

Set-Location $Worktree
git add "automation/results/mpp2-paired-smoke-prediction-supplement-20260712-r001"
git commit -m 'return MPP2 paired smoke prediction supplement'
git push gitee "HEAD:$ResultBranch"
git status --short --branch
```

成功后只需回报 result branch 和 commit。该 supplement 不是新的训练 result envelope，不运行 `result import`；本地审核方将直接验证文件大小、SHA-256、列结构和样本键。
