[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [int[]]$Seed = @(42),
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$python = [string]$configData.python_interpreter
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "配置的服务器 Python 不存在: $python"
}
$arguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'src/inspect_existing_uni2h.py'), '--config', $Config)
foreach ($value in $Seed) {
    if ($value -notin @(42, 43, 44)) { throw "Seed 只允许 42、43、44，实际=$value" }
    $arguments += @('--seed', [string]$value)
}
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $arguments += @('--output-dir', $OutputDir)
}
& $python @arguments
exit $LASTEXITCODE
