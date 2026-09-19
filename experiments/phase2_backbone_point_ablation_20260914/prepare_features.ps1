[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InspectionReport,
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$python = [string]$configData.python_interpreter
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "配置的服务器 Python 不存在: $python"
}
$env:HF_HOME = [string]$configData.paths.hf_home
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$arguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'src/feature_extract.py'), '--config', $Config, '--device', $Device, '--inspection-report', $InspectionReport)
& $python @arguments
exit $LASTEXITCODE
