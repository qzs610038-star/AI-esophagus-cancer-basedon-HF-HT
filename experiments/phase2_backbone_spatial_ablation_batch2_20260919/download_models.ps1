[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [ValidateSet('hoptimus0', 'hoptimus1', 'phikonv2', 'all')][string[]]$Model = @('all'),
    [string]$VerifyDevice = 'cuda',
    [string]$Report,
    [switch]$RegisterOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$python = [string]$configData.python_interpreter
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "配置的服务器 Python 不存在: $python"
}
$env:HF_HOME = [string]$configData.paths.hf_home
$env:HF_HUB_OFFLINE = if ($RegisterOnly) { '1' } else { '0' }
$arguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'src/download_models.py'), '--config', $Config, '--verify-device', $VerifyDevice)
foreach ($value in $Model) { $arguments += @('--model', $value) }
if (-not [string]::IsNullOrWhiteSpace($Report)) { $arguments += @('--report', $Report) }
if ($RegisterOnly) { $arguments += '--register-only' }
& $python @arguments
exit $LASTEXITCODE
