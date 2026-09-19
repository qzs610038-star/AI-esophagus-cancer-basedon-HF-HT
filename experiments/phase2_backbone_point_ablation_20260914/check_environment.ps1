[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$python = [string]$configData.python_interpreter
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "配置的服务器 Python 不存在: $python"
}
$arguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'src/environment_check.py'), '--config', $Config)
if (-not [string]::IsNullOrWhiteSpace($Output)) {
    $arguments += @('--output', $Output)
}
& $python @arguments
exit $LASTEXITCODE
