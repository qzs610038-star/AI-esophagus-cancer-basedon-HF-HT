param(
    [Parameter(Mandatory = $true)][string]$Scope,
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$PythonInterpreter,
    [string]$RunsRoot,
    [string]$WeightsRoot,
    [Nullable[int]]$Seed,
    [string]$EndpointManifest,
    [string]$Device
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$python = if ($PythonInterpreter) { $PythonInterpreter } else { $configData.runtime.python_interpreter }
if (-not $python) { throw '未配置 Python 解释器。' }

$arguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'src/run_experiments.py'), '--config', $Config, '--scope', $Scope)
if ($RunsRoot) { $arguments += @('--runs-root', $RunsRoot) }
if ($WeightsRoot) { $arguments += @('--weights-root', $WeightsRoot) }
if ($null -ne $Seed) { $arguments += @('--seed', $Seed.Value) }
if ($EndpointManifest) { $arguments += @('--endpoint-manifest', $EndpointManifest) }
if ($Device) { $arguments += @('--device', $Device) }

& $python @arguments
exit $LASTEXITCODE
