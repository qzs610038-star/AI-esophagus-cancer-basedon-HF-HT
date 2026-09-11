<#
Evaluate the completed original/seed-42 run on read-only XZY data.
All new artifacts are written below <RunDirectory>\external_xzy only.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [int]$NativeStep = 224,
    [int]$BatchSize,
    [switch]$Resume,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$runPath = (Resolve-Path -LiteralPath $RunDirectory).Path
$configPath = Join-Path $runPath 'config.json'
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "RunDirectory is missing config.json: $runPath"
}
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$python = [string]$config.python_interpreter
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Recorded Python interpreter is unavailable: $python"
}
$arguments = @(
    '-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'xzy_runner.py'),
    '--run-dir', $runPath,
    '--native-step', [string]$NativeStep
)
if ($PSBoundParameters.ContainsKey('BatchSize')) {
    $arguments += @('--batch-size', [string]$BatchSize)
}
if ($Resume) { $arguments += '--resume' }
if ($PlanOnly) { $arguments += '--plan-only' }

& $python @arguments
$exitCode = $LASTEXITCODE
exit $exitCode
