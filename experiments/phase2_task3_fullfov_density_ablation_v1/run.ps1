param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('check-inputs','render-label-audit','smoke','formal')]
  [string]$Action,
  [string]$Python = 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe',
  [string]$OutputDir = '',
  [string]$WeightsDir = '',
  [string]$Device = ''
)

$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Arguments = @((Join-Path $PackageRoot 'runner.py'), '--action', $Action, '--config', (Join-Path $PackageRoot 'config.json'))
if ($OutputDir) { $Arguments += @('--output-dir', $OutputDir) }
if ($WeightsDir) { $Arguments += @('--weights-dir', $WeightsDir) }
if ($Device) { $Arguments += @('--device', $Device) }

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Python解释器不存在: $Python"
}
& $Python @Arguments
exit $LASTEXITCODE

