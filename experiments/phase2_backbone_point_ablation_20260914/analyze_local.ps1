[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string[]]$BatchDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$ExternalTargets,
    [string]$PythonInterpreter
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$python = & (Join-Path $PSScriptRoot 'resolve_local_python.ps1') -PythonInterpreter $PythonInterpreter
$arguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'src/analyze_local.py'), '--config', $Config, '--output-dir', $OutputDir)
foreach ($value in $BatchDir) { $arguments += @('--batch-dir', $value) }
if (-not [string]::IsNullOrWhiteSpace($ExternalTargets)) { $arguments += @('--external-targets', $ExternalTargets) }
& $python @arguments
exit $LASTEXITCODE
