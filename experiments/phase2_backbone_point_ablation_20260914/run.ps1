[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
& (Join-Path $PSScriptRoot 'run_seed42.ps1') -Config $Config -Device $Device
exit $LASTEXITCODE
