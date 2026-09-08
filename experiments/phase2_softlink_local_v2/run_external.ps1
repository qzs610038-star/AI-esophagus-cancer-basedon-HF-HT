param([Parameter(Mandatory=$true)][string]$EndpointManifest,[string]$Config=(Join-Path $PSScriptRoot 'config.json'),[string]$PythonInterpreter,[string]$RunsRoot,[string]$Device)
& (Join-Path $PSScriptRoot 'invoke.ps1') -Scope external -EndpointManifest $EndpointManifest -Config $Config -PythonInterpreter $PythonInterpreter -RunsRoot $RunsRoot -Device $Device
exit $LASTEXITCODE
