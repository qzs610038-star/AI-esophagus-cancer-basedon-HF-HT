param([Parameter(Mandatory=$true)][int]$Seed,[string]$Config=(Join-Path $PSScriptRoot 'config.json'),[string]$PythonInterpreter,[string]$RunsRoot,[string]$WeightsRoot,[string]$Device)
& (Join-Path $PSScriptRoot 'invoke.ps1') -Scope seed -Seed $Seed -Config $Config -PythonInterpreter $PythonInterpreter -RunsRoot $RunsRoot -WeightsRoot $WeightsRoot -Device $Device
exit $LASTEXITCODE
