param([string]$Config=(Join-Path $PSScriptRoot 'config.json'),[string]$PythonInterpreter,[string]$RunsRoot)
& (Join-Path $PSScriptRoot 'invoke.ps1') -Scope precheck -Config $Config -PythonInterpreter $PythonInterpreter -RunsRoot $RunsRoot
exit $LASTEXITCODE
