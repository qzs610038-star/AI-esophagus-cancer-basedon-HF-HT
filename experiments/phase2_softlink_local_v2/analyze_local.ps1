param([Parameter(Mandatory=$true)][string]$BatchDir,[string]$Config=(Join-Path $PSScriptRoot 'config.json'),[string]$PythonInterpreter)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$python = & (Join-Path $PSScriptRoot 'resolve_local_python.ps1') -PythonInterpreter $PythonInterpreter
Write-Host "本地分析解释器: $python"
& $python -X utf8 -u (Join-Path $PSScriptRoot 'src/analyze_local.py') --config $Config --batch-dir $BatchDir
exit $LASTEXITCODE
