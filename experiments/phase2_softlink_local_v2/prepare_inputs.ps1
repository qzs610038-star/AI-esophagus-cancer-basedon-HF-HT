param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$PythonInterpreter
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$python = if ($PythonInterpreter) { $PythonInterpreter } else { $configData.runtime.python_interpreter }
if (-not $python) { throw '未配置 Python 解释器。' }
& $python -X utf8 -u (Join-Path $PSScriptRoot 'src/prepare_inputs.py') --config $Config --package-dir $PSScriptRoot
exit $LASTEXITCODE
