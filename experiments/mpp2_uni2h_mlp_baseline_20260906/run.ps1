param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$PythonInterpreter,
    [string]$RunsRoot
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-JsonFile($Path, $Value) {
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30), $utf8)
}

$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$package = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'package.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$runModePath = Join-Path $PSScriptRoot 'run_mode.json'
if (Test-Path -LiteralPath $runModePath) {
    $runMode = Get-Content -LiteralPath $runModePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $configData | Add-Member -NotePropertyName run_kind -NotePropertyValue $runMode.run_kind -Force
}
if ($PythonInterpreter) { $configData.python_interpreter = $PythonInterpreter }
if ($RunsRoot) { $configData.runs_root = $RunsRoot }
if ($package.experiment_id -notmatch '^[\p{L}\p{N}_-]+$') {
    throw 'experiment_id must contain only letters, digits, underscores or hyphens.'
}
$runId = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$experimentRoot = Join-Path $configData.runs_root $package.experiment_id
$runDir = Join-Path $experimentRoot $runId
$null = New-Item -ItemType Directory -Path $runDir
foreach ($name in @('logs', 'raw', 'checkpoints')) {
    $null = New-Item -ItemType Directory -Path (Join-Path $runDir $name)
}
$runDir = (Resolve-Path -LiteralPath $runDir).Path
$record = [ordered]@{
    experiment_id = $package.experiment_id
    code_version = $package.code_version
    run_id = $runId
    demo_only = $package.demo_only
    run_kind = $configData.run_kind
    status = 'starting'
    started_at = (Get-Date).ToString('o')
    ended_at = $null
    exit_code = $null
    code_directory = $PSScriptRoot
    run_directory = $runDir
    python_interpreter = $configData.python_interpreter
    entrypoint = $package.entrypoint
    arguments = $package.args
}
$recordPath = Join-Path $runDir 'run.json'
$exitCode = 1
try {
    Write-JsonFile $recordPath $record
    Write-JsonFile (Join-Path $runDir 'config.json') $configData
    Write-JsonFile (Join-Path $runDir 'package.json') $package
    if (Test-Path -LiteralPath $runModePath) {
        Copy-Item -LiteralPath $runModePath -Destination (Join-Path $runDir 'run_mode.json')
    }
    [System.IO.File]::WriteAllText((Join-Path $runDir 'logs/startup.log'), "Run directory: $runDir`r`n", $utf8)
    Write-Host "Run directory: $runDir"
    $record.status = 'running'
    Write-JsonFile $recordPath $record
    & $configData.python_interpreter -X utf8 -u (Join-Path $PSScriptRoot 'runner.py') --run-dir $runDir
    $exitCode = $LASTEXITCODE
} catch {
    $message = ($_ | Out-String)
    [System.IO.File]::AppendAllText((Join-Path $runDir 'logs/errors.log'), $message, $utf8)
    [Console]::Error.WriteLine($message)
    $exitCode = 1
} finally {
    $record.status = if ($exitCode -eq 0) { 'succeeded' } else { 'failed' }
    $record.exit_code = $exitCode
    $record.ended_at = (Get-Date).ToString('o')
    Write-JsonFile $recordPath $record
}
Write-Host "Copy this return folder: $runDir"
exit $exitCode
