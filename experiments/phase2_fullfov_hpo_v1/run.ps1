param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [ValidateSet('check-inputs','prepare-features','paired-view','paired-recipe','search-point','search-spatial','extend-point','extend-spatial','freeze','final-ablation','external-eval','analyze-local','export-phase3')]
    [string]$Action = 'check-inputs',
    [string]$BatchDir,
    [string]$Device = 'cuda',
    [string]$PythonInterpreter,
    [string]$RunsRoot,
    [string]$WeightsRoot
)

$ErrorActionPreference = 'Stop'
# A child's nonzero exit is recorded below, not converted into a PowerShell error.
$PSNativeCommandUseErrorActionPreference = $false
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-JsonFile($Path, $Value) {
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30), $utf8)
}

$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$package = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'package.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($PythonInterpreter) { $configData.python_interpreter = $PythonInterpreter }
if ($RunsRoot) { $configData.paths.runs_root = $RunsRoot }
if ($WeightsRoot) { $configData.paths.weights_root = $WeightsRoot }
# The experiment identifier is one directory name, never a path.
if ($package.experiment_id -notmatch '^[\p{L}\p{N}_-]+$') {
    throw 'experiment_id must contain only letters, digits, underscores or hyphens.'
}
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$experimentRoot = [System.IO.Path]::GetFullPath((Join-Path $configData.paths.runs_root $package.experiment_id))
if ($BatchDir) {
    $batchPath = [System.IO.Path]::GetFullPath($BatchDir)
    $prefix = $experimentRoot.TrimEnd('\') + '\'
    if (-not $batchPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $batchPath -PathType Container)) {
        throw 'BatchDir must be an existing batch under the configured experiment runs root.'
    }
} else {
    $batchPath = Join-Path $experimentRoot $stamp
    $null = New-Item -ItemType Directory -Path $batchPath
}
$runId = $Action + '_' + $stamp
$runDir = Join-Path $batchPath $runId
$weightDir = Join-Path (Join-Path (Join-Path $configData.paths.weights_root $package.experiment_id) (Split-Path $batchPath -Leaf)) $runId
$null = New-Item -ItemType Directory -Path $runDir
foreach ($name in @('logs', 'raw')) {
    $null = New-Item -ItemType Directory -Path (Join-Path $runDir $name)
}
$provenanceSrc = Join-Path $runDir 'provenance\src'
$null = New-Item -ItemType Directory -Path $provenanceSrc
Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'src') -Filter '*.py' -File | Copy-Item -Destination $provenanceSrc
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $runDir 'provenance\README.md')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'inputs') -Destination (Join-Path $runDir 'provenance') -Recurse
$runDir = (Resolve-Path -LiteralPath $runDir).Path
$weightDir = [System.IO.Path]::GetFullPath($weightDir)
$record = [ordered]@{
    experiment_id = $package.experiment_id
    code_version = $package.code_version
    run_id = $runId
    action = $Action
    batch_directory = $batchPath
    demo_only = $package.demo_only
    status = 'starting'
    started_at = (Get-Date).ToString('o')
    ended_at = $null
    exit_code = $null
    code_directory = $PSScriptRoot
    run_directory = $runDir
    weight_directory = $weightDir
    weight_registry = (Join-Path $runDir 'model_weights.json')
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
    Write-JsonFile (Join-Path $runDir 'model_weights.json') ([ordered]@{
        schema_version = '1.0'
        experiment_id = $package.experiment_id
        run_id = $runId
        weight_directory = $weightDir
        files = @()
        return_policy = 'server_weights_excluded_from_local_result_copy'
    })
    Write-JsonFile (Join-Path $runDir 'feature_caches.json') ([ordered]@{
        schema_version = '1.0'
        experiment_id = $package.experiment_id
        run_id = $runId
        feature_caches_root = $configData.paths.feature_caches_root
        entries = @()
        return_policy = 'server_feature_caches_excluded_from_local_result_copy'
    })
    [System.IO.File]::WriteAllText((Join-Path $runDir 'logs/startup.log'), "Run directory: $runDir`r`n", $utf8)
    Write-Host "Run directory: $runDir"
    $record.status = 'running'
    Write-JsonFile $recordPath $record
    & $configData.python_interpreter -X utf8 -u (Join-Path $PSScriptRoot 'runner.py') --run-dir $runDir --weights-dir $weightDir --action $Action --device $Device
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
exit $exitCode
