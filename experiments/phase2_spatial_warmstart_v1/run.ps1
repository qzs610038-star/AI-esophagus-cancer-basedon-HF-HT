param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$PythonInterpreter,
    [string]$RunsRoot,
    [string]$WeightsRoot,
    [ValidateSet('train', 'resume')][string]$Mode = 'train',
    [ValidateSet('all', 'spatial_residual_only', 'point_continue', 'spatial_joint')][string]$Arm = 'all',
    [string]$ResumeCheckpoint
)

$ErrorActionPreference = 'Stop'
$planningPackage = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'package.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($planningPackage.implementation_status -ne 'implemented') {
    throw 'PLAN ONLY: training is not implemented. Read docs/implementation_handoff.md.'
}

# A child's nonzero exit is recorded below, not converted into a PowerShell error.
$PSNativeCommandUseErrorActionPreference = $false
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-JsonFile($Path, $Value) {
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30), $utf8)
}

$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$package = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'package.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($PythonInterpreter) { $configData.python_interpreter = $PythonInterpreter }
if ($RunsRoot) { $configData.runs_root = $RunsRoot }
if ($WeightsRoot) { $configData.weights_root = $WeightsRoot }
# The experiment identifier is one directory name, never a path.
if ($package.experiment_id -notmatch '^[\p{L}\p{N}_-]+$') {
    throw 'experiment_id must contain only letters, digits, underscores or hyphens.'
}
if ($configData.batch_id -notmatch '^[\p{L}\p{N}_-]+$') {
    throw 'batch_id must contain only letters, digits, underscores or hyphens.'
}
if ($Mode -eq 'resume' -and (-not $ResumeCheckpoint)) {
    throw 'ResumeCheckpoint is required when Mode=resume.'
}
if ($Mode -eq 'train' -and $ResumeCheckpoint) {
    throw 'ResumeCheckpoint is only valid when Mode=resume.'
}
if ($Mode -eq 'resume' -and $Arm -eq 'all') {
    throw 'Resume mode requires one explicit arm.'
}
$resolvedParentCheckpoint = $null
$parentRun = $null
if ($ResumeCheckpoint) {
    $checkpointItem = Get-Item -LiteralPath $ResumeCheckpoint
    if (-not $checkpointItem.PSIsContainer) {
        $resolvedParentCheckpoint = $checkpointItem.FullName
        $parentRun = $checkpointItem.Directory.Parent.Parent.FullName
    } else {
        throw 'ResumeCheckpoint must be a last.pt file, not a directory.'
    }
}
$runId = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$experimentRoot = Join-Path $configData.runs_root $package.experiment_id
$batchRoot = Join-Path $experimentRoot $configData.batch_id
$runDir = Join-Path $batchRoot $runId
$weightExperimentRoot = Join-Path $configData.weights_root $package.experiment_id
$weightBatchRoot = Join-Path $weightExperimentRoot $configData.batch_id
$weightDir = Join-Path $weightBatchRoot $runId
$null = New-Item -ItemType Directory -Path $runDir
foreach ($name in @('logs', 'raw')) {
    $null = New-Item -ItemType Directory -Path (Join-Path $runDir $name)
}
$null = New-Item -ItemType Directory -Path $weightDir
$runDir = (Resolve-Path -LiteralPath $runDir).Path
$weightDir = (Resolve-Path -LiteralPath $weightDir).Path
$record = [ordered]@{
    experiment_id = $package.experiment_id
    batch_id = $configData.batch_id
    code_version = $package.code_version
    run_id = $runId
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
    mode = $Mode
    arm = $Arm
    parent_checkpoint = $resolvedParentCheckpoint
    parent_run = $parentRun
    resumed_arm = if ($Mode -eq 'resume') { $Arm } else { $null }
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
        batch_id = $configData.batch_id
        run_id = $runId
        weight_directory = $weightDir
        files = @()
        return_policy = 'server_weights_excluded_from_local_result_copy'
    })
    [System.IO.File]::WriteAllText((Join-Path $runDir 'logs/startup.log'), "Run directory: $runDir`r`n", $utf8)
    Write-Host "Run directory: $runDir"
    $record.status = 'running'
    Write-JsonFile $recordPath $record
    $runnerArgs = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'runner.py'), '--run-dir', $runDir, '--weights-dir', $weightDir)
    if ($PSBoundParameters.ContainsKey('Mode') -or $PSBoundParameters.ContainsKey('Arm') -or $ResumeCheckpoint) {
        $runnerArgs += @('--mode', $Mode, '--arm', $Arm)
        if ($ResumeCheckpoint) { $runnerArgs += @('--resume-checkpoint', $ResumeCheckpoint) }
    }
    & $configData.python_interpreter @runnerArgs
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
