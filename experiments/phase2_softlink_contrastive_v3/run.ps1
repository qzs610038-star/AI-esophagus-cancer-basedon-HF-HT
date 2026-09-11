<#
Shared launcher for the v4 original-split and complete-six-fold protocols.
The protocol-specific wrappers call this file; all model behavior is in
runner.py -> src.orchestrator.  Runs and weights intentionally have separate
roots and both include <batch>/<run> below the experiment name.
#>
param(
    [ValidateSet('original', 'lopo6')]
    [string]$Protocol = 'original',
    [int[]]$Seeds,
    [string[]]$Folds,
    [switch]$Resume,
    [string]$ResumeFrom,
    [switch]$PlanOnly,
    [string]$Batch,
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string]$PythonInterpreter,
    [string]$RunsRoot,
    [string]$WeightsRoot
)

$ErrorActionPreference = 'Stop'
# A child's nonzero exit is recorded below, not converted into a PowerShell error.
$PSNativeCommandUseErrorActionPreference = $false
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-JsonFile($Path, $Value) {
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 40), $utf8)
}
function Test-SafeDirectoryName([string]$Name) {
    return ($Name -and $Name -notmatch '[\\/:*?"<>|]' -and $Name -notmatch '^\.\.?$')
}
function Read-JsonFile([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Ensure-ErrorLog([string]$RunDirectory) {
    $logDirectory = Join-Path $RunDirectory 'logs'
    $null = New-Item -ItemType Directory -Path $logDirectory -Force
    $path = Join-Path $logDirectory 'errors.log'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        [System.IO.File]::WriteAllText($path, '', $utf8)
    }
    return $path
}

$runDir = $null
$weightDir = $null
$errorLogPath = $null
$record = $null
$recordPath = $null
$resumeHistoryPath = $null
$resumeMode = [bool]$Resume
$exitCode = 1
try {
    $configData = Read-JsonFile $Config
    $package = Read-JsonFile (Join-Path $PSScriptRoot 'package.json')
    $hasResumeFrom = -not [string]::IsNullOrWhiteSpace($ResumeFrom)
    if ($Resume -and -not $hasResumeFrom) { throw '-Resume requires -ResumeFrom <既有运行目录>.' }
    if (-not $Resume -and $hasResumeFrom) { throw '-ResumeFrom is only valid together with -Resume.' }
    if (-not $configData.runtime) { throw 'config.runtime is required for the server path contract.' }
    if ($package.experiment_id -notmatch '^[\p{L}\p{N}_-]+$') {
        throw 'experiment_id must contain only letters, digits, underscores or hyphens.'
    }
    if ($Resume) {
        if ($PythonInterpreter -or $RunsRoot -or $WeightsRoot -or $Batch) {
            throw '续跑复用既有运行的配置和路径，不能同时覆盖 Batch、PythonInterpreter、RunsRoot 或 WeightsRoot.'
        }
        if (-not (Test-Path -LiteralPath $ResumeFrom -PathType Container)) {
            throw "ResumeFrom directory does not exist: $ResumeFrom"
        }
        $runDir = (Resolve-Path -LiteralPath $ResumeFrom).Path
        $recordPath = Join-Path $runDir 'run.json'
        if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
            throw "ResumeFrom is missing run.json: $runDir"
        }
        $existingRecord = Read-JsonFile $recordPath
        if ([string]$existingRecord.experiment_id -ne [string]$package.experiment_id) {
            throw "Resume experiment_id mismatch: $($existingRecord.experiment_id) != $($package.experiment_id)"
        }
        if ([string]$existingRecord.protocol -ne [string]$Protocol) {
            throw "Resume protocol mismatch: $($existingRecord.protocol) != $Protocol"
        }
        if ([string]::IsNullOrWhiteSpace([string]$existingRecord.batch_id)) {
            throw 'Resume run.json must contain batch_id.'
        }
        $Batch = [string]$existingRecord.batch_id
        $runId = [string]$existingRecord.run_id
        $record = $existingRecord
        if (-not (Test-SafeDirectoryName $Batch)) { throw 'Existing batch_id must be a single safe directory name.' }
        if ($existingRecord.run_directory) {
            $recordRunDir = (Resolve-Path -LiteralPath ([string]$existingRecord.run_directory)).Path
            if (-not [string]::Equals($recordRunDir, $runDir, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'ResumeFrom does not match run.json.run_directory.'
            }
        }
        if ([string]::IsNullOrWhiteSpace([string]$existingRecord.weight_directory)) {
            throw 'Resume run.json must contain weight_directory.'
        }
        $weightDir = [string]$existingRecord.weight_directory
        if (-not [System.IO.Path]::IsPathRooted($weightDir)) { throw 'Existing weight_directory must be absolute.' }
        if (-not (Test-Path -LiteralPath $weightDir -PathType Container)) {
            throw "Existing weight_directory does not exist: $weightDir"
        }
        $weightDir = (Resolve-Path -LiteralPath $weightDir).Path
        if ([string]::Equals($weightDir, $runDir, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'runs and weights directories must remain separate.'
        }
        if ($null -eq $existingRecord.seeds -or @($existingRecord.seeds).Count -eq 0) {
            throw 'Resume run.json must contain the original seeds.'
        }
        $Seeds = @($existingRecord.seeds | ForEach-Object { [int]$_ })
        $Folds = if ($null -eq $existingRecord.folds) { @() } else { @($existingRecord.folds | ForEach-Object { [string]$_ }) }
        $runnerConfigPath = Join-Path $runDir 'config.json'
        if (-not (Test-Path -LiteralPath $runnerConfigPath -PathType Leaf)) {
            throw "ResumeFrom is missing the config snapshot: $runnerConfigPath"
        }
        # Resume uses the immutable config snapshot from the old run, not the
        # current package config or new path overrides.
        $configData = Read-JsonFile $runnerConfigPath
        $errorLogPath = Ensure-ErrorLog $runDir
    } else {
        if ($PythonInterpreter) { $configData.python_interpreter = $PythonInterpreter }
        if ($RunsRoot) { $configData.runs_root = $RunsRoot; $configData.runtime.runs_root = $RunsRoot }
        if ($WeightsRoot) { $configData.weights_root = $WeightsRoot; $configData.runtime.weights_root = $WeightsRoot }
        if (-not $Batch) { $Batch = [string]$configData.batch_id }
        if (-not (Test-SafeDirectoryName $Batch)) { throw 'Batch must be a single safe directory name.' }

        $defaults = $configData.protocol_defaults.$Protocol
        if (-not $defaults) { throw "No protocol_defaults entry for $Protocol." }
        if ($null -eq $Seeds -or $Seeds.Count -eq 0) { $Seeds = @($defaults.seeds | ForEach-Object { [int]$_ }) }
        if ($Protocol -eq 'lopo6') {
            if ($null -eq $Folds -or $Folds.Count -eq 0) { $Folds = @($defaults.folds | ForEach-Object { [string]$_ }) }
            if (@($Folds | Select-Object -Unique).Count -ne $Folds.Count) { throw 'lopo6 fold list contains duplicates.' }
        } else {
            # Original split has no fold argument and can never auto-dispatch LOPO.
            $Folds = @()
        }

        $runRoot = if ($configData.runs_root) { [string]$configData.runs_root } else { [string]$configData.runtime.runs_root }
        $weightsRoot = if ($configData.weights_root) { [string]$configData.weights_root } else { [string]$configData.runtime.weights_root }
        $runId = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
        $experimentRoot = Join-Path $runRoot $package.experiment_id
        $batchRoot = Join-Path $experimentRoot $Batch
        $runDir = Join-Path $batchRoot $runId
        $weightExperimentRoot = Join-Path $weightsRoot $package.experiment_id
        $weightBatchRoot = Join-Path $weightExperimentRoot $Batch
        $weightDir = Join-Path $weightBatchRoot $runId

        $null = New-Item -ItemType Directory -Path $runDir -Force
        foreach ($name in @('logs', 'raw')) {
            $null = New-Item -ItemType Directory -Path (Join-Path $runDir $name) -Force
        }
        # Create errors.log before any later operation can enter catch, so
        # startup/configuration failures are retained in the run directory.
        $errorLogPath = Ensure-ErrorLog $runDir
        $null = New-Item -ItemType Directory -Path $weightDir -Force
        $runDir = (Resolve-Path -LiteralPath $runDir).Path
        $weightDir = (Resolve-Path -LiteralPath $weightDir).Path
        $runnerConfigPath = Join-Path $runDir 'config.json'
    }

    if (-not $Resume) {
        # Snapshot effective CLI choices so a fresh run is auditable.
        $launch = [ordered]@{
            protocol = $Protocol
            seeds = @($Seeds)
            folds = @($Folds)
            resume = $false
            plan_only = [bool]$PlanOnly
            batch_id = $Batch
        }
        $configData | Add-Member -MemberType NoteProperty -Name launch -Value $launch -Force
        $record = [ordered]@{
            experiment_id = $package.experiment_id
            code_version = $package.code_version
            plan_version = $package.plan_version
            run_id = $runId
            batch_id = $Batch
            protocol = $Protocol
            seeds = @($Seeds)
            folds = @($Folds)
            demo_only = $package.demo_only
            status = 'starting'
            started_at = (Get-Date).ToString('o')
            ended_at = $null
            exit_code = $null
            source_code_directory = $PSScriptRoot
            server_code_directory = $configData.runtime.server_code_directory
            run_directory = $runDir
            weight_directory = $weightDir
            weight_registry = (Join-Path $runDir 'model_weights.json')
            python_interpreter = $configData.python_interpreter
            entrypoint = $package.entrypoint
            arguments = $package.args
            plan_only = [bool]$PlanOnly
            resume = $false
        }
        $recordPath = Join-Path $runDir 'run.json'
        Write-JsonFile $recordPath $record
        Write-JsonFile $runnerConfigPath $configData
        Write-JsonFile (Join-Path $runDir 'package.json') $package
        Write-JsonFile (Join-Path $runDir 'model_weights.json') ([ordered]@{
            schema_version = '1.0'
            experiment_id = $package.experiment_id
            code_version = $package.code_version
            plan_version = $package.plan_version
            batch_id = $Batch
            run_id = $runId
            weights_root = (Resolve-Path -LiteralPath $weightsRoot).Path
            weight_directory = $weightDir
            entries = @()
            files = @()
            return_policy = 'server_weights_excluded_from_local_result_copy'
        })
        $startup = "Run directory: $runDir`r`nWeights directory: $weightDir`r`nProtocol: $Protocol`r`nSeeds: $($Seeds -join ',')`r`nFolds: $($Folds -join ',')`r`nPlanOnly: $([bool]$PlanOnly)`r`n"
        [System.IO.File]::WriteAllText((Join-Path $runDir 'logs/startup.log'), $startup, $utf8)
        Write-Host "Run directory: $runDir"
        $record.status = 'running'
        Write-JsonFile $recordPath $record
    } else {
        # Do not rewrite run.json/config/package on resume; the engine refreshes
        # model_weights.json incrementally from completed checkpoints.  Record
        # the actually executing repair version separately for provenance.
        $resumeRecord = [ordered]@{
            event = 'started'
            resumed_at = (Get-Date).ToString('o')
            code_version = $package.code_version
            original_code_version = $existingRecord.code_version
            protocol = $Protocol
            plan_only = [bool]$PlanOnly
            source_code_directory = $PSScriptRoot
        }
        $resumeHistoryPath = Join-Path $runDir 'logs/resume_history.jsonl'
        $resumeLine = ($resumeRecord | ConvertTo-Json -Compress)
        [System.IO.File]::AppendAllText($resumeHistoryPath, ($resumeLine + "`r`n"), $utf8)
        $resumeNote = "Resume directory: $runDir`r`nWeights directory: $weightDir`r`nProtocol: $Protocol`r`nSeeds: $($Seeds -join ',')`r`nFolds: $($Folds -join ',')`r`nPlanOnly: $([bool]$PlanOnly)`r`nCode version: $($package.code_version)`r`n"
        [System.IO.File]::AppendAllText((Join-Path $runDir 'logs/startup.log'), $resumeNote, $utf8)
        Write-Host "Resuming run directory: $runDir"
    }

    $runnerArgs = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'runner.py'), '--config', $runnerConfigPath, '--run-dir', $runDir, '--weights-dir', $weightDir, '--protocol', $Protocol, '--seeds')
    $runnerArgs += @($Seeds | ForEach-Object { [string]$_ })
    if ($Folds.Count -gt 0) {
        $runnerArgs += '--folds'
        $runnerArgs += @($Folds | ForEach-Object { [string]$_ })
    }
    if ($Resume) { $runnerArgs += '--resume' }
    if ($PlanOnly) { $runnerArgs += '--plan-only' }
    & $configData.python_interpreter @runnerArgs
    $exitCode = $LASTEXITCODE
} catch {
    $message = ($_ | Out-String)
    if ($runDir) {
        try {
            if (-not $errorLogPath) { $errorLogPath = Ensure-ErrorLog $runDir }
            [System.IO.File]::AppendAllText($errorLogPath, $message, $utf8)
        } catch {
            # Preserve the original failure on the console if the directory
            # itself is not writable.
        }
    }
    [Console]::Error.WriteLine($message)
    $exitCode = 1
} finally {
    if ($record -and $recordPath -and -not $resumeMode) {
        $record.status = if ($exitCode -eq 0) { 'succeeded' } else { 'failed' }
        $record.exit_code = $exitCode
        $record.ended_at = (Get-Date).ToString('o')
        Write-JsonFile $recordPath $record
    }
    if ($resumeMode -and $resumeHistoryPath -and $package) {
        $resumeCompletion = [ordered]@{
            event = 'completed'
            completed_at = (Get-Date).ToString('o')
            code_version = $package.code_version
            exit_code = $exitCode
            status = if ($exitCode -eq 0) { 'succeeded' } else { 'failed' }
        }
        $completionLine = ($resumeCompletion | ConvertTo-Json -Compress)
        [System.IO.File]::AppendAllText($resumeHistoryPath, ($completionLine + "`r`n"), $utf8)
    }
}
exit $exitCode
