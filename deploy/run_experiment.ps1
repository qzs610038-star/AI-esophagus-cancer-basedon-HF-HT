# run_experiment.ps1
# PFMval standardized experiment launcher for server deployment.
# Usage:
#   .\run_experiment.ps1 -ExperimentId "gfnet_65t_fold1" -Script "train_online_tokens.py" -Arguments "--encoder_type gfnet --fold 1"
#   .\run_experiment.ps1 -ExperimentId "gfnet_65t_fold1" -Script "train_online_tokens.py" -Arguments "..." -CheckRegistry
#
# All English ASCII only -- Windows PowerShell 5.1 GBK encoding requirement.

param(
    [string]$ExperimentId,

    [string]$Script,

    [string]$Arguments,

    [switch]$CheckRegistry,

    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonPath) {
    $pythonPath = "python"
}

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------
if ($Help) {
    Write-Host @"
run_experiment.ps1 — PFMval standardized experiment launcher

Usage:
  .\run_experiment.ps1 -ExperimentId <id> -Script <script> -Arguments "<args>"
  .\run_experiment.ps1 -ExperimentId <id> -Script <script> -Arguments "<args>" -CheckRegistry
  .\run_experiment.ps1 -Help

Parameters:
  -ExperimentId    Unique experiment ID (must exist in experiment_registry.json)
  -Script          Python training script name (e.g. train_online_tokens.py)
  -Arguments       Arguments to pass to the training script (quoted string)
  -CheckRegistry   Deprecated compatibility switch; registry check is always mandatory
  -Help            Show this help message

Preflight checks:
  1. Verify working directory is project root
  2. nvidia-smi GPU status
  3. Set PYTHONIOENCODING=utf-8, HF_HUB_OFFLINE=1, PFMVAL_CONFIG
  4. Verify training script exists

Output:
  Log file: results_nightly/logs/<ExperimentId>_<timestamp>.log
  Exit code: forwarded from training script

Example:
  .\run_experiment.ps1 -ExperimentId "gfnet_65t_fold1" -Script "train_online_tokens.py" -Arguments "--encoder_type gfnet --fold 1" -CheckRegistry
"@
    exit 0
}

# Validate mandatory parameters
if (-not $ExperimentId) {
    Write-Host "[FAIL] -ExperimentId is required. Use -Help for usage."
    exit 1
}
if (-not $Script) {
    Write-Host "[FAIL] -Script is required. Use -Help for usage."
    exit 1
}
if (-not $Arguments) {
    Write-Host "[FAIL] -Arguments is required. Use -Help for usage."
    exit 1
}

# ------------------------------------------------------------
# Preflight 1: verify working directory
# ------------------------------------------------------------
Write-Host "=== Preflight: Working Directory ==="
$CurrentDir = Get-Location
if ($CurrentDir.Path -ne $ProjectRoot) {
    Write-Host "[WARN] Current directory is not project root. Changing to: $ProjectRoot"
    Set-Location $ProjectRoot
}
Write-Host "[PASS] Working directory: $ProjectRoot"

# ------------------------------------------------------------
# Preflight 1b: durable project state gate
# ------------------------------------------------------------
Write-Host "=== Preflight: Durable Project State ==="
$opsScript = Join-Path $ProjectRoot "deploy\pfmval_ops.py"
if (-not (Test-Path $opsScript)) {
    Write-Host "[FAIL] Durable state CLI missing: $opsScript"
    exit 1
}
$stateTask = if ($Script -like "*mpp*") { "training" } else { "general" }
& $pythonPath $opsScript agent start-check --strict --task $stateTask --host-scope server
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Project state/training gate rejected this launch."
    exit 1
}

# ------------------------------------------------------------
# Preflight 2: verify nvidia-smi
# ------------------------------------------------------------
Write-Host "=== Preflight: GPU Status ==="
$nvidiaOutput = & nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] nvidia-smi failed. Is the GPU available?"
    Write-Host $nvidiaOutput
    exit 1
}
Write-Host "[PASS] nvidia-smi OK"

# Check for lingering python processes
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "[WARN] Found $($pythonProcs.Count) lingering python.exe process(es):"
    $pythonProcs | ForEach-Object { Write-Host "  PID=$($_.Id) CPU=$($_.CPU) WS=$([math]::Round($_.WorkingSet64/1MB, 1))MB" }
    Write-Host "[INFO] No process will be terminated automatically. Verify process ownership and GPU usage before any manual action."
}

# ------------------------------------------------------------
# Preflight 3: environment variables
# ------------------------------------------------------------
Write-Host "=== Preflight: Environment Variables ==="
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
Write-Host "[INFO] PYTHONIOENCODING=utf-8"
Write-Host "[INFO] HF_HUB_OFFLINE=1"

# PFMVAL_CONFIG
$configPath = Join-Path $ProjectRoot "configs\config.server.yaml"
if (Test-Path $configPath) {
    $env:PFMVAL_CONFIG = $configPath
    Write-Host "[PASS] PFMVAL_CONFIG=$configPath"
} else {
    Write-Host "[WARN] config.server.yaml not found at: $configPath"
    Write-Host "[INFO] Falling back to default config.yaml in project root"
}

# ------------------------------------------------------------
# Preflight 4: verify script exists
# ------------------------------------------------------------
Write-Host "=== Preflight: Training Script ==="
$scriptPath = Join-Path $ProjectRoot $Script
if (-not (Test-Path $scriptPath)) {
    Write-Host "[FAIL] Training script not found: $scriptPath"
    exit 1
}
Write-Host "[PASS] Training script: $scriptPath"

# ------------------------------------------------------------
# Mandatory: CheckRegistry
# ------------------------------------------------------------
if ($true) {
    Write-Host "=== CheckRegistry: Verifying experiment in registry (mandatory) ==="
    $checkScript = Join-Path $ProjectRoot "scripts\check_project_state.py"
    if (Test-Path $checkScript) {
        $checkArgs = @($checkScript, "--mode", "server", "--check-registry-only", "--experiment-id", $ExperimentId)
        $checkResult = & $pythonPath $checkArgs 2>&1
        $checkExit = $LASTEXITCODE
        Write-Host $checkResult
        if ($checkExit -ne 0) {
            Write-Host "[FAIL] Registry check returned non-zero (exit=$checkExit)."
            Write-Host "[FAIL] Register experiment '$ExperimentId' before running."
            exit 1
        }
    } else {
        Write-Host "[FAIL] check_project_state.py not found; mandatory registry check cannot run."
        exit 1
    }
}

# ------------------------------------------------------------
# Execute training
# ------------------------------------------------------------
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $ProjectRoot "results_nightly\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$logFile = Join-Path $logDir "${ExperimentId}_${timestamp}.log"

Write-Host "============================================================"
Write-Host "=== Starting Experiment: $ExperimentId"
Write-Host "=== Script : $Script"
Write-Host "=== Args   : $Arguments --num_threads 8"
Write-Host "=== Log    : $logFile"
Write-Host "=== Time   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

# Build the full command
$fullArgs = "-u `"$scriptPath`" $Arguments --num_threads 8"

# Execute and tee to log
$outSourceIdentifier = "pfmval.$PID.$timestamp.stdout"
$errSourceIdentifier = "pfmval.$PID.$timestamp.stderr"
$logStream = $null
$proc = $null
$launchFailed = $false
$exitCode = 1
try {
    $procInfo = New-Object System.Diagnostics.ProcessStartInfo
    $procInfo.FileName = $pythonPath
    $procInfo.Arguments = $fullArgs
    $procInfo.UseShellExecute = $false
    $procInfo.RedirectStandardOutput = $true
    $procInfo.RedirectStandardError = $true
    $procInfo.WorkingDirectory = $ProjectRoot
    $procInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $procInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $procInfo

    $logStream = [System.IO.StreamWriter]::new($logFile, $false, [System.Text.Encoding]::UTF8)

    $outputBuilder = New-Object System.Text.StringBuilder
    $errorBuilder = New-Object System.Text.StringBuilder

    Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -SourceIdentifier $outSourceIdentifier -Action {
        $line = $EventArgs.Data
        if ($line -ne $null) {
            Write-Host $line
            $logStream.WriteLine($line)
            $logStream.Flush()
        }
    } | Out-Null

    Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -SourceIdentifier $errSourceIdentifier -Action {
        $line = $EventArgs.Data
        if ($line -ne $null) {
            Write-Host "[STDERR] $line" -ForegroundColor Red
            $logStream.WriteLine("[STDERR] $line")
            $logStream.Flush()
        }
    } | Out-Null

    $proc.Start() | Out-Null
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()
    $proc.WaitForExit()

    $exitCode = $proc.ExitCode

} catch {
    $launchFailed = $true
    Write-Host "[FAIL] Failed to start training process: $_"
} finally {
    # Cleanup must never replace the Python exit code or hide argparse stderr.
    # Use explicit non-empty identifiers instead of capturing a pipeline whose
    # output is discarded by Out-Null.
    Unregister-Event -SourceIdentifier $outSourceIdentifier -ErrorAction SilentlyContinue
    Unregister-Event -SourceIdentifier $errSourceIdentifier -ErrorAction SilentlyContinue
    Get-Job -Name $outSourceIdentifier -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
    Get-Job -Name $errSourceIdentifier -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
    if ($logStream) {
        $logStream.Close()
        $logStream.Dispose()
    }
    if ($proc) {
        $proc.Dispose()
    }
}

if ($launchFailed) {
    exit 1
}

Write-Host "============================================================"
Write-Host "=== Experiment Complete: $ExperimentId"
Write-Host "=== Exit Code: $exitCode"
Write-Host "=== Log     : $logFile"
Write-Host "=== Time    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

exit $exitCode
