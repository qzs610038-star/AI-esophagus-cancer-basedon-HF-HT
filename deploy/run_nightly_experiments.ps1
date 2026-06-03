# ============================================================
# run_nightly_experiments.ps1 — Nightly LoRA validation batch
# ============================================================
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File deploy/run_nightly_experiments.ps1
#
# Checkpoint dir naming (train_online_cls.py L350):
#   run_name = "{mode}_r{lora_rank}_{dataset_name}"
#   dataset_name derived from: online_cls_cross_fold{N} / online_cls_{patient}
# ============================================================

$ErrorActionPreference = "Continue"
$PYTHON = "C:\Users\AIPatho1\pfmval_env\Scripts\python.exe"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = "logs\nightly"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$MAIN_LOG = "$LOG_DIR\batch_${TIMESTAMP}.log"
$RESULT_CSV = "$LOG_DIR\summary_${TIMESTAMP}.csv"

# ── Experiment Queue ──
# Label: human-readable name
# CkptDir: output dir under checkpoints/online_cls/
# DepCkpt: prerequisite checkpoint dir (for stage2/stage3 resume)
# Cmd: extra CLI args for train_online_cls.py
# Priority: P0/P1/P2

$QUEUE = @(
    # ═══ P0: Cross-patient LoRA chain (serial deps) ═══
    [PSCustomObject]@{
        Label   = "P0-1 LoRA r=8 Cross-Fold1"
        CkptDir = "lora_r8_online_cls_cross_fold1"
        DepCkpt = ""
        Priority = "P0"
        Cmd     = "--mode lora --lora_rank 8 --cross_patient --fold 1"
    },
    [PSCustomObject]@{
        Label   = "P0-2 Stage2 r=8 Cross-Fold1"
        CkptDir = "stage2_r8_online_cls_cross_fold1"
        DepCkpt = "lora_r8_online_cls_cross_fold1"
        Priority = "P0"
        Cmd     = "--mode stage2 --lora_rank 8 --cross_patient --fold 1 --resume checkpoints/online_cls/lora_r8_online_cls_cross_fold1/best_model.pth"
    },
    [PSCustomObject]@{
        Label   = "P0-3 Stage3 r=8 Cross-Fold1"
        CkptDir = "stage3_r8_online_cls_cross_fold1"
        DepCkpt = "stage2_r8_online_cls_cross_fold1"
        Priority = "P0"
        Cmd     = "--mode stage3 --lora_rank 8 --cross_patient --fold 1 --resume checkpoints/online_cls/stage2_r8_online_cls_cross_fold1/best_model.pth"
    },
    # ═══ P1: Independent — Fold2/3 + Dropout ═══
    [PSCustomObject]@{
        Label   = "P1-1 LoRA r=8 Cross-Fold2"
        CkptDir = "lora_r8_online_cls_cross_fold2"
        DepCkpt = ""
        Priority = "P1"
        Cmd     = "--mode lora --lora_rank 8 --cross_patient --fold 2"
    },
    [PSCustomObject]@{
        Label   = "P1-2 LoRA r=8 Cross-Fold3"
        CkptDir = "lora_r8_online_cls_cross_fold3"
        DepCkpt = ""
        Priority = "P1"
        Cmd     = "--mode lora --lora_rank 8 --cross_patient --fold 3"
    },
    [PSCustomObject]@{
        Label   = "P1-3 LoRA r=8 Dropout=0.1 Cross-Fold1"
        CkptDir = "lora_r8_online_cls_cross_fold1_d01"
        DepCkpt = ""
        Priority = "P1"
        Cmd     = "--mode lora --lora_rank 8 --lora_dropout 0.1 --cross_patient --fold 1 --dataset_name online_cls_cross_fold1_d01"
    },
    # ═══ P2: Baselines — low rank + Frozen Fold2/3 ═══
    [PSCustomObject]@{
        Label   = "P2-1 LoRA r=4 Cross-Fold1"
        CkptDir = "lora_r4_online_cls_cross_fold1"
        DepCkpt = ""
        Priority = "P2"
        Cmd     = "--mode lora --lora_rank 4 --cross_patient --fold 1"
    },
    [PSCustomObject]@{
        Label   = "P2-2 Frozen r=8 Cross-Fold2"
        CkptDir = "frozen_r8_online_cls_cross_fold2"
        DepCkpt = ""
        Priority = "P2"
        Cmd     = "--mode frozen --lora_rank 8 --cross_patient --fold 2"
    },
    [PSCustomObject]@{
        Label   = "P2-3 Frozen r=8 Cross-Fold3"
        CkptDir = "frozen_r8_online_cls_cross_fold3"
        DepCkpt = ""
        Priority = "P2"
        Cmd     = "--mode frozen --lora_rank 8 --cross_patient --fold 3"
    }
)

# ── Helpers ──

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message"
    Write-Host $line
    Add-Content -Path $MAIN_LOG -Value $line
}

function Get-CkptResult {
    param([string]$CkptDir)
    $historyFile = "checkpoints\online_cls\$CkptDir\training_history.csv"
    if (-not (Test-Path $historyFile)) { return $null }
    $lines = Get-Content $historyFile | Select-Object -Skip 1
    if (-not $lines -or $lines.Count -eq 0) { return $null }
    $bestPCC = -999.0
    $bestEpoch = 0
    $bestLoss = 0
    foreach ($line in $lines) {
        $cols = $line -split ','
        if ($cols.Count -lt 8) { continue }
        $valPCC = try { [double]$cols[6] } catch { continue }
        $valLoss = try { [double]$cols[5] } catch { 999 }
        if ($valPCC -gt $bestPCC) {
            $bestPCC = $valPCC
            $bestEpoch = try { [int]$cols[0] } catch { 0 }
            $bestLoss = $valLoss
        }
    }
    if ($bestPCC -le -900) { return $null }
    return @{PCC=$bestPCC; Epoch=$bestEpoch; Loss=$bestLoss}
}

function Get-SummaryText {
    param([string]$CkptDir)
    $summaryFile = "checkpoints\online_cls\$CkptDir\training_summary.txt"
    if (Test-Path $summaryFile) {
        return (Get-Content $summaryFile -Raw) -replace '\n',' | ' -replace '\r',''
    }
    return ""
}

# ── Env check ──
Write-Log "============================================"
Write-Log "  PFMval Nightly LoRA Validation"
Write-Log "  Host: $env:COMPUTERNAME"
Write-Log "  Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "  Total experiments: $($QUEUE.Count)"
Write-Log "============================================"
Write-Log ""

if (-not (Test-Path $PYTHON)) {
    Write-Log "FATAL: Python not found: $PYTHON"
    exit 1
}
$pyVer = & $PYTHON --version 2>&1
Write-Log "[ENV] $pyVer"
Write-Log "[ENV] Working dir: $(Get-Location)"
Write-Log ""

# ── Main loop ──
$COMPLETED = @{}   # CkptDir -> "PCC=0.xxx"
$FAILED = @{}      # CkptDir -> "reason"
$START_TIME = Get-Date

foreach ($exp in $QUEUE) {
    $label = $exp.Label
    $ckptDir = $exp.CkptDir
    $depCkpt = $exp.DepCkpt
    $pri = $exp.Priority
    $cmd = $exp.Cmd

    Write-Log ("-" * 60)
    Write-Log "[$pri] $label"
    Write-Log "     Output dir: checkpoints\online_cls\$ckptDir"
    Write-Log ("-" * 60)

    # ── Dep check ──
    if ($depCkpt -ne "") {
        if ($FAILED.ContainsKey($depCkpt)) {
            Write-Log "[SKIP] Dep $depCkpt failed ($($FAILED[$depCkpt])), skipping"
            $FAILED[$ckptDir] = "Dep $depCkpt failed"
            continue
        }
        $depPath = "checkpoints\online_cls\$depCkpt\best_model.pth"
        if (-not (Test-Path $depPath)) {
            Write-Log "[SKIP] Dep checkpoint missing: $depPath"
            $FAILED[$ckptDir] = "Dep missing: $depPath"
            continue
        }
        Write-Log "[DEP] Prerequisite $depCkpt -> best_model.pth ready"
    }

    # ── Already done? ──
    $summaryText = Get-SummaryText -CkptDir $ckptDir
    if ($summaryText) {
        Write-Log "[SKIP] Already exists:"
        Write-Log "       $summaryText"
        $COMPLETED[$ckptDir] = "pre-existing"
        continue
    }

    # ── Execute ──
    # cmd.exe is needed because Python path has spaces and we chain with &&
    $cmdArgs = "/c", "set PYTHONIOENCODING=utf-8 && `"$PYTHON`" -u train_online_cls.py $cmd"
    Write-Log "[CMD] cmd.exe /c set PYTHONIOENCODING=utf-8 && ... train_online_cls.py $cmd"

    $expStart = Get-Date
    try {
        $proc = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdArgs -Wait -NoNewWindow -PassThru
        $exitCode = $proc.ExitCode
    } catch {
        Write-Log "[ERROR] Process exception: $_"
        $exitCode = 1
    }
    $expEnd = Get-Date
    $elapsed = ($expEnd - $expStart).TotalMinutes

    # ── Extract result ──
    $result = Get-CkptResult -CkptDir $ckptDir
    if ($exitCode -eq 0 -and $result -ne $null) {
        Write-Log ("[OK] Val PCC={0:F4} | Epoch={1} | Val Loss={2:F4} | Elapsed {3:F1}min" -f $result.PCC, $result.Epoch, $result.Loss, $elapsed)
        $COMPLETED[$ckptDir] = "PCC={0:F4}" -f $result.PCC
    } elseif ($result -ne $null) {
        Write-Log ("[WARN] exit={0} | Val PCC={1:F4}" -f $exitCode, $result.PCC)
        $COMPLETED[$ckptDir] = "PCC={0:F4} (exit={1})" -f $result.PCC, $exitCode
    } else {
        Write-Log "[FAIL] exit=$exitCode | Elapsed ${elapsed:F1}min | No result file found"
        $FAILED[$ckptDir] = "exit=$exitCode, no training_history.csv"
    }
    Write-Log ""
}

# ══════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════
$END_TIME = Get-Date
$TOTAL_HOURS = ($END_TIME - $START_TIME).TotalHours

Write-Log "============================================"
Write-Log "  EXPERIMENT SUMMARY"
Write-Log "============================================"
Write-Log "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Total time: ${TOTAL_HOURS:F1}h"
Write-Log "Done: $($COMPLETED.Count) | Failed: $($FAILED.Count) | Total: $($QUEUE.Count)"
Write-Log ""

# Write CSV
$csvLines = @("label,ckpt_dir,status,pcc,epoch,detail")
foreach ($exp in $QUEUE) {
    $ckptDir = $exp.CkptDir
    if ($COMPLETED.ContainsKey($ckptDir)) {
        $r = Get-CkptResult -CkptDir $ckptDir
        $pcc = if ($r) { '{0:F4}' -f $r.PCC } else { "-" }
        $epoch = if ($r) { $r.Epoch } else { "-" }
        $csvLines += "$($exp.Label -replace ',',' '),$ckptDir,OK,$pcc,$epoch,$($COMPLETED[$ckptDir])"
    } elseif ($FAILED.ContainsKey($ckptDir)) {
        $csvLines += "$($exp.Label -replace ',',' '),$ckptDir,FAIL,-,-,$($FAILED[$ckptDir])"
    }
}
$csvLines | Out-File -FilePath $RESULT_CSV -Encoding utf8

# By priority
foreach ($pri in @("P0", "P1", "P2")) {
    Write-Log "========== $pri =========="
    foreach ($exp in $QUEUE | Where-Object { $_.Priority -eq $pri }) {
        $ckptDir = $exp.CkptDir
        if ($COMPLETED.ContainsKey($ckptDir)) {
            $r = Get-CkptResult -CkptDir $ckptDir
            if ($r) {
                Write-Log ("  {0}: PCC={1:F4} (Epoch {2})" -f $exp.Label, $r.PCC, $r.Epoch)
            } else {
                Write-Log ("  {0}: {1}" -f $exp.Label, $COMPLETED[$ckptDir])
            }
        } elseif ($FAILED.ContainsKey($ckptDir)) {
            Write-Log ("  {0}: FAIL - {1}" -f $exp.Label, $FAILED[$ckptDir])
        }
    }
    Write-Log ""
}

# Key comparisons
Write-Log "========== Key Comparisons =========="
$lora_f1 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold1"
$frozen_f1 = Get-CkptResult -CkptDir "frozen_r8_online_cls_cross_fold1"

if ($lora_f1 -and $frozen_f1) {
    $delta = $lora_f1.PCC - $frozen_f1.PCC
    Write-Log ("LoRA Cross-Fold1: {0:F4} | Frozen: {1:F4} | Delta = {2:+.4f}" -f $lora_f1.PCC, $frozen_f1.PCC, $delta)
} elseif ($lora_f1) {
    Write-Log ("LoRA Cross-Fold1: {0:F4} (Frozen baseline pre-existing)" -f $lora_f1.PCC)
}

$s2 = Get-CkptResult -CkptDir "stage2_r8_online_cls_cross_fold1"
$s3 = Get-CkptResult -CkptDir "stage3_r8_online_cls_cross_fold1"
if ($s2) { Write-Log ("Stage2 Cross-Fold1: {0:F4}" -f $s2.PCC) }
if ($s3) { Write-Log ("Stage3 Cross-Fold1: {0:F4}" -f $s3.PCC) }

$lora_f2 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold2"
$lora_f3 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold3"
if ($lora_f2) { Write-Log ("LoRA Cross-Fold2: {0:F4}" -f $lora_f2.PCC) }
if ($lora_f3) { Write-Log ("LoRA Cross-Fold3: {0:F4}" -f $lora_f3.PCC) }

# 3-fold mean
if ($lora_f1 -and $lora_f2 -and $lora_f3) {
    $mean = ($lora_f1.PCC + $lora_f2.PCC + $lora_f3.PCC) / 3.0
    Write-Log ("LoRA 3-Fold Mean: {0:F4}" -f $mean)
}

$lora_d01 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold1_d01"
$lora_r4 = Get-CkptResult -CkptDir "lora_r4_online_cls_cross_fold1"
if ($lora_d01) { Write-Log ("LoRA Dropout=0.1 Fold1: {0:F4}" -f $lora_d01.PCC) }
if ($lora_r4) { Write-Log ("LoRA r=4 Fold1: {0:F4}" -f $lora_r4.PCC) }

$frozen_f2 = Get-CkptResult -CkptDir "frozen_r8_online_cls_cross_fold2"
$frozen_f3 = Get-CkptResult -CkptDir "frozen_r8_online_cls_cross_fold3"
if ($frozen_f2) { Write-Log ("Frozen Cross-Fold2: {0:F4}" -f $frozen_f2.PCC) }
if ($frozen_f3) { Write-Log ("Frozen Cross-Fold3: {0:F4}" -f $frozen_f3.PCC) }

Write-Log ""
Write-Log "Summary CSV: $RESULT_CSV"
Write-Log "Main log: $MAIN_LOG"
Write-Log "============================================"
