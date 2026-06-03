# ============================================================
# run_nightly_experiments.ps1 — 夜间 LoRA 验证实验批量执行
# ============================================================
# 2026-06-04 | Server: RTX 4080 16GB
# 用法: powershell -File deploy/run_nightly_experiments.ps1
#
# Checkpoint 目录名规则（train_online_cls.py L350）:
#   run_name = "{mode}_r{lora_rank}_{dataset_name}"
#   dataset_name 自动推导: online_cls_cross_fold{N} / online_cls_{patient}
# ============================================================

$ErrorActionPreference = "Continue"
$PYTHON = "C:\Users\AIPatho1\pfmval_env\Scripts\python.exe"
$ENCODING_FLAGS = "PYTHONIOENCODING=utf-8"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = "logs\nightly"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$MAIN_LOG = "$LOG_DIR\batch_${TIMESTAMP}.log"
$RESULT_CSV = "$LOG_DIR\summary_${TIMESTAMP}.csv"

# ── 实验队列 ──
# Label: 人类可读标签
# CkptDir: train_online_cls.py 自动创建的输出目录名 (checkpoints/online_cls/<CkptDir>/)
# DepCkpt: 依赖的 checkpoint 目录名（用于 stage2/stage3 的 --resume）
# Cmd: 训练命令额外参数
# Priority: P0/P1/P2

$QUEUE = @(
    # ═══ P0: 跨患者 LoRA 主干链（串行依赖）═══
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
    # ═══ P1: 独立验证 — Fold2/3 + Dropout ═══
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
    # ═══ P2: 补充基线 — 低 rank + Frozen Fold2/3 ═══
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

# ── 辅助函数 ──
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

# ── 环境检查 ──
Write-Log "============================================"
Write-Log "  PFMval 夜间 LoRA 验证实验"
Write-Log "  机器: $(hostname)"
Write-Log "  开始: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "  实验数: $($QUEUE.Count)"
Write-Log "============================================"
Write-Log ""

if (-not (Test-Path $PYTHON)) {
    Write-Log "FATAL: Python 不存在: $PYTHON"
    exit 1
}
$pyVer = & $PYTHON --version 2>&1
Write-Log "[ENV] $pyVer"
Write-Log "[ENV] 工作目录: $(Get-Location)"
Write-Log ""

# ── 主循环 ──
$COMPLETED = @{}   # $CkptDir -> "PCC=0.xxx"
$FAILED = @{}      # $CkptDir -> "原因"
$START_TIME = Get-Date

foreach ($exp in $QUEUE) {
    $label = $exp.Label
    $ckptDir = $exp.CkptDir
    $depCkpt = $exp.DepCkpt
    $pri = $exp.Priority
    $cmd = $exp.Cmd

    Write-Log ("─" * 60)
    Write-Log "[$pri] $label"
    Write-Log "     输出目录: checkpoints\online_cls\$ckptDir"
    Write-Log ("─" * 60)

    # ── 依赖检查 ──
    if ($depCkpt -ne "") {
        if ($FAILED.ContainsKey($depCkpt)) {
            Write-Log "[SKIP] 依赖 $depCkpt 已失败 ($($FAILED[$depCkpt]))，跳过"
            $FAILED[$ckptDir] = "依赖 $depCkpt 失败"
            continue
        }
        $depPath = "checkpoints\online_cls\$depCkpt\best_model.pth"
        if (-not (Test-Path $depPath)) {
            Write-Log "[SKIP] 依赖 checkpoint 不存在: $depPath"
            $FAILED[$ckptDir] = "依赖缺失: $depPath"
            continue
        }
        Write-Log "[DEP] 依赖 $depCkpt → best_model.pth 已就绪"
    }

    # ── 已完成检查 ──
    $summaryText = Get-SummaryText -CkptDir $ckptDir
    if ($summaryText) {
        Write-Log "[EXISTS] 已有结果，跳过:"
        Write-Log "         $summaryText"
        $COMPLETED[$ckptDir] = "已存在"
        continue
    }

    # ── 执行训练 ──
    $fullCmd = "set $ENCODING_FLAGS && `"$PYTHON`" -u train_online_cls.py $cmd"
    Write-Log "[CMD] $fullCmd"

    $expStart = Get-Date
    try {
        $proc = Start-Process cmd -ArgumentList "/c", $fullCmd -Wait -NoNewWindow -PassThru
        $exitCode = $proc.ExitCode
    } catch {
        Write-Log "[ERROR] 进程异常: $_"
        $exitCode = 1
    }
    $expEnd = Get-Date
    $elapsed = ($expEnd - $expStart).TotalMinutes

    # ── 结果提取 ──
    $result = Get-CkptResult -CkptDir $ckptDir
    if ($exitCode -eq 0 -and $result -ne $null) {
        Write-Log "[OK] Val PCC=$('{0:F4}' -f $result.PCC) | Epoch=$($result.Epoch) | Val Loss=$('{0:F4}' -f $result.Loss) | 耗时 ${elapsed:F1}min"
        $COMPLETED[$ckptDir] = "PCC=$('{0:F4}' -f $result.PCC)"
    } elseif ($result -ne $null) {
        Write-Log "[WARN] 退出码=$exitCode | Val PCC=$('{0:F4}' -f $result.PCC)"
        $COMPLETED[$ckptDir] = "PCC=$('{0:F4}' -f $result.PCC) (exit=$exitCode)"
    } else {
        Write-Log "[FAIL] 退出码=$exitCode | 耗时 ${elapsed:F1}min | 无结果文件"
        $FAILED[$ckptDir] = "退出码=$exitCode, 无 training_history.csv"
    }
    Write-Log ""
}

# ══════════════════════════════════════════════════════════
#  汇总报告
# ══════════════════════════════════════════════════════════
$END_TIME = Get-Date
$TOTAL_HOURS = ($END_TIME - $START_TIME).TotalHours

Write-Log "============================================"
Write-Log "  实 验 汇 总"
Write-Log "============================================"
Write-Log "完成: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "总耗时: ${TOTAL_HOURS:F1}h"
Write-Log "完成: $($COMPLETED.Count) | 失败: $($FAILED.Count) | 总计: $($QUEUE.Count)"
Write-Log ""

# 写入 CSV
$csvLines = @("label,ckpt_dir,status,pcc,epoch,detail")
foreach ($exp in $QUEUE) {
    $ckptDir = $exp.CkptDir
    if ($COMPLETED.ContainsKey($ckptDir)) {
        $r = Get-CkptResult -CkptDir $ckptDir
        $pcc = if ($r) { '{0:F4}' -f $r.PCC } else { "-" }
        $epoch = if ($r) { $r.Epoch } else { "-" }
        $csvLines += "$($exp.Label.Replace(',',' ')),$ckptDir,OK,$pcc,$epoch,$($COMPLETED[$ckptDir])"
    } elseif ($FAILED.ContainsKey($ckptDir)) {
        $csvLines += "$($exp.Label.Replace(',',' ')),$ckptDir,FAIL,-,-,$($FAILED[$ckptDir])"
    }
}
$csvLines | Out-File -FilePath $RESULT_CSV -Encoding utf8

# 分优先级别表
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
            Write-Log ("  {0}: FAIL — {1}" -f $exp.Label, $FAILED[$ckptDir])
        }
    }
    Write-Log ""
}

# 关键对比
Write-Log "========== 关键对比 =========="
$lora_f1 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold1"
$frozen_f1 = Get-CkptResult -CkptDir "frozen_r8_online_cls_cross_fold1"
$frozen_single = Get-CkptResult -CkptDir "frozen_r8_online_cls_HYZ15040"

if ($lora_f1 -and $frozen_f1) {
    $delta = $lora_f1.PCC - $frozen_f1.PCC
    Write-Log ("LoRA 跨患者 Fold1: {0:F4} | Frozen: {1:F4} | Δ = {2:+.4f}" -f $lora_f1.PCC, $frozen_f1.PCC, $delta)
} elseif ($lora_f1) {
    Write-Log ("LoRA 跨患者 Fold1: {0:F4} (Frozen 基线未跑/已存在)" -f $lora_f1.PCC)
}

$s2 = Get-CkptResult -CkptDir "stage2_r8_online_cls_cross_fold1"
$s3 = Get-CkptResult -CkptDir "stage3_r8_online_cls_cross_fold1"
if ($s2) { Write-Log ("Stage2 跨患者 Fold1: {0:F4}" -f $s2.PCC) }
if ($s3) { Write-Log ("Stage3 跨患者 Fold1: {0:F4}" -f $s3.PCC) }

$lora_f2 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold2"
$lora_f3 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold3"
if ($lora_f2) { Write-Log ("LoRA 跨患者 Fold2: {0:F4}" -f $lora_f2.PCC) }
if ($lora_f3) { Write-Log ("LoRA 跨患者 Fold3: {0:F4}" -f $lora_f3.PCC) }

# 三折均值（如果三折都跑完）
if ($lora_f1 -and $lora_f2 -and $lora_f3) {
    $mean = ($lora_f1.PCC + $lora_f2.PCC + $lora_f3.PCC) / 3.0
    Write-Log ("LoRA 三折均值: {0:F4}" -f $mean)
}

$lora_d01 = Get-CkptResult -CkptDir "lora_r8_online_cls_cross_fold1_d01"
$lora_r4 = Get-CkptResult -CkptDir "lora_r4_online_cls_cross_fold1"
if ($lora_d01) { Write-Log ("LoRA Dropout=0.1 Fold1: {0:F4}" -f $lora_d01.PCC) }
if ($lora_r4) { Write-Log ("LoRA r=4 Fold1: {0:F4}" -f $lora_r4.PCC) }

Write-Log ""
Write-Log "📄 汇总CSV: $RESULT_CSV"
Write-Log "📄 主日志:  $MAIN_LOG"
Write-Log "============================================"
