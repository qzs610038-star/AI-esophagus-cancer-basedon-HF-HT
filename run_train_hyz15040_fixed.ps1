<#
.SYNOPSIS
    HYZ15040 修复数据集训练包装脚本
.DESCRIPTION
    使用干净的CSV文件（仅含HYZ15040的2655行数据）重新训练HisToGene-UNI模型。
    通过目录链接(Junction)复用已有的UNI2-h特征缓存，无需重复提取。
    遵循项目约束：只新增文件，不修改现有文件。
.NOTES
    问题背景：原始CSV包含多患者数据(10578行)，三层交集过滤时出现匹配问题，
    导致验证集仅17个样本。使用clean CSV(2655行)后可正确加载2390/265样本。
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\AI空间转录病理研究\PFMval_new"

# ─── 路径定义 ──────────────────────────────────────────────────────────────
$CacheBase       = Join-Path $ProjectRoot "uni2h_cache"
$OriginalCache   = Join-Path $CacheBase "HYZ15040"
$JunctionTarget  = Join-Path $CacheBase "HYZ15040_UNI_fixed"

$CleanCSV = Join-Path $ProjectRoot "data_new_3ST\ssGSEA_zscore\HYZ15040_ssGSEA_zscore_clean.csv"

# ─── 1. 验证干净CSV文件存在 ────────────────────────────────────────────────
if (-not (Test-Path $CleanCSV)) {
    Write-Host "[ERROR] 干净CSV文件不存在: $CleanCSV" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] 干净CSV文件已确认: $CleanCSV" -ForegroundColor Green

# ─── 2. 创建缓存目录链接(Junction) ────────────────────────────────────────
# _infer_base_name("HYZ15040_UNI_fixed") = "HYZ15040_UNI_fixed" (不以"_UNI"结尾)
# 因此缓存路径为 uni2h_cache/HYZ15040_UNI_fixed/train 和 val
# 通过Junction链接到已有的 uni2h_cache/HYZ15040 目录，避免重新提取特征
if (Test-Path $JunctionTarget) {
    Write-Host "[OK] 缓存目录链接已存在: $JunctionTarget" -ForegroundColor Green
} else {
    if (-not (Test-Path $OriginalCache)) {
        Write-Host "[ERROR] 原始缓存目录不存在: $OriginalCache" -ForegroundColor Red
        exit 1
    }
    Write-Host "[INFO] 创建缓存目录链接: $JunctionTarget -> $OriginalCache" -ForegroundColor Cyan
    cmd /c mklink /J "$JunctionTarget" "$OriginalCache"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 创建目录链接失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] 目录链接创建成功" -ForegroundColor Green
}

# ─── 3. 启动训练 ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  启动 HisToGene-UNI 修复数据集训练" -ForegroundColor Yellow
Write-Host "  dataset_name: HYZ15040_UNI_fixed" -ForegroundColor Yellow
Write-Host "  labels_csv:   $CleanCSV" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$PythonExe = "C:\Program Files\Python313\python.exe"

& $PythonExe histogene/train_uni.py `
    --dataset_name HYZ15040_UNI_fixed `
    --labels_csv $CleanCSV `
    --num_epochs 150 `
    --batch_size 64 `
    --lr 1e-4 `
    --early_stop_patience 15
