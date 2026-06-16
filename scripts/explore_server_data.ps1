# explore_server_data.ps1 — Quick scan of D:\AIPatho\qzs data layout for PFMval
# Run on server: powershell -NoProfile -ExecutionPolicy Bypass -File explore_server_data.ps1
# Copy entire output back to Claude for analysis.

$ErrorActionPreference = "Continue"
$BASE = "D:\AIPatho\qzs"

Write-Host "============================================================"
Write-Host " PFMval Server Data Explorer"
Write-Host " Host: $env:COMPUTERNAME | User: $env:USERNAME"
Write-Host " Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"
Write-Host ""

# ── Section 1: Top-level directories under D:\AIPatho\qzs ──
Write-Host "=== SECTION 1: Top-level under $BASE ==="
if (Test-Path $BASE) {
    Get-ChildItem $BASE -Directory | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = if ($size) { [math]::Round($size / 1MB, 1) } else { 0 }
        Write-Host "  DIR  $($_.Name)  (~${sizeMB} MB)"
    }
} else {
    Write-Host "  ERROR: $BASE not found!"
}
Write-Host ""

# ── Section 2: patch directories ──
Write-Host "=== SECTION 2: Patch directories ==="
$patchDirs = @(
    "$BASE\data-phase2\patch",
    "$BASE\pfmval_deploy_git\data_new_3ST\patch_noov_spilt"
)
foreach ($pd in $patchDirs) {
    if (Test-Path $pd) {
        Write-Host "--- $pd ---"
        Get-ChildItem $pd -Directory | ForEach-Object {
            $sub = $_.FullName
            $pngCount = (Get-ChildItem $sub -Recurse -Filter "*.png" -ErrorAction SilentlyContinue).Count
            $trainCount = if (Test-Path "$sub\train_patches") { (Get-ChildItem "$sub\train_patches" -Filter "*.png").Count } else { 0 }
            $valCount   = if (Test-Path "$sub\val_patches")   { (Get-ChildItem "$sub\val_patches" -Filter "*.png").Count } else { 0 }
            Write-Host "  [$($_.Name)]  total_png=$pngCount  train=$trainCount  val=$valCount"
        }
    } else {
        Write-Host "  NOT FOUND: $pd"
    }
}
Write-Host ""

# ── Section 3: ssGSEA label files ──
Write-Host "=== SECTION 3: Label CSVs ==="
$labelDirs = @(
    "$BASE\data-phase2\ssGSEA",
    "$BASE\pfmval_deploy_git\data_new_3ST\ssGSEA_zscore"
)
foreach ($ld in $labelDirs) {
    if (Test-Path $ld) {
        Write-Host "--- $ld ---"
        Get-ChildItem $ld -Filter "*.csv" | ForEach-Object {
            $lines = (Get-Content $_.FullName -TotalCount 0 -ErrorAction SilentlyContinue | Measure-Object).Count
            # Safer: just show size
            $sizeKB = [math]::Round($_.Length / 1KB, 1)
            Write-Host "  $($_.Name)  (${sizeKB} KB)"
        }
        Get-ChildItem $ld -Filter "*.json" | ForEach-Object {
            Write-Host "  $($_.Name)  ($([math]::Round($_.Length/1KB,1)) KB)"
        }
    } else {
        Write-Host "  NOT FOUND: $ld"
    }
}
Write-Host ""

# ── Section 4: Cache directories ──
Write-Host "=== SECTION 4: Feature caches ==="
$cacheBases = @(
    "$BASE\pfmval_deploy_git"
)
$cacheTypes = @("uni2h_cache", "uni2h_cache_tokens", "virchow2_cache_tokens", "omiclip_cache", "egnv2_cache")
foreach ($cb in $cacheBases) {
    if (-not (Test-Path $cb)) { continue }
    foreach ($ct in $cacheTypes) {
        $cp = Join-Path $cb $ct
        if (-not (Test-Path $cp)) { continue }
        Write-Host "--- $cp ---"
        Get-ChildItem $cp -Directory | ForEach-Object {
            $sub = $_.FullName
            $ptCount = (Get-ChildItem $sub -Recurse -Filter "*.pt" -ErrorAction SilentlyContinue).Count
            $pngCount = (Get-ChildItem $sub -Recurse -Filter "*.png" -ErrorAction SilentlyContinue).Count
            Write-Host "  [$($_.Name)]  .pt=$ptCount  .png=$pngCount"
        }
    }
}
Write-Host ""

# ── Section 5: Checkpoints ──
Write-Host "=== SECTION 5: Checkpoints (recent, top-level only) ==="
$ckptDir = "$BASE\pfmval_deploy_git\checkpoints"
if (Test-Path $ckptDir) {
    Get-ChildItem $ckptDir -Directory | ForEach-Object {
        $sub = $_.FullName
        $subCount = (Get-ChildItem $sub -Directory).Count
        Write-Host "  [$($_.Name)]  subdirs=$subCount"
    }
} else {
    Write-Host "  NOT FOUND: $ckptDir"
}
Write-Host ""

# ── Section 6: JFX0729 raw fixed data ──
Write-Host "=== SECTION 6: JFX0729 raw fixed data (source for split) ==="
$jfxSources = @(
    "$BASE\data-phase2\patch\JFX0729_noov",
    "$BASE\data-phase2\patch\JFX0729",
    "$BASE\pfmval_deploy_git\data_new_3ST\JFX_fixed_data\JFX0729"
)
foreach ($js in $jfxSources) {
    if (Test-Path $js) {
        $pngCount = (Get-ChildItem $js -Filter "*.png").Count
        Write-Host "  FOUND: $js  ($pngCount .png files)"
    } else {
        Write-Host "  NOT FOUND: $js"
    }
}
Write-Host ""

# ── Section 7: Python environments ──
Write-Host "=== SECTION 7: Python environments ==="
$pythonPaths = @(
    "C:\Users\AIPatho1\pfmval_env\Scripts\python.exe",
    "C:\Program Files\Python313\python.exe",
    "D:\miniconda\python.exe"
)
foreach ($pp in $pythonPaths) {
    if (Test-Path $pp) {
        $ver = & $pp --version 2>&1
        Write-Host "  $pp  ->  $ver"
    } else {
        Write-Host "  NOT FOUND: $pp"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " SCAN COMPLETE"
Write-Host "============================================================"
