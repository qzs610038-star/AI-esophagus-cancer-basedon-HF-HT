# run_gfnet_experiment.ps1 — GFNet 65-token formal training (server)
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File deploy/run_gfnet_experiment.ps1
# Prerequisites: git pull first to get model_gfnet.py and updated train_online_tokens.py

$ErrorActionPreference = "Stop"
$Python = "C:\Users\AIPatho1\pfmval_env\Scripts\python.exe"
$ProjectDir = "D:\AIPatho\qzs\pfmval_deploy_git"

Set-Location $ProjectDir

# Verify critical files exist
@("model_gfnet.py", "train_online_tokens.py", "model_online_tokens.py") | ForEach-Object {
    if (-not (Test-Path $_)) {
        Write-Error "Missing: $_ — run 'git pull' first"
        exit 1
    }
}
Write-Host "All source files present" -ForegroundColor Green

# Step 1: Self-checks
Write-Host "`n=== Step 1: Self-checks ===" -ForegroundColor Cyan
& $Python model_gfnet.py
if ($LASTEXITCODE -ne 0) { Write-Error "model_gfnet.py self-check failed"; exit 1 }

& $Python model_online_tokens.py
if ($LASTEXITCODE -ne 0) { Write-Error "model_online_tokens.py self-check failed"; exit 1 }

Write-Host "Self-checks passed" -ForegroundColor Green

# Step 2: GFNet 65-token smoke test (1 epoch)
Write-Host "`n=== Step 2: GFNet smoke test (1 epoch) ===" -ForegroundColor Cyan

$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"

& $Python -u train_online_tokens.py `
    --mode frozen `
    --cross_patient --fold 1 `
    --dataset_name online_tokens_cross_fold1_65t_gfnet_smoke `
    --encoder_type gfnet `
    --num_tokens 65 `
    --batch_size 4 --grad_accum_steps 2 `
    --num_epochs 1 --early_stop_patience 1 `
    --lr 1e-4 --weight_decay 1e-4 `
    --num_workers 0 --amp

if ($LASTEXITCODE -ne 0) { Write-Error "GFNet smoke test failed"; exit 1 }
Write-Host "GFNet smoke test passed" -ForegroundColor Green

# Step 3: GFNet 65-token formal training (150 epochs)
Write-Host "`n=== Step 3: GFNet 65-token formal training (150 epochs) ===" -ForegroundColor Cyan
Write-Host "This will take several hours. Monitor with: nvidia-smi -l 2" -ForegroundColor Yellow

& $Python -u train_online_tokens.py `
    --mode frozen `
    --cross_patient --fold 1 `
    --dataset_name online_tokens_cross_fold1_65t_gfnet `
    --encoder_type gfnet `
    --num_tokens 65 `
    --batch_size 4 --grad_accum_steps 2 `
    --num_epochs 150 --early_stop_patience 20 `
    --lr 1e-4 --weight_decay 1e-4 `
    --num_workers 0 --amp

if ($LASTEXITCODE -ne 0) { Write-Error "GFNet training failed"; exit 1 }

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Output: checkpoints/online_tokens/frozen_r8_online_tokens_cross_fold1_65t_gfnet/"
