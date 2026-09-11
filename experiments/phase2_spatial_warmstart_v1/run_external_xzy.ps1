param(
    [string]$Python = 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe',
    [string]$SourceCache = 'D:\AIPatho\qzs\runs\phase2_softlink_contrastive_v3\v4_full_diagnostic\20260909_101623_341_f0de1302\external_xzy\stage1_fixed_e5\frozen_regression_e5.npz',
    [string]$Bundle = 'D:\AIPatho\qzs\weights\phase2_spatial_warmstart_v1\warmstart_no_lora_v1\20260910_190620_252_10e4d946\spatial_joint\seed_42\model_bundle.pt',
    [string]$Output = 'D:\AIPatho\qzs\runs\phase2_spatial_warmstart_v1\warmstart_no_lora_v1\20260910_190620_252_10e4d946\external_xzy\spatial_joint',
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$PackageDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputNpz = Join-Path $Output 'xzy_standardized_input.npz'

& $Python -X utf8 (Join-Path $PackageDir 'src\main.py') prepare-xzy-input `
    --input $SourceCache --output $InputNpz --native-step 224
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -X utf8 (Join-Path $PackageDir 'src\main.py') external-eval `
    --bundle $Bundle --input $InputNpz --output $Output --device $Device
exit $LASTEXITCODE
