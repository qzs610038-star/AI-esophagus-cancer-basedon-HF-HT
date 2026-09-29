[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json')
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$configData = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$python = [string]$configData.python_interpreter
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "配置的服务器 Python 不存在: $python"
}

$bundleRoot = Join-Path $PSScriptRoot 'offline_dependencies'
$bundleName = 'transformers_4.57.1_py313_win_amd64_wheels'
$zip = Join-Path $bundleRoot ($bundleName + '.zip')
$extractRoot = Join-Path $env:TEMP 'pfmval_batch2_transformers_4.57.1'
$wheelDir = Join-Path $extractRoot $bundleName
if (-not (Test-Path -LiteralPath $zip -PathType Leaf)) {
    throw "代码包缺少离线依赖包: $zip"
}

& $python -c "import platform, sys; assert sys.version_info[:2] == (3, 13) and platform.machine().lower() in ('amd64', 'x86_64'), '离线包仅支持 Windows Python 3.13 x64'; print(sys.version.split()[0], platform.machine())"
if ($LASTEXITCODE -ne 0) { throw 'Python 版本或架构与离线依赖包不匹配' }

Expand-Archive -LiteralPath $zip -DestinationPath $extractRoot -Force
& $python -m pip install --disable-pip-version-check --no-index --find-links $wheelDir 'transformers==4.57.1'
if ($LASTEXITCODE -ne 0) { throw 'Phikon-v2 离线依赖安装失败' }

& $python -c "import torch, timm, huggingface_hub, transformers; from transformers import AutoModel; print('torch', torch.__version__, 'timm', timm.__version__, 'huggingface_hub', huggingface_hub.__version__, 'transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { throw '安装后导入检查失败；请不要开始训练' }

Write-Host '依赖导入通过；下一步运行 .\download_models.ps1 -RegisterOnly 完成三模型严格加载。'
