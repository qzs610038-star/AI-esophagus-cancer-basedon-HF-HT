# ============================================================
# setup_server.ps1 — Windows Server 首次环境安装
# ============================================================
# 用法：在服务器上以管理员 PowerShell 运行
#   powershell -ExecutionPolicy Bypass -File setup_server.ps1
#
# 此脚本将：
#   1. 验证/启用 OpenSSH Server
#   2. 安装 Miniconda（如未安装）
#   3. 创建训练 conda 环境
#   4. 配置 Git + 代理
#   5. 验证 CUDA + PyTorch
# ============================================================

param(
    [string]$ProxyHttp = "",
    [string]$ProxyHttps = "",
    [string]$HFEndpoint = "https://hf-mirror.com",
    [string]$CondaDir = "$env:USERPROFILE\miniconda3",
    [string]$ProjectDir = "D:\PFMval_new"
)

$ErrorActionPreference = "Continue"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  PFMval Windows Server 环境安装" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "配置:"
Write-Host "  ProxyHttp    = $(if ($ProxyHttp) { $ProxyHttp } else { '未设置' })"
Write-Host "  HFEndpoint   = $HFEndpoint"
Write-Host "  CondaDir     = $CondaDir"
Write-Host "  ProjectDir   = $ProjectDir"
Write-Host ""

# ---- 辅助函数 ----
function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ---- 1. 基础信息 ----
Write-Host "==============================================" -ForegroundColor Yellow
Write-Host "  1/7: 系统信息" -ForegroundColor Yellow
Write-Host "=============================================="

Write-Host "  Hostname: $env:COMPUTERNAME"
Write-Host "  User: $env:USERNAME"
Write-Host "  Admin: $(if (Test-Admin) { '是' } else { '否（部分功能可能失败）' })"

# ---- 2. OpenSSH Server ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Yellow
Write-Host "  2/7: 验证 OpenSSH Server" -ForegroundColor Yellow
Write-Host "=============================================="

$sshd = Get-Service sshd -ErrorAction SilentlyContinue
if (-not $sshd) {
    Write-Host "  OpenSSH Server 未安装，尝试安装..."
    if (Test-Admin) {
        Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
        Write-Host "  OpenSSH Server 安装完成"
    } else {
        Write-Host "  !!! 需要管理员权限安装 OpenSSH Server" -ForegroundColor Red
        Write-Host "  以管理员运行 PowerShell 后执行:" -ForegroundColor Yellow
        Write-Host "    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"
        Write-Host "    Start-Service sshd"
        Write-Host "    Set-Service -Name sshd -StartupType 'Automatic'"
    }
} else {
    Write-Host "  OpenSSH Server 状态: $($sshd.Status)"
    if ($sshd.Status -ne "Running") {
        Write-Host "  正在启动..."
        Start-Service sshd -ErrorAction SilentlyContinue
    }
    if ($sshd.StartType -ne "Automatic") {
        Write-Host "  设置开机自启..."
        Set-Service -Name sshd -StartupType 'Automatic' -ErrorAction SilentlyContinue
    }
}

# 确保 .ssh 目录存在
$sshDir = "$env:USERPROFILE\.ssh"
if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
    Write-Host "  已创建 $sshDir"
}

Write-Host "  SSH 公钥存放位置: $sshDir\authorized_keys"

# ---- 3. 配置代理 ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Yellow
Write-Host "  3/7: 配置代理" -ForegroundColor Yellow
Write-Host "=============================================="

if ($ProxyHttp) {
    Write-Host "  设置系统环境变量..."
    [Environment]::SetEnvironmentVariable("http_proxy", $ProxyHttp, "User")
    [Environment]::SetEnvironmentVariable("https_proxy", $ProxyHttps, "User")
    [Environment]::SetEnvironmentVariable("HTTP_PROXY", $ProxyHttp, "User")
    [Environment]::SetEnvironmentVariable("HTTPS_PROXY", $ProxyHttps, "User")
    $env:http_proxy = $ProxyHttp
    $env:https_proxy = $ProxyHttps
    $env:HTTP_PROXY = $ProxyHttp
    $env:HTTPS_PROXY = $ProxyHttps

    # Git 代理
    git config --global http.proxy $ProxyHttp 2>$null
    git config --global https.proxy $ProxyHttps 2>$null

    Write-Host "  代理配置完成"
} else {
    Write-Host "  未配置代理（跳过）"
}

# ---- 4. Miniconda ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Yellow
Write-Host "  4/7: 安装 Miniconda" -ForegroundColor Yellow
Write-Host "=============================================="

$condaExe = "$CondaDir\Scripts\conda.exe"
if (Test-Path $condaExe) {
    Write-Host "  conda 已安装: $condaExe"
    & $condaExe --version 2>$null
} else {
    Write-Host "  下载 Miniconda..."
    $installerUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $installerPath = "$env:TEMP\Miniconda3-installer.exe"

    try {
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "  安装到 $CondaDir..."
        Start-Process -FilePath $installerPath -ArgumentList "/S /D=$CondaDir" -Wait
        Remove-Item $installerPath -Force
        Write-Host "  Miniconda 安装完成"
    } catch {
        Write-Host "  !!! 下载失败: $_" -ForegroundColor Red
        Write-Host "  请手动下载安装: https://docs.conda.io/en/latest/miniconda.html"
    }
}

# 初始化 conda（确保在 PATH 中）
$condaScripts = "$CondaDir\Scripts"
$condaBin = "$CondaDir\condabin"
if (Test-Path $condaScripts) {
    if ($env:PATH -notmatch [regex]::Escape($condaScripts)) {
        $env:PATH = "$condaScripts;$condaBin;$env:PATH"
    }
}

# ---- 5. 创建 conda 环境 ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Yellow
Write-Host "  5/7: 创建 Python 训练环境" -ForegroundColor Yellow
Write-Host "=============================================="

function Ensure-CondaEnv {
    param([string]$EnvName, [string]$YmlFile)

    & $condaExe env list 2>$null | Select-String $EnvName | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $EnvName 环境已存在"
    } else {
        $ymlPath = "$ProjectDir\$YmlFile"
        if (Test-Path $ymlPath) {
            Write-Host "  创建 $EnvName 环境（需要5-15分钟）..."
            & $condaExe env create -f $ymlPath 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  !!! YML 创建失败，尝试手动安装..." -ForegroundColor Yellow
                & $condaExe create -n $EnvName python=3.10 -y 2>&1
            }
        } else {
            Write-Host "  !!! $ymlPath 不存在，跳过（请先推送代码）" -ForegroundColor Yellow
        }
    }
}

Ensure-CondaEnv -EnvName "pfmval_histogene" -YmlFile "env_histogene.yml"
Ensure-CondaEnv -EnvName "pfmval_egnv2" -YmlFile "env_egnv2.yml"

# ---- 6. HuggingFace 配置 ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Yellow
Write-Host "  6/7: 配置 HuggingFace" -ForegroundColor Yellow
Write-Host "=============================================="

if ($HFEndpoint) {
    [Environment]::SetEnvironmentVariable("HF_ENDPOINT", $HFEndpoint, "User")
    $env:HF_ENDPOINT = $HFEndpoint
    Write-Host "  HF_ENDPOINT = $HFEndpoint"
}

$hfToken = $env:HF_TOKEN
if (-not $hfToken) {
    Write-Host "  !!! HF_TOKEN 未设置" -ForegroundColor Yellow
    Write-Host "  如需下载受限制模型，请在系统环境变量中设置 HF_TOKEN"
}

# ---- 7. 验证 ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Yellow
Write-Host "  7/7: 环境验证" -ForegroundColor Yellow
Write-Host "=============================================="

# Python
Write-Host "Python:"
$py = Get-Command python -ErrorAction SilentlyContinue
if ($py) {
    python --version 2>&1
} else {
    Write-Host "  (需先激活 conda 环境)"
}

# PyTorch + CUDA
Write-Host ""
Write-Host "PyTorch + CUDA (pfmval_histogene):"
& $condaExe run -n pfmval_histogene python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  (环境尚未就绪)"
}

# GPU
Write-Host ""
Write-Host "GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>$null

# OpenSSH
Write-Host ""
$sshd = Get-Service sshd -ErrorAction SilentlyContinue
Write-Host "OpenSSH Server: $(if ($sshd) { $sshd.Status } else { '未安装' })"

# 磁盘
Write-Host ""
Write-Host "磁盘:"
$disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='D:'"
if ($disk) {
    $free = [math]::Round($disk.FreeSpace / 1GB, 1)
    $total = [math]::Round($disk.Size / 1GB, 1)
    Write-Host "  D: 可用=${free}GB / 总计=${total}GB"
}

# ---- 完成 ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  安装完毕！" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  后续步骤:" -ForegroundColor Green
Write-Host "  1. 确认 SSH 公钥已添加到 $sshDir\authorized_keys" -ForegroundColor Green
Write-Host "  2. 修改 $ProjectDir\config.yaml 指向服务器数据路径" -ForegroundColor Green
Write-Host "  3. conda activate pfmval_histogene && python config_utils.py" -ForegroundColor Green
Write-Host "  4. 传输/下载模型权重和特征缓存到服务器" -ForegroundColor Green
Write-Host "  5. python train_histogene_uni_tokens.py --epochs 5   # 冒烟测试" -ForegroundColor Green
Write-Host ""

# 提示公钥配置
$pubKeyPath = "$sshDir\authorized_keys"
if (Test-Path $pubKeyPath) {
    Write-Host "  authorized_keys 已存在:" -ForegroundColor Green
    Get-Content $pubKeyPath
} else {
    Write-Host "  !!! authorized_keys 不存在，请将本地公钥内容写入此文件:" -ForegroundColor Yellow
    Write-Host "  本地执行: cat ~/.ssh/pfmval_server.pub" -ForegroundColor Yellow
    Write-Host "  服务器执行: notepad $pubKeyPath" -ForegroundColor Yellow
}
