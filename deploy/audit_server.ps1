# ============================================================
# audit_server.ps1 — Windows Server 环境审计
# ============================================================
# 用法：在服务器上以 PowerShell 运行此脚本
#   powershell -ExecutionPolicy Bypass -File audit_server.ps1
# 或远程执行：
#   ssh user@host "powershell -Command -" < audit_server.ps1
# ============================================================

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  PFMval Windows Server 环境审计报告" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# ---- 1. 系统信息 ----
Write-Host "`n--- 系统信息 ---" -ForegroundColor Yellow
Write-Host "Hostname: $env:COMPUTERNAME"
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "OS: $($os.Caption) (Build $($os.BuildNumber))"
Write-Host "Install Date: $($os.InstallDate)"
Write-Host "Last Boot: $($os.LastBootUpTime)"

# ---- 2. CPU 和内存 ----
Write-Host "`n--- CPU ---" -ForegroundColor Yellow
$cpu = Get-CimInstance Win32_Processor
Write-Host "型号: $($cpu.Name -join '; ')"
Write-Host "物理CPU: $($cpu.Count) 颗"
Write-Host "核心/线程: $($cpu.NumberOfCores) 核 / $($cpu.NumberOfLogicalProcessors) 线程"

Write-Host "`n--- 内存 ---" -ForegroundColor Yellow
$mem = Get-CimInstance Win32_ComputerSystem
$totalRAM = [math]::Round($mem.TotalPhysicalMemory / 1GB, 0)
Write-Host "总内存: ${totalRAM} GB"
$os_info = Get-CimInstance Win32_OperatingSystem
$freeRAM = [math]::Round($os_info.FreePhysicalMemory / 1MB, 1)
Write-Host "可用内存: ${freeRAM} GB"

# ---- 3. 磁盘 ----
Write-Host "`n--- 磁盘空间 ---" -ForegroundColor Yellow
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    $total = [math]::Round($_.Size / 1GB, 1)
    $free = [math]::Round($_.FreeSpace / 1GB, 1)
    $used = $total - $free
    $pct = if ($total -gt 0) { [math]::Round(($used / $total) * 100, 1) } else { 0 }
    Write-Host "  $($_.DeviceID) 总计=${total}GB  已用=${used}GB  可用=${free}GB  (${pct}%)"
}

# ---- 4. GPU ----
Write-Host "`n--- GPU ---" -ForegroundColor Yellow
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "驱动信息:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu --format=csv 2>$null
    Write-Host ""
    Write-Host "GPU 进程:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>$null
} else {
    Write-Host "!!! nvidia-smi 不可用 — GPU 驱动可能未安装或不在 PATH 中" -ForegroundColor Red
}

# ---- 5. CUDA ----
Write-Host "`n--- CUDA ---" -ForegroundColor Yellow
$cudaDirs = @("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
foreach ($pattern in $cudaDirs) {
    $found = Get-Item $pattern -ErrorAction SilentlyContinue
    if ($found) { Write-Host "  $($found.FullName)" }
}
if (-not (Get-Item $cudaDirs[0] -ErrorAction SilentlyContinue)) {
    Write-Host "  未找到 CUDA Toolkit 安装目录"
}

# 检查 nvcc
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    Write-Host "  nvcc: $($nvcc.Source)"
} else {
    Write-Host "  nvcc: 未安装或不在 PATH 中"
}

# ---- 6. Python 环境 ----
Write-Host "`n--- Python ---" -ForegroundColor Yellow
$pythonCommands = @("python", "python3", "python310", "python311", "python312")
foreach ($py in $pythonCommands) {
    $cmd = Get-Command $py -ErrorAction SilentlyContinue
    if ($cmd) {
        $ver = & $py --version 2>&1
        Write-Host "  $py : $ver [$($cmd.Source)]"
    }
}

# ---- 7. Conda ----
Write-Host "`n--- Conda ---" -ForegroundColor Yellow
$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) {
    Write-Host "  conda: $($conda.Source)"
    conda --version 2>$null
    Write-Host ""
    Write-Host "已安装环境:"
    conda env list 2>$null
} else {
    Write-Host "  conda 未安装"
}

# ---- 8. Git ----
Write-Host "`n--- Git ---" -ForegroundColor Yellow
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-Host "  git: $($git.Source)"
    git --version 2>$null
} else {
    Write-Host "  git: 未安装"
}

# ---- 9. OpenSSH Server ----
Write-Host "`n--- OpenSSH Server ---" -ForegroundColor Yellow
$sshd = Get-Service sshd -ErrorAction SilentlyContinue
if ($sshd) {
    Write-Host "  OpenSSH Server: $($sshd.Status)"
    Write-Host "  Startup Type: $($sshd.StartType)"
} else {
    Write-Host "  OpenSSH Server: 未安装"
}

$sshAgent = Get-Service ssh-agent -ErrorAction SilentlyContinue
if ($sshAgent) {
    Write-Host "  OpenSSH Agent: $($sshAgent.Status)"
}

# ---- 10. 网络 ----
Write-Host "`n--- 网络 ---" -ForegroundColor Yellow
Write-Host "代理设置:"
Write-Host "  http_proxy = $($env:http_proxy)"
Write-Host "  https_proxy = $($env:https_proxy)"
Write-Host "  HTTP_PROXY = $($env:HTTP_PROXY)"
Write-Host "  HTTPS_PROXY = $($env:HTTPS_PROXY)"

Write-Host ""
Write-Host "外网连通性测试:"
$hosts = @("google.com", "baidu.com", "pypi.org", "huggingface.co")
foreach ($h in $hosts) {
    $result = Test-Connection -ComputerName $h -Count 1 -Quiet -TimeoutSeconds 3 2>$null
    $status = if ($result) { "可连通" } else { "不可达" }
    Write-Host "  $h : $status"
}

# ---- 11. 已运行进程 (GPU) ----
Write-Host "`n--- 当前 GPU 使用情况 ---" -ForegroundColor Yellow
if ($nvidiaSmi) {
    $processes = nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null
    if ($processes) {
        Write-Host $processes
    } else {
        Write-Host "  无 GPU 进程运行中"
    }
}

# ---- 总结 ----
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  审计完毕" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "关键配置项（请在 deploy\config.sh 中填写）:"
Write-Host "  SERVER_HOST = <本机IP>"
Write-Host "  SERVER_USER = $env:USERNAME"
Write-Host "  SERVER_PROJECT_DIR = D:/PFMval_new (建议)"
Write-Host "  SERVER_DATA_BASE = <病理图像所在盘符和路径>"
