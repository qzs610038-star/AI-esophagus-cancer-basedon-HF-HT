# Git 云端自动同步方案评估

> 2026-05-20 | PFMval 项目

## 三种方案对比

| 维度 | Post-Commit Hook | GitHub Actions（定时） | **Task Scheduler + 脚本** |
|------|-----------------|----------------------|---------------------------|
| 触发方式 | 每次本地 commit | 云端 cron (UTC) | 本地定时或事件触发 |
| 中文路径支持 | 高风险，需 LC_ALL=C | N/A（运行在 Linux runner） | 安全，PowerShell 设 UTF-8 |
| 自动推送每次提交？ | 是（不想要） | 否（只同步已推送的远程） | 可配——仅在有未推送提交时推送 |
| 离线本地备份？ | 否 | 否 | 是（robocopy 到磁盘/NAS/OneDrive） |
| 镜像到 Gitee？ | 否 | 是（hub-mirror-action） | 是（在脚本中添加第二个 remote） |
| 维护成本 | 中——hook 不被 git 跟踪 | 低——YAML 在 repo 中 | 低——一个脚本 + 一次任务注册 |
| 失败可见性 | 静默 | GitHub 日志 + 邮件 | 脚本写 log 文件 |
| 费用 | 免费 | 免费（公开仓库） | 免费 |
| **综合评分** | **2/5** | **3/5** | **5/5** |

## 推荐方案：Windows Task Scheduler + PowerShell 脚本

### 原因
1. 仅在有未推送提交时才推送，不干扰手动工作流
2. 安全处理中文路径（PowerShell UTF-8）
3. 双保险：云端推送 + 本地 robocopy 镜像
4. 零成本，一次设置后零维护
5. 可扩展：同时推送到 GitHub 和 Gitee

---

## 实施步骤

### 步骤 1：配置 Git 编码（一次性）

```bash
git config --global core.quotepath false
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
```

### 步骤 2：创建备份脚本

保存为 `.claude/git-backup.ps1`：

```powershell
$ErrorActionPreference = "Stop"
$RepoPath = "D:\AI空间转录病理研究\PFMval_new"
$BackupPath = "D:\Backups\PFMval_new"   # 按需修改
$LogFile = Join-Path $RepoPath ".claude\backup.log"
$MaxLogLines = 200

Set-Location $RepoPath

$env:LC_ALL = "C"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"===== $timestamp =====" | Out-File -Append -Encoding UTF8 $LogFile

# 仅在有未推送提交时推送
$unpushed = git log origin/main..HEAD --oneline 2>&1
if ($unpushed -and $LASTEXITCODE -eq 0) {
    "Pushing unpushed commits..." | Out-File -Append -Encoding UTF8 $LogFile
    git push origin main 2>&1 | Out-File -Append -Encoding UTF8 $LogFile
    if ($LASTEXITCODE -eq 0) { "Push OK" | Out-File -Append -Encoding UTF8 $LogFile }
    else { "Push FAILED (exit $LASTEXITCODE)" | Out-File -Append -Encoding UTF8 $LogFile }
} else {
    "No unpushed commits. Skipping push." | Out-File -Append -Encoding UTF8 $LogFile
}

# 本地镜像备份（排除 .git 和大文件）
if (Test-Path $BackupPath) {
    robocopy $RepoPath $BackupPath /MIR /XD ".git" /XF "*.pth" "*.log" /FFT /R:2 /W:5 /NP /NDL
    "Local mirror OK (robocopy exit $LASTEXITCODE)" | Out-File -Append -Encoding UTF8 $LogFile
}

# 日志裁剪
$lines = Get-Content $LogFile -ErrorAction SilentlyContinue
if ($lines.Count -gt $MaxLogLines) {
    $lines[-$MaxLogLines..-1] | Set-Content $LogFile -Encoding UTF8
}
```

### 步骤 3：注册计划任务

以管理员身份运行 PowerShell：

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"D:\AI空间转录病理研究\PFMval_new\.claude\git-backup.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At 3:00AM
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "PFMval Git Backup" `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "每日 PFMval 项目 Git 推送 + 本地镜像备份"
```

也可使用 `-AtLogOn`（登录时）或 `-AtLock`（锁屏时）触发器。

### 步骤 4：测试

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\AI空间转录病理研究\PFMval_new\.claude\git-backup.ps1"
cat D:\AI空间转录病理研究\PFMval_new\.claude\backup.log
```

### 步骤 5（可选）：添加 Gitee 镜像

```bash
git remote add gitee https://gitee.com/YOUR_USERNAME/PFMval_new.git
```

然后在备份脚本的 GitHub push 后添加 `git push gitee main`。

---

## 应急手动备份命令

```bash
# 准备好时手动推送
git push origin main

# 本地磁盘镜像（仅工作区，跳过大文件）
robocopy "D:\AI空间转录病理研究\PFMval_new" "D:\Backups\PFMval_new" /MIR /XD ".git" /XF "*.pth" "*.log" /FFT /R:2 /W:5 /NP /NDL

# 完整仓库备份（含全部历史，便携 git bundle）
git bundle create "D:\Backups\PFMval_20260520.bundle" --all
```

## 快速验证命令

```bash
git config --global --list | grep -E "quote|encoding"   # 检查编码配置
git log origin/main..HEAD --oneline                      # 检查未推送提交
git push origin main                                     # 手动推送测试
cat .claude/backup.log                                   # 查看备份日志
schtasks /Query /TN "PFMval Git Backup" /V               # 检查计划任务状态
```
