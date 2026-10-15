# VS Code Remote Tunnels 远程连接部署方案

> ⚠️ **historical**：用户最新约束禁止把 Remote Tunnel 作为服务器交互通道。本文仅供历史排障复盘，不得用于当前部署或训练控制。

> 📅 日期：2026-06-17 | 🎯 目标：无需管理员权限，打通本地 AI Agent 到服务器 DESKTOP-HKP0IV0 的远程命令通道
> 📋 上下文：方案 A（端口映射）依赖管理员操作；方案 E 为"无管理员首选方案"，利用服务器**出站**连接建立隧道
> 🔗 关联文档：`SSH远程连接排障与部署方案_20260616.md` §方案 E | memory: `ssh-direct-connection-impossible` | memory: `server-environment-quickref`

---

## 一、为什么选方案 E

### 核心问题

服务器 DESKTOP-HKP0IV0 位于 NAT 内网（10.192.41.155），公网 IP `117.68.10.96:22` 被映射到另一台机器。SSH 直连需管理员新增端口映射，而管理员响应时间不确定。

### 方案 E 的独特优势

| 对比维度 | 方案 A（端口映射） | 方案 E（VS Code Tunnels） |
|---------|:---:|:---:|
| 管理员依赖 | 🟡 需要（NAT 规则配置） | 🟢 **不需要** |
| 实现原理 | 入站端口转发 | **服务器主动出站**连接 Azure 中继 |
| 部署时间 | 等管理员（天数～周） | **30 分钟**（立即可做） |
| 改防火墙 | 需要开放入站端口 | **不需要** |
| 改 NAT | 需要新增映射规则 | **不需要** |
| 连接方式 | 原生 SSH（`ssh user@host`） | VS Code Remote 窗口 / vscode.dev 网页 |

### 架构原理

```
本地 (你的 PC)                    Azure 中继                       服务器 DESKTOP-HKP0IV0
┌────────────────────┐     HTTPS  ┌──────────────────┐     HTTPS  ┌──────────────────────────┐
│ VS Code             │ ←────────→│ global.rel.       │ ←────────→│ code tunnel（出站连接）     │
│ + Remote Tunnels    │   :443    │ tunnels.api.      │   :443    │   ↓                       │
│                     │           │ visualstudio.com  │           │ VS Code Server 进程        │
│ 或 vscode.dev       │           │ tunnelrelay.io    │           │   ↓                       │
│ (浏览器)             │           └──────────────────┘           │ 完整终端 / 文件 / Python   │
└────────────────────┘                                           └──────────────────────────┘
```

**核心机制**：服务器 `code tunnel` 进程主动发起**出站** HTTPS 连接到 Azure 中继。本地 VS Code 通过同一中继找到服务器，所有流量端到端 TLS 加密。**整个链路无入站连接——天然穿透 NAT 和防火墙。**

---

## 二、前提条件检查

在开始部署前，在服务器上确认以下条件（通过 RDP 登录检查）：

```powershell
# 1. 出站 HTTPS 可达性（通过 Clash 代理测试）
curl -x http://127.0.0.1:7890 https://tunnelrelay.io --connect-timeout 10

# 2. Clash 代理端口确认
netstat -an | findstr "7890"

# 3. 用户目录磁盘空间（CLI 约 100MB，VS Code Server 约 200MB）
# 检查 C 盘剩余空间
Get-PSDrive C | Select-Object Used, Free

# 4. 确认当前用户
whoami
# 期望: desktop-hkp0iv0\AIPatho1
```

> ⚠️ 如果 `curl` 测试不通，检查 Clash 是否运行、代理端口是否正确、Clash 规则是否放行 `tunnelrelay.io`。

---

## 三、服务器端部署（Step by Step）

### Step 1: 获取 VS Code CLI（2 分钟）

选择**方式 B**（无需管理员权限）：

```powershell
# 创建工具目录
mkdir C:\Users\AIPatho1\tools\vscode-cli -Force

# 通过代理下载最新 CLI（约 50MB）
curl -x http://127.0.0.1:7890 -L ^
    "https://update.code.visualstudio.com/latest/cli-win32-x64/stable" ^
    -o C:\Users\AIPatho1\tools\vscode-cli\vscode_cli.tar.gz

# 解压
tar -xf C:\Users\AIPatho1\tools\vscode-cli\vscode_cli.tar.gz ^
    -C C:\Users\AIPatho1\tools\vscode-cli

# 验证安装
C:\Users\AIPatho1\tools\vscode-cli\code.exe tunnel --version
# 期望输出: 版本号（如 1.99.x）
```

> 💡 **备用下载方式**：如果服务器下载慢，在本地 PC 用浏览器下载后通过 RDP 复制文件到服务器。
>
> Windows x64 CLI 下载地址：`https://update.code.visualstudio.com/latest/cli-win32-x64/stable`

### Step 2: 首次认证（3 分钟）

由于服务器无图形界面，使用设备码模式认证：

```powershell
# 设置代理环境变量
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"

# 启动 CLI 登录（不尝试打开浏览器）
C:\Users\AIPatho1\tools\vscode-cli\code.exe tunnel login --cli
```

终端会输出类似：
```
To sign in, use a web browser to open the page https://microsoft.com/devicelogin
and enter the code X9Q2M3B7 to authenticate.
```

然后：
1. 在你的**本地电脑浏览器**打开 `https://microsoft.com/devicelogin`
2. 输入终端显示的 8 位代码
3. 用 **GitHub 账号**或 **Microsoft 个人账号**登录
4. 浏览器显示"登录成功"后，服务器终端会自动继续
5. 按提示为这台远程机器命名 → 输入 `DESKTOP-HKP0IV0`（或 `pfmval-server`）

认证成功后，登录会话保存在 `%USERPROFILE%\.vscode-cli\` 目录，后续启动无需重新认证。

> ⚠️ **关键**：本地 VS Code 连接时必须用**完全相同的账号**登录。建议使用 GitHub 账号（与代码仓库一致）。

### Step 3: 启动隧道并验证连通（2 分钟）

```powershell
# 接受许可条款 + 启动隧道
C:\Users\AIPatho1\tools\vscode-cli\code.exe tunnel ^
    --proxy-server=http://127.0.0.1:7890 ^
    --accept-server-license-terms
```

启动成功后终端输出：
```
* Visual Studio Code Server
* Remotely connect to your machines from VS Code.

Open this link in your browser:
https://vscode.dev/tunnel/DESKTOP-HKP0IV0
```

**验证步骤**：
1. 在本地浏览器打开终端输出的链接
2. 用同一账号登录
3. 看到 VS Code 网页版界面，左下角显示 `DESKTOP-HKP0IV0` → ✅ 隧道打通
4. 在网页版终端中执行 `hostname && whoami`，确认返回 `DESKTOP-HKP0IV0` + `AIPatho1`

> ⚠️ 此时先不要关闭服务器上的 PowerShell 窗口——隧道运行在当前会话中。等 Step 4 配置持久化后再关闭。

### Step 4: 配置持久化运行（5 分钟）

隧道不能每次手动启动。根据 AIPatho1 是否有管理员权限，选择对应方式：

#### 方式 A：安装为 Windows 服务（需管理员，最稳定）

```powershell
# ⚠️ 此命令需要管理员权限。如果 AIPatho1 非管理员，会报 Access Denied
code tunnel service install --accept-server-license-terms

# 服务管理命令
code tunnel service log          # 查看日志
code tunnel status               # 查看状态
code tunnel service uninstall    # 卸载服务
```

#### 方式 B：用户级任务计划程序（推荐，无需管理员 ✅）

Windows 非管理员用户可以用 `schtasks` 创建自己的计划任务：

```powershell
# 创建登录时自动启动的隧道任务
schtasks /Create /TN "VS Code Tunnel" ^
    /TR "C:\Users\AIPatho1\tools\vscode-cli\code.exe tunnel --accept-server-license-terms" ^
    /SC ONLOGON ^
    /F

# 验证任务已创建
schtasks /Query /TN "VS Code Tunnel" /FO LIST /V

# 手动触发一次（立即测试）
schtasks /Run /TN "VS Code Tunnel"

# 如需删除
schtasks /Delete /TN "VS Code Tunnel" /F
```

**工作原理**：每次 AIPatho1 登录（包括 RDP 登录）时自动启动 `code tunnel`。服务器通常保持登录状态（通过 RDP 会话），隧道因此持续运行。

**限制**：如果服务器重启且无人 RDP 登录，隧道不会自动启动。此时需要一次 RDP 登录触发任务。

#### 方式 C：创建 start-tunnel.bat + 计划任务（更健壮）

```powershell
# 1. 创建启动脚本
@"
@echo off
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
C:\Users\AIPatho1\tools\vscode-cli\code.exe tunnel --accept-server-license-terms >> C:\Users\AIPatho1\tools\vscode-cli\tunnel.log 2>&1
"@ | Out-File -FilePath C:\Users\AIPatho1\tools\vscode-cli\start-tunnel.bat -Encoding ASCII

# 2. 创建计划任务（指向 bat 脚本）
schtasks /Create /TN "VS Code Tunnel" ^
    /TR "C:\Users\AIPatho1\tools\vscode-cli\start-tunnel.bat" ^
    /SC ONLOGON ^
    /F

# 3. 查看日志
Get-Content C:\Users\AIPatho1\tools\vscode-cli\tunnel.log -Tail 20
```

> ✅ **推荐**：先用方式 C 创建 bat 脚本 + 计划任务。日志落盘方便排查问题。

---

## 四、本地连接配置

### 4.1 VS Code 桌面版连接（推荐，功能最全）

1. 本地 VS Code 安装 **"Remote - Tunnels"** 扩展
   - 扩展 ID: `ms-vscode.remote-server`
   - 或通过命令面板: `Ctrl+Shift+P` → `Extensions: Install Extensions` → 搜索 `Remote - Tunnels`

2. 点击 VS Code 左下角绿色 `><` 图标
   - 或 `Ctrl+Shift+P` → `Remote-Tunnels: Connect to Tunnel...`

3. 选择登录方式 → 用**与服务器端相同的 GitHub/Microsoft 账号**登录

4. 在远程机器列表中选择 `DESKTOP-HKP0IV0`

5. VS Code 打开新窗口，状态栏左下角显示 `>< Tunnels: DESKTOP-HKP0IV0`

**验证连接**：在新窗口的终端（`Ctrl+`` `）中执行：
```powershell
hostname
# 期望: DESKTOP-HKP0IV0

nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
# 期望: NVIDIA GeForce RTX 4080, XXXXX MiB

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 期望: CUDA: True, GPU: NVIDIA GeForce RTX 4080
```

### 4.2 网页版连接（零安装，应急用）

服务器 `code tunnel` 启动后输出的链接可直接在浏览器打开：
```
https://vscode.dev/tunnel/DESKTOP-HKP0IV0
```
登录同一账号后获得完整的 VS Code 网页界面（终端 + 文件浏览 + 编辑）。

### 4.3 多机器管理

如果后续多台服务器都装了 tunnel，在 VS Code 的 Remote Explorer 面板中可以切换。每台机器用不同名称注册。

---

## 五、AI Agent (Claude Code) 适配方案

这是方案 E 与 SSH 方案的核心差异：AI Agent 不能直接 `ssh user@host "command"`。有以下三种适配模式：

### 模式 1：Claude Code 运行在 Tunnel VS Code 窗口中（最直接）

```
本地 VS Code → Remote Tunnels → 远程服务器 DESKTOP-HKP0IV0
     ↑
Claude Code Extension（在此窗口中运行）
     ↑
Bash 工具发出的命令 → 自动在远程服务器终端执行
```

**操作流程**：
1. 本地 VS Code 通过 Remote Tunnels 连接到服务器（§四）
2. 在远程窗口的终端中确认环境正常
3. Claude Code 在这个窗口中的所有 Bash 调用 → 透明作用于远程服务器
4. **Claude Code 感知不到自己在远程**——但所有 `python train_xxx.py`、`nvidia-smi`、文件读写都在服务器上执行

**优势**：零额外配置，Claude Code 无需改造
**劣势**：每次使用需要 VS Code 窗口保持 tunnel 连接；不适合定时/无人值守场景

### 模式 2：tunnel exec 命令行模式（类 SSH 体验）

VS Code CLI 支持通过 tunnel 执行远程命令（类似 `ssh host command`）：

```bash
# 在本地终端通过 tunnel 在远程执行命令
code tunnel exec -n DESKTOP-HKP0IV0 "nvidia-smi"
code tunnel exec -n DESKTOP-HKP0IV0 "hostname && whoami"
code tunnel exec -n DESKTOP-HKP0IV0 "C:\Users\AIPatho1\pfmval_env\Scripts\python.exe -c 'import torch; print(torch.cuda.is_available())'"
```

> ⚠️ `tunnel exec` 功能需要 VS Code CLI ≥ 1.82。如果当前版本不支持，可升级 CLI 或回退到模式 1。使用时需先在本地通过 `code tunnel login` 用同一账号认证。

### 模式 3：组合方案 E + H（生产推荐 ⭐）

**永远不要将长时间训练绑在交互式会话上**（无论是 SSH 还是 Tunnel）。训练应独立于连接运行：

```
本地 AI Agent                     VS Code Tunnel                    服务器
┌──────────────────┐  code tunnel  ┌──────────────────┐  schtasks   ┌──────────────────┐
│ Claude Code       │ ←──────────→ │ Remote 终端        │ ←─────────→ │ 训练任务 (独立)    │
│                   │              │                    │             │                  │
│ 1. 创建训练任务    │              │ 2. schtasks /Run   │             │ python train...  │
│ 3. 轮询日志/GPU   │              │ 4. Get-Content log │             │ > log.txt        │
│ 5. 回收结果       │              │ 6. Copy-Item       │             │ 日志落盘          │
└──────────────────┘              └──────────────────┘             └──────────────────┘
```

**具体命令流程**：

```powershell
# Step 1: AI Agent 在 tunnel 终端中启动训练（独立任务）
schtasks /Create /TN "PFMval_CLS_LoRA_Fold1" ^
    /TR "cmd.exe /c 'set PYTHONIOENCODING=utf-8 && set HF_HUB_OFFLINE=1 && C:\Users\AIPatho1\pfmval_env\Scripts\python.exe -u D:\AIPatho\qzs\pfmval_deploy_git\train_online_cls.py --mode lora --lora_rank 8 --cross_patient --fold 1 --epochs 50 > D:\AIPatho\qzs\logs\cls_lora_fold1.log 2>&1'" ^
    /SC ONCE /ST 00:00 /F

# 立即触发任务
schtasks /Run /TN "PFMval_CLS_LoRA_Fold1"

# Step 2: 轮询训练状态
# 检查任务是否还在运行
schtasks /Query /TN "PFMval_CLS_LoRA_Fold1" /FO LIST | findstr "Status"

# 检查 GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# 查看最新日志
Get-Content D:\AIPatho\qzs\logs\cls_lora_fold1.log -Tail 30

# Step 3: 训练完成后，检查结果
ls D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\online_tokens\cls_lora_fold1\

# Step 4: 回收结果（在本地 VS Code 中通过文件浏览器下载，或用 Git 同步）
```

---

## 六、网络与代理配置详解

### 6.1 Clash 代理确认

服务器通过 Clash 代理上网。确认以下信息：

```powershell
# 确认 Clash 端口（默认 7890）
netstat -an | findstr "LISTENING" | findstr "7890"

# 测试代理连通性
curl -x http://127.0.0.1:7890 https://www.google.com --connect-timeout 10 -I
```

### 6.2 需要放行的 VS Code Tunnel 域名

确保以下域名在 Clash 规则中能正常访问：

| 域名 | 用途 | 端口 | 是否需要 TLS 直连 |
|------|------|:---:|:---:|
| `global.rel.tunnels.api.visualstudio.com` | 隧道管理 API | 443 | ✅ 不能 MITM |
| `*.rel.tunnels.api.visualstudio.com` | 区域中继节点 | 443 | ✅ 不能 MITM |
| `tunnelrelay.io` | 核心中继服务 | 443 | ✅ 不能 MITM |
| `login.microsoftonline.com` | 账号认证 | 443 | 正常代理即可 |
| `update.code.visualstudio.com` | CLI 下载/更新 | 443 | 正常代理即可 |

> ⚠️ **关键**：Tunnel 中继域名的 TLS 不能被中间人解密（MITM）。如果 Clash 或企业防火墙在做 SSL 拦截，需将上述域名加入**直连白名单**（Bypass）。

### 6.3 验证域名可达性

```powershell
# 逐一测试关键域名（通过代理）
curl -x http://127.0.0.1:7890 https://global.rel.tunnels.api.visualstudio.com --connect-timeout 10 -I
curl -x http://127.0.0.1:7890 https://tunnelrelay.io --connect-timeout 10 -I

# 如果返回 HTTP 200 或 302/403（非超时/连接拒绝），说明可达
```

### 6.4 代理故障排查

| 现象 | 可能原因 | 检查方法 |
|------|---------|---------|
| `code tunnel` 启动后卡住不动 | 代理未配置或不通 | `curl -x http://127.0.0.1:7890 https://tunnelrelay.io` |
| SSL 证书错误 | Clash 或其他软件在做 TLS 拦截 | 在 Clash 规则中将 `*.tunnels.api.visualstudio.com` 设为直连 |
| 连接超时 | 代理端口不通或 Clash 未运行 | `netstat -an \| findstr "7890"` 确认端口监听 |
| 隧道建立后频繁断开 | Azure 中继延迟高 | 正常现象（国内→境外），`--no-sleep` 可缓解 |

---

## 七、持久化方案对比

| 方案 | 管理员权限 | 重启后自动启动 | 无人登录时运行 | 稳定性 |
|------|:---:|:---:|:---:|:---:|
| `code tunnel service install` | 🟡 需要 | ✅ 是 | ✅ 是 | ⭐⭐⭐⭐⭐ |
| `schtasks /SC ONLOGON` | 🟢 不需要 | ✅ 登录后自动 | ❌ 需有人登录 | ⭐⭐⭐⭐ |
| `schtasks /SC ONSTART` | 🟡 可能需要 | ✅ 开机自动 | 🟡 取决于权限 | ⭐⭐⭐⭐⭐ |
| `Start-Process` 手动后台 | 🟢 不需要 | ❌ 需手动 | ❌ 需手动 | ⭐⭐⭐ |
| Startup 文件夹快捷方式 | 🟢 不需要 | ✅ 登录后自动 | ❌ 需有人登录 | ⭐⭐⭐ |

**当前推荐**：如果 AIPatho1 能获得一次性管理员协助安装服务 → 方式 A；否则 → 方式 B（`schtasks /SC ONLOGON`），服务器长期保持 RDP 登录状态即可。

---

## 八、故障排查清单

### 8.1 隧道无法建立

```powershell
# 1. 检查 CLI 版本
code tunnel --version

# 2. 检查认证状态
# 查看保存的登录会话
ls C:\Users\AIPatho1\.vscode-cli\

# 3. 重新认证
code tunnel login --cli

# 4. 清除旧隧道会话
code tunnel kill
code tunnel prune

# 5. 检查代理
curl -x http://127.0.0.1:7890 https://global.rel.tunnels.api.visualstudio.com -v

# 6. 检查隧道日志
Get-Content C:\Users\AIPatho1\tools\vscode-cli\tunnel.log -Tail 50
```

### 8.2 本地 VS Code 看不到远程机器

| 排查项 | 检查方法 |
|--------|---------|
| 账号一致 | 服务器认证的账号 = VS Code 登录的账号？ |
| 远程隧道运行中 | 服务器上 `Get-Process code*` 确认进程在跑 |
| 网络可达 | 本地浏览器访问 `https://vscode.dev/tunnel/DESKTOP-HKP0IV0` 能否打开？ |
| Remote Tunnels 扩展 | VS Code 扩展是否已安装并启用？ |
| 账号登录状态 | VS Code 左下角账号图标 → 确认已登录 |

### 8.3 隧道频繁断开

```powershell
# 在启动命令中加入 --no-sleep（防休眠）
code tunnel --no-sleep --accept-server-license-terms

# 在 bat 启动脚本中加入自动重连循环
@"
@echo off
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
:loop
echo [%date% %time%] Starting code tunnel... >> C:\Users\AIPatho1\tools\vscode-cli\tunnel.log
C:\Users\AIPatho1\tools\vscode-cli\code.exe tunnel --accept-server-license-terms >> C:\Users\AIPatho1\tools\vscode-cli\tunnel.log 2>&1
echo [%date% %time%] Tunnel exited, restarting in 30s... >> C:\Users\AIPatho1\tools\vscode-cli\tunnel.log
timeout /t 30 /nobreak
goto loop
"@ | Out-File -FilePath C:\Users\AIPatho1\tools\vscode-cli\start-tunnel-loop.bat -Encoding ASCII
```

### 8.4 终端响应慢

这是正常现象：所有流量经 Azure 中继 + 国内到境外的延迟。预期延迟 200-400ms。
- 写代码/执行命令：完全可接受
- 大文件传输：建议走 Git（代码）或 RDP 文件复制（数据），不要在 tunnel 终端中传输大文件

---

## 九、安全说明

| 安全维度 | 评估 | 说明 |
|---------|:---:|------|
| 传输加密 | ✅ TLS 1.3 端到端 | 微软基础设施加密 |
| 认证 | ✅ OAuth 2.0 | GitHub / Microsoft 账号认证 |
| 授权 | ✅ 同账号绑定 | 只有登录相同账号的设备才能看到隧道 |
| 公网暴露 | ✅ 无 | 无入站端口，服务器不直接暴露于公网 |
| 数据路径 | ⚠️ 经 Azure 中继 | 代码/数据通过微软服务器中转（非端到端直连） |
| 账号安全 | ⚠️ 单点风险 | GitHub 账号被盗 → 获得服务器访问。**务必开启 2FA** |

**建议**：
1. 对 GitHub/Microsoft 账号开启**双因素认证（2FA）**
2. 定期在服务器上 `code tunnel prune` 清除旧会话
3. 敏感数据（患者信息等）确认合规后再通过 tunnel 传输

---

## 十、完整执行清单

### 首次部署（总计约 30 分钟）

- [ ] **P0** 前提条件检查（代理可达 + 磁盘空间 + 用户确认）
- [ ] **P0** 下载 VS Code CLI（`curl -x proxy ...` 或本地传文件）
- [ ] **P0** 认证（`code tunnel login --cli` → 本地浏览器输入设备码）
- [ ] **P0** 启动隧道并验证（`code tunnel --accept-server-license-terms`）
- [ ] **P0** 本地 VS Code 连接验证（`hostname` / `nvidia-smi` / `python -c "import torch"`）
- [ ] **P1** 配置持久化（`schtasks` 或 `service install`）
- [ ] **P1** 创建 bat 启动脚本（`start-tunnel-loop.bat` 含自动重连）
- [ ] **P1** 测试：关闭 tunnel 后重新连接（确认持久化生效）

### 生产就绪

- [ ] **P1** 验证 AI Agent 适配模式（模式 1：Tunnel VS Code 窗口中运行 Claude Code）
- [ ] **P1** 创建训练任务模板（`schtasks` 命令行模板，含完整 Python 命令）
- [ ] **P2** GitHub 账号开启 2FA
- [ ] **P2** Clash 规则优化（tunnel 域名直连）

### 日常使用

- [ ] 启动训练前：`schtasks /Create ...` → `schtasks /Run ...`
- [ ] 训练中轮询：`Get-Content log.txt -Tail 30` + `nvidia-smi`
- [ ] 训练完成：检查结果 → Git 同步代码 / RDP 回收大文件
- [ ] 隧道异常：RDP 登录 → 检查 `tunnel.log` → 手动重启

---

## 十一、附录：关键命令速查

### 服务器端

```powershell
# 隧道管理
code tunnel                              # 启动隧道（交互式）
code tunnel --no-sleep                   # 启动隧道（防休眠）
code tunnel login --cli                  # 设备码认证
code tunnel status                       # 查看隧道状态
code tunnel kill                         # 停止隧道
code tunnel prune                        # 清除旧会话
code tunnel service install              # 安装为 Windows 服务（需管理员）
code tunnel service uninstall            # 卸载服务
code tunnel service log                  # 查看服务日志

# 计划任务
schtasks /Create /TN "任务名" /TR "命令" /SC ONLOGON /F    # 创建登录时触发任务
schtasks /Query /TN "任务名" /FO LIST /V                    # 查看任务详情
schtasks /Run /TN "任务名"                                   # 手动触发
schtasks /Delete /TN "任务名" /F                             # 删除任务

# 代理测试
curl -x http://127.0.0.1:7890 https://tunnelrelay.io --connect-timeout 10 -I
curl -x http://127.0.0.1:7890 https://global.rel.tunnels.api.visualstudio.com -I
```

### 本地

```bash
# VS Code 命令面板
Remote-Tunnels: Connect to Tunnel...     # 连接到远程隧道
Remote-Tunnels: Disconnect from Tunnel   # 断开

# CLI (如果本地也装了 code CLI)
code tunnel exec -n DESKTOP-HKP0IV0 "命令"   # 远程执行命令
code tunnel list                              # 列出可用远程机器
```

---

## 更新历史

| 日期 | 变更 |
|------|------|
| 2026-06-17 | 初始版本：完整部署流程 + 代理配置 + 持久化方案 + AI Agent 适配 + 故障排查 |
