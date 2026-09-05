# PFMval tsnet 用户态 SSH 通道部署方案（v0.1，2026-08-30）

> 状态：`pending_user_review`
>
> 方案类型：服务器传输与运维通道变更讨论稿，非实验部署方案。
>
> 当前停止点：仅形成待审方案；未创建代码、未构建程序、未连接服务器、未修改 Tailnet（Tailscale 私有网络）策略，也未改变项目 `gitee_only` 状态。

## 一、目标与结论

### 1.1 目标

在不修改实验室 NAT（Network Address Translation，网络地址转换）、不安装 Windows 虚拟网卡、不安装系统服务、无需 PFMval 服务器管理员权限的前提下，为本机到 PFMval Windows 服务器建立一个仅 Tailnet 内可达的标准 SSH 通道。

预期链路：

```text
本机 OpenSSH Client
  -> 本机现有 Tailscale 节点
  -> PFMval tsnet 用户态节点 tcp/22
  -> Windows 服务器 127.0.0.1:22
  -> 现有 Windows OpenSSH Server (sshd)
```

`tsnet` 是将 Tailscale 节点嵌入 Go 程序的用户态网络库。它不创建操作系统级 Tailscale 网卡，不要求系统级 daemon（后台服务），也不需要管理员权限。该程序只负责透明转发 TCP 字节；Windows `sshd` 继续负责 SSH 加密、主机身份与用户公钥认证。

### 1.2 推荐结论

推荐先开展一次**前台、瞬时、零训练、零文件传输**的用户态 SSH 概念验证。只有在身份、访问控制、断开回退和稳定性全部通过后，才讨论用户级登录自启及直接文件传输。

本方案不建议：

- 在 PFMval Windows 上尝试非标准便携运行完整 `tailscaled`；
- 启用 Tailscale Funnel（公网暴露）；
- 用 tsnet 替换 Windows `sshd` 或改用 Tailscale SSH；
- 初次试点即开放 SCP、任意命令、训练调度或结果回传；
- 在未修订治理状态前绕过 `gitee_only` 执行。

## 二、当前事实、假设与缺口

### 2.1 已确认事实

1. 当前机器事实源中的服务器传输模式为 `gitee_only`。
2. 当前禁止通道包含 SSH、SCP、HTTP remote command（HTTP 远程命令）和 Remote Tunnel（远程隧道）。
3. 历史排障记录显示目标 Windows 服务器本机 `localhost:22` 可用，而旧公网地址的 TCP 22 被 NAT 映射到另一台机器。
4. 本机已有运行中的 Tailscale Windows 服务；Robot arm&dog 项目的 Ubuntu 节点已通过“Tailscale + 普通 OpenSSH”实测可用。
5. PJ2 跳板机路径已由用户排除。

### 2.2 待现场确认假设

以下内容不是当前服务器机器事实，执行前必须通过 RDP（Remote Desktop Protocol，远程桌面协议）最小只读核验：

- 当前登录用户仍为 `AIPatho1`；
- 当前 Windows `sshd` 服务仍运行；
- `127.0.0.1:22` 仍可达；
- 服务器能出站访问 Tailscale 控制面与 DERP（Designated Encrypted Relay for Packets，加密中继）服务；
- `%LOCALAPPDATA%` 具备至少 150 MB 可用空间；
- Tailnet 所有者可创建一次性认证凭据并修改 Grants（授权规则）。

### 2.3 当前硬阻塞

`DIR-20260730-003` 与 `project_state/current_state.json` 仍把 SSH/Tunnel 列为禁止项。在形成新的、范围更窄的用户指令并通过本地治理预检前，本方案不得进入服务器试点。

## 三、范围与非目标

### 3.1 v0.1 试点范围

- 单个用户态程序：`pfmval-tsnet-ssh.exe`；
- 单个 Tailnet 节点名：建议 `pfmval-tsnet-ssh`；
- 单个虚拟监听端口：Tailnet TCP 22；
- 单个上游：固定 `127.0.0.1:22`；
- 单个来源身份：用户指定的本机 Tailscale 身份；
- 单次前台运行；
- 验证命令仅允许 `hostname`、`whoami` 与版本查询；
- 不传输文件，不训练，不调度作业，不导入或接纳结果。

### 3.2 非目标

- 不替换 Git/Gitee 作为代码版本事实源；
- 不在 v0.1 中重构 experiment/job/result 协议；
- 不改变受保护资产、数据集、缓存或 checkpoint；
- 不建立公网入口；
- 不配置 Windows 服务或开机无人登录自启；
- 不删除任何旧 SSH 密钥、历史配置、二进制或节点状态。

## 四、目标实现设计

### 4.1 最小程序边界

建议源码目录（批准后创建）：

```text
tools/tsnet_ssh_forwarder/
  go.mod
  go.sum
  main.go
  main_test.go
  README.md
```

程序只实现以下能力：

1. 启动一个固定名称的 `tsnet.Server`；
2. 使用独立状态目录保存节点身份，并只声明 `tag:pfmval-ssh`；
3. 在 tsnet 用户态网络中监听 `tcp/:22`；
4. 每个连接只拨号到 `127.0.0.1:22`；
5. 双向复制 TCP 字节并在任一方向结束后关闭连接；
6. 输出启动、认证、连接数和错误类别，不记录 SSH 内容、密钥、Token（令牌）或患者数据；
7. 支持 `--version`，显示构建对应的本地 Git commit；
8. 响应 Ctrl+C 正常停止。

明确不实现：

- 任意目标地址或任意端口参数；
- shell/命令执行接口；
- HTTP API；
- 文件浏览与上传接口；
- Funnel；
- 自动修改 Tailnet 策略、Windows 防火墙、注册表或任务计划；
- 自动删除状态。

### 4.2 路径提案

服务器路径在现场确认并获得用户批准前仅为提案，不写入 `configs/server_paths.yaml`：

```text
%LOCALAPPDATA%\PFMvalTsnetSSH\
  bin\pfmval-tsnet-ssh.exe
  state\
  logs\pfmval-tsnet-ssh.log
```

本地只跟踪源码，不把构建后的 `.exe` 提交到 Git。构建产物进入独立临时 staging 目录，保留文件名、大小、构建命令、Go 版本与 Git commit 作为来源记录。

依据用户规则，本方案不新增项目哈希治理。若后续审查认为二进制完整性必须使用 SHA-256，须另行说明必要性、影响并获得用户批准后再启用。

### 4.3 身份与密钥

1. tsnet 节点使用独立身份 `pfmval-tsnet-ssh`，不复用 Robot Ubuntu 节点身份。
2. 首次注册优先使用一次性、短时、预授权的 Tailscale auth key（认证密钥），并绑定 `tag:pfmval-ssh`。
3. auth key 仅通过临时环境变量传入；不得写入仓库、配置文件、日志或对话。
4. 节点注册成功并形成 `state/` 后立即撤销或等待一次性密钥失效；后续启动复用节点状态，不再需要 auth key。
5. SSH 用户认证继续使用 Windows `authorized_keys`，tsnet 程序不得读取 SSH 私钥或公钥正文。

### 4.4 Tailnet 最小权限

在生产验收前，Tailnet policy（策略文件）必须满足：

- `tagOwners` 仅允许 Tailnet Owner（所有者）管理 `tag:pfmval-ssh`；
- `src` 仅允许用户确认的本机身份；
- `dst` 仅允许 `tag:pfmval-ssh`；
- `ip` 仅允许 `tcp:22`；
- 不允许 `*` 访问全部端口；
- 不允许 Funnel 或公开分享；
- 使用 Tailscale 官方 Grants（授权）语法，并在管理界面保存前通过内置校验。

策略示意仅供审核，执行时必须替换来源身份占位符并由用户确认：

```json
{
  "tagOwners": {
    "tag:pfmval-ssh": ["autogroup:owner"]
  },
  "grants": [
    {
      "src": ["<用户确认的本机Tailscale身份>"],
      "dst": ["tag:pfmval-ssh"],
      "ip": ["tcp:22"]
    }
  ]
}
```

若现有 Tailnet policy 已包含更宽规则，新增窄规则不会覆盖旧规则；必须先审查规则并集，确认不存在其它来源通过旧规则访问该节点。

## 五、治理变更设计

### GOV1—治理授权（Governance Authorization）

**依赖：无｜预计：30–60 分钟｜失败概率：中**

在任何服务器操作前，先形成待用户审核的规范化指令预览。该指令至少应：

1. 对 `DIR-20260730-003` 做窄化补充，而不是删除历史指令；
2. 仅批准 tsnet 前台试点、Tailnet TCP 22 和三条只读身份命令；
3. 明确禁止 SCP、训练、任意远程命令、结果 import/accept、Funnel 与持久化；
4. 单独批准一次精确的 RDP 二进制引导传输，或指定仍通过 Gitee 完成一次引导；
5. 规定失败立即停止，不自动扩大权限；
6. 规定试点结束后由用户决定接受、延期、回退或扩大范围。

只有用户明确批准预览后，才可通过 `deploy/pfmval_ops.py state record-directive` 追加指令。不得手工编辑 `directives.jsonl` 或 `current_state.json`。

在 v0.1 试点通过前，`server_transport.mode` 仍保持 `gitee_only`；新增试点应作为受限例外记录，不提前宣称正式通道已迁移。

### ENV2—环境预检（Environment Preflight）

**依赖：GOV1｜预计：30 分钟｜环境：本地 PowerShell + 服务器 RDP｜磁盘：<150 MB｜失败概率：中**

本地只读检查：

```powershell
python deploy/pfmval_ops.py agent start-check --strict
go version
git status --short
D:\tailscale.exe status
```

服务器 RDP 只读检查（批准后由用户或获授权 Agent 执行）：

```powershell
whoami
Get-Service -Name sshd
Test-NetConnection -ComputerName 127.0.0.1 -Port 22
Get-PSDrive -Name C
```

通过条件：

- 本地严格门禁 `FAIL=0`；
- 本地 Go 工具链满足所选 Tailscale module（模块）`go.mod` 要求；
- 服务器无需提权即可运行普通用户程序；
- `sshd` 为 Running；
- `127.0.0.1:22` 为 True；
- 用户目录可用空间不少于 150 MB。

### DEV3—本地开发（Local Development）

**依赖：ENV2｜预计：90 分钟｜环境：本地 Go 工具链｜失败概率：低**

批准后在独立维护分支/工作树中创建最小源码。建议分支：

```text
codex/tsnet-ssh-pilot-20260830
```

开始构建前必须确认该维护工作树干净；不得把当前主工作区内尚未提交的用户改动混入试点 commit。

实现原则：

- 固定监听 `:22`、固定上游 `127.0.0.1:22`；
- 状态目录由用户级路径参数指定，默认不触碰系统目录；
- 不提供通用代理功能；
- 不引入 Web UI、更新器、守护进程管理或多余重试框架；
- 依赖版本固定进入 `go.mod` / `go.sum`；
- 版本输出嵌入本地 Git commit。

### TST4—最小本地测试（Minimal Local Test）

**依赖：DEV3｜预计：30 分钟｜环境：本地 Go 工具链｜失败概率：低**

仅开展最简必要测试：

1. `go test ./...`：使用本地 echo server 验证双向 TCP 复制；
2. `go vet ./...`：发现明显 Go API/格式问题；
3. `go build`：生成 Windows amd64 单文件；
4. `pfmval-tsnet-ssh.exe --version`：核对 commit 与版本；
5. 不连接真实 Tailnet，不连接服务器，不模拟训练。

建议命令：

```powershell
Set-Location tools\tsnet_ssh_forwarder
go test ./...
go vet ./...
New-Item -ItemType Directory -Force -Path $env:TEMP\pfmval-tsnet-build | Out-Null
go build -trimpath -buildvcs=true -o $env:TEMP\pfmval-tsnet-build\pfmval-tsnet-ssh.exe .
& $env:TEMP\pfmval-tsnet-build\pfmval-tsnet-ssh.exe --version
```

### SEC5—安全预审（Security Review）

**依赖：TST4｜预计：30 分钟｜失败概率：中**

审核清单：

- 搜索确认不存在 Funnel、HTTP listener、任意上游参数或 shell 执行；
- 日志不打印 auth key、节点私钥、SSH 流量和环境变量；
- `state/` 不进入 Git；
- Tailnet policy 通过官方校验；
- 现有宽 Grants/ACL 不会旁路最小规则；
- 二进制来源记录包含 Git commit、Go 版本、依赖清单和文件大小；
- 不新增哈希校验，除非用户另行批准。

### PKG6—构建与引导包（Build and Bootstrap Package）

**依赖：SEC5｜预计：30 分钟｜失败概率：低**

引导包只包含：

```text
pfmval-tsnet-ssh.exe
README-BOOTSTRAP.txt
```

`README-BOOTSTRAP.txt` 记录版本、Git commit、文件大小、启动和停止命令，不包含凭据。

引导传输二选一，必须在 GOV1 中明确：

- **推荐**：用户通过现有 RDP 精确复制这两个文件；
- **兼容现行边界**：仍通过一次 Gitee 受管分支传输，但不得长期跟踪 `.exe`。

未获批准不得自行选择第三种传输方式。

### DEP7—服务器前台部署（Foreground Deployment）

**依赖：PKG6 + Tailnet policy 已确认｜预计：30 分钟｜环境：服务器普通用户 PowerShell｜失败概率：中**

只在用户级目录创建 `bin/`、`state/`、`logs/`。首次启动使用一次性 auth key；不得把密钥写入命令历史、脚本或日志。建议由用户在 RDP 交互窗口中临时注入，程序完成注册后立即清除环境变量。

第一轮只前台运行：

```powershell
& "$env:LOCALAPPDATA\PFMvalTsnetSSH\bin\pfmval-tsnet-ssh.exe" `
  --state-dir "$env:LOCALAPPDATA\PFMvalTsnetSSH\state"
```

启动后必须核对：

- 无 UAC（User Account Control，用户账户控制）提权提示；
- Tailnet 显示节点名 `pfmval-tsnet-ssh`；
- 节点具有预期 tag；
- 日志不含 auth key；
- 未创建 Windows 服务、虚拟网卡、防火墙规则或计划任务。

### E2E8—端到端验收（End-to-End Verification）

**依赖：DEP7｜预计：30 分钟｜环境：本机 OpenSSH｜失败概率：中**

验收分三层，任一层失败即停止：

1. **节点层**：本机 Tailscale 能看到 `pfmval-tsnet-ssh`；
2. **TCP 层**：目标 Tailnet 地址 TCP 22 可达；
3. **SSH 层**：仅执行身份命令。

```powershell
ssh -o BatchMode=yes `
    -o IdentitiesOnly=yes `
    -o ConnectTimeout=15 `
    -i $env:USERPROFILE\.ssh\pfmval_server_nopass `
    AIPatho1@<执行时确认的tsnet节点名或100.x地址> `
    "hostname; whoami"
```

验收标准：

- 返回目标 Windows 主机名与 `AIPatho1`；
- 不出现密码提示；
- 不使用旧公网 IP；
- Windows OpenSSH Host Key 与服务器本机已确认指纹一致；
- 连续两次新建连接成功；
- 关闭前台程序后，Tailnet TCP 22 立即不可达；
- 旧 RDP 与 Gitee 通道不受影响。

本阶段禁止 `scp`、目录遍历、Git 操作和训练命令。

### OBS9—观察与判定（Observation and Decision）

**依赖：E2E8｜预计：24 小时被动观察 + 15 分钟复测｜失败概率：中**

保持前台试点期间不执行科研任务，只记录：

- 节点在线/离线状态；
- Tailscale 直连或 DERP 中继类型；
- 两次 SSH 建连延迟；
- 程序异常退出次数；
- 日志大小与是否含敏感信息；
- RDP、Gitee 和训练环境是否出现副作用。

判定：

- **GO**：E2E 全通过、无公网暴露、无提权、无副作用；
- **CONDITIONAL GO**：功能通过但仅 DERP、中继延迟较高或需人工登录启动；
- **NO-GO**：身份不匹配、访问控制旁路、日志泄密、程序异常、影响既有通道或需管理员权限。

科学性能不在本次判定范围；本试点不得产生实验结论。

### PER10—持久化二次审批（Persistence Approval）

**依赖：OBS9=GO/CONDITIONAL GO + 用户再次批准｜预计：30 分钟｜失败概率：中**

若用户批准持久化，只允许创建当前用户的 `ONLOGON` 任务；不创建系统服务：

```powershell
schtasks /Create `
  /TN "PFMval tsnet SSH" `
  /TR "<经现场确认的完整exe与参数>" `
  /SC ONLOGON `
  /F
```

限制：服务器重启后，在 `AIPatho1` 登录前通道不可用。这是无管理员方案的明确边界。任务创建前必须预览完整 `/TR` 内容，避免路径或引号错误。

### TRN11—传输能力扩展（Transport Expansion）

**依赖：PER10 稳定 + 独立用户批准｜预计：2–4 小时治理设计｜失败概率：高**

只有前述阶段通过后，才讨论减少 Gitee 往返。扩展必须分级：

1. **L1 只读诊断**：允许 allowlisted 命令查询状态；
2. **L2 小型文件**：允许精确路径的配置、日志和诊断包通过 SCP；
3. **L3 作业调度**：允许绑定 `job_id`、`source_commit` 和批准文件的固定入口；
4. **L4 结果回传**：允许先进入本地 inbox，再复用现有 result import/accept 验证；
5. **L5 代码传输**：仍以 Git commit 为版本事实源，是否把 Gitee 从传输介质降为备份需另行设计。

任何扩展不得因 SSH 可用而绕过：

- 正式训练批准；
- W### 实验工作树绑定；
- `source_commit` / `job_id`；
- result inbox/import/accept；
- 原始预测回传；
- 受保护资产与防泄漏边界；
- 本地单写者规则。

### RBK12—回退（Rollback）

**触发：任一步骤失败或用户要求停止｜预计：5–20 分钟｜失败概率：低**

回退顺序：

1. 关闭前台 `pfmval-tsnet-ssh.exe`；
2. 验证 Tailnet TCP 22 已不可达；
3. 在 Tailnet 管理界面撤销节点或其授权；
4. 保留服务器 `bin/`、`state/`、`logs/` 供审查，不自动删除；
5. 若已创建 ONLOGON 任务，用户批准后再删除精确任务；
6. 通过 append-only 状态更新将试点标记为 `failed`、`cancelled` 或 `superseded`；
7. `server_transport.mode` 保持或恢复为 `gitee_only`。

任何文件删除、任务删除或节点移除均需回显精确目标并按用户批准执行。

## 六、依赖图

```text
GOV1 治理授权
  -> ENV2 环境预检
  -> DEV3 本地开发
  -> TST4 最小本地测试
  -> SEC5 安全预审
  -> PKG6 构建与引导包
  -> DEP7 服务器前台部署
  -> E2E8 端到端验收
  -> OBS9 观察与判定
       |-> PER10 持久化二次审批
       |     -> TRN11 传输能力扩展
       \-> RBK12 回退（任一步骤均可进入）
```

## 七、任务总表

| 步骤 | 主要产物 | 依赖 | 预计时间 | 执行环境 | 服务器写入 |
|---|---|---|---:|---|---|
| GOV1 | 指令预览与批准记录 | — | 30–60 分钟 | 本地 | 否 |
| ENV2 | 环境预检记录 | GOV1 | 30 分钟 | 本地 + RDP | 否 |
| DEV3 | 最小 Go 源码 | ENV2 | 90 分钟 | 本地 | 否 |
| TST4 | test/vet/build/version 输出 | DEV3 | 30 分钟 | 本地 | 否 |
| SEC5 | 安全审查清单 | TST4 | 30 分钟 | 本地 + Tailnet 管理页 | 仅策略待用户操作 |
| PKG6 | exe + bootstrap 说明 | SEC5 | 30 分钟 | 本地 | 否 |
| DEP7 | 前台 tsnet 节点 | PKG6 | 30 分钟 | 服务器普通用户 | 是，用户目录 |
| E2E8 | 身份验收记录 | DEP7 | 30 分钟 | 本机 | 否 |
| OBS9 | 观察记录与 GO 判定 | E2E8 | 24 小时观察 | 两端 | 仅日志 |
| PER10 | 用户级 ONLOGON 任务 | OBS9 + 二次批准 | 30 分钟 | 服务器普通用户 | 是 |
| TRN11 | 分级传输治理方案 | PER10 + 独立批准 | 2–4 小时 | 本地治理 | 待定 |
| RBK12 | 停止与回退记录 | 任一步骤 | 5–20 分钟 | 两端 | 默认不删文件 |

## 八、风险矩阵

| 风险 | 概率 | 影响 | 发现方式 | 缓解/停止条件 |
|---|:---:|:---:|---|---|
| 当前治理仍禁止 SSH/Tunnel | 高 | HARD 阻塞 | strict gate + directive 检查 | GOV1 未获批即停止 |
| Tailnet 现有宽规则旁路最小 Grant | 中 | 未授权设备可访问 | 策略并集审查 | 未消除旁路即 NO-GO |
| 服务器无法访问 Tailscale 控制面/DERP | 中 | 节点无法注册或离线 | 前台日志 | 不改系统代理；记录后停止 |
| tsnet 依赖版本与本机 Go 不兼容 | 中 | 无法构建 | `go test/build` | 固定兼容版本；不在服务器装 Go |
| Windows localhost SSH 公钥认证仍失败 | 中 | TCP 通但无法登录 | BatchMode SSH | 单独修复 SSH 认证，不扩大 tsnet 权限 |
| 仅能走 DERP，中继延迟较高 | 中 | 交互体验下降 | Tailscale 连接状态 | CONDITIONAL GO，由用户决定 |
| 前台程序退出或用户注销 | 高 | 通道中断 | 观察日志 | 接受试点边界；持久化需 PER10 |
| auth key 泄露 | 低/中 | 未授权节点注册 | 日志/历史检查 | 一次性短时 key、临时环境变量、立即撤销 |
| 日志包含敏感信息 | 低 | 安全事件 | SEC5 + 日志抽查 | 仅元数据日志；发现即 NO-GO |
| RDP 引导复制违反当前通道边界 | 高 | 治理失败 | GOV1 审核 | 必须在新指令中精确授权，否则仍用 Gitee |
| 直接 SSH 被误用来绕过实验治理 | 中 | 证据链破坏 | 命令/结果审计 | v0.1 仅身份命令；TRN11 独立审批 |

## 九、验收证据与记录位置

方案获批后，建议记录：

- 本地源码与测试：`tools/tsnet_ssh_forwarder/`；
- 治理批准：`project_state/directives.jsonl` 的新指令及状态更新；
- 试点诊断记录：`automation/diagnostics/diagnostic-<date>-tsnet-ssh-pilot-r001/`；
- 安全/验收回执：`project_state/governance/PFMval_tsnet_ssh_pilot_receipt_<date>.md`；
- 服务器非证据日志：`%LOCALAPPDATA%\PFMvalTsnetSSH\logs\`；
- 若路径正式启用，再通过受管流程更新 `configs/server_paths.yaml`。

试点记录不得进入 experiment Registry，也不得生成 accepted scientific evidence（已接纳科学证据）。

## 十、用户审核决策点

本方案需要按顺序获得四次独立决定：

1. **D1 方案定版**：批准、修改或否决本 v0.1；
2. **D2 试点授权**：批准 GOV1 的窄化指令及一次引导传输方式；
3. **D3 持久化授权**：OBS9 通过后决定是否创建用户级 ONLOGON 任务；
4. **D4 通道扩展授权**：决定是否开放 SCP、诊断、作业调度和结果回传。

D1 不等于 D2；D2 不等于训练授权；任何一步未明确批准均停在当前阶段。

## 十一、参考来源

- Tailscale `tsnet` 概览：<https://tailscale.com/docs/features/tsnet>
- Tailscale `tsnet.Server` API：<https://tailscale.com/docs/reference/tsnet-server-api>
- Tailscale Grants 语法：<https://tailscale.com/docs/reference/syntax/grants>
- Tailscale 设备连接与 DERP：<https://tailscale.com/docs/reference/device-connectivity>
- Windows OpenSSH 配置：<https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration>
- 项目历史 SSH 根因：`01_指南与解读/部署方案/SSH远程连接排障与部署方案_20260616.md`
- 当前服务器传输事实：`project_state/current_state.json`
- 当前服务器路径事实源：`configs/server_paths.yaml`

## 十二、当前审批状态

- [ ] D1 用户批准本部署方案定版
- [ ] D2 用户批准窄化 tsnet SSH 前台试点与引导传输
- [ ] GOV1 新指令已通过受管 CLI 追加
- [ ] ENV2 环境预检完成
- [ ] DEV3–SEC5 本地实现与审核完成
- [ ] DEP7–E2E8 前台试点完成
- [ ] OBS9 观察完成
- [ ] D3 用户批准或否决持久化
- [ ] D4 用户批准或否决传输能力扩展

在 D1 与 D2 明确完成前，本文件不得作为服务器执行授权。
