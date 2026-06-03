# SSH 远程操作 — 初学者指南

> **目标读者**：从未用过 SSH 的初学者。读完本文你将理解：SSH 是什么、本地电脑如何操控远程服务器、我们项目中遇到的 SSH 问题及解决思路。
> **预计阅读时间**：15 分钟。

---

## 一、SSH 是什么？

### 一句话

**SSH（Secure Shell）是一条加密隧道，让你在本地电脑的终端里操作远程服务器，就像坐在那台电脑面前一样。**

### 类比理解

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   你的笔记本电脑                远程服务器            │
│   ┌──────────┐                ┌──────────┐         │
│   │ 终端窗口  │ ═══ SSH 隧道 ═══│  真实系统  │        │
│   │ $ ls     │  ← 加密传输 →  │ 执行 ls   │         │
│   │ $ python │                │ 运行 python│        │
│   └──────────┘                └──────────┘         │
│                                                     │
│   你敲的命令在本地显示，但实际在服务器上执行             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

就像远程桌面（RDP），不过是**命令行版本**——更快、更轻量、可脚本化。

---

## 二、SSH 连接的三要素

每次 SSH 连接需要三个信息：

| 要素 | 含义 | 示例 | 类比 |
|------|------|------|------|
| **主机地址** | 服务器在哪 | `117.68.10.96` | 房子的地址 |
| **端口号** | 哪个门可以进 | `22`（默认） | 房子的哪个门 |
| **身份凭证** | 证明你有权进入 | 密码 或 SSH 密钥 | 钥匙 |

### 最基本的连接命令

```bash
# 密码方式（Windows PowerShell / Linux / Mac 终端均可）
ssh 用户名@服务器IP

# 实际例子
ssh AIPatho1@117.68.10.96
# 然后输入密码

# 指定端口（默认是 22，如果不是 22 需要 -p 参数）
ssh AIPatho1@117.68.10.96 -p 22330
```

---

## 三、密码 vs SSH 密钥

### 密码认证（最简单，安全性低）

```
你 → 输入密码 → 服务器验证 → 允许登录
```

- 优点：不需要任何配置
- 缺点：每次都要输密码、易被暴力破解、脚本无法自动化

### SSH 密钥认证（推荐，更安全且可自动化）

```
你本地生成一对密钥：
  🔑 私钥（自己保管，绝不外传）  →  id_rsa（本机 ~/.ssh/ 目录下）
  🔓 公钥（放到服务器上）        →  id_rsa.pub（复制到服务器的 authorized_keys）

连接时：
  服务器用公钥加密一段随机信息 → 你的私钥解密并返回 → 匹配则放行
```

**为什么项目要用密钥？**
- 自动化脚本（`deploy/run.sh`）无人值守，不可能每次输密码
- 更安全：私钥 2048 位加密，暴力破解几乎不可能

### 密钥设置步骤（一次性操作）

```bash
# 步骤 1：本地生成密钥（只需做一次）
ssh-keygen -t rsa -b 4096 -f ~/.ssh/pfmval_server
# 一路回车即可，生成 pfmval_server（私钥）和 pfmval_server.pub（公钥）

# 步骤 2：把公钥内容复制到服务器的 authorized_keys 文件
# 方式 A — 如果 ssh 密码登录可用：
ssh-copy-id -i ~/.ssh/pfmval_server.pub AIPatho1@117.68.10.96

# 方式 B — 手动复制：把 pfmval_server.pub 的内容粘贴到
# 服务器的 C:\Users\AIPatho1\.ssh\authorized_keys 文件中

# 步骤 3：本地配置免密连接（编辑 ~/.ssh/config）
Host pfmval_server
    HostName 117.68.10.96
    User AIPatho1
    Port 22330
    IdentityFile ~/.ssh/pfmval_server

# 步骤 4：一键连接
ssh pfmval_server
```

---

## 四、我们项目中 SSH 为什么失败了？

我们服务器上遇到的**真实情况**：

```
服务器有三个"门"：

  端口 22    →  被 WSL2/Ubuntu 的 SSH 占用（返回 "OpenSSH_9.6p1 Ubuntu"）
  端口 22330 →  Windows OpenSSH 在这里监听（但外网防火墙不开放）
  端口 8080  →  我们自建的 HTTP 命令服务（同样被防火墙挡了）
```

**根因链条**：

```
① 无管理员权限
   ↓
② 无法查看 SSH 日志 → 不知道为什么密钥被拒
   ↓
③ 无法修改 sshd_config → 无法调整监听端口
   ↓
④ 无法添加防火墙规则 → 任何入站端口都被封
   ↓
⑤ 即使我们自己写了 cmd_server.py 在 8080 端口跑起来了
   外网也无法访问
```

这就是为什么我们在组会上申请**开放一个入站端口**或**安装 Tailscale**——只需一个出口，整个自动化管线就通了。

---

## 五、SSH 的高级用法（自动化必备）

### 5.1 远程执行命令（不登录）

```bash
# 在服务器上执行一条命令，结果返回本地
ssh pfmval_server "python train.py --epochs 1"

# 这就是 deploy/run.sh 的核心原理
```

### 5.2 文件传输

```bash
# 本地 → 服务器（上传）
scp my_code.py pfmval_server:D:/AIPatho/qzs/

# 服务器 → 本地（下载结果）
scp -r pfmval_server:D:/AIPatho/qzs/results/ ./

# deploy/push.sh 和 deploy/pull.sh 的底层机制
```

### 5.3 端口转发（反向隧道）

当服务器**没有公网端口**但可以**出站连接**时：

```bash
# 在服务器上执行（把本地 8080 暴露到公网）
ssh -R 0:localhost:8080 serveo.net
# 输出: Forwarding access to https://xxxx.serveo.net
# 然后你在本地访问 https://xxxx.serveo.net 就能调用服务器的 8080 服务

# 这就是我们尝试 bore/cloudflared/serveo 的原理
```

### 5.4 保持连接不中断

```bash
# 训练可能跑几个小时，SSH 断开训练就会被杀
# 解决方案：后台运行

# 方式 1：nohup（Linux）
ssh pfmval_server "nohup python train.py > train.log 2>&1 &"

# 方式 2：screen/tmux（推荐）
ssh pfmval_server
screen -S training        # 创建名为 training 的会话
python train.py           # 启动训练
# 按 Ctrl+A 然后按 D 退出（训练继续跑）
# 下次登录：screen -r training 恢复查看

# 方式 3：Windows 服务器（我们的情况）
# 用 PowerShell Start-Process 后台运行
ssh pfmval_server 'Start-Process python -ArgumentList "train.py --epochs 150" -NoNewWindow'
```

---

## 六、SSH 不可用时的替代方案

当 SSH 完全不可用时（我们当前的情况），有序选择：

```
优先级 1: Tailscale VPN → 零端口开放，Mesh VPN 穿透防火墙
优先级 2: Cloudflare Tunnel → 服务器出站连接 Cloudflare 建立隧道
优先级 3: frp 反向代理 → 自建中转服务器
优先级 4: HTTP 命令服务 + 开放端口 → 自写的 cmd_server.py
优先级 5: RDP 手动操作 → 当前状态，但效率极低
```

---

## 七、关键概念速查

| 概念 | 一句话解释 |
|------|-----------|
| SSH | 加密的远程命令行连接 |
| 端口 | 服务器上不同服务的"门牌号"（22=SSH, 80=HTTP, 443=HTTPS） |
| 公钥/私钥 | 一对匹配的密钥，公钥放服务器，私钥自己持有 |
| authorized_keys | 服务器上存放允许登录的公钥列表 |
| sshd | SSH 服务端守护进程（SSH Daemon） |
| 入站规则 | 防火墙允许外部连接到本机某端口的规则 |
| 反向隧道 | 服务器主动出站建立连接，让外部通过中转访问内部服务 |
| scp | 基于 SSH 的文件传输命令 |
| known_hosts | 本地记录"信任哪些服务器"的文件，防止中间人攻击 |

---

## 八、我们的项目自动化蓝图

```
                        SSH / Tailscale / HTTP
  本地电脑  ════════════════════════════════════  服务器 (RTX 4080)
  ┌──────────┐           加密通道               ┌──────────────────┐
  │          │                                  │                  │
  │ deploy/  │ ── push.sh ──→ 代码同步           │  D:\AIPatho\qzs\ │
  │          │                                  │                  │
  │          │ ── cmd_client ─→ 启动训练         │  python train.py │
  │          │                                  │                  │
  │          │ ←── pull.sh ── 回收结果           │  checkpoints/    │
  │          │                                  │  predictions.csv │
  └──────────┘                                  └──────────────────┘

  当前状态：通道待打通（等待管理员开放端口或安装 Tailscale）
  其余一切就绪：代码 ✓  环境 ✓  数据 ✓  脚本 ✓
```

---

> 📎 相关文档：[服务器部署问题提问清单](../../02_组会汇报/服务器部署问题提问清单_20260530.md) | [服务器迁移完整策略](../../.claude/../../memory/project_server_migration_strategy.md) | [服务器远程连接配置](SSH远程操作_初学者指南.md)
