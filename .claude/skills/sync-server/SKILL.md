---
name: sync-server
description: Interactive guide for syncing code and results with the remote GPU server. Git is the primary channel (local SSH push, server HTTPS pull/push). RDP is fallback for gitignored large files only.
argument-hint: [--push|--pull|--status|--check]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Write
---

# Sync Server — 服务器同步向导

Git 双向同步 + RDP 大文件兜底。

> **2026-06-04 更新**：本地已切换为 SSH (`git@github.com:qzs610038-star/...`)，推送不再依赖代理。服务器通过 HTTPS + Clash 代理 (7897) 正常 pull/push。Git 已成为主通道，RDP 仅用于 gitignored 大文件。

## 调用格式

```
/sync-server --push     # 推送代码到服务器（本地 commit + push → 服务器 pull）
/sync-server --pull     # 从服务器拉取训练结果（服务器 push → 本地 pull + post-train）
/sync-server --status   # 检查本地 vs 远程 Git 同步状态
/sync-server --check    # 验证服务器环境就绪
```

## Git 架构

```
本地 (SSH) ──git push──→ GitHub ──git pull──→ 服务器 (HTTPS+代理)
本地 (SSH) ←──git pull── GitHub ←──git push── 服务器 (HTTPS+代理)
```

| 端 | 协议 | 代理 | 密钥 |
|----|------|------|------|
| 本地 → GitHub | SSH | 无（直连） | `~/.ssh/pfmval_server` |
| 服务器 → GitHub | HTTPS | Clash 7897 | user/pass (git credential) |

## 服务器信息

| 项目 | 值 |
|------|-----|
| IP / 用户 | 117.68.10.96 / AIPatho1 |
| 项目目录 | `D:\AIPatho\qzs\pfmval_deploy_git` |
| Python | `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe` |
| GPU | RTX 4080 (16GB), CUDA 12.9 |
| 代理 | Clash Verge :7897 |
| Git user | qzs610038-star / 3369651601@qq.com |
| 访问方式 | RDP 远程桌面 |

## --push: 同步代码到服务器

### 流程

```
本地 git add/commit/push (SSH) → 服务器 git pull (HTTPS)
```

### 本地端

```bash
# 常规 git 工作流
git add <changed_files>
git commit -m "feat: <描述>"
git push origin main
```

### 服务器端（训练前执行一次）

```powershell
cd D:\AIPatho\qzs\pfmval_deploy_git
git pull origin main
```

> ⚠️ `config.yaml` 在服务器有独立版本（路径不同），git pull 如有冲突，保留服务器版。

### 大文件（gitignored）

以下文件不走 Git，仍用 RDP 拖拽：
- `checkpoints/**/*.pth`（模型权重）
- `checkpoints/**/results_vis/**/predictions.csv`（大型预测 CSV）
- 新增 Python 依赖包（`pip install` 在服务器重装）

## --pull: 拉取训练结果

### 流程

```
服务器 git add/commit/push (HTTPS) → 本地 git pull (SSH) → /post-train
```

### 服务器端（训练完成后）

```powershell
cd D:\AIPatho\qzs\pfmval_deploy_git

# 提交训练结果（.pth 和 predictions.csv 已被 .gitignore 排除）
git add checkpoints/*/results_vis/
git add checkpoints/*/training_history.csv
git add checkpoints/*/per_pathway_pcc.csv
git add checkpoints/*/args.json
git commit -m "results: <模型>_<患者>_<PCC>"
git push origin main
```

### 本地端

```bash
cd "d:/AI空间转录病理研究/PFMval_new"
git pull origin main           # SSH，稳定不受代理影响
/post-train --latest           # 自动处理：指标提取 → 报告 → 排名 → 文档更新
```

### 大型 checkpoint 需要时

```powershell
# 服务器端：RDP 拖拽 checkpoints/<name>/*.pth 到本地对应目录
# 或打包后传输：
tar -czf checkpoint_<name>.tar.gz checkpoints/<name>/*.pth
```

## --status: 同步状态检查

```bash
# 本地 vs 远程 Git 状态
git log --oneline main...origin/main --left-right

# 等价：检查本地是否领先/落后
git status
```

示例输出：
```
本地领先远程: 0 commits | 落后远程: 3 commits (服务器有新结果)
最后同步: 2026-06-04 00:50
```

## --check: 服务器环境验证

```powershell
# 1. Git 连通性
cd D:\AIPatho\qzs\pfmval_deploy_git
git fetch origin
git status

# 2. Python 环境
C:\Users\AIPatho1\pfmval_env\Scripts\Activate.ps1
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# 3. 磁盘空间
Get-PSDrive D | Select-Object Used,Free

# 4. GPU 状态
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

## 注意事项

- ⚠️ **不要覆盖服务器的 `config.yaml`**（路径不同，服务器版有独立配置）
- ⚠️ 首次使用前，服务器需 `git pull` 同步最新历史（2026-06-04 本地做过 cherry-pick 历史修复）
- 大文件（特征缓存、预训练权重）优先在服务器端重新生成，不传输
- 推送故障时参考 `git-rescue` 技能（`.claude/skills/git-rescue/SKILL.md`）
- 服务器 git push 如果遇到 push protection，参照本地经验：token 已替换为 `<HF_TOKEN>`
