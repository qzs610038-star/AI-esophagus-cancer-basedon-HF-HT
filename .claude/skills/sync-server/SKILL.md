---
name: sync-server
description: Interactive guide for syncing code and results with the remote GPU server (manual sync via RDP/shared drive). Step-by-step checklist.
argument-hint: [--push|--pull|--status|--check]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Write
---

# Sync Server — 服务器同步向导

手动同步流程的交互式引导（RDP/共享目录方式）。

> SSH 不可用时的备用方案。SSH 通道建立后，优先使用 `deploy/push.sh` + `deploy/pull.sh`。

## 调用格式

```
/sync-server --push     # 推送代码/数据到服务器
/sync-server --pull     # 从服务器拉取训练结果
/sync-server --status   # 检查本地 vs 服务器同步状态
/sync-server --check    # 验证服务器环境就绪
```

## 服务器信息

| 项目 | 值 |
|------|-----|
| 项目目录 | `D:\AIPatho\qzs` |
| Python | `C:\ProgramData\miniconda3\python.exe` |
| venv | `C:\Users\AIPatho1\pfmval_env\Scripts\Activate.ps1` |
| GPU | RTX 4080 (16GB) |
| 访问方式 | RDP 远程桌面 |

## --push: 推送流程

### Step 1: 检测本地变更

```bash
# 列出最近修改的 .py 训练脚本
git diff --name-only HEAD -- "*.py"

# 列出新的 checkpoint（未在 experiments_log.csv 中）
# 对比本地和上次同步时的文件列表
```

### Step 2: 生成同步清单

自动生成需要同步的文件列表：
- `.py` 训练/评估脚本（代码变更）
- `config.yaml`（⚠️ 通常不覆盖服务器版！）
- 新模型权重 `.pth`（如有）
- 部署脚本 `deploy/*`

### Step 3: 打包

```bash
# 生成带时间戳的增量包
bash deploy/sync_list.sh
# 或手动打包:
tar -czf pfmval_sync_$(date +%Y%m%d_%H%M%S).tar.gz <文件列表>
```

### Step 4: 传输到服务器

方式（按优先级）：
1. RDP 驱动器重定向 → 复制到 `D:\AIPatho\qzs`
2. 共享目录 / U 盘
3. 百度网盘

### Step 5: 服务器端解压 & 验证

```powershell
# 在服务器 PowerShell 中执行：
cd D:\AIPatho\qzs
tar -xzf pfmval_sync_*.tar.gz

# 验证 Python 环境
C:\Users\AIPatho1\pfmval_env\Scripts\Activate.ps1
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 冒烟测试
$env:PYTHONIOENCODING = "utf-8"
python train_histogene_uni_tokens_augmix.py --patient HYZ15040 --epochs 1
```

## --pull: 拉取流程

### Step 1: 确认服务器端训练完成

检查服务器上 checkpoint 目录是否有新结果。

### Step 2: 打包结果

```powershell
# 在服务器上打包训练结果
cd D:\AIPatho\qzs
tar -czf results_$(Get-Date -Format 'yyyyMMdd_HHmmss').tar.gz checkpoints/*/results_vis/*/
```

### Step 3: 传输回本地

解压到本地对应目录，不覆盖已有文件。

### Step 4: 运行 post-train 流水线

```bash
/post-train --scan
```

## --status: 同步状态检查

```
本地最新 commit:  b2ff1da (before virchow2&oli training)
本地最新 checkpoint: CrossPatient_Fold1_..._20260530
服务器最新 checkpoint: 待确认（上次同步: 2026-05-29）
待同步: 2 个 .py 文件, 0 个 .pth 权重
```

## --check: 服务器环境验证

在服务器上运行以下命令并报告结果：

```powershell
# 1. Python 环境
C:\Users\AIPatho1\pfmval_env\Scripts\Activate.ps1
python --version

# 2. CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# 3. 关键目录
ls D:\AIPatho\qzs\config.yaml
ls D:\AIPatho\qzs\data_new_3ST\patch_noov_spilt\

# 4. 磁盘空间
Get-PSDrive D | Select-Object Used,Free
```

## 注意事项

- ⚠️ **不要覆盖服务器的 `config.yaml`**（路径不同）
- ⚠️ 服务器 `config.yaml` 中 `paths:` 和 `patients:` 节需要单独配置
- 大文件（特征缓存、预训练权重）优先在服务器端重新生成，不传输
- 首次同步使用 `deploy/pack_deploy.sh` 生成完整部署包