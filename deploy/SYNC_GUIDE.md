# PFMval 手动同步操作指南

> 网络不通时的替代方案：本地开发 + 手动复制代码到服务器训练 + 回收结果。

---

## 快速参考

| 方向 | 传输内容 | 方法 |
|------|----------|------|
| 本地 → 服务器 | `.py` 脚本、`config.yaml`（手动合并路径） | U盘 / 共享文件夹 / IM |
| 服务器 → 本地 | `.pth` 权重、`results_vis/`、`predictions*.csv` | 同上 |

---

## 一、服务器环境确认

首次在服务器上操作时，确认以下环境就绪：

```powershell
# 1. Python 环境
"C:\ProgramData\miniconda3\python.exe" --version

# 2. PyTorch + CUDA
"C:\ProgramData\miniconda3\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期: True, NVIDIA GeForce RTX 4080

# 3. 项目目录存在
dir D:\AIPatho\qzs\train_histogene_uni_tokens_augmix.py
```

---

## 二、代码同步（本地 → 服务器）

> **🚫 警告：绝对禁止在服务器上执行 `git clean -fd`！**
> 2026-06-04 已因此丢失 lora_r4 Fold1 + Fold3 等训练结果。`git clean` 会删除 checkpoints/ 下未 commit 的结果文件。
> 正确的服务器同步命令见下方。

### Git 方式（推荐，增量同步）

**本地**（双推：Gitee 给服务器用，GitHub 备份）：
```bash
git push gitee main     # 服务器通过此 remote 拉取
git push origin main    # GitHub 备份
```

**服务器**：
```powershell
# ✅ 安全同步（仅这两条，不包含 git clean）
git fetch gitee main --force
git reset --hard gitee/main

# ❌ 绝对不要加 git clean -fd！会删除 checkpoints/ 下训练结果
```

### 手动方式（网络不通时的替代）

```bash
# 1. 生成同步文件清单
cd d:\AI空间转录病理研究\PFMval_new
bash deploy/sync_list.sh

# 2. 将清单中的文件复制到 U 盘或共享目录
# 注意：排除 config.yaml（除非服务器路径也有更新）

# 3. 标记同步基准点
bash deploy/sync_list.sh --mark
```

### 服务器操作

```powershell
# 1. 将文件从 U 盘复制到项目目录，覆盖同名文件
Copy-Item -Path "U:\sync_files\*" -Destination "D:\AIPatho\qzs\" -Force

# 2. 确认文件到位
dir D:\AIPatho\qzs\train_*.py
```

---

## 三、训练执行（服务器）

### 单 Fold 训练

```powershell
cd D:\AIPatho\qzs
$env:PYTHONIOENCODING = "utf-8"

# 冒烟测试（1 epoch）
& "C:\ProgramData\miniconda3\python.exe" train_histogene_uni_tokens_augmix.py --cross_patient --fold 1 --epochs 1

# 完整训练
& "C:\ProgramData\miniconda3\python.exe" train_histogene_uni_tokens_augmix.py --cross_patient --fold 1 --epochs 150 --tv_weight 0.01 --tv_mode l2
```

### 三折 CV（三个窗口同时跑）

```powershell
# 窗口 1 — Fold 1 (JFX+LMZ→HYZ)
$env:PYTHONIOENCODING = "utf-8"
& "C:\ProgramData\miniconda3\python.exe" train_histogene_uni_tokens_augmix.py --cross_patient --fold 1 --epochs 150 --tv_weight 0.01 --tv_mode l2

# 窗口 2 — Fold 2 (HYZ+LMZ→JFX)
$env:PYTHONIOENCODING = "utf-8"
& "C:\ProgramData\miniconda3\python.exe" train_histogene_uni_tokens_augmix.py --cross_patient --fold 2 --epochs 150 --tv_weight 0.01 --tv_mode l2

# 窗口 3 — Fold 3 (HYZ+JFX→LMZ)
$env:PYTHONIOENCODING = "utf-8"
& "C:\ProgramData\miniconda3\python.exe" train_histogene_uni_tokens_augmix.py --cross_patient --fold 3 --epochs 150 --tv_weight 0.01 --tv_mode l2
```

> 三个 Fold 互不依赖，可同时跑。16GB VRAM 同时跑 2 个轻量模型可行；AugMix+TV 约需 4-6GB/fold，三个同时跑需 ~12-18GB，建议先试 2 个并行，确认不 OOM 再加第三个。

---

## 四、结果回收（服务器 → 本地）

### 服务器打包

```powershell
cd D:\AIPatho\qzs

# 打包本次训练的权重和结果
# 找到最新生成的 results_vis 子目录
dir checkpoints\results_vis\ | Sort-Object LastWriteTime -Descending | Select-Object -First 10

# 打包到 U 盘
Compress-Archive -Path @(
    "checkpoints\results_vis\CrossPatient_Fold1_*",
    "checkpoints\results_vis\CrossPatient_Fold2_*",
    "checkpoints\results_vis\CrossPatient_Fold3_*",
    "checkpoints\*.pth",
    "predictions*.csv"
) -DestinationPath "U:\pfmval_results_$(Get-Date -Format 'yyyyMMdd').zip"
```

### 本地解压

```bash
# 解压到项目根目录
unzip /path/to/pfmval_results_20260529.zip -d d:/AI空间转录病理研究/PFMval_new/
```

---

## 五、训练任务分配速查

### 本地 RTX 4060 (8GB)

| 任务 | 命令模板 |
|------|----------|
| 冒烟测试 | `python <script>.py --cross_patient --fold 1 --epochs 1` |
| 单患者快速验证 | `python <script>.py --patient HYZ15040 --epochs 50` |
| EGN-v2 单 fold | `python train_egnv2_uni.py --fold 1 --epochs 150` |
| 消融实验 | `python <script>.py --cross_patient --fold 1 --epochs 30` |

### 服务器 RTX 4080 (16GB)

| 任务 | 命令模板 |
|------|----------|
| 完整 3 折 CV | 三个窗口分别跑 Fold 1/2/3, `--epochs 150` |
| 超参扫参 | 多个 `--tv_weight` / `--tv_mode` 组合，每 Fold 1 跑一遍 |
| Virchow2 3 折 | `python train_histogene_virchow2_tokens.py --fold {1,2,3}` |
| 原版 ViT | `python train_cross_patient_histogene.py` |

---

## 六、常见问题

**Q: 服务器训练报 `FileNotFoundError` 找不到缓存文件？**
A: 检查 `config.yaml` 中 `paths:` 节路径是否正确。服务器路径格式如 `D:/data/caches/`（正斜杠）。

**Q: 服务器上 `PYTHONIOENCODING=utf-8` 还需要吗？**
A: 需要。Windows 中文版控制台默认 GBK，不加会导致打印 Unicode 字符时崩溃。

**Q: 如何确认服务器训练进度？**
A: 查看 `training_status_*.txt` 文件（训练脚本自动生成），或查看 `checkpoints/results_vis/` 下最新的日志。

**Q: config.yaml 怎么同步？**
A: 不要直接覆盖。本地和服务器各维护一份。如果新增了配置项，手动在服务器版上添加对应条目。
