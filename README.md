# PFMval_new — 食管癌病理空间转录组预测

> 基于食管癌 H&E 病理切片，使用深度学习模型预测 30 条基因通路的 ssGSEA 活性评分（Z-score 标准化）。

---

## 🎯 项目目标

输入：H&E 病理切片图像（patch 级，224×224）
输出：30 条基因通路的 **ssGSEA 活性评分**（已 Z-score 标准化）
评估指标：**PCC**（Pearson 相关系数，逐通路计算后取均值）

---

## 📊 数据集概况

3 个食管癌患者（**非肺癌**）的空间转录组 + H&E 病理数据：

| 患者 ID | Patch 数量 | 用途 |
|---------|-----------|------|
| HYZ15040 | 2,655 | 主验证集 / 跨患者测试集 |
| JFX0729 | 7,838 | 训练集 / 跨患者训练 |
| LMZ12939 | 7,513 | 训练集 / 跨患者训练 |

数据路径：`data_new_3ST/patch_noov_spilt/{patient}_noov_split/`
（路径 typo `spilt` 为原始数据，**不可修正**）

---

## 🧬 模型演进时间线

| 时间 | 模型 | 特征 | 备注 | 最优 PCC |
|------|------|------|------|---------|
| Phase 0 | EGN-v1 | ResNet50 (2048d) | **已淘汰** | — |
| Phase 1 | HisToGene 原版 | ViT 直接输入 | 基线 | — |
| Phase 2 | HisToGene-UNI | UNI2-h CLS (1536d) | 单 token | 0.5336 (单患者) |
| Phase 3 | **HisToGene-UNI Token** | UNI2-h Tokens [265,1536] | **主力模型** | **0.5217** (HYZ AugMix) |
| Phase 3.5 | HisToGene-UNI + GAT | UNI2-h + 图结构 | 实验完成（提升不显著） | 0.4068 (Fold1) |
| Phase 4 | **OmiCLIP (Loki)** | coca_ViT-L-14 [255,768] | **新增（特征已提取）** | 训练验证中 |

---

## 🏆 当前最优结果

| 场景 | 模型 | PCC |
|------|------|-----|
| 单患者验证（HYZ15040） | HisToGene-UNI Token AugMix | **0.5217** |
| 跨患者 Fold1（JFX+LMZ→HYZ） | HisToGene-UNI Token AugMix | **0.4142** |
| 跨患者 3 折平均 | HisToGene-UNI Token | 0.3812 |

---

## ⚡ 快速启动

### 环境一：HisToGene 系列训练（主力）

```powershell
# Python 3.13, PyTorch 2.6.0+cu118
$env:PYTHONIOENCODING = "utf-8"
& "C:\Program Files\Python313\python.exe" extract_uni_tokens.py --patient HYZ15040
& "C:\Program Files\Python313\python.exe" train_histogene_uni_tokens_augmix.py --patient HYZ15040 --epochs 50
```

### 环境二：OmiCLIP 特征提取（新）

```powershell
# Python 3.9, open_clip 2.26.1
& "D:\conda_envs\loki_env\python.exe" extract_omiclip_features.py --patient HYZ15040
# 训练仍用 Python313
& "C:\Program Files\Python313\python.exe" train_histogene_omiclip.py --patient HYZ15040 --epochs 50
```

### 通用：可视化结果

```powershell
& "C:\Program Files\Python313\python.exe" visualize_results.py --model_dir .
```

### 服务器迁移

项目已全面支持 **Linux 服务器迁移**，只需编辑 `config.yaml` 即可适配不同数据路径：

```bash
# 1. 创建 conda 环境
conda env create -f env_histogene.yml
# 2. 编辑 config.yaml 中的路径
# 3. 验证配置
python config_utils.py
# 4. 开始训练
python train_histogene_uni_tokens_augmix.py --patient HYZ15040 --epochs 150
```

详见 **[服务器迁移指南（初学者版）](01_指南与解读/服务器迁移指南_初学者版.md)**。

---

## 📁 目录结构

```
PFMval_new/
├── 01_指南与解读/                  # 各类设计文档与初学者指南
│   ├── 服务器迁移指南_初学者版.md   # 服务器迁移手把手教程
│   └── 项目全貌与迁移指南.md       # 跨 AI 平台上下文文档
├── 02_组会汇报/                    # 周报与汇报材料
├── .qoder/
│   ├── basic_rule.md              # 项目硬性规则（AI 自动加载）
│   ├── experience.md              # 经验索引（AI 自动加载）
│   ├── skills/                    # 分领域经验
│   └── repowiki/zh/content/       # 结构化技术参考（7文件，按需查阅）
├── config.yaml                     # 🔧 统一配置文件（数据路径、训练参数）
├── config_utils.py                 # 🔧 配置工具库（所有脚本通过它读取路径）
├── env_histogene.yml               # conda 环境定义（HisToGene 系列）
├── env_egnv2.yml                   # conda 环境定义（EGN-v2 / GAT）
├── data_new_3ST/                   # 三患者原始数据
├── uni2h_cache_tokens/             # UNI2-h tokens 缓存
├── uni2h_cache_tokens_aug/         # AugMix 增强缓存
├── omiclip_cache/                  # OmiCLIP 特征缓存
├── pretrained_omiclip/             # OmiCLIP 权重
├── histogene/  egnv1/  egnv2/      # ⚠️ 只读目录，禁止修改
├── extract_uni_tokens.py           # UNI tokens 提取
├── extract_omiclip_features.py     # OmiCLIP 特征提取
├── train_histogene_uni_tokens_augmix.py  # 主力训练脚本
├── model_uni_tokens.py             # HisToGeneUNITokens 模型
├── dataset_uni_tokens_augmix.py    # 主数据集类
├── config_utils.py / notify_utils.py
├── visualize_results.py / split.py / zscore.py
└── README.md
```

---

## ⚠️ 重要约束（绝对不可违反）

1. **只读目录**：`histogene/`、`egnv1/`、`egnv2/` 目录下文件**禁止修改**，所有适配通过根目录新建独立文件实现
2. **路径 typo**：`patch_noov_spilt`（spilt）是原始数据路径拼写，**不可修正**
3. **最优 epoch 选取**：以 **val_loss 最小**为准，不是 val_pcc 最大
4. **predictions.csv 列名**：必须为 `true_xxx` / `pred_xxx` 格式（visualize_results.py 约定）
5. **Windows 编码**：所有训练命令前缀 `$env:PYTHONIOENCODING = "utf-8"`（Linux 不需要）
6. **PowerShell 语法**：不支持 `&&`，使用 `;` 分隔
7. **路径统一管理**：所有训练/提取脚本的数据路径通过 `config_utils.py` 的函数获取（`get_patient_paths()`、`get_project_root()` 等），禁止在脚本中硬编码绝对路径
8. **config.yaml 是唯一配置入口**：迁移到服务器时只需修改此文件，如需新增路径配置项，扩展 `config.yaml` + `config_utils.py`，不得在各脚本中散落硬编码路径

---

## 🚀 下一步计划

- **短期（本周）**：OmiCLIP 训练验证 + 三折交叉验证
- **中期**：Virchow2 / BLEEP 探索；扩展至 5 患者
- **长期**：多中心验证；论文投稿

---

## 📚 文档索引

### AI 自动加载（每次会话必读）
- [`CLAUDE.md`](CLAUDE.md) — Claude Code 项目指南（含模型体系、铁律、环境、路径）
- [`.qoder/basic_rule.md`](.qoder/basic_rule.md) — 项目硬性规则
- [`.qoder/experience.md`](.qoder/experience.md) — 经验索引 + 踩坑记录

### 技术参考（按需查阅）
- [`.qoder/repowiki/zh/content/项目概述.md`](.qoder/repowiki/zh/content/项目概述.md) — 项目目标、数据集、模型矩阵
- [`.qoder/repowiki/zh/content/环境与配置.md`](.qoder/repowiki/zh/content/环境与配置.md) — Python 环境、CUDA、编码
- [`.qoder/repowiki/zh/content/数据系统.md`](.qoder/repowiki/zh/content/数据系统.md) — 数据路径、config_utils API
- [`.qoder/repowiki/zh/content/模型体系/模型体系.md`](.qoder/repowiki/zh/content/模型体系/模型体系.md) — 模型架构详情
- [`.qoder/repowiki/zh/content/训练系统/训练指南.md`](.qoder/repowiki/zh/content/训练系统/训练指南.md) — 训练脚本与命令
- [`.qoder/repowiki/zh/content/推理评估/推理与评估.md`](.qoder/repowiki/zh/content/推理评估/推理与评估.md) — 推理评估规范
- [`.qoder/repowiki/zh/content/故障排除.md`](.qoder/repowiki/zh/content/故障排除.md) — 常见问题速查

### 指南与方案（存档参考）
- [`01_指南与解读/`](01_指南与解读/) — 分析报告、设计方案、部署记录
- [`02_组会汇报/`](02_组会汇报/) — 历史组会汇报
- [`docs/`](docs/) — 实验报告与修复记录
