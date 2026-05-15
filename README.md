# PFMval — 病理基础模型验证

基于食管癌H&E病理切片+空间转录组数据，使用深度学习模型预测30条基因通路ssGSEA活性评分。

## 数据集

3个食管癌患者：**HYZ15040**、**JFX0729**、**LMZ12939**（不是肺癌）。

```bash
data_new_3ST/patch_noov_spilt/
├── HYZ15040_noov_split/    # 训练/验证集
├── JFX0729_noov_split/     # 跨患者训练
└── LMZ12939_noov_split/    # 跨患者训练
```

UNI2-h预提取特征：[MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)，输出1536维。

## 模型体系

| 模型 | 特征 | 单患者PCC | 跨患者PCC | 状态 |
|------|------|-----------|-----------|------|
| HisToGene-UNI Token | 1536维 UNI2-h | 0.5336 | 0.3812 (3折平均) | **主力** |
| HisToGene-UNI+GAT | 1536维 UNI2-h | — | 0.4068 | 实验完成 |
| EGN-v2+UNI | 1536维 UNI2-h | — | 0.1950 | 活跃 |
| HisToGene 原版 | ViT直接输入 | — | — | 基线 |
| EGN-v1 | ResNet50 | — | — | **已淘汰** |

## 环境配置

两个独立Python环境：

| 用途 | 路径 | 说明 |
|------|------|------|
| HisToGene系列训练 | `C:\Program Files\Python313\python.exe` | torch 2.6.0+cu118 |
| EGN-v2/GAT系列 | `D:\conda_envs\pfmval_py310\python.exe` | torch_geometric 2.7.0 |

## 快速开始

```bash
# 1. 特征提取（UNI2-h Token）
python extract_uni_tokens.py --patient HYZ15040

# 2. 训练
python train_histogene_uni_tokens.py --patient HYZ15040 --epochs 50

# 3. 可视化结果
python visualize_results.py --model_dir .
```

## 项目结构

```
PFMval_new/
├── train_*.py               # 各模型训练脚本
├── dataset_*.py             # 数据加载
├── model_*.py               # 模型定义
├── extract_*.py             # 特征提取
├── visualize_*.py           # 结果可视化
├── split.py / zscore.py     # 数据预处理
├── config_utils.py          # 配置工具
├── scripts/                 # PowerShell运行脚本
├── tools/                   # 调试/诊断/分析工具
├── data/                    # 原始CSV数据
├── docs/                    # 报告与文献
├── .qoder/                  # 项目文档与规则（AI助手必读）
└── CLAUDE.md                # AI编程助手指南
```

## 文档索引

- [`.qoder/basic_rule.md`](.qoder/basic_rule.md) — 项目硬性规则
- [`.qoder/experience.md`](.qoder/experience.md) — 经验索引
- [`.qoder/skills/`](.qoder/skills/) — 各领域详细经验（数据处理、训练、可视化、组会汇报）
- [`.qoder/repowiki/`](.qoder/repowiki/) — 项目百科（架构、系统、流程文档）
- [`CLAUDE.md`](CLAUDE.md) — AI编程助手快速指南
