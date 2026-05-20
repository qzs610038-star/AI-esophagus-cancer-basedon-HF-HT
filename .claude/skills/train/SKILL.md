---
name: train
description: Launch PFMval model training with correct Python environment, PYTHONIOENCODING, and parameters. Supports UNI/Virchow2/OmiCLIP backbones.
argument-hint: <backbone> <patient-config> [--epochs N] [--lr LR] [--batch B] [--mixup alpha]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob
---

# PFMval 训练启动器

统一训练入口，自动选择 Python 解释器、设置环境变量、拼接路径。

## 调用格式

```
/train <backbone> <patient-config> [options]
```

**backbone**: `uni` | `virchow2` | `omiclip`
**patient-config**: `HYZ15040` | `JFX0729` | `LMZ12939` | `cross`（3折交叉验证）

## Python 环境自动选择

| Backbone | Python 路径 | 说明 |
|----------|------------|------|
| `uni` | `C:\Program Files\Python313\python.exe` | torch 2.6.0+cu118 |
| `virchow2` | `C:\Program Files\Python313\python.exe` | 同 UNI |
| `omiclip` | `C:\Program Files\Python313\python.exe` | 特征已提取，训练同 UNI |

## 训练脚本映射

| Backbone | 训练脚本 |
|----------|---------|
| `uni` | `train_histogene_uni_tokens_augmix.py` |
| `virchow2` | `train_histogene_virchow2_tokens.py` |
| `omiclip` | `train_histogene_omiclip.py` |

## 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 5e-5 | 学习率 |
| `--dropout` | 0.5 | Dropout 比率 |
| `--mixup_alpha` | 0.2 | MixUp alpha（仅 UNI 默认开启） |
| `--batch_size` | 16 | 批大小 |
| `--num_epochs` | 50 | 单患者；Cross 默认 80 |

## 执行模板

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" \
    "d:\AI空间转录病理研究\PFMval_new\<训练脚本>" \
    --patient <patient-config> \
    --lr <lr> \
    --dropout <dropout> \
    --batch_size <batch_size> \
    --num_epochs <num_epochs>
```

## 示例

```bash
# UNI 单患者训练
/train uni HYZ15040 --epochs 50

# Virchow2 跨患者3折
/train virchow2 cross --epochs 80 --lr 5e-05

# 快速测试（2 epoch）
/train uni HYZ15040 --epochs 2 --batch 8
```
