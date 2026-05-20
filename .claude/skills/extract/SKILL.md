---
name: extract
description: Extract features from pathology foundation models (UNI2-h / Virchow2 / OmiCLIP) for specified patients. One-command feature caching.
argument-hint: <backbone> <patient> [--mode lite|full] [--rebuild]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob
---

# PFMval 特征提取器

一键提取病理基础模型的特征 token/embedding，缓存为 `.pt` 文件供训练使用。

## 调用格式

```
/extract <backbone> <patient> [options]
```

**backbone**: `uni` | `virchow2` | `omiclip`
**patient**: `HYZ15040` | `JFX0729` | `LMZ12939` | `all`（全部3患者）

## 提取脚本与 Python 环境

| Backbone | 提取脚本 | Python 环境 |
|----------|---------|------------|
| `uni` | `extract_uni_tokens.py` | `C:\Program Files\Python313\python.exe` |
| `virchow2` | `extract_virchow2_tokens.py` | `C:\Program Files\Python313\python.exe` |
| `omiclip` | `extract_omiclip_features.py` | `D:\conda_envs\loki_env\python.exe` ⚠️ Python 3.9 |

## 缓存目录

| Backbone | 缓存目录 | Lite 大小/患者 |
|----------|---------|---------------|
| `uni` | `uni2h_cache_tokens/` | ~2.3 GB |
| `virchow2` | `virchow2_cache_tokens/` | ~2.3 GB |
| `omiclip` | `omiclip_cache/` | ~待探查 |

## 选项

| 选项 | 说明 |
|------|------|
| `--mode lite` | 仅提取部分 token（默认，推荐） |
| `--mode full` | 提取全部 token（磁盘占用大 4×） |
| `--rebuild` | 强制重建已有缓存 |

## 执行模板

```bash
# UNI / Virchow2
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" \
    "d:\AI空间转录病理研究\PFMval_new\extract_<backbone>_tokens.py" \
    --patient <patient> --mode lite

# OmiCLIP（必须用 loki_env Python 3.9）
"D:\conda_envs\loki_env\python.exe" \
    "d:\AI空间转录病理研究\PFMval_new\extract_omiclip_features.py" \
    --patient <patient> --mode lite
```

## 执行逻辑

1. 检查缓存目录是否存在已有文件 → 如非 `--rebuild` 则跳过
2. 如果 backbone 对应提取脚本不存在 → 提示用户先完成部署
3. 如果 Python 环境不匹配 → 提示正确路径
4. 提取完成后打印缓存统计（文件数、样本 shape、磁盘占用）

## 示例

```bash
# 提取 UNI token（HYZ15040）
/extract uni HYZ15040

# 提取 Virchow2 token（全部患者，重建缓存）
/extract virchow2 all --rebuild

# 提取 OmiCLIP（必须先搭建 loki_env）
/extract omiclip HYZ15040
```
