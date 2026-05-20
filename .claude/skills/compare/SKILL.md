---
name: compare
description: Compare PFMval training results across experiments. Parse training history, predictions CSV, generate comparison tables.
argument-hint: <exp-dir-1> [exp-dir-2] ... [--per-pathway] [--summary]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep
---

# PFMval 结果比较器

解析训练历史和预测输出，生成跨实验对比表。

## 调用格式

```
/compare <exp1> [exp2] [exp3] ... [--per-pathway] [--summary]
```

**exp**: checkpoint 目录名或路径，例如：
- `checkpoints/HisToGene_UNI_HYZ15040/`
- 简写：`uni_HYZ` → 自动匹配最新时间戳目录

## 对比内容

| 维度 | 数据源 | 说明 |
|------|--------|------|
| 最佳 Epoch | `training_history.csv` | val_loss 最小的 epoch |
| Val PCC/MAE/R² | `training_history.csv` | 最佳 epoch 对应的验证指标 |
| 逐通路 PCC | `predictions.csv` | 30 条通路的 per-pathway PCC |
| 收敛速度 | `training_history.csv` | 达到最佳 epoch 的速度 |

## 输出格式

```
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Experiment          │ Val PCC  │ Val MAE  │ Best Ep  │
├─────────────────────┼──────────┼──────────┼──────────┤
│ UNI_HYZ15040        │  0.4095  │  0.0417  │    42    │
│ Virchow2_HYZ15040   │  0.4183  │  0.0399  │    38    │  ← +2.1%
│ OmiCLIP_HYZ15040    │  0.3921  │  0.0432  │    55    │
└─────────────────────┴──────────┴──────────┴──────────┘
```

`--per-pathway` 模式额外输出每实验的 Top3/Bottom3 通路。

## 执行逻辑

1. 定位每个实验的 `training_history.csv` 和 `predictions.csv`
2. 解析 CSV，找到 `val_loss` 最小的 epoch
3. 提取该 epoch 的 `val_pcc`, `val_mae`, `val_r2`
4. 计算变化率（相对于第一个实验）
5. 格式化为对比表格

## 执行命令

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" -c "
import pandas as pd, os, json
# 读取各实验的 training_history.csv
# 生成对比表和逐通路指标
"
```

## 示例

```bash
# 对比3个 backbone 在 HYZ15040 的效果
/compare uni_HYZ virchow2_HYZ omiclip_HYZ

# 逐通路详细对比
/compare uni_HYZ virchow2_HYZ --per-pathway
```
