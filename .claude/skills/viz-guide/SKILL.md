---
name: viz-guide
description: Visualization and output standards for PFMval training results — file checklist, naming, format rules.
argument-hint: [checklist|format|timestamps|metrics]
disable-model-invocation: true
allowed-tools: Read, Bash
---

# Viz Guide — 可视化与输出规范

训练结果可视化、报告生成的标准规范。

## 一、时间戳隔离

- 每次可视化结果保存到独立时间戳子目录
- 路径格式：`{model_dir}/checkpoints/results_vis/{dataset}_{timestamp}/`
- **`generate_full_report` 只能全局调用一次**，避免重复创建时间戳目录

## 二、最佳 Epoch 标准

- 基于 **val_loss 最小**（非 val_pcc 最大），两者可能对应不同 epoch
- `model_params.txt` 可能记录中间 checkpoint 值，以 CSV 为准
- 汇总对比时统一使用 val_loss 最小 epoch

## 三、标准输出文件清单

每次训练完成后必须生成：

| # | 文件 | 内容 |
|---|------|------|
| 1 | `model_params.txt` | 模型参数 + 关键指标 |
| 2 | `training_history_{dataset}.csv` | 逐 epoch 训练记录 |
| 3 | `predictions.csv` | 逐样本预测（`true_xxx`/`pred_xxx` 列名） |
| 4 | `training_curves.png` | Loss/PCC/MAE 训练曲线 |
| 5 | `pcc_barplot.png` | 逐通路 PCC 柱状图 |
| 6 | `per_pathway_pcc.csv` | 逐通路指标表（pathway/pcc/r²/mae/rank，按 PCC 降序） |

## 四、predictions.csv 列名格式

- 必须为 `true_通路名`/`pred_通路名`
- 反向格式（`通路名_true`）会导致 `visualize_results.py` 解析失败
