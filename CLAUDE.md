# CLAUDE.md — PFMval 项目指南

## 项目概述

基于食管癌H&E病理切片+空间转录组数据，预测30条基因通路ssGSEA活性评分。3个患者数据集：HYZ15040、JFX0729、LMZ12939。

## 模型体系

| 模型 | 状态 | 特征 | 说明 |
|------|------|------|------|
| HisToGene-UNI | 活跃 | 1536维 | 主力，跨患者泛化最强 |
| EGN-v2+UNI | 活跃 | 1536维 | 跨患者PCC 0.195 |
| HisToGene-UNI+GAT | 实验完成 | 1536维 | 提升不显著(-0.27%) |
| HisToGene 原版 | 活跃 | ViT | 基线，~70.6M参数 |
| EGN-v1 | **已淘汰** | — | 所有对比分析排除 |

当前最佳：HisToGene-UNI Token 跨患者3折平均PCC=0.3812。

## 核心铁律

1. **受保护目录禁改**：`histogene/`、`egnv1/`、`egnv2/` 下文件严禁修改，适配通过根目录独立文件实现
2. **EGN-v1已淘汰**：除非用户明确要求，否则排除
3. **路径不修正typo**：`patch_noov_spilt` 是原始拼写
4. **val_loss选最优epoch**：非val_pcc最大
5. **predictions.csv列名**：`true_xxx`/`pred_xxx` 格式
6. **generate_full_report只调用一次**：避免重复创建时间戳目录

## 运行环境

| 用途 | Python路径 |
|------|-----------|
| HisToGene系列 | `C:\Program Files\Python313\python.exe` |
| EGN-v2/GAT(需PyG) | `D:\conda_envs\pfmval_py310\python.exe` |

## 关键经验

- `dataset_name`默认值陷阱：从`--patient`自动推导，不硬编码
- 跨患者训练：二折交叉（2患者训→1患者测），衰减-31.7%~-77.2%
- UNI特征对EGN-v2提升显著(+28%~+50%)，对HisToGene提升有限(+2.3%~+11.8%)
- 三层交集(patches∩CSV∩cache)样本少，二层交集(cache∩labels)样本更多效果好
- evaluate()必须同时输出PCC、MAE、R²

## 数据路径

```
data_new_3ST/patch_noov_spilt/{patient}_noov_split/   # 三患者patch
uni2h_cache/{patient}/train/ 和 val/                    # UNI特征缓存
{patient}_ssGSEA_scores_zscore.csv                       # ssGSEA标签
```

## Git规则

忽略：`data_new_3ST/`、`*.pth`、`.venv/`、缓存目录、`*.log`、`training_status_*.txt`、`temp_*.py`

训练输出：保存至`{model_dir}/checkpoints/results_vis/{dataset}_{timestamp}/`，不覆盖历史结果。
