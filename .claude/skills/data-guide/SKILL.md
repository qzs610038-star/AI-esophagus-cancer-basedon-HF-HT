---
name: data-guide
description: Data loading, path configuration, and CSV format reference for PFMval. Use when encountering data-related errors.
argument-hint: [layers|predictions|paths|typo]
disable-model-invocation: true
allowed-tools: Read, Bash
---

# Data Guide — 数据处理经验

数据加载、路径配置、CSV 格式相关的踩坑经验速查。

## 调用格式

```
/data-guide              # 完整经验
/data-guide layers       # 三层 vs 二层交集
/data-guide predictions  # predictions.csv 列名规则
/data-guide paths        # 数据集特殊路径
```

## 一、三层交集 vs 二层交集

| 交集类型 | 条件 | 样本量 | 效果 |
|---------|------|:---:|:---:|
| 三层（传 `--patches_dir`） | patches PNG ∩ CSV标签 ∩ .pt缓存 | 少 | 差 |
| 二层（不传 `--patches_dir`） | .pt缓存 ∩ CSV标签 | 多 | **好** ✅ |

**建议**：EGN-v2+UNI 训练默认不传 `--patches_dir`。

## 二、predictions.csv 列名格式

- `visualize_results.py` 要求列名为 **`true_通路名`/`pred_通路名`**
- 部分旧脚本输出 `通路名_true` 格式 → 解析失败
- **新脚本统一使用 `true_xxx`/`pred_xxx` 格式**

## 三、数据集特殊路径

```
data_new_3ST/patch_noov_spilt/HYZ15040_noov_split/
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/
data_new_3ST/patch_noov_spilt/LMZ12939_noov_split/
```

⚠️ `data_new_3ST/JFX0729` **不存在**，必须走 `patch_noov_spilt/` 子目录。
⚠️ `patch_noov_spilt` 是原始拼写（含 typo `spilt` 而非 `split`），**不要修正**。

## 四、标签文件

- ssGSEA z-score 标签：`{patient}_ssGSEA_scores_zscore.csv`
- 位于 `data_new_3ST/ssGSEA_zscore/` 目录
- 30 条通路，列名为通路全名（如 `HALLMARK_OXIDATIVE_PHOSPHORYLATION`）