---
name: onboard
description: Onboard a new patient to PFMval. Validates data structure, extracts features for all backbones, updates configs, and runs test training.
argument-hint: <patient-name> [--skip-extract] [--backbone uni|virchow2|all]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Write, Edit
---

# PFMval 新病例上线

引导式添加新患者数据：检查结构 → 提取特征 → 更新配置 → 试训练。

## 调用格式

```
/onboard <patient-name> [options]
```

**patient-name**: 新患者名称（如 `NEW001`）

Options:
- `--skip-extract` — 跳过特征提取（如已手动提取）
- `--backbone uni` — 仅提取指定 backbone（默认 `all`）

## 执行步骤

### 步骤 1：目录结构检查

验证以下路径存在：

```
data_new_3ST/patch_noov_spilt/<patient>_noov_split/
├── train_patches/          # 训练集 PNG patches
└── val_patches/            # 验证集 PNG patches
data_new_3ST/ssGSEA_zscore/<patient>_ssGSEA_zscore.csv  # 标签
```

检查项：
- [ ] PNG 文件数量（train + val）
- [ ] CSV 列数（应 = 31：1 patch_id + 30 pathways）
- [ ] CSV 中的 patch_id 是否与 PNG 文件名对应
- [ ] 通路名称是否与现有 30 通路一致

### 步骤 2：更新特征提取脚本

在以下文件的 `PATIENT_PATHS` 字典中添加新患者：

- `extract_uni_tokens.py`
- `extract_virchow2_tokens.py`（如存在）
- `extract_omiclip_features.py`（如存在）

```python
"NEW001": {
    "train": "data_new_3ST/patch_noov_spilt/NEW001_noov_split/train_patches",
    "val":   "data_new_3ST/patch_noov_spilt/NEW001_noov_split/val_patches",
},
```

### 步骤 3：提取特征

```bash
# UNI
/extract uni NEW001
# Virchow2
/extract virchow2 NEW001
# OmiCLIP（如环境就绪）
/extract omiclip NEW001
```

### 步骤 4：更新训练脚本

在对应训练脚本的 PATIENT_CONFIG 中添加新患者配置。

### 步骤 5：试训练（2 epoch）

```bash
/train uni NEW001 --epochs 2
```

验证：训练无报错、loss 正常下降、checkpoint 正常保存。

### 步骤 6：更新 CLAUDE.md（可选）

在 CLAUDE.md 的数据路径或患者列表中添加新患者信息。

## 检查清单输出

完成后打印：

```
新病例上线检查清单 — NEW001
✅ 数据目录结构正确 (train: 3526 patches, val: 1510 patches)
✅ 标签 CSV 格式正确 (31 columns, 30 pathways, 5036 labels)
✅ 通路名称与现有数据集一致
✅ PATIENT_PATHS 已更新 (3 个文件)
✅ UNI 特征提取完成 (5036 .pt files, 2.3 GB)
⏸ Virchow2 特征提取待执行
⏸ OmiCLIP 特征提取待执行（loki_env 未搭建）
⏸ 试训练待执行
```

## 新增后训练命令

新患者的跨患者训练配置：

```bash
# 单患者训练
/train uni NEW001 --epochs 50

# 跨患者3折（加入新患者后可配置更多fold组合）
# 编辑训练脚本中的 cross-patient 字典
```
