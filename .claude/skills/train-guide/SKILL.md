---
name: train-guide
description: Training engineering reference — environment config, hyperparameter tuning, cross-patient strategies, and known pitfalls.
argument-hint: [env|pitfalls|cross-patient|gat|metrics]
disable-model-invocation: true
allowed-tools: Read, Bash
---

# Train Guide — 训练工程经验

模型训练、环境配置、调参的踩坑经验速查。

## 一、dataset_name 默认值陷阱 ⚠️

- **历史教训**：`train_egnv2_uni.py` 默认 `dataset_name="HYZ15040_UNI"` → LMZ12939 训练输出目录误标
- HYZ15040_UNI 检查点被覆盖，需重新训练
- **修复方案**：默认值设为 `None` + 从 `--patient` 自动推导
- `histogene/train_uni.py:167` 同样有此问题（受保护目录不改，手动指定 `--dataset_name` 规避）

## 二、UNI 特征的差异化价值

| 维度 | HisToGene | EGN-v2 |
|------|-----------|--------|
| 单患者提升 | +2.3% ~ +11.8% | +28% ~ +50% |
| 跨患者提升 | +235% | +81% |

EGN-v2 收益最大（替换冻结 ResNet-50），HisToGene 提升有限（ViT 本身有学习能力）。

## 三、跨患者泛化训练

- 训练集：JFX0729 + LMZ12939，测试集：HYZ15040
- 所有模型跨患者泛化显著衰减（-31.7% ~ -77.2%）
- HisToGene-UNI 衰减最小 (-31.7%)，HisToGene 原版衰减最大 (-77.2%)

## 四、GAT 空间建模实验

- HisToGene-UNI Token + GAT (13.46M)，Fold1 PCC=0.4068 vs 基线 0.4095 (-0.27%)
- **结论**：GAT 单独提升有限，空间邻域建模对跨患者贡献不大
- 原因：UNI2-h 特征已含充分局部信息 + 跨患者空间模式差异大

## 五、训练指标完整性

`evaluate()` 函数必须同时输出 **PCC、MAE、R²** 三项。
`training_history.csv` 必须包含列：`train_loss, val_loss, val_pcc, val_mae, val_r2`。

## 六、环境配置速查

| 用途 | Python 路径 |
|------|-----------|
| HisToGene/UNI/Virchow2/OmiCLIP | `C:\Program Files\Python313\python.exe` |
| EGN-v2/GAT (需 PyG) | `D:\conda_envs\pfmval_py310\python.exe` |
| 服务器全部训练 | `C:\ProgramData\miniconda3\python.exe` |

- Conda 激活：`(& "D:\miniconda\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression`
- 所有训练命令加 `PYTHONIOENCODING=utf-8` 前缀
- **最佳 epoch 以 val_loss 最小为准**（非 val_pcc 最大）

## 七、训练监控

委派子代理训练时：只看前 1-2 epoch 确认正常（loss 下降、无报错），随后返回，不持续跟踪。