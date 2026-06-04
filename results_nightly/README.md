# 夜间 LoRA 验证实验结果

> 执行时间：2026-06-04 01:42 ~ 10:50  
> 服务器：RTX 4080 16GB, PyTorch 2.6.0+cu124  
> 代码版本：commit `eacd41c`（已含 CPU 限线程 + NaN 保护）  
> 待重跑实验需先 `git pull` 更新代码

## 数据说明

`online_cls/{实验名}/` 下包含：
- `training_history.csv` — 逐 epoch 训练/验证 Loss/PCC/R²/MAE + LR
- `training_summary.txt` — 最佳 epoch + 超参数摘要
- `per_pathway_pcc.csv` — 30 通路逐条 PCC/R²/MAE/排名
- `predictions.csv` — 逐样本 true_xxx / pred_xxx

## 实验结果

### Cross-Fold1 四模式对比（JFX+LMZ → HYZ）

| 模式 | Best Val PCC | Best Epoch | Val Loss | Δ vs Frozen |
|------|:---:|:---:|:---:|:---:|
| `frozen_r8_online_cls_cross_fold1` | 0.4113 | 1 | 0.3268 | — |
| `lora_r8_online_cls_cross_fold1` | **0.4322** | 2 | 0.3212 | **+0.0209** |
| `stage2_r8_online_cls_cross_fold1` | 0.4118 | 1 | 0.3292 | +0.0005 |
| `stage3_r8_online_cls_cross_fold1` | 0.4056 | 1 | 0.3325 | -0.0057 |

### Cross-Fold2（HYZ+LMZ → JFX）

| 模式 | Best Val PCC | Best Epoch | Val Loss |
|------|:---:|:---:|:---:|
| `lora_r8_online_cls_cross_fold2` | 0.3726 | 2 | 0.3650 |

### 单患者（HYZ15040）

| 模式 | Best Val PCC | Best Epoch |
|------|:---:|:---:|
| `frozen_r8_online_cls_HYZ15040` | 0.5236 | 1 |
| `lora_r8_online_cls_HYZ15040` (local) | 0.5462 | 2 |

### 失败实验（待重跑）

| 实验 | 原因 | 修复状态 |
|------|------|:---:|
| `lora_r8_online_cls_cross_fold3` | LMZ 标签极端值 numpy overflow | ✅ 已修复 |
| `lora_r8_online_cls_cross_fold1_d01` | 级联失败（GPU 状态未恢复） | ✅ 已修复 |
| `lora_r4_online_cls_cross_fold1` | 同上 | ✅ 已修复 |
| `frozen_r8_online_cls_cross_fold2` | 同上 | ✅ 已修复 |
| `frozen_r8_online_cls_cross_fold3` | 同上 | ✅ 已修复 |

## 逐通路 Top/Bottom 10（LoRA Cross-Fold1）

**Top 10**: ECM(0.725), MYC(0.719), Fibrosis(0.688), OxPhos(0.685), tls(0.642), Wound_Healing(0.599), Glycolysis(0.580), IL6(0.573), EMT(0.545), toxic(0.537)

**Bottom 10**: Interferon_Alpha(0.049), ifng(0.162), tgfb(0.165), Apoptosis(0.201), TNF(0.246), Coagulation(0.317), icp(0.311), DNA_Damage(0.307), ROS(0.305), Inflammatory(0.357)

## 过拟合分析

所有模式均表现出极快的过拟合：

| 模式 | Epoch2 Train PCC | Epoch2 Val PCC | Final Train PCC | Final Val PCC |
|------|:---:|:---:|:---:|:---:|
| Frozen | 0.575 | 0.374 | 0.654 | 0.345 |
| LoRA | 0.604 | **0.432** | 0.942 | 0.327 |
| Stage2 | 0.683 | 0.402 | 0.967 | 0.320 |
| Stage3 | 0.730 | 0.394 | 0.976 | 0.335 |

**规律**：可训练参数越多 → Train-Val Gap 越宽 → 泛化越差。有效训练窗口仅 1-2 epoch。

## 策略结论

1. ✅ LoRA Stage1 跨患者有效（+5.1%），方向成立
2. ❌ Stage2/Stage3 已放弃 — 解冻 backbone 必然损害泛化
3. 🎯 下一阶段：正则化路线（dropout↑ / rank↓ / AugMix / TV Loss）
4. ⏳ Fold3 待重跑（LMZ 溢出已修复），Fold2/3 frozen 基线待补齐
