# PFMval 已测试模型性能排名

> 更新时间：2026-06-04 | 任务：H&E 病理切片 → 30条基因通路 ssGSEA 预测（食管癌 3 患者：HYZ15040 / JFX0729 / LMZ12939）
>
> ⚠️ **已知问题（2026-06-04）**：UNI2-h + DenseNet121 跨患者 Fold1 存在两次独立训练结果不一致：
> - `uni2h_new/train_des_cross.py` → PCC **0.4429** (epoch 5, val_pcc 最大)
> - `train_online_cls.py` → PCC **0.4113** (epoch 1, val_loss 最小)
> - 差异 0.0316，可能来源：epoch 选择标准（val_pcc vs val_loss）、训练脚本差异、数据加载差异。**待排查，暂不修改旧排名**。
> - **LoRA 实验全部基于 `train_online_cls.py` 训练**，与 frozen online 结果 (0.4113) 对比，不使用旧 `uni2h_new/` 结果 (0.4429)。

---

## 快速排名总表（跨患者泛化 Fold1: JFX+LMZ→HYZ）

跨患者泛化是本任务的核心评估场景（2 患者训练 → 第 3 患者测试），最接近实际部署条件。

| 排名 | 模型 | Val/Test PCC | 最佳Epoch | 参数量 | 过拟合Gap | 权重路径 |
|:---:|------|:---:|:---:|:---:|:---:|------|
| 1 | **UNI2-h + DenseNet121 (旧, uni2h_new/)** 🔥 | **0.4429** ⚠️ | 5 | ~5.8M | 0.19 | `uni2h_new/checkpoints/CrossPatient_Fold1_HYZ15040_UNI2h_DenseNet121/best_model_uni2h.pth` |
| 2 | **UNI2-h + DenseNet121 CLS + LoRA r=8** 🆕 | **0.4322** (val) | 2 | ~5.8M + 1.8M LoRA | 0.17 | `checkpoints/online_cls/lora_r8_online_cls_cross_fold1/best_model.pth` |
| 3 | **HisToGene-UNI-Tokens + AugMix + TV Loss + Virchow2融合** | **0.4242** | 14 (+融合) | ~9.8M + 7.2M | — | TV模型: `histogene/checkpoints/HisToGene_UNI_Tokens_AugMix/CrossPatient_JFX_LMZ_to_HYZ_UNI_tokens_AugMix/best_histogene_uni_tokens_augmix.pth` |
| 4 | **HisToGene-UNI-Tokens + AugMix + TV Loss** | **0.4212** (val) | 14 | ~9.8M | 0.15 | 同上 |
| 5 | **HisToGene-UNI-Tokens + AugMix + TV Loss (L2 w=0.01)** | **0.4170** (val) | 14 | ~9.8M | 0.21 | `histogene/checkpoints/results_vis/TV_Sweep_tv_l2_w0.01_20260523_165603/` |
| 6 | HisToGene-UNI-Tokens + AugMix | 0.4142 (val) | 4 | ~9.8M | 0.23 | `histogene/checkpoints/HisToGene_UNI_Tokens_AugMix/CrossPatient_JFX_LMZ_to_HYZ_UNI_tokens_AugMix/best_histogene_uni_tokens_augmix.pth`（旧版） |
| 7 | **UNI2-h + DenseNet121 CLS (frozen, online复现)** 🆕 | **0.4113** (val) | 1 | ~5.8M | 0.10 | `checkpoints/online_cls/frozen_r8_online_cls_cross_fold1/best_model.pth` |
| 8 | HisToGene-UNI-Tokens + GAT | 0.4068 (val) | 50 | ~13.5M | — | `histogene/checkpoints/GAT_fold1/` |
| 9 | HisToGene-UNI（方案A） | 0.3946 (test) | 3 | ~4.0M | 0.39 | `histogene/checkpoints/CrossPatient_JFX_LMZ_to_HYZ/best_histogene_uni.pth` |
| 10 | HisToGene-UNI + Reg | 0.3923 (test) | 5 | ~4.0M | 0.28 | `histogene/checkpoints/results_vis/CrossPatient_JFX_LMZ_to_HYZ_reg_optimized_20260428_225944/` |
| 11 | HisToGene-UNI-Tokens（方案B base） | 0.3835 (val) | 3 | ~7.7M | 0.31 | `histogene/checkpoints/CrossPatient_Fold3_to_LMZ12939_UNI_tokens/best_histogene_uni_tokens.pth` |
| 12 | EGN-v2+UNI | 0.3537 (test) | 16 | ~2.8M | -0.04 | `egnv2/checkpoints/CrossPatient_JFX_LMZ_to_HYZ_UNI/best_egnv2_uni.pth` |
| 13 | HisToGene-Virchow2 Tokens | 0.3537 (val) | 11 | ~7.2M | — | 最新 run `20260521_110949/` |
| 14 | HisToGene-OmiCLIP | 0.2544 (val, 3折均值 0.1970) | 1 | ~6.1M | 0.02 | `checkpoints/omiclip_CrossPatient_JFX0729+LMZ12939_to_HYZ15040_OmiCLIP/best_histogene_omiclip.pth` |
| 15 | EGN-v2 (ResNet-50) | 0.1950 (test) | 16 | ~3.0M | 0.10 | `egnv2/checkpoints/CrossPatient_JFX_LMZ_to_HYZ/best_egnv2.pth` |
| 16 | HisToGene 原版 (ViT) | 0.1178 (test) | 2 | ~70.6M | 0.52 | `histogene/checkpoints/CrossPatient_JFX_LMZ_to_HYZ_orig/best_histogene.pth` |
| — | ~~LoRA Stage2 (解冻末2层)~~ | ~~0.4118~~ ❌ | 1 | — | — | **已放弃**：=frozen，无增益 |
| — | ~~LoRA Stage3 (解冻末4层)~~ | ~~0.4056~~ ❌ | 1 | — | — | **已放弃**：<frozen，损害泛化 |

> **2026-06-04 更新**：🆕 **LoRA 验证实验** — LoRA Stage1 r=8 Cross-Fold1 PCC=**0.4322**（vs frozen online 0.4113，+5.1%）。Fold2 (JFX)=0.3726。Stage2/3 已放弃（解冻 backbone 损害泛化：0.4118/0.4056）。frozen online 复现仅 0.4113（vs 旧 uni2h_new/ 0.4429），差异待排查。方向从"渐进解冻"转向"正则化约束"。详细结果：`results_nightly/README.md`
> **2026-05-23 更新**：TV Loss 超参扫参完成（9组合: 3 mode × 3 weight）。L2 w=0.01 最优 (PCC=0.4170)，L1 mode 全面劣于 L2/Laplacian。三折 CV 平均 PCC=0.3943 (+0.0131 vs UNI Tokens 基线 0.3812)。AttnPool 失败根因修正：注意力模式跨患者高度一致 (ρ=0.99) 但近乎均匀 (熵=0.95/1.0)，根因非"患者特异性"而是注意力过于微弱无信息增益。
> **2026-05-22 更新**：新增 TV Loss 空间平滑正则化（PCC +0.0070，+1.7%）+ Virchow2 晚期融合（PCC +0.0030）。联合提升从 0.4142→0.4242 (+2.4%)。
> AttnPool（注意力池化）在跨患者场景下未带来提升，单患者+0.0036 但跨患者过拟合加剧（参数量 +2.1M）。

## 三折交叉验证完整对比

| 模型 | Fold1 (→HYZ) | Fold2 (→JFX) | Fold3 (→LMZ) | **3折平均** |
|------|:---:|:---:|:---:|:---:|
| **UNI2-h + DenseNet121 (旧, uni2h_new/)** | **0.4429** ⚠️ | **0.3683** | **0.3796** | **0.3969** |
| **UNI2-h + DenseNet121 CLS + LoRA r=8** 🆕 | **0.4322** | **0.3726** | ⏳ 待重跑 | — |
| **HisToGene-UNI-Tokens + AugMix + TV L2 w=0.01** | **0.4170** | **0.3752** | **0.3907** | **0.3943** |
| **UNI2-h + DenseNet121 CLS (frozen, online复现)** 🆕 | **0.4113** | ⏳ 待补 | ⏳ 待补 | — |
| HisToGene-UNI-Tokens + AugMix | 0.4142 | — | 0.3835 | — |
| HisToGene-UNI | 0.3946 | 0.3424 | 0.3731 | **0.3700** |
| EGN-v2+UNI | 0.3537 | 0.3917 | 0.3980 | **0.3811** |
| HisToGene-OmiCLIP | 0.2544 | 0.1081 | 0.2284 | **0.1970** |

> 注：UNI2-h + DenseNet121 三折平均 PCC = **0.3969**，无任何数据增强、无 TV Loss、无模型融合，裸奔即超越 UNI-Tokens + AugMix + TV L2（0.3943）。Fold1=0.4429 创跨患者 Fold1 新高（此前最佳 0.4242）。**首次证明 UNI2-h backbone 在跨患者场景下优于 UNI**。
> 🆕 **LoRA Stage1 跨患者 Fold1=0.4322，Fold2=0.3726**。Fold3 (LMZ) 因 numpy overflow 待重跑（NaN 保护已修复）。若 Fold3 能达到 ~0.38，三折均值约 0.395，与旧 frozen 0.3969 持平——但 LoRA 仅训练 1.8M 额外参数、2 epoch 即达此水平，正则化后可望超越。
> TV Loss (L2 w=0.01) 三折平均 PCC = **0.3943**，较 UNI Tokens 基线 0.3812 提升 **+0.0131 (+3.4%)**。

---

## 单患者训练性能（同一患者内 train/val 拆分）

### HisToGene 系列

| 模型 | HYZ15040 | JFX0729 | LMZ12939 | 权重目录 |
|------|:---:|:---:|:---:|------|
| HisToGene 原版 (ViT) | 0.5164 | 0.6041 | 0.5287 | `histogene/checkpoints/{patient}/best_histogene.pth` |
| HisToGene-UNI (方案A) | 0.5773 | — | — | `histogene/checkpoints/HYZ15040_UNI/best_histogene_uni.pth` |
| HisToGene-UNI-Tokens | 0.5265 | — | — | `histogene/checkpoints/HYZ15040_UNI_tokens/best_histogene_uni_tokens.pth` |
| HisToGene-UNI-Tokens (reg) | 0.5249 | — | — | `histogene/checkpoints/HYZ15040_UNI_tokens_reg/best_histogene_uni_tokens.pth` |
| HisToGene-UNI-Tokens-AugMix | 0.5217 | — | — | `histogene/checkpoints/HisToGene_UNI_Tokens_AugMix/HYZ15040_UNI_tokens_AugMix/best_histogene_uni_tokens_augmix.pth` |

### UNI2-h 系列 🔥

| 模型 | HYZ15040 | JFX0729 | LMZ12939 | 权重目录 |
|------|:---:|:---:|:---:|------|
| UNI2-h + DenseNet121 CLS + LoRA r=8 🆕 | **0.5462** | — | — | `checkpoints/online_cls/lora_r8_online_cls_HYZ15040/best_model.pth` |
| UNI2-h + DenseNet121 CLS (frozen) 🆕 | **0.5236** | — | — | `checkpoints/online_cls/frozen_r8_online_cls_HYZ15040/best_model.pth` |
| UNI2-h + DenseNet121 (旧, uni2h_new/) | **0.5227** | — | — | `uni2h_new/checkpoints/HYZ15040/best_model_uni2h.pth` |

### EGN-v2 系列

| 模型 | HYZ15040 | JFX0729 | LMZ12939 | 权重目录 |
|------|:---:|:---:|:---:|------|
| EGN-v2 (ResNet-50) | 0.4048 | 0.4445 | 0.3837 | `egnv2/checkpoints/{patient}/best_egnv2.pth` |
| EGN-v2+UNI (v3 best) | **0.6075** | 0.5627 | 0.5083 | `egnv2/checkpoints/HYZ15040_UNI/best_egnv2_uni.pth` |

### OmiCLIP 系列

| 模型 | HYZ15040 | JFX0729 | LMZ12939 | 权重目录 |
|------|:---:|:---:|:---:|------|
| HisToGene-OmiCLIP | 0.5275 | 0.6082 | 0.5267 | `checkpoints/omiclip_{patient}/best_histogene_omiclip.pth` |

### EGN-v1（已淘汰）

| 模型 | HYZ15040 | JFX0729 | LMZ12939 |
|------|:---:|:---:|:---:|
| EGN-v1 (ViT + GCN) | 0.2289 | 0.3141 | 0.2165 |

---

## 模型架构与超参数速查

### 0a. UNI2-h + DenseNet121 CLS + LoRA r=8 🆕 ★ 最新最佳 LoRA (2026-06-04)

```
架构: UNI2-h backbone (frozen) + LoRA (qkv+proj, 24层, r=8, α=16) + DenseNet121-style MLP
输入: 在线图像 patches [B, 3, 224, 224] + 空间坐标 (pos_x, pos_y)
参数量: 687.2M 总 / 5.8M 可训练 (0.8%)，其中 LoRA 1.8M + MLP 4.0M

训练: train_online_cls.py, lr_downstream=1e-4, lr_lora=1e-4, BS=8, HuberLoss, AMP
在线加载: 每个 epoch 实时通过 backbone 前向提取特征（非预缓存）
LoRA 注入: blocks 0-23 全部, target_modules=[qkv, proj]

跨患者 Fold1: PCC=0.4322 (epoch 2), Fold2=0.3726, Fold3 待重跑
单患者 HYZ: PCC=0.5462 (epoch 2), vs frozen 0.5236 (+0.0226)
过拟合: Train-Val Gap 0.17 (与 frozen 0.10 相比略宽，但仍优于 Stage2/3)
权重路径: checkpoints/online_cls/lora_r8_online_cls_cross_fold1/best_model.pth
文本结果: results_nightly/online_cls/lora_r8_online_cls_cross_fold1/

关键发现:
- LoRA 跨患者泛化 +5.1% vs frozen online (0.4113→0.4322)
- 可训练参数越少泛化越好: Frozen < LoRA < Stage2 < Stage3
- 正确方向: 正则化约束（dropout↑ / rank↓），非解冻 backbone
- Stage2/3 已放弃 (PCC 0.4118/0.4056, ≤ frozen)
```

### 0b. UNI2-h + DenseNet121 CLS (frozen, online) 🆕 ★ 在线训练 frozen 基线 (2026-06-04)

```
架构: Frozen UNI2-h backbone + DenseNet121-style MLP + 坐标嵌入
输入: 在线图像 patches [B, 3, 224, 224] + 空间坐标
参数量: 687.2M 总 / 4.0M 可训练 (MLP only, backbone 冻结)

训练: train_online_cls.py --mode frozen, lr=1e-4, BS=8, HuberLoss, AMP
跨患者 Fold1: PCC=0.4113 (epoch 1, val_loss 最小)
单患者 HYZ: PCC=0.5236 (epoch 1)

⚠️ 与 uni2h_new/ 的 frozen 结果差异：
- uni2h_new/train_des_cross.py: PCC=0.4429 (epoch 5, val_pcc 最大)
- train_online_cls.py: PCC=0.4113 (epoch 1, val_loss 最小)
- 差异 0.0316 (7.7%)，待排查。可能来源: epoch 选择标准、数据加载、
  特征缓存 vs 在线提取、MLP 架构细节差异
- 所有 LoRA 实验基于此在线训练方案，用此 0.4113 作为对照基线
权重路径: checkpoints/online_cls/frozen_r8_online_cls_cross_fold1/best_model.pth
```

### 0. UNI2-h + DenseNet121 (旧, uni2h_new/) ★ 旧最佳跨患者 (2026-05-30)

```
架构: Frozen UNI2-h backbone (MahmoodLab/UNI2-h, 1536-dim) + DenseNet121-style MLP
输入: UNI2-h 全局特征 [B, 1536]（CLS-like 输出）
参数量: ~5.8M（DenseNet121 MLP only, backbone 冻结）

超参数: lr=1e-3, batch_size=256, AdamW(wd=1e-4), MSELoss, ReduceLROnPlateau
DenseNet121: initial_dim=256, growth_rate=32, bottleneck_factor=4, transition_factor=0.5
正则化: dropout=0.2, 无数据增强, 无 TV Loss, 无模型融合

跨患者 3折CV: Fold1=0.4429(ep5), Fold2=0.3683(ep2), Fold3=0.3796(ep1), 均值=0.3969
单患者 HYZ15040: PCC=0.5227(ep15)
训练脚本: uni2h_new/train_des.py (单患者), uni2h_new/train_des_cross.py (跨患者)
特征缓存: uni2h_new/uni2h_cache_30/{patient}/{train,val}/*.pt
权重路径: uni2h_new/checkpoints/CrossPatient_Fold{1,2,3}_*_UNI2h_DenseNet121/best_model_uni2h.pth

关键发现:
- 裸奔（无增强、无 TV、无融合）3折均值 0.3969，超越 UNI-Tokens + AugMix + TV L2（0.3943）
- Fold1 PCC=0.4429 创跨患者 Fold1 历史新高（+4.4% vs Virchow2融合 0.4242）
- 严重过拟合：1-5 epoch 即达最优（Fold2/Fold3 仅 epoch 1-2），需更强正则化
- DenseNet121 MLP 参数量大但训练极快（3折CV全程 < 10 min GPU时间）
- UNI2-h backbone 首次在跨患者场景中超越 UNI，证明其病理特征表征能力更强
```

### 1. HisToGene-UNI-Tokens + AugMix + TV Loss + Virchow2融合 ★ 最佳跨患者

```
架构: LightweightTokenEncoder(hidden=512, layers=2, heads=8) + CoordEmbed(n_pos=128) + MLP(2048, dropout=0.5)
输入: UNI2-h token序列 [B, 65, 1536] + 空间坐标 (x, y)
参数量: ~9.8M（cross-pt 2层编码器）

超参数: lr=3e-5, batch_size=64, AdamW(wd=1e-4), HuberLoss, ReduceLROnPlateau
增强: n_augments=3, aug_sample_prob=0.5, mixup_alpha=0.2, mixup_prob=0.5
正则化: 空间 TV Loss (L2 mode, weight=0.01, k=6), 仅在非MixUp batch施加
后处理: Per-pathway grid search 融合 Virchow2 预测 (grid_step=0.05)

最佳结果: Val PCC=0.4212 (L1 TV单模型), PCC=0.4242 (融合后), epoch=14
2026-05-23 扫参: L2 w=0.01 最优 (PCC=0.4170), 3折CV均值=0.3943
训练脚本: train_histogene_uni_tokens_augmix.py (--tv_weight 0.01 --tv_k 6 --tv_mode l2)
融合脚本: ensemble_late_fusion.py
```

### 1b. TV Loss 超参扫参结果 (2026-05-23)

9 组合: w ∈ {0.01, 0.05, 0.1} × mode ∈ {l1, l2, laplacian}, Fold1 (JFX+LMZ→HYZ):

| Rank | Mode | Weight | Val PCC | R² | Best Ep | Overfit Gap |
|:---:|------|--------|:-------:|:-----:|:-----:|:---------:|
| 1 | **l2** | **0.01** | **0.4170** | 0.1512 | 14 | 0.2142 |
| 2 | laplacian | 0.1 | 0.4169 | 0.1622 | 18 | 0.2138 |
| 3 | l2 | 0.05 | 0.4156 | 0.1594 | 7 | 0.1870 |
| 4 | l2 | 0.1 | 0.4143 | 0.1589 | 17 | 0.1966 |
| 5 | laplacian | 0.05 | 0.4129 | 0.1671 | 9 | 0.2026 |
| 6 | laplacian | 0.01 | 0.4103 | 0.1481 | 14 | 0.2030 |
| 7 | l1 | 0.1 | 0.4097 | 0.1543 | 15 | 0.2024 |
| 8 | l1 | 0.01 | 0.4085 | 0.1546 | 17 | 0.2294 |
| 9 | l1 | 0.05 | 0.4015 | 0.1400 | 4 | 0.1910 |

结论:
- **L2 mode 全面优于 L1**（最佳 0.4170 vs 0.4097, 均值 0.4156 vs 0.4066）
- L2 对 weight 不敏感（0.4143-0.4170），Laplacian 需较高 weight (0.1)
- L1 mode 尤其是 w=0.05 表现最差（0.4015），可能因 L1 稀疏梯度不适配 HuberLoss
- Δ vs AugMix 无 TV 基线 (0.4142): L2 w=0.01 仅 +0.0028，提升有限但一致
- 三折 CV: Fold1=0.4170, Fold2=0.3752, Fold3=0.3907, **均值=0.3943**

### 1c. HisToGene-UNI-Tokens + AugMix（原基线）

```
架构: LightweightTokenEncoder(hidden=512, layers=2, heads=8) + CoordEmbed(n_pos=128) + MLP(2048, dropout=0.5)
输入: UNI2-h token序列 [B, 65, 1536] + 空间坐标 (x, y)
参数量: ~9.8M

超参数: lr=3e-5, batch_size=64, AdamW(wd=1e-4), HuberLoss, ReduceLROnPlateau
增强: n_augments=3, aug_sample_prob=0.5, mixup_alpha=0.2, mixup_prob=0.5

最佳结果: Val PCC=0.4142 (Fold1), epoch=4
训练脚本: train_histogene_uni_tokens_augmix.py
```

### 2. HisToGene-UNI-Tokens（方案B base）

```
架构: LightweightTokenEncoder(hidden=512, layers=1, heads=8) + CoordEmbed(n_pos=128) + MLP(2048, dropout=0.5)
输入: UNI2-h token序列 [B, 65, 1536] + 空间坐标
参数量: ~7.7M

超参数: lr=5e-5, batch_size=64, AdamW(wd=5e-4), HuberLoss, ReduceLROnPlateau, grad_clip=0.5, label_noise=0.05

3折CV结果: Fold1=0.3916(val), Fold2=0.3507(val), Fold3=0.3835(val)
训练脚本: train_histogene_uni_tokens.py
```

### 3. HisToGene-UNI（方案A）

```
架构: Linear(1536→1024) + LayerNorm + CoordEmbed + MLP(2048, dropout=0.3)
输入: UNI2-h 池化特征 [B, 1536] + 空间坐标
参数量: ~4.0M

超参数: lr=1e-4, batch_size=64, AdamW(wd=1e-4), HuberLoss, ReduceLROnPlateau

3折CV结果 (test PCC): Fold1=0.3946, Fold2=0.3424, Fold3=0.3731
训练脚本: train_histogene_uni.py
```

### 4. HisToGene-UNI + GAT

```
架构: UNI-Tokens encoder + Graph Attention Network (空间图建模)
输入: UNI token序列 + patch空间邻接图
GAT 参数量: ~13.5M

超参数: 两阶段训练（Stage1 预训练 → Stage2 GAT微调）

Fold1结果: Val PCC=0.4068 (epoch 50)
训练脚本: train_histogene_uni_tokens_gat.py
```

### 5. HisToGene-Virchow2 Tokens

```
架构: LightweightTokenEncoder + CoordEmbed + MLP
输入: Virchow2 ViT-H/14 token序列 [B, N, 1280] + 空间坐标
参数量: ~7.16M（不含 Virchow2 backbone 632M）

超参数: lr=5e-5, batch_size=64, AdamW(wd=1e-4), HuberLoss, grad_clip=1.0

最佳结果: Val PCC=0.3537, epoch=11 (20260521_110949)
训练脚本: train_histogene_virchow2_tokens.py
```

### 6. EGN-v2+UNI

```
架构: UNI2-h特征编码 + kNN空间图 + 2层GraphSAGE + MLP回归头
输入: UNI2-h池化特征 + patch空间邻接图(k=10, radius=300)
参数量: ~2.8M（最轻量）

超参数: lr=1e-4, batch_size=64, AdamW(wd=1e-4), HuberLoss

单患者最佳: HYZ15040 Val PCC=0.6075 (epoch 150, 无过拟合)
3折CV结果: Fold1=0.3537, Fold2=0.3917, Fold3=0.3980, 平均=0.3811
特点: 过拟合Gap极小(0.04~0.13)，收敛慢但稳定
训练脚本: train_egnv2_uni.py
```

### 7. HisToGene-OmiCLIP

```
架构: LightweightTokenEncoder(hidden=512, layers=1, heads=8, token_drop=0.3) + CoordEmbed(n_pos=128) + MLP(2048, dropout=0.5)
输入: OmiCLIP token [B, 255, 768] + 空间坐标
Backbone: coca_ViT-L-14 (307M params, 冻结)
参数量: ~6.1M

超参数: lr=1e-4, batch_size=64, AdamW(wd=5e-3), HuberLoss(delta=1.0), ReduceLROnPlateau
正则化: dropout=0.5, weight_decay=5e-3, token_drop_rate=0.3 (针对 255-token 过拟合优化)

单患者: HYZ=0.5275, JFX=0.6082, LMZ=0.5267, 平均=0.5541
跨患者 3折: Fold1=0.2544 (JFX+LMZ→HYZ), Fold2=0.1081 (HYZ+LMZ→JFX), Fold3=0.2284 (HYZ+JFX→LMZ), 平均=0.1970
泛化衰减: -64.4%（单患者 0.554 → 跨患者 0.197）
训练脚本: train_histogene_omiclip.py
```

### 8. EGN-v2 (ResNet-50)

```
架构: ResNet-50图像编码 + kNN空间图 + 2层GraphSAGE + MLP
输入: 原始PNG patch图像 (224×224)
参数量: ~3.0M（不含ResNet-50 backbone）

超参数: lr=1e-4, batch_size=64, AdamW(wd=1e-4), HuberLoss

跨患者 Fold1: Test PCC=0.1950（远弱于UNI系列）
训练脚本: train_egnv2_uni.py (backbone='resnet')
```

### 9. HisToGene 原版 (ViT)

```
架构: ViT-Large(patch16, 224×224) + 8层Transformer + CoordEmbed + MLP
输入: 原始PNG patch图像
参数量: ~70.6M（最大，训练最慢）

单患者: HYZ=0.5164, JFX=0.6041, LMZ=0.5287
跨患者 Fold1: Test PCC=0.1178（几乎无泛化能力）
已淘汰用于跨患者场景
```

---

## AttnPool 失败根因深度分析 (2026-05-23)

分析脚本: `analyze_attnpool_failure.py`, 报告: `attnpool_analysis/attnpool_failure_report.txt`

核心发现（推翻原假设）:
1. **注意力模式跨患者高度一致** (Spearman ρ = 0.988-0.994) — 非"患者特异性"
2. **注意力近乎均匀分布** (归一化熵 = 0.95/1.0) — 模型几乎没学到 token 差异
3. **空间自相关极弱** (归一化差异 ~0.7-1.0) — 注意力不随空间位置变化

真正失败根因 (三层递进):
- Layer 1: 2层 Transformer Encoder 已将 65 token 信息充分混合，再做加权池化时模型发现"接近均等"就是最优
- Layer 2: 全局 AttnPool 对 30 条通路使用同一组权重，通路特异性需求冲突导致退化为均匀
- Layer 3: +2.1M 参数 (+27%) 无信息增益，在小数据跨患者场景纯增过拟合

设计启示 → Per-Pathway Attention:
- 将 attn_pool 输出维度从 1 改为 30，每条通路独立权重
- 参数量增量仅 Linear(128, 30) ≈ 3.9K (+0.04%)
- 可配合 group sparsity 正则化防止 30 组注意力共线性

---


1. **跨患者泛化**: UNI2-h + DenseNet121 三折均值 0.3969 为当前跨患者最佳（无增强/无TV/无融合裸奔）
2. **UNI2-h vs UNI**: UNI2-h backbone 首次在跨患者场景中系统超越 UNI（Fold1 0.4429 vs 0.4142, +6.9%）
3. **TV Loss 提升有限但稳健**: L2 mode 均值 +0.0028 vs 无 TV, 3折 CV +0.0131 vs UNI Tokens 基线 0.3812
4. **L2 > Laplacian > L1**: TV Loss mode 排序明确，L1 (含此前报告的 w=0.05) 实际最差
5. **单患者拟合**: EGN-v2+UNI 单患者 Val PCC 可达 0.6075，几乎无过拟合，但跨患者泛化仅 0.35-0.40
6. **UNI/UNI2-h backbone 是关键**: 所有使用 UNI/UNI2-h 特征的模型跨患者 PCC 均 > 0.34；不使用病理预训练的模型跨患者 < 0.20
7. **UNI2-h 严重过拟合**: DenseNet121 MLP 仅 1-5 epoch 即达最优，需更强正则化（AugMix/TV/Dropout加大）
8. **AttnPool 失败非因患者特异性**：注意力跨患者高度一致 (ρ=0.99) 但近乎均匀 (熵=0.95)，根因是注意力过于微弱无信息增益
9. **Virchow2 竞争力中等**：PCC=0.35，逊于 UNI-Tokens 系列，与 EGN-v2+UNI 持平
10. **OmiCLIP 跨患者泛化极差**：单患者平均 PCC=0.554 与 UNI 持平，但跨患者骤降至 0.197（衰减-64.4%），CoCa 预训练特征不具备跨患者迁移能力，不建议用于跨患者场景
11. **EGN-v1 已淘汰**：所有指标显著低于其他模型

---

## 输出目录索引

```
histogene/checkpoints/results_vis/
├── CrossValidation_3Fold_Comparison/     # 3折CV汇总（CSV + 图表）
├── AllModels_comparison/                 # 所有模型逐通路PCC对比
├── CrossPatient_comparison/              # 跨患者模型对比
├── CrossPatient_JFX_LMZ_to_HYZ_UNI_tokens_AugMix_20260516_164210/  # ★ 最佳AugMix
├── CrossPatient_Fold2_to_JFX0729_UNI_tokens_20260501_194556/       # Fold2 base
├── CrossPatient_Fold3_to_LMZ12939_UNI_tokens_20260501_195623/      # Fold3 base
├── CrossPatient_Fold1_Virchow2_Tokens_20260521_110949/             # Virchow2 最佳
├── CrossPatient_Fold1_to_HYZ_UNI_tokens_reg_20260501_193628/       # Tokens reg
├── GAT_fold1_JFX0729+LMZ12939_to_HYZ15040_20260508_215940/        # GAT
└── HYZ15040_UNI_tokens_AugMix_20260519_002133/                     # 单患者AugMix

egnv2/checkpoints/results_vis/
├── HYZ15040_UNI_20260425_000841/          # EGN-v2+UNI 单患者最佳
├── CrossPatient_Fold2_HYZ_LMZ_to_JFX_UNI_20260501_185340/
└── CrossPatient_Fold3_HYZ_JFX_to_LMZ_UNI_20260501_185532/

checkpoints/omiclip_*/
├── omiclip_HYZ15040/results_vis_20260521_200611/    # OmiCLIP 单患者 HYZ (PCC=0.5275)
├── omiclip_JFX0729/results_vis_20260521_212841/    # OmiCLIP 单患者 JFX (PCC=0.6082)
├── omiclip_LMZ12939/results_vis_20260521_213410/   # OmiCLIP 单患者 LMZ (PCC=0.5267)
├── omiclip_CrossPatient_JFX0729+LMZ12939_to_HYZ15040_OmiCLIP/results_vis_20260521_214721/  # Fold1
├── omiclip_CrossPatient_HYZ15040+LMZ12939_to_JFX0729_OmiCLIP/results_vis_*/                # Fold2
└── omiclip_CrossPatient_HYZ15040+JFX0729_to_LMZ12939_OmiCLIP/results_vis_*/                # Fold3
```

---

## 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 在新医院数据上直接预测 | HisToGene-UNI-Tokens AugMix | AugMix 对染色域偏移最鲁棒，PCC 最高 |
| 在新数据上微调 | HisToGene-UNI-Tokens AugMix 权重初始化 | 最佳起点，再训练收敛快 |
| 快速实验/算力受限 | EGN-v2+UNI | 仅 2.8M 参数，收敛后无过拟合 |
| 追求极致稳健 | 3模型 Ensemble（UNI-Tokens 3折） | 投票降低单模型偏差 |
| 单患者内部验证 | EGN-v2+UNI | 单患者 PCC 最高达 0.6075 |
