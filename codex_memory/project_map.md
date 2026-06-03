# PFMval 项目架构摘要

更新日期：2026-05-30

## 项目目标

PFMval_new 是食管癌 H&E 病理切片到空间转录组通路活性预测的研究项目。

- Phase 2 当前主线：H&E patch -> 30 条通路 ssGSEA Z-score，核心指标为 PCC。
- Phase 3 下一主线：H&E -> 虚拟通路分数 -> 患者预后二分类，核心指标为 AUC。
- 当前 ST 数据为 3 个患者：HYZ15040、JFX0729、LMZ12939。
- 9 患者完整数据预计 2026 年 6 月底前到齐。

## 关键目录

- `01_指南与解读/`：项目学习指南、部署方案、分析报告。
- `02_组会汇报/`：周报与汇报材料。
- `.claude/`：Claude Code 自动化、skills、hooks、维护方案。
- `.qoder/`：硬性规则、经验索引、结构化 repowiki。
- `experiments/`：实验脚本与结果。
- `reports/`：报表与图。
- `deploy/`：本地到服务器的同步、部署、远程训练脚本。
- `data_new_3ST/`：三患者原始数据。注意路径拼写 `patch_noov_spilt` 不可修正。
- `histogene/`、`egnv1/`、`egnv2/`：受保护只读目录，默认禁止修改。
- `uni2h_cache_tokens/`、`uni2h_cache_tokens_aug/`、`omiclip_cache/`、`virchow2_cache_tokens/`：特征缓存。

## 核心代码入口

- `config.yaml`：唯一配置入口，服务器迁移也优先改这里。
- `config_utils.py`：路径和配置读取工具，新脚本必须通过它取路径。
- `train_histogene_uni_tokens_augmix.py`：当前主力训练脚本，支持 AugMix、MixUp、TV Loss、跨患者和三折。
- `model_uni_tokens.py`：HisToGene UNI token 模型，轻量 Transformer token encoder + 坐标嵌入 + MLP 回归头。
- `dataset_uni_tokens_augmix.py`：UNI token 数据集，支持训练时增强 token 随机采样。
- `spatial_tv_loss.py`：空间 TV / Laplacian 平滑正则。
- `ensemble_late_fusion.py`：UNI 与 Virchow2 等模型晚期融合。
- `visualize_results.py`：结果可视化，依赖 `predictions.csv` 的 `true_xxx` / `pred_xxx` 列名约定。

## 当前模型判断

- 当前跨患者最佳：HisToGene-UNI-Tokens + AugMix + TV L2 w=0.01 + Virchow2 融合，Fold1 PCC=0.4242。
- 当前跨患者最佳单模型三折均值：UNI AugMix + TV L2 w=0.01，PCC=0.3943。
- UNI backbone 是跨患者泛化关键，非 UNI 模型跨患者通常明显掉点。
- OmiCLIP 单患者表现可观，但跨患者均值约 0.197，不建议作为跨患者主线。
- AttnPool 在跨患者上已有负向经验，除非有强证据，不应贸然复活为默认路径。
- EGN-v1 已淘汰，除非用户明确要求，不进入新分析主线。

## 项目铁律

- 不修改 `histogene/`、`egnv1/`、`egnv2/` 下文件。
- 不修正 `patch_noov_spilt` 拼写。
- 最优 epoch 以 `val_loss` 最小为准，不以 `val_pcc` 最大为准。
- `predictions.csv` 列名必须保持 `true_xxx` / `pred_xxx`。
- Windows 训练命令需设置 `PYTHONIOENCODING=utf-8`。
- PowerShell 不使用 `&&`，用 `;` 分隔。
- 新脚本禁止硬编码项目绝对路径，必须通过 `config_utils.py`。

## 当前工作区注意事项

2026-05-30 初次读取时，工作区已有大量未提交改动，涉及 `.claude`、`.qoder`、文档重组、训练脚本、结果文件和新增在线训练相关脚本。Codex 不应回滚或覆盖这些改动，除非用户明确要求。
