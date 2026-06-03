# CLAUDE.md — PFMval

必读上下文：`.qoder/basic_rule.md`（硬性规则）→ `.qoder/experience.md`（踩坑经验）→ `README.md`（项目门面）
维护方案：`.claude/maintenance-plan.md` | 技术参考：`.qoder/repowiki/zh/content/`（按需查阅）

## 项目概述

食管癌 H&E 病理切片 → 30 条基因通路 ssGSEA 活性评分预测。3 患者：HYZ15040 / JFX0729 / LMZ12939。

## 模型体系

| 模型 | 状态 | 特征 | 单患者Val PCC | 跨患者Test PCC |
|------|------|------|:---:|:---:|
| UNI2-h + DenseNet121 CLS (frozen) | **主力** | 1536维 | 0.5236 | **0.3969** (三折均值) / Fold1 0.4113 |
| UNI2-h LoRA 渐进解冻 | **实验** | 1536维 | **0.5462** | 待验证 |
| HisToGene-UNI Token + AugMix | 活跃 | 1536维 | 0.5217 | 0.4142 (Fold1) |
| HisToGene-UNI+GAT | 实验完成 | 1536维 | — | 0.4068 (提升不显著) |
| EGN-v2+UNI | 活跃 | 1536维 | — | 0.1950 |
| Virchow2 | **已关闭** | 1280维 | — | 0.3516 (天花板~0.35) |
| OmiCLIP | **暂停(P3)** | 768维 | 0.55 | 0.19 (单患者强跨患者崩) |
| EGN-v1 | **已淘汰** | — | — | 所有对比排除 |

> 当前最佳 frozen 基线：UNI2-h + DenseNet121 CLS 三折均值 **0.3969**，Fold1 **0.4113**（已确认）。LoRA Stage 1 单患者 **0.5462**（+0.0235 vs frozen），跨患者 Fold1 为下一关键验证。

## 核心铁律

1. **受保护目录禁改**：`histogene/`、`egnv1/`、`egnv2/` 下文件严禁修改
2. **EGN-v1 已淘汰**：除非用户明确要求，否则排除
3. **路径不修正 typo**：`patch_noov_spilt` 是原始拼写
4. **val_loss 选最优 epoch**：非 val_pcc 最大
5. **predictions.csv 列名**：`true_xxx`/`pred_xxx` 格式
6. **generate_full_report 只调用一次**：避免重复创建时间戳目录
7. **禁止硬编码路径**：所有路径通过 `config_utils` 获取，迁移只需改 `config.yaml`

## 运行环境

| 用途 | Python 路径 |
|------|-----------|
| HisToGene/UNI/Virchow2/LoRA 在线训练 | `C:\Program Files\Python313\python.exe` |
| EGN-v2/GAT (需 PyG) | `D:\conda_envs\pfmval_py310\python.exe` |
| 服务器 (RTX 4080) | `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe` |

- **CUDA**：11.8 (本机 RTX 4060 8GB) / 12.4 (服务器 RTX 4080 16GB)
- **编码铁律**：训练命令必须加 `PYTHONIOENCODING=utf-8`，在线训练额外加 `-u`（无缓冲输出，否则 epoch 日志延迟）：
  ```bash
  PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" -u train_online_cls.py ...
  ```

## 关键经验

- 跨患者衰减严重：2 患者训→1 患者测，PCC 衰减 31%~77%
- `dataset_name` 从 `--patient` 自动推导，不硬编码
- evaluate() 必须同时输出 PCC、MAE、R²
- 在线训练过拟合极快：有效窗口仅 1-2 epoch，Train-Val Gap 是最有用的监控指标
- HF 离线加载需 `HF_HUB_OFFLINE=1`（网络不可达时避免重试等待）
- LoRA 训练需 `batch_size=1 + grad_accum=2 + AMP` 适配 8GB 显存

## 下一步方向

| 优先级 | 任务 |
|:---:|------|
| **P0** | LoRA 跨患者 Fold1 验证 + 服务器 VPN 连通 + Phase 3 预后最小闭环 |
| **P1** | LoRA Stage 2/3 + 正则化（rank↓/Dropout↑/AugMix/TV Loss） |
| **P2** | 9 患者到齐（~6月底）后重训 + RL 方案 D BO 通路选择 |
| **P3** | 暂停：GAT 深化 / AttnPool / OmiCLIP 跨患者 / 多教师融合 |

## 在线训练脚本速查

| 文件 | 用途 |
|------|------|
| `train_online_cls.py` | CLS 模式训练（frozen/lora/stage2/stage3） |
| `train_online_tokens.py` | Token 模式训练（同上） |
| `model_online_cls.py` / `model_online_tokens.py` | 模型定义 |
| `lora_utils.py` | 手工 LoRA（注入/merge/解冻/参数管理，~400行） |
| `dataset_online.py` | 在线图像 Dataset（坐标解析 + 多患者合并） |

## 训练控制

- **暂停训练**：创建空文件 `PAUSE_TRAINING`
- **断点续训**：`--resume <checkpoint_path>`
- **监控原则**：子代理执行训练时，观察 1-2 epoch 确认正常即返回，不持续跟踪

## 数据与路径

所有路径通过 `config_utils.py` 获取：`get_patient_paths(patient, backbone)` / `get_fold_config(fold)` / `get_project_root()`。
核心数据：
```
data_new_3ST/patch_noov_spilt/{patient}_noov_split/   # HE 切片 patches
uni2h_cache/{patient}/{train,val}/                      # UNI 特征缓存
{patient}_ssGSEA_scores_zscore.csv                      # ssGSEA 标签
```
预训练权重：`pretrained_omiclip/checkpoint.pt` (7.1GB) | `virchow2_repo/model.safetensors` (2.4GB)

## 服务器 (2026-06-03 部署完成)

| 项目 | 值 |
|------|-----|
| IP / 用户 | 117.68.10.96 / AIPatho1 |
| 项目路径 | `D:\AIPatho\qzs\pfmval_deploy_git\` |
| Python venv | `C:\Users\AIPatho1\pfmval_env\Scripts\python.exe` |
| PyTorch | 2.6.0+cu124, CUDA 12.9 |
| GPU | RTX 4080 (16GB) |
| HF 缓存 | `D:\AIPatho\shared\.cache\huggingface\` (HF_HOME) |
| GitHub | `qzs610038-star/AI-esophagus-cancer-basedon-HF-HT` (公开) |
| 代理 | Clash Verge 7897 (HTTP/HTTPS) |
| 数据 patches | `D:\AIPatho\qzs\data-phase2\patch\{患者}_noov_split\` |
| ssGSEA 标签 | `D:\AIPatho\qzs\data-phase2\ssGSEA_zscore\{患者}_ssGSEA_zscore.csv` |

> 训练命令模板和配置参数速查：[[server-environment-quickref]]。部署日志：[[server-deploy-status]]。
> **代码同步**：本地通过 **SSH** 直推 GitHub（`git@github.com:qzs610038-star/...`）。HTTPS + Clash 代理在大流量传输时会 TLS 断连，已切换为 SSH 协议（密钥：`~/.ssh/pfmval_server`）。服务器同步用 `git pull`。推送故障排查参考 `.claude/skills/git-rescue/SKILL.md`。

## 部署脚本规范（2026-06-04 踩坑总结）

- **PowerShell 5.1 编码铁律**：服务器 PowerShell 脚本必须 **全英文**，不得包含中文等非 ASCII 字符。Windows 中文版默认编码为 GBK，UTF-8 without BOM 的文件会被错误解码导致解析器崩溃
- **命令链执行**：PowerShell 5.1 不支持 `&&`，需用 `Start-Process cmd.exe -ArgumentList "/c", "cmd1 && cmd2"` 包装
- **Checkpoint 目录名规则**：`checkpoints/online_cls/{mode}_r{rank}_{dataset_name}/`。仅改变 `--lora_dropout` 不改变 mode/rank/dataset_name → 目录名不变 → 会覆盖先前结果。必须显式传 `--dataset_name` 区分（如 `online_cls_cross_fold1_d01`）
- **批量实验调度**：参考 `deploy/run_nightly_experiments.ps1` — 队列定义 + 依赖检查 + 已完成跳过 + 自动汇总 CSV
- **SSH 密钥维护**：`~/.ssh/pfmval_server` / `pfmval_server_local` 可能过期，定期验证连通性

## PowerShell / Git

- PowerShell 不支持 `&&`，用 `;` 分隔或用 `cmd.exe /c` 包装；Conda 激活需 `conda.exe shell.powershell hook`
- Git 忽略：`data_new_3ST/`、`*.pth`、`.venv/`、缓存目录、`*.log`
- 训练输出保存至 `{model_dir}/checkpoints/results_vis/{dataset}_{timestamp}/`，不覆盖历史
