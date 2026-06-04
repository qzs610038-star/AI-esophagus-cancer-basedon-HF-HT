# PFMval 经验索引

> 🔒 **必读文件**：本文件在每次会话开始时必须读取。
> 内容仅保留指向性说明，详细经验请按需阅读对应skill文件。

---

## 📂 Skill文件索引

| 文件 | 适用场景 | 路径 |
|------|---------|------|
| 数据处理经验 | 数据加载、路径配置、CSV格式 | `.qoder/skills/数据处理经验.md` |
| 训练工程经验 | 模型训练、调参、环境配置 | `.qoder/skills/训练工程经验.md` |
| 可视化与输出规范 | 结果可视化、报告生成 | `.qoder/skills/可视化与输出规范.md` |
| 组会汇报撰写指南 | 撰写组会汇报文档 | `.qoder/skills/组会汇报撰写指南.md` |

---

## ⚠️ 关键提醒（每次必看）

1. **EGN-v1已淘汰** — 所有对比分析排除EGN-v1
2. **受保护目录禁改** — `histogene/`、`egnv1/`、`egnv2/` 只能新增独立文件适配
3. **路径typo** — `patch_noov_spilt` 是原始拼写，不修正
4. **predictions.csv** — 列名必须为 `true_xxx`/`pred_xxx` 格式
5. **val_loss选模型** — 最佳epoch以val_loss最小为准，非val_pcc最大

---

## 📊 当前最佳性能基线

| 模型 | 单患者Val PCC | 跨患者Test PCC |
|------|-------------|---------------|
| UNI2-h + DenseNet121 CLS LoRA r=8 (Stage1) | **0.5462** | **0.4322 (Fold1)** / 0.3726 (Fold2) / Fold3 待重跑 |
| UNI2-h + DenseNet121 CLS (frozen) | 0.5236 | 0.4113 (Fold1) / Fold2/3 待补 |
| LoRA Stage2 (解冻末2层) | — | 0.4118 (Fold1, =frozen) ❌ |
| LoRA Stage3 (解冻末4层) | — | 0.4056 (Fold1, <frozen) ❌ |
| HisToGene-UNI Token + AugMix | 0.5217 | 0.4142 (Fold1) |
| HisToGene-UNI Token + GAT | — | 0.4068 (Fold1, 不显著) |
| EGN-v2+UNI | — | 0.1950 |

> **2026-06-04 结论**：LoRA Stage1 跨患者有效（+5.1% vs frozen）。**Stage2/3 已放弃**（解冻 backbone 损害泛化）。方向从"渐进解冻"转向"正则化约束"（dropout↑ / rank↓ / AugMix）。详细结果：`results_nightly/online_cls/`

---

## 🔬 下一步方向

| 优先级 | 任务 |
|:---:|------|
| **P0** | LoRA Fold3 重跑（已修复 NaN 保护+CPU 限线程） → 三折均值 |
| **P0** | LoRA Dropout 0.1/0.2/0.3 Cross-Fold1 |
| **P1** | LoRA 低 rank (r=4, r=2) Cross-Fold1 |
| **P1** | Frozen Cross-Fold2/3 基线补齐 |

---

## LoRA 夜间实验结果详情（2026-06-04）

### Cross-Fold1 四模式对比

| 模式 | Val PCC | Epoch | Δ Frozen | Train-Val Gap | 判定 |
|------|:---:|:---:|:---:|:---:|:---:|
| Frozen | 0.4113 | 1 | — | 窄 | 基线 |
| LoRA | **0.4322** | 2 | **+0.0209** | 中 | ✅ 最优 |
| Stage2 | 0.4118 | 1 | +0.0005 | 宽 | ❌ 退回基线 |
| Stage3 | 0.4056 | 1 | -0.0057 | 最宽 | ❌ 低于基线 |

### 关键规律
- **过拟合速度**：Frozen < LoRA < Stage2 < Stage3（参数越多过拟合越快）
- **有效窗口**：仅 1-2 epoch，之后 Train-Val Gap 急剧扩大
- **Fold 差异**：Fold1 (HYZ)=0.4322 > Fold2 (JFX)=0.3726，JFX 更难预测
- **Fold3 失败原因**：LMZ12939 标签极端 z-score → numpy overflow（已修复 NaN 保护）
- **逐通路最强**：ECM(0.725)、MYC(0.719)、Fibrosis(0.688)
- **逐通路最弱**：Interferon_Alpha(0.049)、ifng(0.162)、tgfb(0.165)

### 实验文件存档
- 文本结果：`results_nightly/online_cls/{实验名}/`（training_history / predictions / per_pathway_pcc / summary）
- 批量日志：`results_nightly/batch_*.log` / `summary_*.csv`
- 训练脚本：`train_online_cls.py`（已加 CPU 限线程 + NaN 保护）
- 批量调度：`deploy/run_nightly_experiments.ps1`

---

## 服务器批量实验部署经验（2026-06-04）

### PowerShell 脚本编码
- Windows 中文版 PowerShell 5.1 默认编码为 **GBK**，UTF-8 without BOM 文件中的中文会被错解码 → 字符串边界断裂 → 解析器崩溃
- **铁律**：服务器 .ps1 脚本必须全英文，不含中文/emoji
- Python 训练仍需 `PYTHONIOENCODING=utf-8`（处理日志中的中文路径）

### 命令链执行
- PowerShell 5.1 不支持 `&&` 语句分隔（仅 PowerShell 7+ 支持）
- 需要 `&&` 链接多条命令时，用 `Start-Process cmd.exe -ArgumentList "/c", "cmd1 && cmd2"`
- 不要用 `cmd /c` 的单字符串形式，容易出现引号转义问题

### Checkpoint 目录名冲突
- `train_online_cls.py` 输出目录由 `--mode`、`--lora_rank`、`--dataset_name` 决定
- 仅改 `--lora_dropout` 不改 `--dataset_name` → 同名目录覆盖 → 结果丢失
- **解决**：变体实验必须显式传 `--dataset_name online_cls_cross_fold1_d01`

### 批量实验调度模式
- 脚本：`deploy/run_nightly_experiments.ps1`（全英文，可复用）
- 特性：优先级队列 + 依赖检查 + 已完成跳过 + 自动汇总 CSV
- 启动方式：`powershell -NoProfile -ExecutionPolicy Bypass -File deploy/run_nightly_experiments.ps1`

### SSH 连接
- `~/.ssh/pfmval_server` 密钥认证失败（2026-06-04），需排查服务器端 authorized_keys
- GitHub SSH 推送正常工作（密钥不同）

## OmiCLIP (Loki) 部署经验（2026-05-19）

### 架构探查结论
- 模型：coca_ViT-L-14（CoCa架构），通过 open_clip 2.26.1 加载
- 加载方式：`model = open_clip.create_model('coca_ViT-L-14'); model.load_state_dict(ckpt['state_dict'])`
- checkpoint key 是 `state_dict`（非 `model`）
- `loki.predex.OmiCLIP_Predictor` 在 loki==0.0.1 中不存在
- vision encoder：`model.visual`（307M参数）
- token输出：`model.visual(x)[1]` → [B, 255, 768]
- CLS输出：`model.encode_image(x)` → [B, 768]
- 预处理：224×224，OPENAI_DATASET_MEAN/STD

### 关键坑点
1. PyTorch 2.0.1 无法处理中文路径写文件 → 用 io.BytesIO + Python open() 绕过
2. total_mem属性 → 应为 total_memory
3. HuggingFace Python库国内卡住 → curl.exe + hf-mirror.com
4. dataset_uni_tokens.py 有1536维硬编码断言 → 自定义OmiCLIPTokensDataset

### 文件位置
- 提取脚本：extract_omiclip_features.py（loki_env运行）
- 训练脚本：train_histogene_omiclip.py（Python313运行）
- 权重：pretrained_omiclip/checkpoint.pt（7.14GB）
- 缓存：omiclip_cache/{patient}/{train|val}/*.pt
