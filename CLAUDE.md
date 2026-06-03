# CLAUDE.md — PFMval 项目指南

## 自动加载层

本文件由 Claude Code 每次会话自动加载。以下文件同为必读上下文，请按顺序加载：

| 优先级 | 文件 | 说明 |
|--------|------|------|
| 1 | `CLAUDE.md`（本文件） | 项目总指南 |
| 2 | `.qoder/basic_rule.md` | 项目硬性规则（Qoder 迁移） |
| 3 | `.qoder/experience.md` | 经验索引 + 关键踩坑记录 |
| 4 | `README.md` | 项目门面，供人类阅读 |

> **维护方案**：`.claude/maintenance-plan.md` — 分 P0-P3 四级，触发式更新优先于定期巡检。
> `.qoder/repowiki/` 为结构化技术参考（当前状态），按需查阅，不自动加载。

## 项目概述

基于食管癌H&E病理切片+空间转录组数据，预测30条基因通路ssGSEA活性评分。3个患者数据集：HYZ15040、JFX0729、LMZ12939。

速览项目，请阅读 README.md
技术参考，请查阅 `.qoder/repowiki/zh/content/`

## 模型体系

| 模型 | 状态 | 特征 | 说明 |
|------|------|------|------|
| HisToGene-UNI (frozen) | 活跃 | 1536维 | 主力，跨患者泛化最强 |
| UNI2-h LoRA 渐进解冻 | **实验** | 1536维 | 在线训练，单患者Val PCC=0.5462，跨患者待验证 |
| EGN-v2+UNI | 活跃 | 1536维 | 跨患者PCC 0.195 |
| HisToGene-UNI+GAT | 实验完成 | 1536维 | 提升不显著(-0.27%) |
| Virchow2 | **已关闭** | 1280维 | 四轮实验 PCC 天花板~0.35，低于UNI~0.40 |
| OmiCLIP | **暂停(P3)** | 768维 | 单患者0.55→跨患者0.19(-64%)，等待LoRA验证完成后恢复 |
| HisToGene 原版 | 活跃 | ViT | 基线，~70.6M参数 |
| EGN-v1 | **已淘汰** | — | 所有对比分析排除 |

当前最佳：UNI2-h + DenseNet121 CLS (frozen) 跨患者3折平均PCC=**0.3969**；UNI2-h LoRA Stage 1 单患者Val PCC=**0.5462**。

## 核心铁律

1. **受保护目录禁改**：`histogene/`、`egnv1/`、`egnv2/` 下文件严禁修改，适配通过根目录独立文件实现
2. **EGN-v1已淘汰**：除非用户明确要求，否则排除
3. **路径不修正typo**：`patch_noov_spilt` 是原始拼写
4. **val_loss选最优epoch**：非val_pcc最大
5. **predictions.csv列名**：`true_xxx`/`pred_xxx` 格式
6. **generate_full_report只调用一次**：避免重复创建时间戳目录
7. **路径统一管理**：所有脚本的数据路径通过 `config_utils` 函数获取（`get_patient_paths()` / `get_project_root()` 等），**禁止在脚本中硬编码绝对路径**。迁移服务器只需修改 `config.yaml`

## 运行环境

| 用途 | Python路径 |
|------|-----------|
| HisToGene系列 | `C:\Program Files\Python313\python.exe` |
| EGN-v2/GAT(需PyG) | `D:\conda_envs\pfmval_py310\python.exe` |

- **CUDA 11.8**：已从 C 盘迁移至 D 盘，C 盘原路径为目录链接（junction）→ D 盘。PyTorch 2.6.0+cu118 + RTX 4060 Laptop GPU，验证通过（2026-05-19）。
- **编码**：Windows 中文控制台默认 GBK，训练脚本中 Unicode 字符会触发 `UnicodeEncodeError`。运行所有训练命令必须加 `PYTHONIOENCODING=utf-8` 前缀：

```bash
PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" train_histogene_uni_tokens_augmix.py ...
```

## 关键经验

- `dataset_name`默认值陷阱：从`--patient`自动推导，不硬编码
- 跨患者训练：二折交叉（2患者训→1患者测），衰减-31.7%~-77.2%
- UNI特征对EGN-v2提升显著(+28%~+50%)，对HisToGene提升有限(+2.3%~+11.8%)
- 三层交集(patches∩CSV∩cache)样本少，二层交集(cache∩labels)样本更多效果好
- evaluate()必须同时输出PCC、MAE、R²

## 当前最佳性能基线

| 模型 | 单患者Val PCC | 跨患者Test PCC |
|------|-------------|---------------|
| **UNI2-h + DenseNet121 CLS (frozen) 三折平均** | — | **0.3969** |
| HisToGene-UNI Token（方案B AugMix） | 0.5217 | 0.4142 (Fold1) |
| HisToGene-UNI Token 三折平均 | — | 0.3812 |
| HisToGene-UNI Token + GAT | — | 0.4068 (Fold1, 提升不显著) |
| EGN-v2+UNI | — | 0.1950 |

### 在线训练（LoRA 渐进解冻）基线

| 模型 | 单患者Val PCC | 说明 |
|------|:---:|------|
| UNI2-h LoRA Stage 1 (Epoch 2) | **0.5462** | +0.0235 vs frozen 基线 0.5227 |
| UNI2-h + DenseNet121 CLS (frozen) | 0.5227 | 当前 frozen 单患者基线 |
| OmiCLIP (frozen, 参考) | 0.55 | 单患者强但跨患者崩(-64%) |

## 下一步方向

- **P0 短期**：LoRA 渐进解冻跨患者验证 + 服务器连通性 + Phase 3 预后最小闭环
- **P1 中期**：LoRA Stage 2/3（末层解冻）、正则化对抗过拟合（rank↓ + Dropout↑ + AugMix + TV Loss）
- **P2 远期**：9 患者到齐（预计 2026年6月底）后重训 + RL 通路选择（方案 D BO 搜索）
- **P3 暂停**：GAT 深化 / AttnPool / OmiCLIP 跨患者 / 大规模多教师融合

## 训练控制机制

- **暂停训练**：创建 `PAUSE_TRAINING` 空文件（PowerShell: `New-Item -Path 'PAUSE_TRAINING' -ItemType File -Force`）
- **断点续训**：`--resume <checkpoint_path>` 参数
- **通知方式**：plyer 桌面 Toast 弹窗，禁用 Ctrl+C 中断
- **监控节约原则**：委派子代理执行训练时，只需观察前 1-2 轮 epoch 确认正常启动（loss 正常下降、无报错），随后立即返回，不持续跟踪整个训练过程

## OmiCLIP (Loki) 经验（2026-05-19）

- 模型：coca_ViT-L-14（CoCa 架构），open_clip 2.26.1 加载
- checkpoint key 是 `state_dict`（非 `model`）
- vision encoder：`model.visual`（307M 参数）
- token 输出：`model.visual(x)[1]` → [B, 255, 768]
- CLS 输出：`model.encode_image(x)` → [B, 768]
- 预处理：224×224，OPENAI_DATASET_MEAN/STD
- 关键坑点：PyTorch 2.0.1 无法处理中文路径写文件；total_mem→total_memory；HF 国内用 hf-mirror.com；dataset_uni_tokens.py 有 1536 维硬编码断言需自定义 Dataset
- 权重：`pretrained_omiclip/checkpoint.pt`（7.14GB）
- 缓存：`omiclip_cache/{patient}/{train|val}/*.pt`

## PowerShell 注意事项

- 不支持 `&&`，使用 `;` 分隔语句
- Conda 激活：`(& "D:\miniconda\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression`

## 数据路径

```
data_new_3ST/patch_noov_spilt/{patient}_noov_split/   # 三患者patch
uni2h_cache/{patient}/train/ 和 val/                    # UNI特征缓存
{patient}_ssGSEA_scores_zscore.csv                       # ssGSEA标签
```

## 路径配置系统（2026-05-19 迁移重构）

所有数据路径由 `config_utils.py` 统一管理，`config.yaml` 是唯一配置入口。

### 关键函数（config_utils.py）

| 函数 | 用途 | 示例 |
|------|------|------|
| `get_project_root()` | 获取项目根目录 | 替代 `r"d:\AI..."` |
| `get_patient_paths(patient, backbone)` | 获取患者完整路径字典 | `get_patient_paths('HYZ15040', 'uni_tokens')` |
| `get_fold_config(fold)` | 获取三折交叉验证配置 | `get_fold_config(1)` → `{train: [JFX, LMZ], test: HYZ}` |
| `get_output_dir(subdir)` | 获取输出目录 | checkpoint 路径 |
| `get_omiclip_checkpoint_path()` | OmiCLIP 权重路径 | 替代硬编码 `D:/AI.../checkpoint.pt` |
| `get_histogene_dir()` / `get_egnv2_dir()` | 子项目目录 | 替代 `str(_PROJECT_ROOT / "histogene")` |

### 路径解析优先级

患者路径按以下顺序查找，未配置则自动 fallback：
1. `config.yaml` → `patients.{name}.patches_dir` / `labels_path`（绝对值覆盖）
2. `config.yaml` → `paths.patch_base` / `paths.ssgsea_base` + 患者 subdir
3. 项目根目录下的默认路径（本地零配置运行）

### 编写新脚本规则

```python
# ✅ 正确：通过 config_utils 获取路径
from config_utils import get_patient_paths, get_project_root, get_fold_config
pc = get_patient_paths('HYZ15040', backbone='uni_tokens')
train_dir = pc['train_patches']  # 自动解析为正确路径

# ❌ 错误：硬编码路径
_PATCH_BASE = str(_PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt")  # 禁止
BASE_DIR = r"d:\AI空间转录病理研究\PFMval_new"                        # 禁止
```

### config.yaml 结构

- `data:` 节：受保护文件（histogene/、egnv2/）路径，保持向后兼容
- `paths:` 节：新脚本的数据根目录（服务器上取消注释填写实际路径）
- `patients:` 节：每位患者的子路径配置，支持绝对路径覆盖

## Git规则

忽略：`data_new_3ST/`、`*.pth`、`.venv/`、缓存目录、`*.log`、`training_status_*.txt`、`temp_*.py`

训练输出：保存至`{model_dir}/checkpoints/results_vis/{dataset}_{timestamp}/`，不覆盖历史结果。
