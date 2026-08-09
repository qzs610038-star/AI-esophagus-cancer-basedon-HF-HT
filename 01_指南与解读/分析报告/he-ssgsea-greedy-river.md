# PFMval 项目服务器迁移方案

## Context

将 PFMval 项目从本地 Windows 迁移到 Linux 服务器训练。核心问题：21+ 个文件硬编码本地路径，`histogene/`、`egnv1/`、`egnv2/` 受保护禁改。目标：迁移后只需编辑 config.yaml 即可运行所有训练，且**本地训练完全不受影响**。

---

## 兼容性分析：为什么本地训练不会被破坏

**受保护文件对 config.yaml 的依赖链**：

| 受保护文件 | 调用的函数 | 读取的 config 键 |
|-----------|-----------|-----------------|
| `histogene/train.py:37-43` | `get_data_paths()` | `data.train_patches_dir`, `data.val_patches_dir`, `data.labels_csv_zscore` |
| `histogene/train_uni.py:36-42` | `get_data_paths()` | 同上 |
| `egnv2/train.py:47-49` | `get_data_paths()` | 同上 |
| `egnv1/train.py:56-58` | `get_data_paths()` | 同上 |
| `histogene/infer.py:30-32` | `get_data_paths()` | 同上 |
| `histogene/infer_uni.py:28-30` | `get_data_paths()` | 同上 |

这些文件**不能修改**。它们依赖 `config.data` 节中的 `train_patches_dir`、`val_patches_dir`、`labels_csv_raw`、`labels_csv_zscore` 四个键。

**约束 → 策略**：采用**纯增量方案**——只加不改。保留现有 config.yaml 的 `data:` 节不动，在下方新增 `paths:` 和 `patients:` 节。新增的 `get_patient_paths()` 函数在无显式配置时，自动计算与当前硬编码完全一致的本地路径。Phase 3 的重构将硬编码路径替换为函数调用，而这些函数在本地环境的默认行为完全相同。

**本地训练零影响验证逻辑**：
```
修改前: PATIENT_CONFIG['HYZ15040']['train_patches'] 
        = _PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "train_patches"

修改后: get_patient_paths('HYZ15040', 'uni_tokens')['train_patches']
        → 读取 config.yaml，paths.patch_base 未配置 → fallback 到默认值
        = _PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt" / "HYZ15040_noov_split" / "train_patches"
        
结果: 完全相同的路径
```

---

## Phase 1: config.yaml — 纯增量扩展

**文件**: `config.yaml`

保留现有 `data:` 节完全不变，在其后新增 `paths:` 和 `patients:` 节：

```yaml
# ========== 以下为现有内容，保持不变 ==========
data:
  train_patches_dir: "./data_new_3ST/patch_noov_spilt/HYZ15040_noov_split/train_patches"
  val_patches_dir: "./data_new_3ST/patch_noov_spilt/HYZ15040_noov_split/val_patches"
  labels_csv_raw: "./data_new_3ST/ssGSEA_zscore/HYZ15040_ssGSEA_zscore.csv"
  labels_csv_zscore: "./data_new_3ST/ssGSEA_zscore/HYZ15040_ssGSEA_zscore.csv"

huggingface:
  token: ""
  local_only: false

training:
  device: "auto"

# ========== 以下为新增内容 ==========
# ---- 路径根目录配置（不配置则使用项目根目录下的默认路径）----
paths:
  # 补丁与标签根目录。留空或注释 = 使用默认值（与现有硬编码一致）
  # patch_base: "/data/server/HE_patches"
  # ssgsea_base: "/data/server/ssGSEA_scores"
  #
  # 特征缓存根目录
  caches:
    # uni_tokens: "/data/caches/uni_tokens"
    # uni_tokens_aug: "/data/caches/uni_tokens_aug"
    # uni_cls: "/data/caches/uni_cls"
    # omiclip: "/data/caches/omiclip"
    # virchow2: "/data/caches/virchow2"
  #
  # 外部资源
  resources:
    # omiclip_checkpoint: "/data/models/omiclip/checkpoint.pt"
    # hf_cache: "/data/huggingface_cache"

# ---- 患者级路径覆盖（服务器上数据位置不同时使用）----
patients:
  HYZ15040:
    # 以下为默认值，留空使用 paths.patch_base + patches_subdir 拼接
    patches_subdir: "HYZ15040_noov_split"
    labels_csv: "HYZ15040_ssGSEA_zscore.csv"
    # 服务器覆盖示例（取消注释并填写实际路径）：
    # patches_dir: "/data/HE_images/HYZ15040"       # 直接指定补丁目录
    # labels_path: "/data/ssGSEA/HYZ15040_scores.csv" # 直接指定标签文件
  JFX0729:
    patches_subdir: "JFX0729_noov_split"
    labels_csv: "JFX0729_ssGSEA_zscore.csv"
  LMZ12939:
    patches_subdir: "LMZ12939_noov_split"
    labels_csv: "LMZ12939_ssGSEA_zscore.csv"
```

**关键设计**：所有新增配置项默认注释或为空。本地环境无需任何修改即可正常运行。迁移服务器时只需取消注释并填写服务器实际路径。

---

## Phase 2: config_utils.py — 纯增量扩展

**文件**: `config_utils.py`

保留所有现有函数不变（`get_project_root`、`load_config`、`resolve_path`、`get_data_paths`、`get_hf_config`、`get_device`），新增以下函数：

### 2a. `get_patient_paths(patient, backbone=None, config=None)`

核心函数，返回单患者完整路径字典。路径解析的 fallback 链确保本地零配置运行：

```
patches_dir 解析优先级：
  1. config.yaml patients.{patient}.patches_dir（绝对路径覆盖，服务器用）
  2. config.yaml paths.patch_base + patients.{patient}.patches_subdir
  3. {PROJECT_ROOT}/data_new_3ST/patch_noov_spilt/{patient}_noov_split（本地默认）

labels_csv 解析优先级：
  1. config.yaml patients.{patient}.labels_path
  2. config.yaml paths.ssgsea_base + patients.{patient}.labels_csv
  3. {PROJECT_ROOT}/data_new_3ST/ssGSEA_zscore/{patient}_ssGSEA_zscore.csv

cache 目录解析优先级：
  1. config.yaml paths.caches.{backbone_key}
  2. {PROJECT_ROOT}/{default_cache_dir}/{patient}/{train,val}（本地默认）
```

返回 dict 键名与当前训练脚本中 `PATIENT_CONFIG` 完全一致：
- `train_patches`, `val_patches`, `labels_csv`
- `token_cache_train`, `token_cache_val`（当 backbone 非 None）
- `token_aug_train`, `token_aug_val`（当 backbone='uni_tokens_aug'）
- `cache_train`, `cache_val`（别名，兼容 omiclip 命名）

### 2b. 辅助函数

| 函数 | 用途 |
|------|------|
| `get_paths_config(config=None)` | 返回 paths 节所有路径（带 fallback 默认值） |
| `get_all_patient_configs(patients, backbone, config)` | 批量获取多患者路径，返回 `{patient: path_dict}` |
| `get_fold_config(fold)` | 返回三折交叉验证配置 `{train: [...], test: ...}` |
| `get_output_dir(subdir_name)` | 返回输出目录绝对路径 |
| `get_omiclip_checkpoint_path()` | 返回 OmiCLIP checkpoint 路径（默认 `{root}/pretrained_omiclip/checkpoint.pt`） |
| `get_hf_cache_dir()` | 返回 HF 缓存目录 |

### 2c. 更新自检入口

扩展 `if __name__ == "__main__"` 块，增加对三个患者 + 各 backbone 路径的验证输出。

---

## Phase 3: 重构根目录训练脚本（9 个文件）

**通用模式**：将模块级 `_PATCH_BASE` / `_SSGSEA_BASE` / `PATIENT_CONFIG` 常量替换为函数调用。函数在本地环境的默认行为完全相同。

以 `train_histogene_uni_tokens_augmix.py` 为例：

```python
# 修改前（57-90 行）：
_PATCH_BASE = str(_PROJECT_ROOT / "data_new_3ST" / "patch_noov_spilt")
_SSGSEA_BASE = str(_PROJECT_ROOT / "data_new_3ST" / "ssGSEA_zscore")
_TOKEN_CACHE_BASE = str(_PROJECT_ROOT / "uni2h_cache_tokens")
_TOKEN_AUG_CACHE_BASE = str(_PROJECT_ROOT / "uni2h_cache_tokens_aug")
PATIENT_CONFIG = {
    'HYZ15040': {
        'train_patches': os.path.join(_PATCH_BASE, "HYZ15040_noov_split", "train_patches"),
        ...
    },
    ...
}

# 修改后：
from config_utils import get_patient_paths, get_fold_config

def build_patient_configs(backbone='uni_tokens_aug'):
    """延迟构建，与 config.yaml fallback 一致"""
    return {
        p: get_patient_paths(p, backbone=backbone)
        for p in ['HYZ15040', 'JFX0729', 'LMZ12939']
    }
# 在 main() 中调用: PATIENT_CONFIG = build_patient_configs()
```

| 文件 | backbone | 额外改动 |
|------|----------|---------|
| `train_histogene_uni_tokens.py` | `uni_tokens` | 移除 53-79 行 |
| `train_histogene_uni_tokens_augmix.py` | `uni_tokens_aug` | 移除 57-90 行 |
| `train_histogene_uni_tokens_gat.py` | `uni_tokens` | 替换 checkpoint 路径用 `get_output_dir()` |
| `train_histogene_omiclip.py` | `omiclip` | 用 `get_omiclip_checkpoint_path()` |
| `train_histogene_virchow2_tokens.py` | `virchow2` | |
| `train_egnv2_uni.py` | `uni_cls` | 替换 `_EGNV2_DIR` 为 `get_egnv2_dir()` |
| `train_cross_patient_histogene.py` | — | |
| `train_cross_patient_histogene_uni.py` | `uni_cls` | |
| `train_cross_patient_egnv2.py` | — | |

---

## Phase 4: 重构特征提取脚本（5 个文件）

同模式，将硬编码 `PATIENT_PATHS` 替换为 `get_patient_paths()`。

| 文件 | 改动 |
|------|------|
| `extract_uni_tokens.py` | `get_patient_paths(p, backbone='uni_tokens')` |
| `extract_uni_tokens_augmented.py` | `get_patient_paths(p, backbone='uni_tokens_aug')` |
| `extract_uni_features_3st.py` | `get_patient_paths(p, backbone='uni_cls')` |
| `extract_omiclip_features.py` | `get_patient_paths(p, backbone='omiclip')` + `get_omiclip_checkpoint_path()` |
| `extract_virchow2_tokens.py` | `get_patient_paths(p, backbone='virchow2')` |

---

## Phase 5: 重构分析和工具脚本（~18 个文件）

统一模式：`BASE_DIR = r"d:\AI空间转录病理研究\PFMval_new"` → `get_project_root()`

涉及文件：
- **根目录可视化/分析**：`visualize_all_models.py`, `visualize_cross_patient.py`, `visualize_model_comparison.py`, `compare_datasets.py`, `compare_single_vs_multi.py`, `analyze_cross_validation.py`, `organize_existing_results.py`, `data_distribution_analysis.py`
- **tools/ 目录**（需 `sys.path.insert(0, parent)` 找到 config_utils）：`debug_dataset.py`, `analyze_intersection.py`, `check_csv_patient.py`, `diagnose_csv_mismatch.py`, `debug_dataset_loading.py`, `generate_clean_csv.py`, `simulate_train_uni_loading.py`, `FINAL_DIAGNOSIS_REPORT.py`, `final_report.py`, `verify_clean_csv.py`
- **杂项**：`inspect_omiclip.py`, `inspect_omiclip2.py`, `scripts/download_omiclip.py`

---

## Phase 6: 服务器环境

- 新增 `env_histogene.yml`（PyTorch + torchvision + transformers + timm + sklearn + pandas + Pillow + tqdm + pyyaml + opencv-python-headless）
- 新增 `env_egnv2.yml`（上述 + torch_geometric + pytorch-scatter + pytorch-sparse）
- 新增 `setup_server.sh`（可选，conda 创建 + config 验证）

---

## Phase 7: 数据迁移

| 数据 | 大小 | 操作 | 说明 |
|------|------|------|------|
| HE 补丁 + ssGSEA CSV | ~1.5 GB | 服务器已有，config.yaml 指向实际路径 | 分别在 `data:` 节（给受保护文件用）和 `patients:` 节（给新脚本用）配置 |
| UNI token cache | ~6.8 GB | 拷贝 | .pt 无内嵌路径，可移植 |
| UNI AugMix cache | ~27 GB | 服务器重新生成 | 太大；`extract_uni_tokens_augmented.py` 已就绪 |
| UNI CLS cache | ~214 MB | 拷贝 | 体积小 |
| OmiCLIP/Virchow2 cache | ~20 GB | 按需拷贝或重新生成 | 视带宽 |
| OmiCLIP pretrained | 8.7 GB | 服务器 redownload | `scripts/download_omiclip.py` 已就绪 |

### 受保护目录 → 符号链接

`histogene/train.py` 等受保护文件内部也通过 `_PROJECT_ROOT / "data_new_3ST"` 拼接路径。服务器上如数据不在项目根目录下，创建符号链接：

```bash
cd /path/to/PFMval
ln -s /data/actual/HE_patches     data_new_3ST
ln -s /data/actual/caches/tokens  uni2h_cache_tokens
# ... 其他缓存目录同理
```

这样受保护文件完全不受影响。

---

## 迁移操作清单

1. 代码推送到 git → 服务器 clone
2. 编辑 `config.yaml`：
   - 更新 `data:` 节四个路径为服务器实际位置（给受保护文件用）
   - 如需，取消注释 `paths:` 和 `patients:` 中的服务器路径（给新脚本用）
3. 运行 `python config_utils.py` 验证所有路径存在
4. 创建 conda 环境：`conda env create -f env_histogene.yml`
5. 拷贝特征缓存 + 创建符号链接
6. 冒烟测试：`python train_histogene_uni_tokens.py --patient HYZ15040 --epochs 5`

---

## 验证方法

1. **本地回归测试**（修改后立即执行）：
   ```bash
   PYTHONIOENCODING=utf-8 "C:\Program Files\Python313\python.exe" config_utils.py
   # 确认输出与修改前一致，所有路径标注"存在"
   ```
2. `python -c "from config_utils import get_patient_paths; print(get_patient_paths('HYZ15040', 'uni_tokens')['train_patches'])"` — 路径应与当前硬编码完全一致
3. 本地运行 `train_histogene_uni_tokens.py --patient HYZ15040 --epochs 2` 确认训练正常启动
4. 服务器：5 epoch 短训练确认 GPU + 数据加载正常

---

## 不改动的文件

- `histogene/*`、`egnv1/*`、`egnv2/*` — 受保护，通过服务器符号链接适配
- `dataset_uni_tokens.py`、`dataset_uni_tokens_augmix.py` — 已参数化，路径由构造函数传入
- `model_uni_tokens.py`、`model_uni_tokens_gat.py` — 纯模型，无路径依赖
- `config_utils.py` 中现有 5 个函数 — 签名和返回值不变
