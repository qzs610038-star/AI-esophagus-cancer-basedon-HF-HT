# ssGSEA 标签预处理与 JFX 修复数据部署方案

创建日期：2026-06-16  
来源文件：`01_指南与解读/分析报告/数据分布与Z-score适用性分析.md`  
当前状态：已从旧版 HYZ15040 / 8 通路分布分析，重写为可由 Claude 直接执行的 JFX0729 fixed data 部署方案。

---

## 0. 给 Claude 的执行口令

> 请严格按 `01_指南与解读/部署方案/ssGSEA标签预处理与JFX修复数据部署方案_20260616.md` 执行。  
> 本次只处理 repaired JFX0729 fixed data，不要运行旧 `split.py` / `zscore.py`。  
> JFX 修复数据 patch 已无重叠问题，不需要再按 350px 或其他空间距离过滤。  
> 标准 z-score 先作为历史兼容基线落到原训练路径；Yeo-Johnson + z-score 作为 P0 对照分支；Quantile-normal 作为 P1 消融。  
> 训练命令使用 `--num_epochs`，不要使用过时的 `--epochs`。  
> 每一步完成后，把命令、输出路径、样本数、失败/通过状态写入交接记录。

---

## 1. 执行结论

本项目当前患者数据处理方案总体合理，可以继续作为历史基线，但不建议在此刻直接把 z-score 全局替换掉。

推荐部署策略：

1. **JFX fixed data 先按现有训练接口生成标准 z-score 标签**  
   输出仍写入 `data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore.csv`，保证 `config.yaml` 和现有训练脚本无需立刻改动。

2. **JFX patch 划分采用固定随机 90/10，无空间距离过滤**  
   repaired JFX 数据已确认 patch 无重叠问题，旧 `split.py` 的距离过滤不再适用；本次应复制文件，不要移动源文件。

3. **新增标签变换对照，不替代主路径**  
   P0：`Yeo-Johnson + z-score`。  
   P1：`QuantileTransformer(output_distribution="normal")`。  
   不建议把 `RobustScaler` 单独作为第一优先级，因为它只是线性缩放，不能改善偏态/长尾形状，而且会放大部分极端值的绝对尺度。

4. **跨患者实验暂时保持标准 z-score 为主线**  
   当前仓库只直接看到 JFX fixed raw ssGSEA；HYZ/LMZ 主要是已 z-score 文件。若只把 JFX 改成 Yeo-Johnson，而 HYZ/LMZ 仍是标准 z-score，跨患者训练/验证的标签空间会不一致。因此 Yeo-Johnson 分支先做 JFX 单患者对照；跨患者分支只有在 HYZ/LMZ raw ssGSEA 或可靠反变换参数齐备后再扩展。

---

## 2. 旧文件中过时内容的处理

旧 `数据分布与Z-score适用性分析.md` 的主要问题：

| 旧内容 | 当前状态 | 处理方式 |
|---|---|---|
| 基于 `HYZ15040_ssGSEA_scores.csv`、10578 样本、8 通路 | 已不代表当前 3ST / 30 通路任务 | 不再作为执行依据 |
| 使用旧 `zscore.py` | 脚本硬编码 `NUM_TARGET_COLS=8` 和旧 CSV 路径 | 禁止用于本次 JFX fixed data |
| 使用旧 `split.py` | 脚本硬编码 HYZ 路径、`shutil.move`、距离过滤 | 禁止用于本次 JFX fixed data |
| 只从“是否适合 z-score”角度讨论 | 当前需要可落地的数据修复、训练复跑和消融对照 | 改写为部署方案 |

保留的有效认识：

- ssGSEA 分布存在偏态、长尾和异常值风险。
- z-score 对模型训练便利，但不能改变分布形状。
- 需要把标签预处理方式作为实验变量，而不是隐性数据清洗步骤。

---

## 3. 当前数据事实

### 3.1 JFX fixed data 输入

输入目录：

```text
D:\AI空间转录病理研究\PFMval_new\data_new_3ST\JFX_fixed_data
```

应包含：

```text
data_new_3ST/JFX_fixed_data/JFX0729/
data_new_3ST/JFX_fixed_data/JFX0729_ssGSEA.csv
```

当前应验收为：

| 项目 | 期望值 |
|---|---:|
| patch PNG 数 | 7788 |
| raw ssGSEA 行数 | 7788 |
| raw ssGSEA 列数 | 31 |
| ID 列 | `barcode`，处理后改名为 `patch_id` |
| 通路列数 | 30 |
| patch stem 与 CSV ID 交集 | 7788 / 7788 |
| 缺失值 | 0 |

### 3.2 JFX fixed data 分布摘要

对 30 个通路列的 fixed JFX raw ssGSEA 检查结果：

| 指标 | 数值 |
|---|---:|
| `abs(skew) > 1` 的通路数 | 10 |
| `abs(skew) > 2` 的通路数 | 2 |
| `abs(kurtosis) > 3` 的通路数 | 2 |

标准 z-score 后的异常值诊断：

| 指标 | 数值 |
|---|---:|
| z-score 后异常值比例 `>5%` 的通路数 | 5 |
| z-score 后异常值比例 `>10%` 的通路数 | 2 |

最需要关注的通路如下。`skew` 和 `kurtosis` 来自 raw ssGSEA；异常值比例来自标准 z-score 后的诊断。

| pathway | skew | kurtosis | z-score 后异常值比例 |
|---|---:|---:|---:|
| `toxic` | 2.2586 | 7.5346 | 8.3847% |
| `icp` | 2.2557 | 7.2518 | 13.7519% |
| `mhc` | -1.9464 | 1.8478 | 18.2075% |
| `G2M_Checkpoint` | 1.4627 | 1.8023 | 5.7524% |
| `tls` | 1.0831 | 1.8024 | 5.9579% |

### 3.3 标签变换对比

在 JFX fixed raw ssGSEA 上的变换对比：

| transform | max_abs | median_abs_skew | `abs(skew)>1` | median_abs_kurt | `abs(kurt)>3` | 结论 |
|---|---:|---:|---:|---:|---:|---|
| standard z-score | 8.083 | 0.629 | 10 | 0.922 | 2 | 历史基线，兼容性最好 |
| robust scaler | 38.005 | 0.629 | 10 | 0.922 | 2 | 不优先，形状不变且极值更大 |
| Yeo-Johnson + z-score | 5.994 | 0.240 | 2 | 0.700 | 0 | P0 对照，最适合先试 |
| quantile normal | 5.199 | 0.002 | 0 | 0.110 | 0 | P1 消融，分布最正态但可能压平生物差异 |

---

## 4. 输出路径约定

### 4.1 标准主路径

这些路径是现有训练代码默认读取的位置，必须优先生成：

```text
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/train_patches
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/val_patches
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/split_manifest.csv
data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore.csv
data_new_3ST/ssGSEA_zscore/JFX0729_zscore_params.csv
data_new_3ST/ssGSEA_zscore/JFX0729_zscore_params.json
```

注意：保留目录名 `patch_noov_spilt`，不要擅自改成 `split`。这是历史路径拼写，训练代码和配置依赖它。

### 4.2 标签变换对照路径

新增 transform 分支统一写入：

```text
data_new_3ST/label_transforms/JFX0729_v20260616/
```

建议文件：

```text
JFX0729_ssGSEA_standard_zscore.csv
JFX0729_ssGSEA_yeojohnson_zscore.csv
JFX0729_ssGSEA_quantile_normal.csv
JFX0729_label_transform_qc.csv
JFX0729_label_transform_params.json
```

---

## 5. Claude 执行步骤

### Step 0：确认环境和变更范围

```powershell
cd "D:\AI空间转录病理研究\PFMval_new"
$env:PYTHONUTF8 = "1"
git status --short
```

要求：

- 不回滚与本任务无关的已有改动。
- 如果 `scripts/prepare_jfx_fixed_data.py` 已存在，先复用它完成标准路径生成。
- 不使用 `split.py` 和 `zscore.py`。
- 不把 repaired JFX patch 再做 overlap/distance filter。

### Step 1：原始数据一致性检查

```powershell
@'
from pathlib import Path
import pandas as pd

root = Path(r"D:\AI空间转录病理研究\PFMval_new")
patch_dir = root / "data_new_3ST" / "JFX_fixed_data" / "JFX0729"
csv_path = root / "data_new_3ST" / "JFX_fixed_data" / "JFX0729_ssGSEA.csv"

patch_ids = {p.stem for p in patch_dir.glob("*.png")}
df = pd.read_csv(csv_path)
csv_ids = set(df.iloc[:, 0].astype(str))

print("patch_count", len(patch_ids))
print("csv_shape", df.shape)
print("id_col", df.columns[0])
print("target_cols", df.shape[1] - 1)
print("missing_values", int(df.iloc[:, 1:].isna().sum().sum()))
print("intersection", len(patch_ids & csv_ids))
print("patch_not_in_csv", len(patch_ids - csv_ids))
print("csv_not_in_patch", len(csv_ids - patch_ids))

assert len(patch_ids) == 7788
assert df.shape == (7788, 31)
assert df.shape[1] - 1 == 30
assert int(df.iloc[:, 1:].isna().sum().sum()) == 0
assert len(patch_ids & csv_ids) == 7788
'@ | & "C:\Program Files\Python313\python.exe" -
```

通过标准：

```text
patch_count 7788
csv_shape (7788, 31)
target_cols 30
missing_values 0
intersection 7788
```

### Step 2：生成 JFX no-filter split + 标准 z-score

首选复用仓库中的专用脚本：

```powershell
& "C:\Program Files\Python313\python.exe" scripts\prepare_jfx_fixed_data.py `
  --patch-source data_new_3ST/JFX_fixed_data/JFX0729 `
  --label-source data_new_3ST/JFX_fixed_data/JFX0729_ssGSEA.csv `
  --patch-output data_new_3ST/patch_noov_spilt/JFX0729_noov_split `
  --label-output data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore.csv `
  --params-dir data_new_3ST/ssGSEA_zscore
```

说明：

- 当前脚本默认按 `seed=42`、`val=778`、`train=7010` 执行。
- 即使脚本暴露了 `--seed` / `--val-fraction` / `--ddof`，本次不要改默认值，避免和历史复跑口径不一致。
- 如果目标文件已存在，先确认是否需要备份；不要删除 raw fixed data。

验收：

```powershell
@'
from pathlib import Path
import pandas as pd

root = Path(r"D:\AI空间转录病理研究\PFMval_new")
split_dir = root / "data_new_3ST" / "patch_noov_spilt" / "JFX0729_noov_split"
csv_path = root / "data_new_3ST" / "ssGSEA_zscore" / "JFX0729_ssGSEA_zscore.csv"

train = sorted((split_dir / "train_patches").glob("*.png"))
val = sorted((split_dir / "val_patches").glob("*.png"))
df = pd.read_csv(csv_path)
target_cols = list(df.columns[1:])

print("train", len(train))
print("val", len(val))
print("csv_shape", df.shape)
print("max_abs_mean", float(df[target_cols].mean().abs().max()))
print("max_abs_std_delta", float((df[target_cols].std(ddof=1) - 1).abs().max()))
print("max_abs_value", float(df[target_cols].abs().max().max()))
print("train_val_overlap", len({p.stem for p in train} & {p.stem for p in val}))

assert len(train) == 7010
assert len(val) == 778
assert df.shape == (7788, 31)
assert float(df[target_cols].mean().abs().max()) < 1e-8
assert float((df[target_cols].std(ddof=1) - 1).abs().max()) < 1e-8
assert len({p.stem for p in train} & {p.stem for p in val}) == 0
'@ | & "C:\Program Files\Python313\python.exe" -
```

### Step 3：新增 transform 对照输出

如果仓库已有 transform 分析脚本，复用并扩展；否则新建一个小脚本，例如：

```text
scripts/analyze_label_transforms.py
```

实现要求：

- 输入：`data_new_3ST/JFX_fixed_data/JFX0729_ssGSEA.csv`
- ID 列统一改名 `patch_id`
- target columns = 第 2 列到最后一列，必须正好 30 列
- 输出目录：`data_new_3ST/label_transforms/JFX0729_v20260616/`
- 至少输出三类标签：
  - `standard_zscore`
  - `yeojohnson_zscore`
  - `quantile_normal`
- 保存 `JFX0729_label_transform_qc.csv`
- 保存 `JFX0729_label_transform_params.json`
- 对所有输出检查 NaN/Inf、行数、列数、patch ID 交集。

建议实现细节：

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

DDOF = 1  # 与 prepare_jfx_fixed_data.py 和 zscore.py 的历史主线保持一致

def sample_zscore(values):
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=DDOF)
    zero_std = std == 0
    std[zero_std] = 1.0
    z = (values - mean) / std
    return z, mean, std, zero_std

# standard_zscore:
#   sample_zscore(raw_values)
#
# yeojohnson_zscore:
#   PowerTransformer(method="yeo-johnson", standardize=False)
#   then sample_zscore(yeojohnson_values)
#
# quantile_normal:
#   QuantileTransformer(
#       n_quantiles=min(1000, n_samples),
#       output_distribution="normal",
#       random_state=42,
#       subsample=None,
#   )
```

注意：不要直接用 `sklearn.preprocessing.StandardScaler` 生成 z-score 分支。`StandardScaler` 内部使用 `ddof=0`，而项目主线 `prepare_jfx_fixed_data.py` / `zscore.py` 使用 `ddof=1`。两者在 7788 个样本上数值差异很小，但为了让 standard z-score 与 Yeo-Johnson + z-score 的对照只反映变换本身，必须统一使用 `ddof=1`。

P0 验收阈值：

| transform | 验收条件 |
|---|---|
| standard_zscore | `max_abs_value <= 8.2`，无 NaN/Inf |
| yeojohnson_zscore | `max_abs_value <= 6.2`，`abs(skew)>1` 通路数不超过 3，无 NaN/Inf |
| quantile_normal | `max_abs_value <= 5.3`，无 NaN/Inf |

### Step 4：训练前配置检查

标准 z-score 主线不需要改 `config.yaml`，因为现有配置已经指向：

```yaml
patients:
  JFX0729:
    patches_subdir: "JFX0729_noov_split"
    labels_csv: "JFX0729_ssGSEA_zscore.csv"
```

如果要跑 Yeo-Johnson 单患者对照，不要直接覆盖主 `config.yaml`。复制一个临时配置：

```powershell
New-Item -ItemType Directory -Force configs | Out-Null
```

```text
configs/config.jfx_yj_20260616.yaml
```

在复制文件中只改 JFX 标签路径：

```yaml
patients:
  JFX0729:
    patches_subdir: "JFX0729_noov_split"
    labels_path: "data_new_3ST/label_transforms/JFX0729_v20260616/JFX0729_ssGSEA_yeojohnson_zscore.csv"
```

运行时用环境变量切换：

```powershell
$env:PFMVAL_CONFIG = "configs/config.jfx_yj_20260616.yaml"
& "C:\Program Files\Python313\python.exe" config_utils.py
Remove-Item Env:PFMVAL_CONFIG
```

要求 `config_utils.py` 输出的 JFX label path 指向 Yeo-Johnson 文件，HYZ/LMZ 不受影响。

### Step 5：冒烟训练

标准 z-score JFX 单患者 CLS 冒烟：

```powershell
& "C:\Program Files\Python313\python.exe" train_online_cls.py `
  --mode frozen `
  --patient JFX0729 `
  --dataset_name jfx_fixed_standard_zscore_smoke `
  --num_epochs 3 `
  --batch_size 8 `
  --num_workers 0
```

标准 z-score JFX 单患者 token 冒烟：

```powershell
& "C:\Program Files\Python313\python.exe" train_online_tokens.py `
  --mode frozen `
  --patient JFX0729 `
  --dataset_name jfx_fixed_standard_zscore_tokens_smoke `
  --num_epochs 3 `
  --batch_size 4 `
  --grad_accum_steps 2 `
  --num_workers 0
```

Yeo-Johnson 单患者 CLS 冒烟：

```powershell
$env:PFMVAL_CONFIG = "configs/config.jfx_yj_20260616.yaml"
& "C:\Program Files\Python313\python.exe" train_online_cls.py `
  --mode frozen `
  --patient JFX0729 `
  --dataset_name jfx_fixed_yeojohnson_zscore_smoke `
  --num_epochs 3 `
  --batch_size 8 `
  --num_workers 0
Remove-Item Env:PFMVAL_CONFIG
```

冒烟通过标准：

- 数据集加载样本数：train=7010，val=778。
- loss 无 NaN/Inf。
- checkpoint 输出目录创建成功。
- 生成训练历史文件或日志中可见 train/val loss。
- 若冒烟失败，先排查标签 CSV 路径和 patch ID 交集，不要直接改模型。

### Step 6：正式复跑顺序

第一批必须使用标准 z-score 主线，用于恢复 JFX fixed baseline：

1. JFX 单患者 CLS frozen baseline。
2. JFX 单患者 token frozen / GFNet 当前推荐设置。
3. Fold2：HYZ15040 + LMZ12939 → JFX0729。
4. Fold1 / Fold3。
5. MultiPatient 三患者合并训练。

第二批为标签 transform 对照：

1. JFX 单患者标准 z-score vs Yeo-Johnson + z-score，同一模型、同一 seed、同一训练轮数。
2. 如果 Yeo-Johnson 在 JFX 单患者稳定提升，再补 Quantile-normal。
3. 只有在 HYZ/LMZ raw ssGSEA 或可靠 transform 参数齐备后，才扩展到跨患者 transform 对照。

---

## 6. 当前预处理方案合理性评估

### 合理部分

| 方案 | 判断 | 理由 |
|---|---|---|
| 每患者独立 z-score | 基线合理 | 与历史实验、配置路径、训练代码兼容 |
| 30 通路统一回归 | 合理 | 当前模型、loss、指标均围绕 30 targets 设计 |
| HuberLoss | 合理 | 当前训练脚本已经使用 HuberLoss，对异常值比 MSE 更稳 |
| JFX fixed data 不做 overlap filter | 合理 | 用户已确认 repaired patch 无重叠问题，旧过滤会损失样本 |
| 使用 manifest 固定 split | 必要 | 后续复跑和 Codex/Claude 交接需要可审计样本集合 |

### 主要风险

| 风险 | 影响 | 改进 |
|---|---|---|
| z-score 不改变偏态/长尾 | `icp`、`toxic`、`mhc` 等通路仍有极端值 | 增加 Yeo-Johnson + z-score 对照 |
| 旧脚本硬编码 8 通路 / HYZ 路径 | 容易误处理 JFX fixed data | 只使用 `scripts/prepare_jfx_fixed_data.py` 或新专用脚本 |
| 标准化参数未统一管理 | 难以复现实验 | 保存 params JSON/CSV、QC CSV、manifest |
| 只替换 JFX 标签变换会破坏跨患者标签空间一致性 | cross-patient 结果不可解释 | transform 分支先限 JFX 单患者；跨患者需三患者同方法 |
| 全样本 fit 标准化包含 val 标签分布 | patch-level 验证可能有轻微乐观偏差 | P1 增加 train-only fit transform 消融 |

---

## 7. 可选改进路线

### P0：最小风险改进

- 保持主线标准 z-score。
- 增加 JFX single-patient Yeo-Johnson + z-score 对照。
- 顺序必须是 raw ssGSEA → Yeo-Johnson → `ddof=1` z-score；不要先对 raw ssGSEA 做 z-score 后再做 Yeo-Johnson。
- 所有结果必须在实验名中显式写入 label transform，例如：
  - `jfx_fixed_standard_zscore_*`
  - `jfx_fixed_yeojohnson_zscore_*`

### P1：更严格的数据科学口径

- 对单患者 patch-level 实验，新增 train-only fit：
  - transform 参数只在 train split 上 fit。
  - val split 只 transform，不参与 fit。
- 输出：
  - `JFX0729_ssGSEA_yeojohnson_zscore_trainfit.csv`
  - `JFX0729_trainfit_transform_params.json`
- 该分支更接近真实泛化评估，但会和历史全样本 z-score 基线不完全可比。

### P2：跨患者一致标签空间

只有在 HYZ/LMZ raw ssGSEA 可用时执行：

- 三患者都生成同名 transform 标签。
- 每个患者保存独立 transform 参数。
- cross-patient 实验中训练患者和测试患者使用同一种 transform 方法。
- 报告中把 `label_transform` 作为一级实验变量。

---

## 8. 回滚方案

如果 JFX fixed data 处理失败：

1. 不删除 `data_new_3ST/JFX_fixed_data/`。
2. 保留失败输出目录并加后缀：

```text
data_new_3ST/patch_noov_spilt/JFX0729_noov_split_failed_YYYYMMDD_HHMM
data_new_3ST/label_transforms/JFX0729_v20260616_failed_YYYYMMDD_HHMM
```

3. 如果需要恢复旧 JFX z-score，可从备份恢复：

```text
data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore_old_20260616.csv
```

4. 恢复后必须重新运行 `config_utils.py` 和一次 dataset loading smoke check。

---

## 9. 最终交付清单

Claude 完成本方案后，应交付：

```text
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/train_patches/
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/val_patches/
data_new_3ST/patch_noov_spilt/JFX0729_noov_split/split_manifest.csv
data_new_3ST/ssGSEA_zscore/JFX0729_ssGSEA_zscore.csv
data_new_3ST/ssGSEA_zscore/JFX0729_zscore_params.csv
data_new_3ST/ssGSEA_zscore/JFX0729_zscore_params.json
data_new_3ST/label_transforms/JFX0729_v20260616/JFX0729_ssGSEA_yeojohnson_zscore.csv
data_new_3ST/label_transforms/JFX0729_v20260616/JFX0729_label_transform_qc.csv
configs/config.jfx_yj_20260616.yaml
checkpoints/.../jfx_fixed_standard_zscore_smoke/
checkpoints/.../jfx_fixed_yeojohnson_zscore_smoke/
```

最终验收语句：

```text
JFX0729 fixed data preprocessing PASS:
- raw patch/CSV count = 7788/7788
- train/val split = 7010/778
- standard z-score baseline generated in existing training path
- Yeo-Johnson transform branch generated and QC passed
- CLS smoke training completed without NaN/Inf
- cross-patient transform branch not started unless HYZ/LMZ raw labels are available
```
