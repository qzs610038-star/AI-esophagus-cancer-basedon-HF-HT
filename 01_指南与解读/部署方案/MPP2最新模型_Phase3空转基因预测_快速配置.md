# 给 Phase 3 队友的 MPP2 模型使用说明（无需了解 Git）

> 目标：把已经训练完成的 MPP2 模型复制到你自己的 Phase 3 工作目录，用新的患者切片预测 30 个空转基因通路信号；再把这些预测作为**候选辅助特征**交给 pCR/MPR 治疗响应预测流程。  
> 2026-08-01 用户审核修正：Phase 3 不是预后或生存预测；当前已运行终点仅确认为 `pCR / non_pCR`。队列约 78 例，准确数目待 patient manifest 核验。本文件只说明候选接口，不证明 MPP2 对 Phase 3 已有增益。
> 2026-08-01 用户批准的 MPP2 输入边界：本说明只允许使用 accepted 的 `mpp2_barcode_repair_v003_frozen_baseline_20260711` 原始输出。Ridge r003 外部门失败且证据被拒绝，`CalibratedMPP2` 不得复制、部署或作为 Phase 3 输入。用户已决定后续研究采用 H&E+MPP2；当前比较直接拼接与特征融合，具体定义和基线结果待队友 Phase 3 基线包核验。
> 本文只要求你会运行 PowerShell 和 Python 命令。不要修改项目 Git 分支、不要进入正在训练的目录、不要移动原始模型。

## 先记住这五件事

1. 你要拿的“最新 MPP2 模型”是一个文件：`best_checkpoint.pth`。它已经训练好，**复制**到你的 Phase 3 目录即可，不能剪切/移动。
2. 它不能直接读取 PNG 图片。先用同一 UNI2-h 图像编码器把每个病理 patch 变成一条 `[1536]` 特征；模型再把特征预测为 30 个通路值。
3. 输出有两套数值：`pred_z_*`（模型原始 z-score 输出）和 `pred_raw_*`（换算回原始 ssGSEA 尺度）。两套都保留，不要自行重新标准化。
4. 预测的通路值是治疗响应模型的**候选辅助变量**，不是已经证明的治疗响应结论。必须按患者聚合并在独立队列上验证，不能让同一患者的切片同时出现在训练和测试中。
5. 不得使用 `CalibratedMPP2`。它的内部校准门虽通过，但 XZY 外部门失败；当前只允许测试 accepted 冻结基线原始输出是否有独立增量。

---

## 0. 你需要的三个东西分别在哪里

| 名称 | 它做什么 | 服务器位置 | 你是否需要复制 |
|---|---|---|---|
| **MPP2 预测头** | 真正的已训练回归模型；把 `[1536]` 特征变成 30 个通路预测 | `D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp2_barcode_repair_v003_frozen_baseline_20260711\best_checkpoint.pth` | **是，复制到自己的 Phase 3 模型包** |
| **UNI2-h 图像编码器缓存** | 把病理图片提取成 `[1536]` 特征 | `D:\AIPatho\shared\.cache\huggingface` | **通常否**；同一服务器上只读复用。新患者必须重新提取自己的特征，不能复用别人的 feature cache |
| **MPP2 修复版标准化参数** | 把模型的 z-score 输出换回原始 ssGSEA 尺度；同时锁定 30 个通路顺序 | `D:\AIPatho\Patch\mpp_zscore_repair_staging\barcode-repair-20260711-d626ad8-v003\group_2` | **是，复制其中的参数和通路顺序到自己的模型包** |

不要使用旧的 `mpp2_std10val_xzy_ext_uni2h_mlp_20260706`，也不要使用旧 `mpp_standard_splits` 中的参数。

---

# 快速操作：按顺序做完 4 步

## 第 1 步：把模型包复制到你自己的 Phase 3 目录

直接使用你现有的 Phase 3 服务器工作目录。例如，以下 `<队友现有 Phase 3 目录>` 必须替换为你的真实目录：

```text
D:\<队友现有 Phase 3 目录>\
```

不要把脚本、结果或新患者数据写进以下两个目录：

```text
D:\AIPatho\qzs\pfmval_deploy_git          # 这里只读 checkpoint；代码版本较旧
D:\AIPatho\qzs\pfmval_automation          # MPP1/3/4/5 正在这里运行，禁止触碰
```

在你的 Phase 3 目录中新建 `prepare_phase3_mpp2_package.py`，粘贴以下代码。它会：

- 检查源 checkpoint 的哈希；
- 复制 checkpoint 和 MPP2 修复版 z-score 参数；
- 从修复标签的**表头**提取正确的 30 个通路顺序（不会复制患者标签数据）；
- 生成一个 `model_provenance.json`，供以后追溯。

如果目标模型包已存在，脚本会直接停止，避免覆盖任何已有版本。

```python
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98"
SOURCE_CKPT = Path(r"D:\AIPatho\qzs\pfmval_deploy_git\checkpoints\mpp_uni2h_mlp\mpp2_barcode_repair_v003_frozen_baseline_20260711\best_checkpoint.pth")
REPAIR_GROUP = Path(r"D:\AIPatho\Patch\mpp_zscore_repair_staging\barcode-repair-20260711-d626ad8-v003\group_2")

# 改成你现有的 Phase 3 工作目录；不要填 pfmval_deploy_git 或 pfmval_automation。
PACKAGE = Path(r"D:\<队友现有 Phase 3 目录>\mpp2_model_package_20260711")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_columns(frame: pd.DataFrame) -> list[str]:
    skip = {"x", "y", "spot", "id", "spot_id", "barcode", "index", "patch_id", "filename"}
    return [c for c in frame.select_dtypes(include=["number"]).columns if c.lower() not in skip]


if PACKAGE.exists():
    raise FileExistsError(f"model package already exists; do not overwrite: {PACKAGE}")
if not SOURCE_CKPT.is_file():
    raise FileNotFoundError(SOURCE_CKPT)
if sha256(SOURCE_CKPT) != EXPECTED_SHA256:
    raise RuntimeError("source checkpoint SHA-256 mismatch; stop and contact the model owner")

params = REPAIR_GROUP / "zscore_params_from_train.json"
label_csv = next((REPAIR_GROUP / "labels").glob("train/*/*_ssGSEA_zscore.csv"), None)
if not params.is_file() or label_csv is None:
    raise FileNotFoundError("repaired MPP2 parameters or label header is unavailable")
pathways = target_columns(pd.read_csv(label_csv))
if len(pathways) != 30:
    raise RuntimeError(f"expected 30 pathways, got {len(pathways)}")

PACKAGE.mkdir(parents=True)
shutil.copy2(SOURCE_CKPT, PACKAGE / "best_checkpoint.pth")
shutil.copy2(params, PACKAGE / "zscore_params_from_train.json")
(PACKAGE / "pathway_order.json").write_text(json.dumps(pathways, ensure_ascii=False, indent=2), encoding="utf-8")
(PACKAGE / "model_provenance.json").write_text(json.dumps({
    "experiment_id": "mpp2_barcode_repair_v003_frozen_baseline_20260711",
    "checkpoint_sha256": EXPECTED_SHA256,
    "checkpoint_epoch": 15,
    "input_contract": "UNI2-h CLS float32 [N,1536]",
    "output_contract": "30 pathways in z-score and repaired MPP2 raw ssGSEA scale",
    "repair_manifest": "barcode-repair-20260711-d626ad8-v003:1204018178a4d355",
    "source_checkpoint": str(SOURCE_CKPT),
    "source_repair_group": str(REPAIR_GROUP),
}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"PASS: model package created at {PACKAGE}")
```

先把代码里的 `<队友现有 Phase 3 目录>` 改成你的真实目录名，再在该目录运行：

```powershell
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' .\prepare_phase3_mpp2_package.py
```

成功后，你的目录应包含：

```text
mpp2_model_package_20260711\
├── best_checkpoint.pth
├── zscore_params_from_train.json
├── pathway_order.json
└── model_provenance.json
```

这就是 Phase 3 可自行持有的“模型包”。原模型仍保留在 PFMval 目录，未被移动。

## 第 2 步：检查模型包和新患者特征是否可用

在自己的 Phase 3 目录新建 `phase3_mpp2_preflight.py`：

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

EXPECTED_SHA256 = "c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98"
PACKAGE = Path(r"D:\<队友现有 Phase 3 目录>\mpp2_model_package_20260711")


class MPPMLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(1536, 1024), nn.GELU(), nn.Dropout(0.3), nn.Linear(1024, 30)
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


parser = argparse.ArgumentParser()
parser.add_argument("--features", type=Path, help="optional Phase 3 .pt feature tensor")
args = parser.parse_args()

ckpt = PACKAGE / "best_checkpoint.pth"
if not ckpt.is_file() or sha256(ckpt) != EXPECTED_SHA256:
    raise RuntimeError("model package checkpoint is missing or has the wrong SHA-256")
pathways = json.loads((PACKAGE / "pathway_order.json").read_text(encoding="utf-8"))
params = json.loads((PACKAGE / "zscore_params_from_train.json").read_text(encoding="utf-8"))["pathways"]
if len(pathways) != 30 or any(name not in params for name in pathways):
    raise RuntimeError("pathway order or repaired z-score parameters are invalid")

checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
model = MPPMLPHead()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
if checkpoint.get("epoch") != 15:
    raise RuntimeError(f"unexpected checkpoint epoch: {checkpoint.get('epoch')}")

if args.features:
    features = torch.load(args.features, map_location="cpu").float()
    if features.ndim == 3:
        features = features[:, 0, :]
    if features.ndim != 2 or features.shape[1] != 1536 or not torch.isfinite(features).all():
        raise RuntimeError(f"invalid Phase 3 features: shape={tuple(features.shape)}")

print(f"PASS: epoch=15, pathways={len(pathways)}, package={PACKAGE}")
```

把其中 `<队友现有 Phase 3 目录>` 改成真实目录名，然后执行：

```powershell
# 先检查模型包
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' .\phase3_mpp2_preflight.py

# 准备好新患者特征后再检查特征
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' .\phase3_mpp2_preflight.py --features D:\your_phase3\phase3_uni2h_cls_features.pt
```

两次都显示 `PASS` 才继续。若报错，停止并把完整报错交给模型所有者；不要更换旧 checkpoint、不要用旧标准化参数凑合。

## 第 3 步：对新患者切片生成通路预测

### 3.1 先做图片特征

对每张新患者切片切出的每个 patch，使用与 MPP2 训练一致的 UNI2-h 图像编码器和 transform，得到一个 `[1536]` CLS 特征。批量保存为：

```text
D:\your_phase3\phase3_uni2h_cls_features.pt    # shape: [N,1536]
D:\your_phase3\phase3_patch_metadata.csv       # N 行，至少有 patch_id
```

`phase3_patch_metadata.csv` 推荐列：

| patch_id | patient_id | slice_id | x | y |
|---|---|---|---:|---:|
| `P001_0001` | `P001` | `P001_S1` | 1234 | 5678 |

这张表非常重要：模型不知道患者或空间位置；它只会输出每个 patch 的 30 个数值。后续要回到切片、患者和 pCR/MPR 治疗响应标签，完全依赖这张映射表。

> `D:\AIPatho\shared\.cache\huggingface` 是服务器共享的 UNI2-h 权重缓存。通常只读复用即可；不要复制、删除或清理它。`mpp_uni2h_cache` 里的旧 feature cache 是其他数据的特征，不能用于新患者。

### 3.2 运行预测

在自己的目录新建 `phase3_mpp2_infer.py`：

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PACKAGE = Path(r"D:\<队友现有 Phase 3 目录>\mpp2_model_package_20260711")
FEATURES = Path(r"D:\your_phase3\phase3_uni2h_cls_features.pt")
PATCH_META = Path(r"D:\your_phase3\phase3_patch_metadata.csv")
OUT = Path(r"D:\your_phase3\phase3_mpp2_predictions.csv")


class MPPMLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(1536, 1024), nn.GELU(), nn.Dropout(0.3), nn.Linear(1024, 30)
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features = torch.load(FEATURES, map_location="cpu").float()
if features.ndim == 3:
    features = features[:, 0, :]
if features.ndim != 2 or features.shape[1] != 1536 or not torch.isfinite(features).all():
    raise ValueError(f"Expected finite [N,1536] UNI2-h CLS features, got {tuple(features.shape)}")

meta = pd.read_csv(PATCH_META)
if "patch_id" not in meta or len(meta) != len(features):
    raise ValueError("metadata must contain patch_id and have exactly N rows")

pathways = json.loads((PACKAGE / "pathway_order.json").read_text(encoding="utf-8"))
params = json.loads((PACKAGE / "zscore_params_from_train.json").read_text(encoding="utf-8"))["pathways"]
if len(pathways) != 30 or any(name not in params for name in pathways):
    raise ValueError("model package pathway information is invalid")

checkpoint = torch.load(PACKAGE / "best_checkpoint.pth", map_location=device, weights_only=False)
model = MPPMLPHead().to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()  # 关闭 Dropout，保证同一输入重复预测一致
with torch.inference_mode():
    pred_z = model(features.to(device)).cpu().numpy()
if not np.isfinite(pred_z).all():
    raise RuntimeError("model output contains NaN/Inf")

out = meta.copy()
for i, pathway in enumerate(pathways):
    out[f"pred_z_{pathway}"] = pred_z[:, i]
    out[f"pred_raw_{pathway}"] = pred_z[:, i] * params[pathway]["std"] + params[pathway]["mean"]
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print({"device": str(device), "checkpoint_epoch": checkpoint.get("epoch"), "rows": len(out), "output": str(OUT)})
```

修改 `<队友现有 Phase 3 目录>`、`FEATURES`、`PATCH_META` 和 `OUT` 后运行：

```powershell
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' .\phase3_mpp2_infer.py
```

成功后得到 `phase3_mpp2_predictions.csv`。每一行对应一个 patch，保留原始元数据，并新增：

- `pred_z_<pathway>`：模型训练空间的预测值，适合比较模型输出；
- `pred_raw_<pathway>`：使用 MPP2 训练集修复版 mean/std 反变换后的数值，适合和原始 ssGSEA 尺度对照。

---

# 第 4 步：把预测结果用于患者治疗响应辅助建模

MPP2 输出是 patch 级数据，pCR/MPR 治疗响应标签是患者级数据。正确流程是：

```text
病理 patch
  -> UNI2-h 特征 [1536]
  -> MPP2：每个 patch 的 30 通路预测
  -> 按 slice / patient 聚合成患者特征
  -> 与临床变量、病理变量一起输入治疗响应模型
  -> 在独立患者集评估
```

## 推荐给治疗响应模块的数据格式

保留两张表，不要只保留最终分数：

1. `phase3_mpp2_predictions.csv`：patch 级完整输出，便于回溯到具体切片区域。
2. `phase3_patient_pathway_features.csv`：由 Phase 3 团队按患者聚合后的特征表，一行一个患者，例如：

| patient_id | pathway_A_mean | pathway_A_median | pathway_A_p90 | pathway_A_sd | ... | survival_time | event |
|---|---:|---:|---:|---:|---|---:|---:|

聚合规则应在治疗响应模型训练前固定，例如每个通路的 mean、median、90% 分位数和标准差；不要看到测试集 pCR/non-pCR 结果后再挑选对自己有利的聚合方式。

## 三条必须遵守的治疗响应边界

1. **患者级划分**：同一患者所有 patch/切片只能出现在训练集或测试集的一侧，不能跨两侧。
2. **训练集内拟合**：缺失值处理、特征缩放、通路筛选、阈值选择与治疗响应模型调参，只能在训练患者上完成；验证/测试患者只能应用这些已固定规则。
3. **比较整合方式**：同一锁定患者级划分、同一输入和同一 MIL 主干下，比较“直接拼接”与“特征融合”。不得因为两组使用了不同数据划分、临床变量、训练预算或 checkpoint 选择规则而把差异归因于融合结构。

模型的历史 XZY 评估（PCC `0.6549`、raw MAE `1176.2114`、raw R² `-0.0880`）只说明它在该外部空转任务上的表现，**不能直接当作 Phase 3 pCR/MPR 治疗响应性能**。

---

## 常见问题

### Q1：我只复制 `best_checkpoint.pth` 可以吗？

不可以。预测 z-score 虽然能运行，但没有 `zscore_params_from_train.json` 和 `pathway_order.json`，无法可靠得到 raw-scale 输出，也无法保证 30 个通路列与训练时一致。

### Q2：能不能把现有 `mpp_uni2h_cache` 复制给新患者使用？

不能。feature cache 与具体 patch 一一对应。新患者必须从自己的切片重新提取 UNI2-h 特征。

### Q3：为什么不直接在 `pfmval_deploy_git` 里运行？

该目录保存的 checkpoint 正确，但代码版本是旧 `main@73db0ef`。它缺少后续状态管理脚本和修复标签路径配置；在里面运行会产生误导性报错，也可能碰到已有未跟踪结果。

### Q4：治疗响应模块是否可以直接用 30 个 patch 预测做二分类？

不可以直接把 patch 当成独立患者样本。必须先根据 `patient_id` 聚合，并按患者划分训练/验证/测试。

---

## 追溯信息（供负责人核对）

| 项目 | 值 |
|---|---|
| 实验 ID | `mpp2_barcode_repair_v003_frozen_baseline_20260711` |
| checkpoint SHA-256 | `c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98` |
| checkpoint epoch | `15` |
| 修复 manifest | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` |
| 修复标签根 | `D:\AIPatho\Patch\mpp_zscore_repair_staging\barcode-repair-20260711-d626ad8-v003` |
| 模型输入 | UNI2-h CLS `float32[N,1536]` |
| 模型输出 | 30 通路 z-score + MPP2 train-only raw-scale 反变换 |
