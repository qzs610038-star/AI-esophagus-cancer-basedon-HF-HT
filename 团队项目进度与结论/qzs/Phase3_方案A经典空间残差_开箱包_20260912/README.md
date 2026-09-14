# Phase 3 方案 A 开箱包（经典空间残差 → 30 维通路特征）

给 Phase 3 同学用的**特征生成包**：用已定稿的 Phase 2「方案 A」权重，把每张切片的 UNI2-h 图像特征变成 **30 维通路 z-score**，再按你们现有管线接到疗效预测模型。

本包**不训练** Phase 2，也**不训练** Phase 3。它只做推理和导出。

更完整的交接背景（权重合同、标度边界）见仓库内：

`experiments/phase2_softlink_local_v2/给phase3队友的交接说明_方案A经典中间残差.md`

本包与实验室服务器上原方案包的**区别与联系**见 [`同步说明.md`](./同步说明.md)。

---

## 是否还依赖服务器上的原方案代码？

**不依赖。** 把本开箱包拷到本机或服务器任意工作目录，配好输入路径即可跑；权重、通路清单、z-score 参数和推理入口都在包内。

只有在你想对照「当初是怎么训 / 怎么构图 / 模型源码长什么样」时，才需要去服务器只读查看原方案包（不要改、不要覆盖已接纳 `runs`）：

| 用途 | 服务器路径 |
| --- | --- |
| 原方案代码目录 | `D:\AIPatho\qzs\code\phase2_softlink_local_v2` |
| 已接纳批次运行根 | `D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10` |
| 正式权重原件（seed 42 / spatial） | `D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10\train_42_spatial\checkpoints\formal_best.pt` |
| 端点清单 | `D:\AIPatho\qzs\runs\phase2_softlink_local_v2\20260908_005325_810_8d306c10\formal_endpoints.json` |

包内也有一份只读源码快照：`reference_phase2_source/`（方便离线对照；默认仍以本包 `src/phase3_package.py` 为运行入口）。

---

## 你拿到的是什么

| 内容 | 作用 |
| --- | --- |
| `weights/formal_best_seed42.pt` | 方案 A 正式权重（`spatial` / 种子 42 / formal / epoch 30） |
| `assets/pathway_manifest.json` | 30 条通路的**固定顺序与名称** |
| `assets/zscore_params_from_train.json` | 仅训练集拟合的 mean/std（用于可选的 raw 逆变换） |
| `src/phase3_package.py` | 日常入口：读特征 → 构图 → 推理 → 写出 CLAM 风格目录 |
| `configs/*.example.json` | 改路径就能跑的配置模板 |
| `examples/custom_adapter.py` | 输入格式对不上时，只改适配器、不改模型核心 |
| `reference_phase2_source/` | Phase 2 源码快照，**只读对照**，不是默认入口 |
| `run.ps1` | Windows 一键调用 |

默认交付给 Phase 3 的通路特征是：**30 维训练 z-score**（不是 raw ssGSEA，也不是 Top-k 离散标签）。

---

## 5 分钟上手

1. 确认本机有 `torch`、`numpy`；读 CLAM H5 还需要 `h5py`（`scipy` 有则更快，没有也能跑）。
2. 复制一份配置，只改输入 / 输出路径：

```powershell
cd <本开箱包目录>
Copy-Item .\configs\clam_slide.example.json .\configs\phase3.local.json
# 用编辑器改 feature_root、slide_manifest、output.root
.\run.ps1 -Config .\configs\phase3.local.json -Mode preflight
.\run.ps1 -Config .\configs\phase3.local.json -Mode all
```

3. 看运行目录里的 `run_manifest.json`：
   - `completed`：干净跑完
   - `completed_with_warnings`：有结果，但务必看 `warnings.jsonl`
   - `failed`：权重不可用，或一张有效切片都没有

服务器上可从 `configs/clam_slide.server.example.json` 起步（示例输出指向 `D:\AIPatho\qzs\feature_caches\phase3_scheme_a`，可按你们实验名改）。

入口不会自动装依赖。若本机没有服务器解释器，会退回系统 `python`；也可用 `-PythonInterpreter` 指定。

---

## 输入怎么选

你们只要把「每张切片上每个 patch 的 1536 维 UNI2-h 特征 + 坐标」喂进来即可。三种方式选一种：

### 方式 A：CLAM 每切片 H5（最常见）

配置：`input.mode = "clam_slide"`

```text
ESCC_uni2h_features/
└─ h5_files/
   ├─ slide_001.h5    # features=[N,1536], coords=[N,2]
   └─ ...
```

建议提供 `slide_manifest`（至少 `slide_id,patient_id`）。没有也能跑：用文件名当 `slide_id`，并暂时当作 `patient_id`，同时写警告。

文件名带前后缀时，用 `slide_id_regex` 抽取 ID。

### 方式 B：逐 patch 点表

配置：`input.mode = "per_patch"`  
点表至少有：`slide_id, x, y, feature_path`；推荐再加 `patient_id, patch_id`。

- 单个 patch 文件可以是 `[1536]`，或 token 矩阵（默认取第 0 行当 CLS）
- 多行共用一个 `[N,1536]` 文件时，请尽量给 `feature_index`
- 列名不同时只改 `column_map`，不用改 Python 核心

### 方式 C：自定义适配器

两种都不合适时，复制 `configs/custom_adapter.example.json` + `examples/custom_adapter.py`。  
适配器只需按切片返回：`features, coords, slide_id, patient_id, spot_ids`。

---

## 输出怎么接到 Phase 3

每次运行都会**新建**目录；同名已存在则自动变成 `xxx_2`、`xxx_3`，不会覆盖旧结果。

```text
<run>/
├─ features_z/{pt_files,h5_files}/   ← 推荐：每切片 [N,30] 通路 z-score
├─ fused_z/{pt_files,h5_files}/      ← 可选：默认 [病理1536 | 通路30] = [N,1566]
├─ predictions/                      ← 逐点 CSV，同时含 pred_z_* 与 pred_raw_*
├─ run_manifest.json
├─ feature_caches.json
├─ warnings.jsonl
├─ slide_summary.csv
└─ ...
```

**接 Phase 3 时请默认用 `features_z`（或融合后的 `fused_z`），读 z-score，不要默认读 raw。**

| 配置项 | 常见取值 | 说明 |
| --- | --- | --- |
| `output.feature_spaces` | `["z"]`（默认） | 也可 `["raw"]` 或两者都写 |
| `output.write_fused` | `true` | 是否写出 1566 维拼接特征 |
| `output.fusion_order` | `pathology_gene`（默认） | 前 1536 病理、后 30 通路。旧 CLAM 多模态通常依赖这个顺序，**不要随便改** |

把 Phase 3 代码里原来读「MPP2 / 旧通路特征」的目录，改成指向本次 `features_z` 或 `fused_z` 即可。`gene_dim` 仍是 30。

---

## 哪些地方可以改，哪些地方不要动

### 可以改（为了对齐你们最新的 Phase 3 管线）

- 输入路径、清单列名、`column_map`、H5 dataset 名、文件名正则
- 自定义 `adapter_module`
- 输出根目录、`run_id`、是否写融合特征、融合顺序（改顺序要在实验记录里写明）
- `runtime.device` / `batch_size`
- 严格模式：`.\run.ps1 ... -Strict`（重复坐标等问题会更早停）

### 不要随便改（会改变「是不是还叫方案 A」）

- 权重文件、`expected_arm/kind/seed`
- `model.input_dim / hidden_dim / output_dim / dropout`（须与正式权重一致）
- `graph.*` 图半径、邻居数、边权公式相关参数  
  改了也能跑，但会标 `contract_variant=true`，结果**不能**再静默称为原方案 A
- `assets` 里的通路顺序与训练集 z-score 参数  
  **禁止**用 Phase 3 测试集 / 外部队列重估 mean、std

本包里的模型是**冻结推理**。若要微调的是你们 Phase 3 下游疗效模型，请在 Phase 3 自己的训练代码里做，不要在本包里改 Phase 2 权重来“凑效果”。

---

## 会影响实验结果的关键约定

1. **接口是 30 维 z-score**  
   Phase 3 当前通路分支应吃 `pred_z` / `features_z`。只有下游明确要求原始 ssGSEA 量级时，才用包内训练集参数做一次逆变换得到 `pred_raw`。

2. **通路列顺序固定**  
   必须与 `assets/pathway_manifest.json` 的 `pathway_names` 一致（`tls` … `ECM_Organization`）。顺序错了等于换特征。

3. **必须构图，不能当普通 MLP 用**  
   方案 A = 中心预测 + 空间残差。只对中心特征做线性头、不传邻居，会静默退化成无空间模型。

4. **新通路特征 ≠ 旧 MPP2 特征**  
   分布不同。旧 Phase 3 checkpoint 通常不能直接沿用；正式比较请用新特征、同一套 folds **重训** Phase 3。

5. **坏切片不会悄悄填假数**  
   缺文件、坏维度、NaN、重复坐标等会写进 `warnings.jsonl` / `dropped_patches.csv`，并尽量继续处理其他切片。只有「权重完全不可用」或「整次运行零有效输出」才会全局失败。

6. **报告结果时请写清来源**  
   建议标注：方案 A / `phase2_softlink_local_v2` / 批次 `20260908_005325_810_8d306c10` / seed 42 / `spatial` / `formal_best` / 输入为 30 维 z-score。

---

## 本地自检（可选）

在开箱包根目录：

```powershell
python -m pytest tests/test_public_behavior.py tests/test_frozen_model_equivalence.py -q
```

需要 `h5py` 才会跑通 CLAM H5 相关用例；没有时该用例会 skip，其余仍可验证。
