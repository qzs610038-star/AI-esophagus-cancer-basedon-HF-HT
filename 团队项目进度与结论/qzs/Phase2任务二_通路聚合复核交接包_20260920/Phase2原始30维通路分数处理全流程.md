# Phase 2 原始30维通路分数：已记录的处理流程与上游待核事项

## 一、概述与定位

本文件记录 Phase 2 主线使用的**已有原始30维 ssGSEA CSV → 点位划分 → 训练集标准化 → V003 标签**这段有代码与元数据可追溯的处理流程。它**不覆盖更上游的“哪些原始基因、怎样算出30维 CSV”**；原始表达矩阵、基因集和 ssGSEA 参数仍需负责同学对照原始记录核实。也请复核本段已有流程的输入是否正确：本地只有标签与历史审计元数据，尚未在本轮读取服务器原始 CSV。

本规范用于配合 [`给队友的交接说明.md`](给队友的交接说明.md) 与 [`给智能体的交接合同.md`](给智能体的交接合同.md)，明确回答：
1. **原始 30 维通路分数从服务器何处获取**（底层 Raw ssGSEA CSV 资产）；
2. **训练集、内部验证集、外部测试集是如何划分的**（基于空间块抽样的防泄漏协议）；
3. **如何进行严格的训练集无偏 z-score 标准化与截断**（`ddof=1` 与冻结参数应用）；
4. **319 个 pre-ssGSEA 候选基因与原始30维 CSV 的关系哪些已知、哪些待审**。

---

## 二、原始 30 维通路分数的服务器物理存储位置

项目登记的原始30维分数位置如下（MPP=2 / Group 2）。路径及文件大小来自 2026-07-11 历史审计快照，**不是本轮服务器在线核验结果**；当前存在性、列头和数值仍应在服务器只读检查。

### 1. 各患者原始文件清单（Raw ssGSEA CSV）

| 患者 ID | 角色划分 | 登记的 Raw ssGSEA 路径 | 历史字节数 | 预期列结构（待现场核对） |
|---|---|---|---|---|
| **HYZ15040** | 内部训练/验证 | `D:\AIPatho\Patch\visiumhd_patch\2\HYZ15040\HYZ15040_ssGSEA.csv` | 852,029 | 31 列（首列 barcode + 30 通路） |
| **JFX** | 内部训练/验证 | `D:\AIPatho\Patch\visiumhd_patch\2\JFX\JFX_ssGSEA.csv` | 1,172,642 | 31 列（首列 barcode + 30 通路） |
| **LMZ12939** | 内部训练/验证 | `D:\AIPatho\Patch\visiumhd_patch\2\LMZ12939\LMZ12939_ssGSEA.csv` | 1,958,186 | 31 列（首列 barcode + 30 通路） |
| **TGC** | 内部训练/验证 | `D:\AIPatho\Patch\visiumhd_patch\2\TGC\TGC_ssGSEA.csv` | 640,637 | 31 列（首列 barcode + 30 通路） |
| **XSL** | 内部训练/验证 | `D:\AIPatho\Patch\visiumhd_patch\2\XSL\XSL_ssGSEA.csv` | 884,838 | 31 列（首列 barcode + 30 通路） |
| **ZHZ** | 内部训练/验证 | `D:\AIPatho\Patch\visiumhd_patch\2\ZHZ\ZHZ_ssGSEA.csv` | 615,625 | 31 列（首列 barcode + 30 通路） |
| **XZY** | 独立外部测试 | `D:\AIPatho\Patch\visiumhd_patch\2\XZY\XZY_ssGSEA.csv` | 594,498 | 31 列（首列 barcode + 30 通路） |

### 2. 项目配置与审计证据出处
- **项目路径登记**：[`configs/server_paths.yaml`](../../../configs/server_paths.yaml) 中的 `mpp_patient_raw_ssgsea` 和 `mpp_external_xzy_raw_ssgsea` 字段；这是路径登记，不证明当前资产仍在。
- **交接包元数据登记**：本交接包内的元数据文件 [`aggregation_review/current_phase2_labels/zscore_manifest.json`](aggregation_review/current_phase2_labels/zscore_manifest.json) 的 `raw_label_source` 字段逐项记录了上述绝对路径。
- **历史审计清单**：`project_state/evidence/mpp/barcode-repair-20260711-d626ad8-v003/server_asset_audit_manifest.json` 保留了这些文件在当时的字节数。本轮不计算哈希。

### 3. 数据特征说明
- **期望列**：首列为点位标识（脚本允许将首列重命名为 `barcode`），后续应为与冻结 manifest 同名同序的30个数值通路列；其中首条为 `tls`，末条为 `ECM_Organization`。实际首列名、是否恰为31列以及每列值需现场核对，不能由下游标签反推。
- **数值含义**：这里的“Raw”仅指**尚未执行 V003 的训练集 z-score**，不证明它最初使用了何种基因表达尺度或 ssGSEA 软件、基因集。

---

## 三、数据集划分机制：空间网格块抽样协议

开发患者在同一切片内采用空间块划分；XZY 作为外部患者单独保留。这里描述**实际冻结的划分**，不建议重新抽样来替换现有 manifest。

### 1. 划分算法代码出处
- **代码文件**：[`scripts/generate_standard_splits.py`](../../../scripts/generate_standard_splits.py)；实际采用值以本包 [`split_info.json`](aggregation_review/current_phase2_labels/split_info.json) 和 [`split_manifest.csv`](aggregation_review/current_phase2_labels/split_manifest.csv) 为准。
- **核心算法逻辑**：
  1. **空间坐标解析**：使用正则表达式 `x(\d+)_y(\d+)` 从每张切片的 `barcode` 中提取物理网格坐标 $(x, y)$。
  2. **网格分块（Block Partitioning）**：
     脚本曾比较 `672 / 1120 / 1568 px` 候选；MPP2 冻结划分**实际采用 `block_size=672 px`**（$3\times224$，不是1120）。将坐标离散化为块 ID：
     $$\text{bx} = \lfloor x / 672 \rfloor, \quad \text{by} = \lfloor y / 672 \rfloor, \quad \text{block\_id} = \text{patient}:\text{bx}:\text{by}$$
  3. **块级抽样（Block Sampling）**：
     manifest 登记 `seed=42`、目标验证比例10%，候选评分参考8%–12%区间。代码逐患者使用 `RandomState(seed + hash(patient) % 100000)`，打乱块顺序后优先选择至少3个点位的块（没有此类块时回退），直至达到目标；不是保证连续相邻块。Python 的 `hash(patient)` 可随进程变化，仅有 seed=42 **不足以重新抽样得到完全相同的划分**；请直接复用冻结的 `split_manifest.csv`。
  4. **外部测试集绝对物理隔离**：
     患者 `XZY` 作为完全独立的异源切片，整体直接标定为 `external_test`，完全不参与内部划分抽样。

### 2. 划分中间契约文件（`split_manifest.csv`）
划分完成后，生成唯一的点位划分清单 [`aggregation_review/current_phase2_labels/split_manifest.csv`](aggregation_review/current_phase2_labels/split_manifest.csv)：
- **训练集（train）点位数**：9,472 点（由 6 位内部患者的训练组织块构成）；
- **内部验证集（internal_val）点位数**：1,078 点（由 6 位内部患者的验证组织块构成）；
- **外部测试集（external）点位数**：XZY 切片共提取 1,039 个评估点位。

---

## 四、z-score 标准化变换算法与防泄漏机制

在 2026-07-11 条形码修复行动中，通过专项转换脚本生成了冻结至今的 V003 标签。

### 1. 标准化变换代码出处
- **代码文件**：[`scripts/rebuild_zscore_from_manifest.py`](../../../scripts/rebuild_zscore_from_manifest.py)。脚本中的 `--source-commit 012345...` 是**说明用占位命令**，不是服务器实际执行记录；包内没有完整的现场命令日志。历史 `server_verification.json` 仅记录当时 staging 验证，不替代当下对 raw 输入的审查。本轮只读检查，不运行此重建脚本。

### 2. 核心数学公式与计算步骤
1. **仅在训练集上拟合（Train-Only Fit）**：
   严格仅提取 `split_manifest.csv` 中标记为 `split == "train"` 的 9,472 个样本点，逐通路 $p \in [1, 30]$ 计算样本均值 $\mu_p$ 和样本标准差 $\sigma_p$（**必须使用无偏自由度 `ddof=1`**）：
   $$\mu_p = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} y_{i, p}$$
   $$\sigma_p = \sqrt{\frac{1}{N_{\text{train}} - 1} \sum_{i \in \text{train}} (y_{i, p} - \mu_p)^2}$$
   *注：若计算得到的 $\sigma_p = 0$ 或非有限数，代码自动强制置 $\sigma_p = 1.0$ 以防除零。*
2. **冻结参数统一应用（Apply z-score）**：
   使用上述从 9,472 个训练点拟合得到的单一参数对 $(\mu_p, \sigma_p)$，无差别地应用于训练集、内部验证集以及外部测试集（XZY）：
   $$z_{i, p} = \frac{y_{i, p} - \mu_p}{\sigma_p}$$
3. **极值防护与截断（Clipping）**：
   - 替换所有 `NaN` 和 `±Inf` 为 `0.0`；
   - 实行对称硬截断：$z_{i, p} = \text{clip}(z_{i, p}, -100.0, 100.0)$。
4. **拟合统计记录（Fit Verification）**：
   脚本在**未截断**的训练 z 值上计算均值和标准差，并写入 `zscore_manifest.json`。现存记录的均值接近0、样本标准差接近1；代码并未将上述 $10^{-4}$ 区间写成运行停止门槛。验证与 XZY 不参与拟合，其均值/方差可以偏离0和1。

### 3. 生成的落地资产（V003 冻结目录）
- **服务器根目录**：`D:\AIPatho\Patch\mpp_zscore_repair_staging\barcode-repair-20260711-d626ad8-v003\group_2\`
  - `labels/train/{patient}/{patient}_ssGSEA_zscore.csv`（6 份，每份首列为 barcode，后接 30 列 z 分数）
  - `labels/val/{patient}/{patient}_ssGSEA_zscore.csv`（6 份）
  - `labels/external/XZY/XZY_ssGSEA_zscore_by_group_2_train.csv`（1 份）
  - `zscore_params_from_train.json`（记录每条通路的 mean、std 与 count）
  - `zscore_manifest.json`（记录通路顺序、拟合元数据与源文件映射）
- **本地包的复核副本**：[`aggregation_review/current_phase2_labels/`](aggregation_review/current_phase2_labels/) 保存了标签和冻结元数据，便于审查；不宣称与当前服务器所有文件逐字节相同。服务器实际资产仍需现场只读核对。

---

## 五、与 Phase 2 任务二原始基因（319 候选基因）输入的对齐与协同

在任务二的“基因重构臂”与“聚合复核”中，已有一组**登记的319基因候选矩阵**。它是待核查的输入线索，不能因为文件名含 `pre_ssgsea` 就认定是最初生成主线30维 CSV 的完整原始基因输入。

### 1. 候选基因矩阵登记的服务器路径（待现场确认）
- **基因矩阵根目录**：`D:\AIPatho\Patch\genes_tag\`
- **候选基因映射表**：`D:\AIPatho\Patch\genes_tag\gene_to_pathway_mapping.csv`（实际条目、版本与成员数待现场核对）
- **7 名患者表达文件**：
  - `D:\AIPatho\Patch\genes_tag\HYZ15040_genes_pre_ssgsea.csv`
  - `D:\AIPatho\Patch\genes_tag\JFX_genes_pre_ssgsea.csv`
  - `D:\AIPatho\Patch\genes_tag\LMZ12939_genes_pre_ssgsea.csv`
  - `D:\AIPatho\Patch\genes_tag\TGC_genes_pre_ssgsea.csv`
  - `D:\AIPatho\Patch\genes_tag\XSL_genes_pre_ssgsea.csv`
  - `D:\AIPatho\Patch\genes_tag\ZHZ_genes_pre_ssgsea.csv`
  - `D:\AIPatho\Patch\genes_tag\XZY_genes_pre_ssgsea.csv`

### 2. 基因数据读取与格式规范
- **列结构与过滤规则**：
  根据旧任务二 [`preflight.py`](legacy_task2_reference/preflight.py) 与 [`gene_expression.py`](legacy_task2_reference/src/targets/gene_expression.py) 的读取约定，核查单患者 CSV 时应先排除可能存在的元数据列：
  `skip_cols = {"barcode", "patient", "spatial_x", "spatial_y", "n_bins_aggregated"}`
  再以实际列头确认是否确为**319个有序基因表达列**，不能把元数据列当基因；避免将合并表 `all_patients_genes_pre_ssgsea.csv` 当作单患者文件。若实际名单/列数与历史不符，先弄清来源并更新有序名单，不要默默强制截成319列。
- **点位与样本的一致性对应**：
  先现场核对7位患者基因矩阵与 raw30/V003 的点位覆盖；连接使用 `patient + barcode` 复合键，核对重复和缺失，可用时再检查坐标/划分。历史对比能对齐内部验证与 XZY 共2,117点，**不能据此断言服务器所有基因点位均一一匹配**；严禁按行号盲连。

### 3. 数值尺度与关键边界提醒
- **尺度定义**：
  历史诊断文件 [`pathway_reconstruction_metrics.json`](aggregation_review/historical_319_candidate/pathway_reconstruction_metrics.json) 将**旧候选聚合入口**登记为 `pre_ssgsea_log1p_after_inverse_train_gene_zscore`。这说明旧代码声称在基因训练集 z-score 逆变换后使用 log1p 输入，**不能证明磁盘上的原始 CSV 就是此尺度**。请结合真实文件与最初生成记录核实 counts、归一化值和 log1p 处理。
- **重要事实边界警示**：
  项目目前登记了319基因候选矩阵，但**未确认**最初 raw30 CSV 的基因全集、源表达尺度、基因集版本与软件参数，也未在本轮穷尽服务器全转录组资产。请先审查319基因是否是所需输入及其聚合合同，再与 V003 作**独立一致性诊断**；V003 逐值一致不是任务二开跑硬门槛。若证据说明必须使用其他基因背景才能解释主线差异，再向课题组询问相应表达矩阵，不预设一定需要原始全转录组 counts。若两套30维目标不同，仍可用核实后的319基因目标开展两臂内部自洽消融，但不得把它与 V003 主线成绩直接比较。

---

## 六、如何在服务器只读核查（不重建、不覆盖）

先在 `AIPatho1` 会话中按[给队友的交接说明](给队友的交接说明.md#服务器账户与虚拟环境)预检解释器。以下 PowerShell 只看路径、历史文件大小对照和表头；它**不能证明** raw30 数值或最初的基因聚合是正确的。

```powershell
$patients = @('HYZ15040', 'JFX', 'LMZ12939', 'TGC', 'XSL', 'ZHZ', 'XZY')

Write-Host "=== 1. 核查原始未标准化 30 维 ssGSEA 文件 (Raw) ===" -ForegroundColor Cyan
foreach ($p in $patients) {
    $path = "D:\AIPatho\Patch\visiumhd_patch\2\$p\${p}_ssGSEA.csv"
    $item = Get-Item $path -ErrorAction SilentlyContinue
    [PSCustomObject]@{
        Patient    = $p
        Exists     = [bool]$item
        SizeBytes  = if ($item) { $item.Length } else { $null }
        LastWrite  = if ($item) { $item.LastWriteTime } else { $null }
        Header     = if ($item) { Get-Content -LiteralPath $path -TotalCount 1 } else { $null }
    }
}

Write-Host "`n=== 2. 核查 30 维 V003 z-score 标准化标签 (13 个 CSV) ===" -ForegroundColor Cyan
$v003_labels = "D:\AIPatho\Patch\mpp_zscore_repair_staging\barcode-repair-20260711-d626ad8-v003\group_2\labels"
$files = if (Test-Path -LiteralPath $v003_labels) { @(Get-ChildItem -LiteralPath $v003_labels -Recurse -Filter "*.csv") } else { @() }
Write-Host "Found V003 CSVs: $($files.Count) (Expected: 13)"

Write-Host "`n=== 3. 核查 319 原始基因表达矩阵与映射关系 ===" -ForegroundColor Cyan
$gene_root = "D:\AIPatho\Patch\genes_tag"
$mapping = "$gene_root\gene_to_pathway_mapping.csv"
Test-Path -LiteralPath $mapping
if (Test-Path -LiteralPath $mapping) { Get-Content -LiteralPath $mapping -TotalCount 1 }
foreach ($p in $patients) {
    $path = "$gene_root\${p}_genes_pre_ssgsea.csv"
    $item = Get-Item $path -ErrorAction SilentlyContinue
    [PSCustomObject]@{
        Patient    = $p
        Exists     = [bool]$item
        SizeBytes  = if ($item) { $item.Length } else { $null }
        Header     = if ($item) { Get-Content -LiteralPath $path -TotalCount 1 } else { $null }
    }
}
```

若已将本包复制到服务器，在**本包根目录**用登记的解释器运行随包的只读核查脚本；它逐患者读取实际 raw30 CSV，对照冻结清单选点，重新计算训练集 `ddof=1` 参数，并把应用冻结参数后的每个已选点位/通路与随包 V003 副本比较：

```powershell
& 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe' -B '.\tests\check_raw30_server.py'
```

脚本只在标准输出打印 `PASS/MISMATCH`、分组点数与最大差异，不写文件、不读取基因矩阵，也不调用原重建脚本。`PASS` 只表示“**现场 raw30 → 本包 V003 副本**”在报告容差内一致；它**不表示**原始基因/基因集/ssGSEA 全链正确，也不代替核对服务器当前 V003 文件与本地副本。`MISMATCH` 或原始文件缺失时请记录差异并复核来源，不要覆盖既有标签。

请再请原负责同学补全更上游四项：原始表达矩阵的基因范围和数值尺度；30条通路的映射来源与成员；ssGSEA 软件、版本和所有影响分数的选项；生成各患者 raw30 CSV 时的脚本/记录。先在开发患者确认合同，XZY 只在合同冻结后检查，不用来选择聚合参数。如果发现当初 raw30 本身有错，单独报告它可能影响主线标签与既有模型的范围，不自动替换 V003 或把旧成绩当作更正后成绩。
