# Phase2 全视野修复与调参：操作说明

本包只处理 `MPP2/group_2` 的原 30 通路标签。首轮计划是 111 次头部训练；**运行任何命令前先确认当前使用的是本包，默认命令只检查输入**。特征准备、训练、外部评估和 Phase3 导出都要分别显式启动。本包不会下载模型，不会修改既有 Phase3 消费端。

## 1. 放置代码包并核对环境

在本机将整个 `D:\AI空间转录病理研究\PFMval_new\experiments\phase2_fullfov_hpo_v1\` 文件夹，按项目现行的手动传输方式放到服务器 `D:\AIPatho\qzs\code\phase2_fullfov_hpo_v1\`。只复制此代码包；原始图像、标签、旧特征缓存和模型权重留在原目录。

以下命令在**服务器 PowerShell** 中运行：

```powershell
Set-Location 'D:\AIPatho\qzs\code\phase2_fullfov_hpo_v1'
$Python = 'C:\Users\AIPatho1\pfmval_env\Scripts\python.exe'
Test-Path -LiteralPath $Python
& $Python -c "import torch, torchvision, timm, numpy, pandas, PIL, safetensors; print('Python dependencies OK'); print(torch.__version__, timm.__version__)"
```
`Test-Path` 应输出 `True`。如服务器 Python 路径不同，先核对实际环境，再在后续命令中给 `run.ps1` 传 `-PythonInterpreter '<实际绝对路径>'`，或更新本包 `config.json` 的 `python_interpreter`。`requirements.txt` 是版本记录；启动脚本不会自动安装或升级依赖。

三个模型的服务器快照路径已预填在 `inputs/model_manifest.json`：UNI 和 Virchow2 来自 2026-09-15 消融运行回传；UNI2-h 来自状态为“未核验”的服务器路径登记。查看登记值：

```powershell
$Models = Get-Content -LiteralPath '.\inputs\model_manifest.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$Models.models.uni2h  | Select-Object repo_id, revision, snapshot_path, path_source
$Models.models.uni    | Select-Object repo_id, revision, snapshot_path, path_source
$Models.models.virchow2 | Select-Object repo_id, revision, snapshot_path, path_source
```

这些是**已记录路径**，不是当前服务器文件存在性的证明。历史 UNI2-h 特征缓存也没有证明其权重版本与预填快照相同。请用下一步输入检查确认当前文件；特征准备动作会核对模型快照文件和缓存合同，需要新提取特征时才严格加载对应模型，失败时不会回退联网或复用身份不匹配的缓存。

## 2. 创建批次，执行只读输入检查

首个动作不传 `-BatchDir`，脚本会创建一个新批次。本次已于 2026-09-16 执行输入检查，终端打印的运行目录为 `D:\AIPatho\qzs\runs\phase2_fullfov_hpo_v1\20260916_231813_101_7b1b9a79\check-inputs_20260916_231813_101_7b1b9a79`。以下命令是本次已执行的命令；继续本批次时**不要重新运行它**，否则会创建另一个批次。

```powershell
.\run.ps1 -Action check-inputs
```

本次批次目录是上述运行目录的父目录。每次打开新的服务器 PowerShell 窗口，先设置以下变量：

```powershell
$BatchDir = 'D:\AIPatho\qzs\runs\phase2_fullfov_hpo_v1\20260916_231813_101_7b1b9a79'
Test-Path -LiteralPath $BatchDir
```

查看本次检查结果：

```powershell
$CheckRun = 'D:\AIPatho\qzs\runs\phase2_fullfov_hpo_v1\20260916_231813_101_7b1b9a79\check-inputs_20260916_231813_101_7b1b9a79'
Get-Content -LiteralPath (Join-Path $CheckRun 'action_result.json') -Raw -Encoding UTF8
```

逐项看 `packaged_inputs`、`server_assets` 和三模型的 `status`、`required_files`。模型应达到 `snapshot_files_present`；该状态只表示路径、固定 revision 和所需文件已核对，不等于权重已成功加载。新提取特征时，`prepare-features` 才会加载对应权重；复用完整缓存时只严格核对其身份和元数据。`slide_geometry.csv` 的 `patch_coverage_size` 目前为空，因此物理 patch 覆盖范围仍为 WARN；来源组与观测网格不能证明真实切片范围或划分缓冲。还要在服务器结合图像和标签来源记录核对这一点，不能把 WARN 当成已核实。

`config.json` 中的图像、v003 标签、`runs`、`weights`、`feature_caches` 根目录由本包使用。需要查某个值时运行：

```powershell
$Config = Get-Content -LiteralPath '.\config.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$Config.paths | Format-List
```

若要改服务器路径，在准备特征前修改**本包** `config.json`，不要移动或覆盖原始数据、旧缓存和旧权重。

## 3. 显式准备四套特征缓存

在输入路径和快照文件核对后运行：

```powershell
.\run.ps1 -Action prepare-features -BatchDir $BatchDir
```

该动作准备四套 CLS（整块汇总向量）缓存：UNI2-h 历史裁剪、UNI2-h 全视野、UNI 全视野、Virchow2 全视野。需要新提取时会严格加载对应固定快照；已有完整缓存则核对合同后复用。全视野变换拒绝非方图。缓存位于 `D:\AIPatho\qzs\feature_caches\phase2_fullfov_hpo_v1\<编码器>\<预处理协议>\all\`；每套成功缓存有 `COMPLETE` 和元数据。完整且身份、模型版本、快照/权重路径、变换、维度都一致的缓存会复用，不会覆盖旧缓存。

检查终端打印的 `Run directory` 下 `action_result.json` 和 `feature_caches.json`；正常应登记 **4** 套缓存。特征文件本体留在服务器 `feature_caches` 根目录，后续常规回传只带路径登记。

## 4. 两批配对实验：各 12 次

先运行历史配方，再运行候选配方。每批包含 UNI2-h 的两种视野 × 单点/空间头 × 种子 42/43/44。两批使用相同图像编码器权重和预定种子，从头训练 H/C/B（共享层、通路读出层、空间残差层）。

```powershell
.\run.ps1 -Action paired-view -BatchDir $BatchDir
.\run.ps1 -Action paired-recipe -BatchDir $BatchDir
```


每个动作的 `action_result.json` 应显示 `planned: 12`、`completed: 12`、`status: completed`。历史配方是 AdamW、学习率 `3e-4`、批 256、权重衰减 `1e-4`；候选配方是 Adam、学习率 `1e-4`、批 32、权重衰减 0。两批最多 60 轮，正式检查点从第 6 轮选择，早停从第 16 轮计数。`model_weights.json` 登记 warmup、formal、last 检查点路径和状态。

## 5. UNI2-h 两阶段搜索：各 30 次

先完成单点搜索。初始 24 组由 2 锚点、6 固定种子随机配置、16 TPE（根据既有试验推荐参数）配置组成，种子均为 42；前三组再补种子 43/44，总计 30 次。阶段开始时要求 Optuna **5.0.0** 才使用 TPE；若当时不可用，整个阶段固定使用预生成分层随机配置。阶段中途不会切换方法。若在新 PowerShell 窗口操作，`$Python` 不会自动保留；**尚未启动该阶段时**，先单独运行版本检查：

```powershell
$Python = (Get-Content -LiteralPath '.\config.json' -Raw -Encoding UTF8 | ConvertFrom-Json).python_interpreter
& $Python -m pip show optuna
```
现在在这一步。

确认版本或接受随机回退后，再单独启动搜索：

```powershell
.\run.ps1 -Action search-point -BatchDir $BatchDir
```

`pip show` 未找到或版本不是 5.0.0 时，本包仍可按随机回退完成搜索；如计划使用 TPE，须**在启动该阶段前**在服务器环境中准备好指定版本。实际采用的方法写入批次目录的 `search_point.json` 与动作结果。

单点阶段完成后，空间阶段从单点前三名头部配方出发：24 组里 18 组固定所选头部并调整图参数，6 组联合调整头部与图参数。两组锚点使用单点前两名和原图参数。运行：

```powershell
.\run.ps1 -Action search-spatial -BatchDir $BatchDir
```

空间方法及候选记录写入 `search_spatial.json`。每个搜索动作正常应显示 `initial_configs: 24`、`seed42_complete: 24`、`top3_replicated: 3`、`status: completed`。两阶段最多 120 轮，正式选模从第 1 轮开始，早停从第 41 轮计数；只看内部验证集患者—通路等权 PCC（相关系数），同分用患者—通路 zMSE（标准化均方误差）。搜索预算有限，不代表全局最优。

### 可选扩展：每阶段最多一次

仅在需要使用预留额度时**另外显式调用**。建议在单点搜索后、空间搜索前决定是否执行单点扩展；空间扩展应在冻结前决定。每次扩展为 12 个新配置加前三组的种子 43/44 复核，共最多 18 次。

```powershell
# 只在决定使用单点预留额度时执行，随后再启动 search-spatial
.\run.ps1 -Action extend-point -BatchDir $BatchDir

# 只在 search-spatial 完成且决定使用空间预留额度时执行
.\run.ps1 -Action extend-spatial -BatchDir $BatchDir
```

不需要扩展时跳过这两行。扩展计划分别写入 `search_point_extension.json`、`search_spatial_extension.json`；每阶段不得重复创建第二个扩展计划。

## 6. 冻结配方并做 27 次最终消融

两个搜索阶段及已启动的扩展完成后，先冻结内部验证集选出的单点与空间配方，再执行三编码器最终实验。外部 XZY 分数不参与冻结、选检查点或择种子。

```powershell
.\run.ps1 -Action freeze -BatchDir $BatchDir
.\run.ps1 -Action final-ablation -BatchDir $BatchDir
```

冻结文件是批次目录下的 `frozen_selection.json`。最终消融是 UNI2-h、UNI、Virchow2 × 单点/空间/去 B × 种子 45/46/47，共 **27** 次；其他编码器沿用 UNI2-h 已选配方，只调整必需的输入维度，不另做搜索。最终动作结果应显示 `planned: 27`、`completed: 27`。

## 7. 外部正式评估与 Phase3 导出

外部评估先检查首批 24 次配对及最终 27 次实验的正式检查点是否齐全，然后预测 XZY，共 **51** 份预测。外部标签只在预测固定后按身份接入评分文件。确认所有训练任务已完成后执行：

```powershell
.\run.ps1 -Action external-eval -BatchDir $BatchDir
```

查看本动作运行目录的 `external_predictions.json`：应有 `count: 51`，并逐项列出预测文件、正式检查点和特征缓存路径。原始预测在该运行目录的 `raw\*.npz`。

Phase3 交接使用预定的 UNI2-h 全视野空间模型种子 45；导出时还核对同臂种子 45/46/47 的正式来源记录。它只写合同，不复制模型权重：

```powershell
.\run.ps1 -Action export-phase3 -BatchDir $BatchDir
```

导出文件是该动作运行目录的 `phase3_export\phase3_contract.json`，包含动态输入/隐藏维度、CLS 取法、全视野变换、图参数与坐标单位、通路顺序、训练集拟合标准化、三种子来源和种子 45 正式检查点路径。现有 Phase3 消费端仍需后续单独适配；本包提供 `src/export_phase3.py` 的 `predict_from_phase3_contract(contract_path, table, device='cuda')` 推理函数。

## 8. 失败处理与运行目录

每次 `run.ps1` 调用都会在同一批次下创建新的动作运行目录。终端打印精确路径；目录内优先查看 `run.json`（退出状态）、`action_result.json`（计划/完成数）、`logs\errors.log`（错误）、`model_weights.json` 和 `feature_caches.json`（服务器绝对路径登记）。训练单任务的记录在批次目录 `tasks\<task_id>.json`。

动作失败或部分完成时，修正明确的输入/环境问题后，以**相同 `-Action` 和相同 `$BatchDir` 再执行一次**。成功任务会复用；失败任务创建新的 `attemptNN`，保留旧尝试的日志和产物，并从头训练。本包不声称恢复优化器、学习率日程或随机状态。不要手工删除旧运行、缓存或权重来“清理”失败。

服务器目录分工如下，常规回传不带大文件本体：

| 内容 | 服务器位置 | 回传内容 |
|---|---|---|
| 动作运行、原始预测和状态 | `D:\AIPatho\qzs\runs\phase2_fullfov_hpo_v1\<批次编号>\` | 整个批次目录及 `tasks`、搜索/冻结记录 |
| 模型权重 | `D:\AIPatho\qzs\weights\phase2_fullfov_hpo_v1\<批次编号>\` | 仅 `model_weights.json` 中的路径，不复制 `.pt` |
| 新特征缓存 | `D:\AIPatho\qzs\feature_caches\phase2_fullfov_hpo_v1\` | 仅 `feature_caches.json` 中的路径，不复制 `.npy` |

## 9. 手动回传与本地派生分析

按项目的手动回传方式，将服务器 `runs\phase2_fullfov_hpo_v1\<批次编号>\` **整个目录**复制到本机 `D:\AI空间转录病理研究\PFMval_new\experiments\results\phase2_fullfov_hpo_v1\<相同批次编号>\`。保持动作目录、`tasks`、搜索计划和冻结记录的层级；不改写旧结果。权重和特征缓存文件仍留在服务器。

以下在**本机 PowerShell** 运行，待本批次结果回传后使用。分析输出必须是一个尚不存在的新文件：

```powershell
Set-Location 'D:\AI空间转录病理研究\PFMval_new\experiments\phase2_fullfov_hpo_v1'
$LocalBatch = 'D:\AI空间转录病理研究\PFMval_new\experiments\results\phase2_fullfov_hpo_v1\20260916_231813_101_7b1b9a79'
$AnalysisDir = Join-Path $LocalBatch 'analysis'
New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null
python .\src\local_report.py --prediction-dir $LocalBatch --output (Join-Path $AnalysisDir 'local_metrics.json')
```

此入口递归读取回传的内部最佳预测和外部正式预测；非预测用途的 `.npz` 会在报告中列为跳过。报告分别命名：①逐患者逐通路 PCC 后等权平均的 `patient_macro_pathway_pcc`；②全部点位×通路 z 分数展平一次计算的 `pooled_pcc`。内部/外部、模型、实验臂和配方分开统计种子均值/标准差，不把两种 PCC 混作同一指标。若只分析外部预测，把 `--prediction-dir` 指向本次 `external-eval` 动作运行目录即可。

## 10. 本地代码验收与边界

交付时在本机运行以下命令，使用合成四边标记图和小张量验证视野、非方图拒绝、缓存合同、身份/通路顺序、构图与形态边权、H/C/B 梯度、优化器组、两套选模窗口、搜索配额、失败覆盖及训练端/导出端预测一致性：

```powershell
Set-Location 'D:\AI空间转录病理研究\PFMval_new\experiments\phase2_fullfov_hpo_v1'
python -m pytest tests -q
```

本轮结果为 **21 passed**；未下载模型、未使用 GPU、未提取正式特征，也未启动正式训练。审查记录见 `audit_summary.md`，代码与输入来源登记见 `package.json`。来源组只用于身份与构图分组，不充当物理切片范围证明；本包没有新的科研性能结论。
