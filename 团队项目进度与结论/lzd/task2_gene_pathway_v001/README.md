# 任务二：基因→通路重构审计与候选共享真值准备

本目录是**自包含、只读源数据**的审计 / 候选生成工具，不训练模型。`audit` 只读；`reconstruct` 仅能写入本目录内的全新输出子目录，拒绝现有路径、符号链接和目录外路径。写入期间的 `run_state.json` 保留 `INCOMPLETE` / `FAILED_INCOMPLETE` 状态供中断恢复判断；本工具不声称输出原子或不可变。不会更改本目录外的源码或资产。

## 状态边界

- 任何结果一律为 `provenance_pending` 候选，**不是真实来源确认结果、不是正式标签，也不得以正式真值引用**。只有重构科学合同被正式冻结/批准后，方可引用。
- 当前真实输入的 `*_genes_pre_ssgsea.csv` 319 基因列，按本机 `ssgsea_calculation_tutorial.md` 的说明已完成 bin 面积校正及 `log1p`。工具直接使用这 319 列，**不再次 `log1p`，不再除/乘 `n_bins_aggregated`**。教程只支持候选来源说明，不替代正式合同。
- 冻结的通路顺序来自包内 `aggregation_review/current_phase2_labels/zscore_manifest.json`；点位身份/内部分割来自对应 `split_manifest.csv`；XZY 仅作外部集。
- 严格要求 6 个内部患者与冻结 manifest 的点位完全一致、XZY 的基因/原始 30 通路身份完全互相匹配；不会按坐标猜测、模糊匹配或静默去重。
- barcode 只接受 literal `patch_x<digits>_y<digits>`，可选的明确 `.csv` 文件后缀可剥离；不剥离 `-1`、下划线等任何其他后缀。`patient + barcode` 是身份键，并校验 barcode 坐标、gene CSV 空间坐标和 split manifest 坐标一致。
- 当前 mapping 有 401 个无重复 pathway/gene pair；mapping 内 342 个不同基因中，23 个不在源 319 列，按精确符号交集过滤。逐通路保留/过滤成员写入 `audit_report.json` 与 `parameters.json`。例如 `mhc` 在当前 319 列中只保留 `B2M`（其余 HLA 成员缺失）；这是显式暴露的候选限制，不做别名替换或推断。
- 当前环境记录的评分为 gseapy 1.1.11、ssGSEA、`rank`、`min_size=1`、ES；其他显式参数见输出 `parameters.json`。权重指数/历史版本/全部默认参数的正式合同尚未冻结；这里的当前调用只为可复现候选，结果不具有正式科学合同地位。
- `audit_report.json.status=INPUT_STRUCTURE_PASS` 只表示输入身份/结构审计通过，不代表 raw30 数值核验通过；raw30 比对状态单独记为 `FAIL_DIAGNOSTIC` / `PASS`，且不阻断候选生成。
- 若可读取包内 V003，候选 z 对 V003 z 的诊断按 `internal_val`、`XZY` 分组，各自独立计算30通路统计，并分别记录预期/实际身份数量、患者覆盖及 `COMPLETE_PAIRED` / `PARTIAL` / `UNASSESSABLE`。任何缺失患者都会降低组状态，任一组不完整时总状态不得称为完整配对。XZY 仅用于探索性审计，**不得用于重构合同或参数选择**。V003 train 明确记为 `NOT_VALIDATED_DUPLICATE_IDENTITIES`，不作数值比较，也不猜测去重方式。该诊断与 raw30 比对分开报告。
- 不能把候选当成 V003 的复现或替代，也不能由 raw30 比对结果单独推断差异成因。

## 环境与运行

建议使用已有 `aipatho` 环境（需 numpy、pandas、gseapy；不需要 pytest）：

```bash
cd /home/lzd/ws/aipatho/Phase2任务二_通路聚合复核交接包_20260920/task2_gene_pathway_v001
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B -m unittest -v
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B task2_gene_pathway.py audit
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B task2_gene_pathway.py reconstruct --output-dir outputs_v003
```

路径默认值为本机提供的 `/mnt/d/AIPatho/Patch/...`；包内 split/z-score manifest 默认由脚本位置解析，不依赖当前工作目录。可通过选项逐项覆盖只读来源，例如：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B task2_gene_pathway.py audit \
  --gene-dir /data/genes_tag \
  --raw30-dir /data/visiumhd_patch/2 \
  --mapping /data/genes_tag/gene_to_pathway_mapping.csv \
  --split-manifest /data/current_phase2_labels/split_manifest.csv \
  --zscore-manifest /data/current_phase2_labels/zscore_manifest.json \
  --v003-root /data/current_phase2_labels
```

`--output-dir` 可指定本目录下的全新子目录（如 `outputs_v003`）；相对路径相对脚本目录解析，父目录须已存在。默认 `outputs/` 因原始交接结果已存在，会被安全拒绝；不覆盖任何已有文件。碰撞时只允许一个进程取得目录。失败后检查目录内 `run_state.json`；`INCOMPLETE` 表示进程可能中断，`FAILED_INCOMPLETE` 表示捕获到失败，不能当作完整结果。

不需要 V003 时可附 `--skip-v003`。重构比对默认使用 `atol=1e-6, rtol=1e-6`，逐项判据为 `abs(candidate - raw30) <= atol + rtol * abs(raw30)`；默认要求每个通路所有点都在容差内才标记该通路 `PASS`。此门槛只作诊断，不会将构造物自动晋级为正式标签。可用 `--atol`、`--rtol` 和 `--max-outside-tolerance-fraction` 显式改门槛。

## 流程与可测试函数

1. 审计七患者 gene/raw30 输入、唯一身份、冻结点位全覆盖/无多余、metadata、列名/顺序、319 gene→401 pair mapping、非有限值/负表达/n_bins 非正、空集及过滤成员。
2. 固定过滤后的 gene set 与通路顺序运行 GSEApy。严格要求每个输入 patch × 30 通路对应唯一且有限的 `Name/Term/ES`，然后按输入身份顺序和冻结通路列顺序输出。
3. 只用 split manifest 的 train 点拟合 pathway raw-ES 均值/std（`ddof=1`）；internal validation 与 XZY 仅 transform。遵从包内 zscore manifest 在 transform 后的裁剪设置。
4. 同时仅用 train 点拟合 per-gene 均值/std（`ddof=1`），作为模型预测聚合的可复用参数。`inverse_gene_zscores` 和 `aggregate_predicted_gene_zscores` 可测试；预测聚合函数先 inverse，再按固定 mapping 评分，并拒绝 `stage="truth"`。评估阶段应读取一次性输出的共同真值，不在分支里重算。

## 输出

`reconstruct` 在本目录内指定的全新输出目录写入（本次完成的结果为 `outputs_v002/`；上方示例使用尚未创建的 `outputs_v003/`）。已有目录会报错，不会覆盖：

- `candidate_raw.csv`、`candidate_z.csv`：候选通路 raw ES 和 train-scale z；ID/metadata 列之后均保持冻结通路顺序。
- `candidate_identity.csv`：患者、barcode/patch_stem、坐标、MPP、split、block 与 bin 计数身份清单。
- `parameters.json`：train-only gene/pathway 均值/std、裁剪及 ssGSEA 调用参数、mapping 过滤成员和尺度证据。
- `audit_report.json`：输入结构审计与 mapping 信息。
- `raw30_pointwise_comparison.csv`、`raw30_comparison_by_pathway.csv`：逐点逐通路 raw ES 有符号差/绝对差、点位容差判据、各通路 abs 差统计和门槛状态。
- `v003_pointwise_diagnostic.csv`、`v003_diagnostic_by_pathway.csv`：候选 z 对包内 V003 z 的独立诊断；摘要每通路按 `split=internal_val` 和 `split=XZY` 各一行，并携带覆盖数量/状态；缺失组也保留 `UNASSESSABLE` 摘要行，不阻断重构。
- `sha256_audit.json`、`run_manifest.json`、`run_state.json`：来源/内容文件 SHA-256、当前工具与预编辑快照哈希、运行依赖/参数和执行完成状态。为避免循环，SHA清单不哈希自身、run manifest、可变 run state；run manifest 记录 SHA 清单哈希，完成状态文件记录 run manifest 与 SHA 清单哈希。
- `outputs/tool_source_snapshot.py`：修改前脚本快照；每次新重构都会验证其 SHA-256 与原 `outputs/run_manifest.json:tool_sha256` 一致。原始 `outputs/` 中已有结果文件保持不变。

## 合成单元测试

`test_task2_gene_pathway.py` 使用 `unittest` 和小型合成数据覆盖缺失/重复身份、错序、明确 barcode 规则、不能二次 log/缩放、train-only `ddof=1` 与 inverse、预测先 inverse、真实标签阶段禁止重算、ssGSEA 输出的完整唯一性和冻结顺序。不访问真实病例数据；目录守卫测试仅在本工具目录内使用临时目录。
