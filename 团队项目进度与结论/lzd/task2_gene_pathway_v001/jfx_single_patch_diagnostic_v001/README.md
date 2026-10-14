# JFX 单点 18132 / 319 基因背景对照

独立只读诊断，固定点位 `JFX / patch_x10080_y10080`。脚本用 `anndata.read_h5ad(..., backed="r")` 分别从 E8、E11 H5AD 读取该点全量基因行；不读取/写回其他 patch，不改 H5AD、旧 notebook、源 CSV、mapping 或现有候选。打分输入直接使用 scored H5AD 与 319-gene CSV 中的已存值，不再做 bin 缩放或 `log1p`。

## 运行

```bash
cd /home/lzd/ws/aipatho/Phase2任务二_通路聚合复核交接包_20260920/task2_gene_pathway_v001/jfx_single_patch_diagnostic_v001
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B diagnose_jfx_patch.py
```

输入路径在脚本顶部显式固定为 E8 `output/2/JFX/JFX_scored.h5ad`、E11 `h5ad/JFX_scored.h5ad`、当前 JFX 319-gene CSV、mapping、raw30 和 `outputs_v002` 候选。输出只写本目录内新建的 `results_patch_x10080_y10080/`；如目录已存在，脚本拒绝覆盖。

两次评分都使用旧 `process_mpp_patches.ipynb` 的30 sets 与显式 ssGSEA 参数：`outdir=None, sample_norm_method="rank", no_plot=True, min_size=1, threads=16`；其他参数不覆盖，按 aipatho 中 gseapy 1.1.11 的默认值。脚本同时通过 AST 比较旧 notebook、E11 exporter 与当前 mapping 的 pathway/gene pairs。

## 本点实测结果

- E8 与 E11 都是 2029×18132；选中点的 `var_names` 顺序相同，18132维值最大绝对差为0，bin数均为404。
- 319-gene CSV 与 E11 H5AD 同点对应基因值最大绝对差为 `8.88e-16`。旧30通路集合有342个符号并集，其中23个不在该 H5AD，实际测得交集319；CSV 的319列恰为该交集。旧 notebook、E11 exporter、当前mapping的401个 pathway/gene pair 完全一致。
- 在当前 gseapy 1.1.11 与同一调用下，完整18132背景的30个ES与该点 raw30 最大绝对差约 `4.55e-13`（中位数0）；319背景与 `outputs_v002/candidate_raw.csv` 最大绝对差约 `7.11e-15`。
- 319背景对比完整背景的30通路绝对差中位数约 `794.81`、最大约 `3713.69`。由于两次输入的每个通路实际可用成员相同，这个单点控制对照表明，仅缩小 rank 背景（18132→319）即可产生本次所见偏差；这解释此点，不将单点结论外推为所有病例唯一历史成因。
- `mhc` 在两个实测背景中都只有 `B2M` 一个有效成员；`tls`、`Glycolysis`、`mhc` 的逐通路背景ES、raw30、候选 raw/z、成员数和背景分数比见对照 CSV。

## 输出与限制

- `pathway_comparison.csv`：30维完整点位分数与每通路背景比、成员数、差值。
- `summary.json`：身份/metadata、E8/E11/CSV值核对、30维差异统计、代表通路、调用参数、输入 SHA-256、脚本 SHA-256及解释边界。
- `sha256_audit.json`：源输入、脚本与生成内容文件校验和；自身不纳入自身哈希。

旧 notebook 未给出历史 gseapy 版本记录；本诊断记录当前版本，并以同点 raw30 / E8 / E11 复现结果交叉核对。此结果仅为单点机制诊断，不是正式标签、不是冻结合同，也不能据此为全队列或其他版本宣称唯一因果。

`ratio_subset319_over_full18132` 是有符号 ES 的算术比值，仅作数值描述；负值、符号翻转或分母接近0时不应作生物学比例解释。
