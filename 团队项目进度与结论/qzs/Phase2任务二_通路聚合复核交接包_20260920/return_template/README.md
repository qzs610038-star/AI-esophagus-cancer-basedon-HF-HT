# 可选的本地数据整理模板

完整合同、映射、脚本和中间数据不要求回传。队友可以使用自己的论文整合目录；如果希望沿用本包结构，可复制本目录为新的、有时间标识的本地工作目录。

可选的本地布局：

```text
return_<YYYYMMDD_HHMMSS>/
├─ aggregation_contract.json
├─ gene_to_pathway_mapping.csv
├─ gene_universe.txt
├─ identity_manifest.csv
├─ dependency_versions.txt
├─ difference_report.md
├─ candidate_raw/
│  ├─ train/<patient>/*.csv
│  ├─ val/<patient>/*.csv
│  └─ external/XZY/*.csv
├─ candidate_z/
│  ├─ train/<patient>/*.csv
│  ├─ val/<patient>/*.csv
│  └─ external/XZY/*.csv
└─ scripts/
   ├─ README.md
   └─ <single_reproducible_entrypoint>
```

## 如果选择使用该布局

1. 可将 `aggregation_contract.template.json` 复制为 `aggregation_contract.json`，在本地记录实际参数。
2. `gene_to_pathway_mapping.csv` 至少包含 `pathway,gene`，并在合同中记录来源、版本、ID 体系、别名和重复规则。
3. `gene_universe.txt` 按实际排序背景逐行记录基因；不能只写“319 genes”。
4. `identity_manifest.csv` 至少包含 `patient_id,patch_id,x,y,split,slide_id,source_file`。
5. raw 与 z 文件都使用 `barcode + 30通路`，顺序与当前主线名称表一致；z 应使用本次共同重构真值**训练点**拟合的参数，raw 和 z 不得混在同名文件。
6. `dependency_versions.txt` 记录实际解释器、Python、GSEApy、NumPy、pandas、SciPy 和 scikit-learn 版本。
7. `difference_report.md` 简述基因/聚合核查结果和共同真值与 V003 是否等价；如需声称逐值等价，请记录身份、通路、数值容差与差异，PCC 高不等于逐值一致。
8. `scripts/README.md` 给出唯一复跑命令、输入与输出，不自动安装依赖。

启动前由队友核实输入基因、映射、算法及身份/顺序，用真实基因只生成一次共同重构30维真值，并只用其训练点拟合标准化。两臂读取同一静态真值，基因臂只聚合预测基因。与 V003 的比较是独立诊断：若不同，仍可对共同重构目标开跑，但不能直接与主线成绩比优劣。合同冻结后才核对 XZY，且 XZY 不得用于选择聚合参数。

完整 mapping、脚本、依赖和中间数据仍只需在队友本地保存，不构成回传或再次审批门槛。

实验结束后只需发送一份 `最简结果说明模板.md`，无需附带上述目录。
