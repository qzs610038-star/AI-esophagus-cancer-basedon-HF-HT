# Task 2 基因预测通路评估交接

## 目标与当前状态

本交付把 **319 基因背景下、用真实基因表达聚合得到的 30 维通路候选真值**固定为 gene-prediction 评估的共享只读目标；预测基因先用预测运行自身的 train-only gene scaler 逆变换，再按同一 319 基因成员及顺序做 ssGSEA，最后使用候选真值对应的 train-fitted pathway scaler 转成通路 z 值。候选真值仍为 `provenance_pending`，来源预测运行是 `smoke`，以下性能数字只作探索性复核，不是正式性能结论。

## 文件导航

```text
delivery_20260923/
├── HANDOFF.md
├── task2_gene_pathway.py
├── outputs_v002/
│   ├── candidate_raw.csv
│   ├── candidate_z.csv
│   ├── candidate_identity.csv
│   ├── parameters.json
│   └── run_manifest.json
├── inputs/
│   ├── split_manifest.csv
│   ├── genes_tag/
│   │   ├── {JFX,XSL,HYZ15040,LMZ12939,TGC,XZY,ZHZ}_genes_pre_ssgsea.csv
│   │   └── gene_to_pathway_mapping.csv
│   └── prediction_run/
│       ├── run.json
│       └── raw/
│           ├── predictions_internal_val.csv
│           ├── predictions_external.csv
│           ├── target_spec.json
│           └── gene_standardization_from_train.json
├── gene_prediction_eval_v001/
│   ├── evaluate_gene_predictions.py
│   └── test_evaluate_gene_predictions.py
├── results/
│   ├── predictions_internal_val.csv
│   ├── predictions_XZY.csv
│   ├── metrics_by_patient_pathway.csv
│   ├── metrics_by_split_pathway.csv
│   ├── evaluation_summary.json
│   └── gene_scaler_comparison.csv
└── science_check/
    └── pathway_comparison.csv
```

- 根目录 `task2_gene_pathway.py` 是评估脚本导入 `score_expression` 的同级计分器；交付评估不会重新构建真值。
- `outputs_v002/` 是唯一共享的候选通路真值和身份表。`candidate_raw.csv` 与 `candidate_z.csv` 分别保存 raw ES 和 pathway train-scale z；`candidate_identity.csv` 冻结其样本、坐标和 split；`parameters.json` 记录成员过滤、gene/pathway scaler 与评分参数；`run_manifest.json` 记录该候选运行状态。
- `inputs/genes_tag/` 放七个病例各自的 319 基因 pre-ssGSEA 输入及映射表，没有放合并后的全病例表。`inputs/prediction_run/` 是本次 smoke 预测及 scaler/target spec；`inputs/split_manifest.csv` 是冻结 split。
- `gene_prediction_eval_v001/` 是评估器与单元测试；`results/` 是已存在的评估输出；`science_check/pathway_comparison.csv` 是 JFX 单点 319 基因与旧背景的诊断表。
- **不要把预测文件里的 `true_<gene>` 当成真实通路真值。**它是模型输入预测表中的真实基因 z 值，用于检查基因 scaler 逆变换；评估器不对这些 `true_gene` 行重算 ssGSEA。通路真值只读自 `outputs_v002/candidate_raw.csv` / `candidate_z.csv`，预测通路则由 `pred_<gene>` 逆变换后评分。

## 科学解释与边界

1. **旧 raw30 与当前 319 基因候选不在同一数值口径。**旧 raw30 的 ssGSEA 在 18,132 基因背景下计算；当前候选用 319 基因背景。尽管两者在已保留的 gene-set 成员上相同，ssGSEA 排名背景不同，ES 数值不能视为可直接对齐。六个 `mhc` 映射符号中，旧 H5AD 只有 `B2M` 存在、其余五个缺失；当前 319 候选过滤后 `mhc` 也只保留 `B2M`。因此这里的 `mhc` 是 B2M 单基因代理，不应解读为完整 MHC 通路分数。JFX 单点对照见 `science_check/pathway_comparison.csv`。若原工程仍可访问，更多旧 raw30 诊断见原工程 `task2_gene_pathway_v001/outputs_v002/raw30_comparison_by_pathway.csv`；该文件及大型逐点差异、H5AD 未包含在本交付中。
2. 冻结样本量为 train **9,472**、internal validation **1,078**、XZY external **1,039**。本次评估覆盖 internal 的六位患者及外部 XZY；XZY 仅探索报告，禁止用于选模、调参或选择科学合同。
3. 已有 smoke 结果中，患者×通路等权的 internal PCC 宏平均为 **0.496128**，XZY 为 **0.416020**。患者×通路等权的 z-MAE 均值约为 internal **0.9361**、XZY **0.6763**。相关性不等于数值精准：例如 internal 的 Fibrosis 通路患者宏平均 PCC/R² 约为 **0.731 / 0.057**，TGFB 约为 **-0.101 / -4.608**；按患者计算再对患者取宏平均后，30 条通路中只有 Fibrosis 的 R² 宏平均为正，且这不表示每位患者都为正。总体 PCC 不能掩盖明显的通路差异，也不替代 MAE/R²。
4. 这些指标对应 `run_kind=smoke` 且真值 `provenance_pending`，不得作为正式结论；也**不能与旧 V003 的 direct 臂指标直接比较**，除非先确认两臂共享完全相同的标签、split、预处理与指标口径。

## 接手人的下一步

1. 先确认并冻结科学合同：319 基因宇宙、GSEApy 1.1.11 的 ssGSEA/ES 设置、gene-set 成员过滤规则，以及尤其是 `mhc` 只剩 B2M 时的命名和生物学解释。当前交付保存了用于评估的运行参数与成员清单，但候选状态并不等于科学批准。
2. 两臂的**最终评估**必须对齐同一份 `outputs_v002/candidate_raw.csv` / `candidate_z.csv` 真值、身份与指标口径。gene 臂训练目标仍是基因表达；direct-pathway 臂则应以该共享通路候选的 train-scale z 值作为训练目标，不能沿用以旧 V003 为目标的模型并直接比较分数。严格只用 `split=train` 行训练、`internal_val` 行选模；锁定选择后再单独报告 XZY，不用 XZY 反向选模。
3. 若要形成正式比较，应使用正式训练而非 smoke 运行，并核实来源与目标构建。本目录复制了现成 smoke 预测表和 319 基因输入，可不依赖外部挂载复现**这次 gene 预测评估**；没有训练模型权重/特征缓存，不能重新训练。若要从 319 基因表重新生成共享真值，需在原工程使用其通路重构工具及运行约定；若要追溯到空间 bin 原始数据，还需要原始 H5AD/矩阵。因此本交付不是全链条独立复现包。

## 复现已有 gene 评估

以下命令从本交付目录运行，所有评估输入路径都显式传入。`--run-dir` **不会**自动改写其它参数的默认值，因此不能只传 `--run-dir`。命令末尾的 `--output-dir` 是评估器目录下一个全新的直接子目录（该选项按评估脚本位置解析）；如该目录已存在，请改成另一个新目录名。依赖为已有环境中的 numpy、pandas、gseapy，无需安装新库。

```bash
cd /home/lzd/ws/aipatho/Phase2任务二_通路聚合复核交接包_20260920/task2_gene_pathway_v001/delivery_20260923
PYTHONDONTWRITEBYTECODE=1 /home/lzd/miniconda3/envs/aipatho/bin/python -B \
  gene_prediction_eval_v001/evaluate_gene_predictions.py \
  --run-dir inputs/prediction_run \
  --prediction-internal inputs/prediction_run/raw/predictions_internal_val.csv \
  --prediction-external inputs/prediction_run/raw/predictions_external.csv \
  --run-json inputs/prediction_run/run.json \
  --target-spec inputs/prediction_run/raw/target_spec.json \
  --source-scaler inputs/prediction_run/raw/gene_standardization_from_train.json \
  --gene-dir inputs/genes_tag \
  --split-manifest inputs/split_manifest.csv \
  --v002-dir outputs_v002 \
  --output-dir results_delivery_recheck_20260923
```

## 搬运说明

已复制的 JSON 中可能保留原始运行时的绝对路径和当时状态；它们是原始运行记录，不代表交付移动后的当前文件位置，移动文件本身不改变科学结论。评估输入与既有结果已复制到本目录，阅读与复现 gene 评估不需要挂载外部数据盘；但重建共享真值仍依赖上文所述的原始材料与审计。
