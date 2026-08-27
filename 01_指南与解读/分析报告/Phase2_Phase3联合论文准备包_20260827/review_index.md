# Phase 2–3 联合论文准备包审核索引

> 交付状态：`pending_user_review`
> 交付整理日期：2026-08-28（材料与检索冻结于 2026-08-27）
> 执行边界：本地 `main`；未连接项目服务器、未训练、未下载模型/训练数据、未推送远端、未写 Google Drive。

## 1. 建议审核顺序

1. 先读 [中文稿](manuscript_zh.md) 或 [英文稿](manuscript_en.md)，确认论文故事与结论边界。
2. 查看 [证据台账](evidence_ledger.csv)，逐条审核允许表述和禁止外推。
3. 查看 [新颖性矩阵](novelty_matrix.csv) 与 [引用库](references.bib)，核验文献定位。
4. 查看 [方法—代码映射](method_code_mapping.md) 与 [复算数据索引](reanalysis_data_index.md)，核验实现和派生指标。
5. 查看 [反方审查](adversarial_review.md) 与 [报告清单](reporting_checklist.md)，决定是否进入投稿定稿。

## 2. 核心文件

| 交付物 | 文件 | 当前用途 |
|---|---|---|
| 双语 IMRaD 初稿 | `manuscript_zh.md`、`manuscript_en.md` | 通用期刊型 Phase 2+3 叙事 |
| 证据台账 | `evidence_ledger.csv` | claim→来源→治理状态→允许/禁止表述 |
| 新颖性审查 | `novelty_matrix.csv` | 44 篇官方来源的差异、研究设计等级与来源适配度矩阵 |
| 引用资产 | `references.bib`、`literature_search_log.md` | 39 个唯一 BibTeX 键；检索与纳排记录 |
| 云端材料路由 | `cloud_materials_ledger.md` | `internal_audit_only`；4 份 2026-08-26 Google Docs 只作为设计建议，公开版移除私有链接 |
| 方法证据 | `method_code_mapping.md` | 论文方法→代码/配置→证据状态 |
| 图表 | `figures/`、`figures_tables_plan.md` | 1 张框架图、3 张量化待审图及后续计划 |
| 复算 | `reanalysis_data_index.md` | 版本化 CSV、状态与可复算入口 |
| 风险审查 | `adversarial_review.md`、`editorial_ethics_review.md`、`reporting_checklist.md` | 拒稿风险、独立编辑/伦理处置、报告规范缺口 |
| 术语 | `terminology_bilingual.csv` | 31 组中英术语 |

## 3. W004 治理接纳结果

- W004 在 Experiment Registry 中仅一条，protocol revision=2。
- 四臂共同绑定 `W004-STAGE2-A001-A004-GROUP-V001`，保留四个 `result_ids`。
- `evidence_status=accepted`，但永久保留 `evidence_scope=COMPATIBILITY_ONLY`。
- A002−A001：pCR `+0.023182`；MPR `+0.003958`。
- A003−A004：pCR `+0.002955`；MPR `−0.000625`。
- `raw_mpp2_claim=false`；不得宣称患者独立泛化、融合协同、临床效用或稳定空间增益。
- A001 保留历史导入；A002–A004 使用 2026-08-27 本次实际登记时间。

最小 SHA-256 仅覆盖四个回传包、关键协议与事务状态，没有扩展为全仓哈希治理；原回传包未改写。

## 4. 探索性复算摘要

### Phase 2

- 840 条 patient×pathway 明细，患者等权聚合为 240 条 pathway 主表。
- internal_val 为 6 位患者等权；外部病例 E1 为单外部病例。
- 内部逐通路均值：RCC 的 PCC/CCC/raw R² 为 0.6504/0.6112/0.2058；CPGCR 为 0.6503/0.6103/0.1990，差异很小。
- 外部病例 E1 四臂逐通路 PCC 为 0.5165–0.5194、CCC 为 0.3978–0.3993，raw R² 全为负。
- 840 条 Moran’s I 可计算；热点是无置换检验的描述性象限计数。

### Phase 3

- 320 条 per-fit、16 条 pooled、4 条固定 AUC 差值、620 条 `case_id` 事后描述。
- pCR pooled test：ORD AUPRC/Brier 为 0.8236/0.1361；ABS 为 0.8132/0.1458。
- pooled 620 行来自重复 fit，不是 620 位患者；校准和 AUPRC/Brier 只作描述。
- W004 prediction CSV 无坐标列，Moran’s I/hotspot 为 `not_computable`；未重建受保护数据。

全部新复算均为 `exploratory_reanalysis / pending_user_review`，没有写入 Registry。

## 5. 新颖性与来源

- 纳入 44 篇官方来源；32 篇发表于 2024–2026，43 篇同行评审，1 篇预印本已降级标注。
- PEaRL 已正式发表于 WACV 2026；HEtoSGEBench 对应 Nature Communications 2025 正式论文；DeepPathway 为 Bioinformatics 2026 accepted manuscript。
- 可写创新点是“连续通路几何软目标 + 冻结锚点/零起点残差 + 匹配四臂 + 治理化负证据桥接”的组合，不是“首次通路对比学习”或 SOTA。
- Google Drive 2026-08-26 四份材料只用于指标与实验设计，不作为项目结果。

关键官方来源：[PEaRL（CVF）](https://openaccess.thecvf.com/content/WACV2026/html/Majumder_PEaRL_Pathway-Enhanced_Representation_Learning_for_Gene_and_Pathway_Expression_Prediction_WACV_2026_paper.html)、[H&E→SGE benchmark（Nature Communications）](https://www.nature.com/articles/s41467-025-56618-y)、[TRIPOD+AI（BMJ）](https://www.bmj.com/content/385/bmj-2023-078378)、[PROBAST+AI（BMJ）](https://www.bmj.com/content/388/bmj-2024-082505)。

## 6. 本地提交

| 提交 | 内容 | 回退方式 |
|---|---|---|
| `605af63` | W004 接纳用户指令 | 新增 revert 提交 |
| `7104bfe` | recovery terminal 校验修复 | 新增 revert 提交 |
| `54958ea` | W004 `check/apply` 与 grouped result 接口 | 新增 revert 提交 |
| `9859a3c` | W004 四臂受限结果组正式接纳 | 新增 revert 提交 |
| `9e89ab6` | RER6 复算脚本与公式测试 | 新增 revert 提交 |
| `3d0cad7` | RER6 待审数据、报告与量化图 | 新增 revert 提交 |
| `5843043` | 固定 W004 迁移前测试夹具，避免正式接纳后的基线漂移 | 新增 revert 提交 |

未使用 `reset`，未改写既有历史。

## 7. 验证结果

- W004 迁移/幂等性最小测试：2/2 通过；覆盖原子 apply、重复 apply 幂等及 grouped result schema。
- 指标公式测试：3/3 通过；包含“不等 spot 数时患者仍等权”的合成测试。另有 32 条 pandas 性能警告，不影响公式与结果正确性。
- RER6 输出：W007 8 张 prediction CSV、W004 320 张 prediction CSV；未读写 Registry。
- 最终严格门禁、Git 范围审计与 `diff --check`：待论文包提交完成后记录。

## 8. 已知局限与必须后续完成的工作

1. W007 仅 seed=42；internal 6 位患者，外部病例 E1 仅 1 位患者。
2. W004 是重复切片级留出，患者可能跨 train/test；`case_id` 聚合不是患者独立 OOF。
3. W004 使用 legacy z-score compatibility 输入，未直接绑定 W007/CPGCR 输出。
4. 当前不能证明 CPGCR 性能最佳、raw MPP2、同源 H&E 双分支融合协同、稳定空间增益或临床效用。
5. 伦理审批编号、知情同意/豁免、队列完整纳排标准、扫描设备与批次信息待作者填写；在补齐前不得投稿。
6. 证据升级需要多 seed W007、患者级 Phase 3 OOF、容量匹配负对照、多患者外部验证和校准。

## 9. 建议用户本轮重点审核

- 是否接受把主线固定为“方法候选 + 多指标可信度 + 受限桥接”，而非性能领先。
- 是否接受把 W004 放入次要结果/补充材料，并始终标 `COMPATIBILITY_ONLY`。
- 是否确认伦理、队列与扫描信息由作者后续补齐；当前文件不填入任何推测内容。
- 是否批准下一轮制定患者独立 Phase 3 与多 seed W007 的部署方案。
