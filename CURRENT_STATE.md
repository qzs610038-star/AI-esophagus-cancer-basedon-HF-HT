# PFMval Current State

> 按机器状态和用户决策维护；本次补充独立实验包与手动回传决定。
> State revision: `258` | Updated: `2026-09-05T23:52:50+08:00`
## Current directives

- `DIR-20260905-003`：用户确认本地与服务器直接传递文件夹；后续给出准确代码/运行文件夹路径，不自动生成本地或服务器压缩包，必要时用户自行压缩。取消诊断包的自动回传ZIP，保留原始回传和旧文件。已读取用户回传的v001诊断：47 PASS、3 WARN、0 FAIL，仅作为服务器抽样读取观察，不作完整训练就绪或科研结果确认。

- `DIR-20260905-002`：用户批准独立实验包与手动回传策略：工作树要求彻底退役，仅用户明确指令可创建；新实验在 experiments/<实验名> 自包含代码包内维护；PFMval Windows 服务器 D:\AIPatho\qzs 下分 code/runs，手动上传与回传，修复替换代码保留运行结果；Gitee 暂停，GitHub 仅备份和网页查阅，提交推送由用户触发，不做日常实验前 Git 预检；服务器只产出原始结果与训练选择必需指标，其余指标和登记移到本地；指定文档目录 Markdown 正文允许 Git 跟踪。本次仅规则、配置、通用零训练模板，不迁移旧实验、不训练、不操作服务器、不提交推送。

- `DIR-20260905-001` [phase2/meeting_decision]: Phase 2选定软对比联合学习作为当前研究主线，优先针对本数据调优，使其相对冻结UNI2-h加两层MLP的对照获得更清楚的指标改善；具体模型改动后续讨论。全部已列指标暂时待定，后续实验保留计算和记录，逐通路平均PCC与展平/整体PCC均保留，论文采用及主副线后续再定。指标主要用于本项目内部前后与实验对照，暂不开展跨论文实验数值优劣比较。六折留一患者继续暂缓；基因重建、稠密/稀疏验证、多基础模型和Phase 2/3衔接补充实验后置，在软对比指标改善后再推进。本次只登记决策，不启动训练或改变既有结果确认状态。

- `DIR-20260710-003` [automation/repair_loop_rollout]: Keep the automated repair loop as a separate later task; validate Codex CLI first and retain adapter interfaces for Claude Code and other verified non-interactive CLIs.
- `DIR-20260711-005` [server_maintenance/launcher_argument_contract]: Preserve training argparse option spelling exactly, keep explicit compatibility aliases only, and never let launcher event cleanup mask Python stderr or exit codes; retire failed pre-training job mpp2-repair-v003-frozen-20260711 and require a new source commit/job ID.
- `DIR-20260712-002` [mpp_training/mpp2_lora_paired_smoke_decision]: MPP2 paired S0/S1 smoke passed engineering and safety gates but failed the preregistered candidate-effectiveness gate; do not auto-launch identical formal seed 42, multi-seed, rank or hyperparameter searches. Retrieve missing prediction CSVs through a separate Gitee supplement for diagnostics; any further training requires a new explicit approval and must select changes only from internal validation.
- `DIR-20260726-001` [mpp_training/mpp2_user_forced_local_exploration_archive]: 用户强制指定：将 MPP2 本地 explore 源码与历史记录纳入版本管理；所有相关结果仅作为待审查候选，未经用户后续明确批准不得登记为 accepted evidence，亦不得改变训练或部署决策。
- `DIR-20260801-005` [phase3/mpp2_phase3_fusion_strategy_and_baseline_dependency_v1]: 用户于2026-08-01修订MPP2后续排序：删除此前用于判断是否采用MPP2输入的P1验证，因为项目已决定在后续Phase 3研究探索中采用H&E图像与accepted修复版冻结MPP2原始输出联合输入；当前核心比较改为特征融合与直接拼接两种整合方式，不再以H&E-only/pathway-only消融作为是否采用MPP2的决策门。相关结果仍属于研究探索边界，不自动升级为accepted性能证据。后续研究开展前，用户将向队友索取一份Phase 3基线实验及其协议/结果作为新的事实输入；在该基线被本项目核验前，不登记或执行后续实验。CalibratedMPP2 rejected/no-deployment边界保持不变。任务3应重点复核MPP2独立审查报告中的方案1 Track B、方案2 CHRep启发后校准、方案3 VPT与SPIRAL/OT，并区分论文原方法与项目迁移假设。本指令只授权状态与讨论文档修订，不授权训练、服务器操作、实验登记或结果接纳。
- `DIR-20260801-006` [phase3/phase3_suspected_data_quality_exclusion_review_v1]: 用户于2026-08-01补充Phase 3数据处理边界：部分表现不佳的数据可能存在真实质量问题，病例或切片剔除不再写成完全禁止，而作为低优先级、待后续组会讨论确认的数据质量审查候选。在用户组会后明确通知前，不改变当前队列、fold或主性能口径，也不启动剔除。未来如获确认，必须依据预先定义且可审计的客观质量标准，不能仅因病例或fold的AUC较低而排除；应保留原始全队列结果、排除清单及调整后结果，并明确主分析与敏感性分析关系，不得替换或隐去低AUC fold。该指令只授权更新学习指南与状态，不授权实验、训练、服务器操作、数据删除或结果接纳。
- `DIR-20260803-001` [phase3/phase3_clam_raw_slide_primary_protocol_v2]: 用户于2026-08-03确认Phase 3可直接使用当前pCR/MPR标签；不存在可补充的原始临床病理报告，当前标签已经医生核验。主评估单位采用切片级，以扩展有效训练/评估样本量；患者级路线此前探索效果不佳，保留为次要或敏感性分析。MPP2输入采用329张ESCC切片的raw output；共享pCR/MPR患者级分组五折作为显式split_dir，训练复核采用PPT所列seed 1、2、42、61，每个seed 5 folds。0721-新划分.pptx中的历史结果只作方案与配置证据，不自动成为accepted实验结果。本指令授权本地状态和服务器配置登记，不授权服务器操作、训练、结果导入或accepted证据晋级。
- `DIR-20260815-001` [w007_result_review/four_arm_import_and_analysis]: 仅用 W007 四臂专用兼容入口登记 FBR/RCC/HCR/CPGCR 已有结果并开展分析；不做通用治理重构，不启动训练，不依据 XZY 追加训练或选配置，稳定结论须经用户审核
- `DIR-20260827-001` [phase3,project_maintenance,paper_output/w004_local_result_group_acceptance_and_phase2_phase3_paper_package_v1]: 用户于2026-08-27明确授权在当前干净main直接执行W004本地结果治理对账：恢复c4143历史治理真值，校验并正式导入A001-A004四个本地回传包，将其作为一个COMPATIBILITY_ONLY四臂结果组接纳；A002-A004导入时间必须使用本次实际时间，不得回填虚假历史。接纳不授权raw MPP2、患者独立泛化、融合协同、临床效用或稳定空间增益主张。额外SHA-256仅限四个回传包、关键协议和原子事务状态，不开展全仓哈希治理。并授权在本地生成Phase2-Phase3通用期刊型中英双语论文准备包，保持pending_user_review，不连接项目服务器、不训练、不下载模型或训练数据、不推送远端、不写Google Drive、不修改现有W004外部工作树。
- `DIR-20260829-001` [phase2,paper_output,project_maintenance/phase2_outline_mutable_design_truth_source_v1]: 用户于2026-08-29明确指定《基于HE图像的空间通路活性重建用于增强食管鳞癌新辅助免疫治疗疗效预测》论文大纲为Phase 2论文问题、章节范围、建议补充实验和当前作者意图的重要事实源。涉及Phase 2论文整理、实验补充建议、方案设计和结果映射时必须读取该文件。该大纲是可迭代文档，后续可按实际情况微调；除用户另行明确要求外，Agent不得直接修改。大纲中的待办、占位、模型描述或性能表述不得自动视为已完成实验、正式代码合同或accepted结果；若与用户后续提供的实际代码结构、数据定义、正式实验结果或结论冲突，以用户提供的当前事实及相关机器Registry证据为准，并显式报告冲突。大纲登记不授权实验代码写入、W工作树绑定、训练、服务器操作、结果导入或结论晋级。
- `DIR-20260829-002` [phase2,paper_output,external_readonly/phase2_outline_feishu_readonly_source_v1]: 用户于2026-08-29补充指定飞书知识库链接 https://wcnuw50cb7nc.feishu.cn/wiki/EAtUwvpOJieU0tkayMKcCjIYnCY 为Phase 2论文大纲的远程来源，与本地Markdown共同计入项目。Agent可使用user身份和只读权限访问该链接以获取最新大纲内容；飞书版本用于确认当前作者意图，本地文件作为项目内快照，二者不自动互相覆盖。若飞书与本地内容不同，必须报告差异，不得静默同步或修改任一版本。未经用户针对具体写入动作的显式授权，禁止编辑正文、标题、评论、节点结构、权限或进行移动、复制、删除等任何飞书写操作。该远程大纲仍只对论文问题、章节范围与建议实验具有设计权威；实际代码结构、数据定义、正式实验结果和结论继续以用户提供的当前事实及相关机器Registry证据为准。
- `DIR-20260904-003` [project_maintenance,团队项目进度与结论目录/remove_team_and_qzs_skills_keep_minimal_user_directed_rule_v1]: 用户于2026-09-04明确要求删除项目级 team-progress-maintainer 与 qzs-stable-conclusion-writer 两个 Skill 及其相关路由、状态登记和约束，以避免误导模型。项目级 Skill 仅保留 pfmval-governance 与 pfmval-audit。团队项目进度与结论共享目录仅保留一条规则：根据用户指令修改，智能体不自主修改。该规则不引入额外审核、模板、只追加、证据写作或专用校验流程，也不授权智能体在没有用户指令时修改团队目录。
- `DIR-20260904-004` [project_maintenance/minimal_core_governance_v1]: 用户于2026-09-04批准将项目治理精简为四条核心约束：用户授权与作用域、非破坏性操作、科研防泄漏、事实与结论诚实。除此之外，严格预检、Registry、W###、双阶段方案审批、批准文件、source commit、性能门控、哈希和固定同步流程均改为按需工具或WARN，不得自行阻断用户已授权工作。Gitee仅为当前已配置同步通道，未来可按用户指令增加手动压缩包、SSH或其他通道；同步和一般维护不要求哈希。团队目录仍仅根据用户指令修改，智能体不自主修改。历史实验事实继续保留，但不构成通用治理门禁。
- `DIR-20260904-005` [phase2/supplemental_experiment_preparation_v1]（补充计划仍保留；优先级由9月5日会议修订）: 用户于2026-09-04要求在全面推进Phase 2补充实验前开展项目文档与实验结论治理维护。以团队共享论文基线和Phase 2/3任务原文为当前任务来源，以mpp2_barcode_repair_v003_frozen_baseline_20260711为已接纳量化锚点；登记基因预测后通路重建、稀疏训练稠密测试、多病理模型公平比较三项planned实验，但在输入合同和协议明确前不启动训练、不声称已有结果。六折留一患者设计继续暂缓，等待团队讨论。刷新experiment registry、document registry、current state及派生视图，不修改团队共享文档正文。

## Active plans

- `mpp_training`: [project_state/plans/mpp_training.md](project_state/plans/mpp_training.md)
- `mpp2_pathway_ridge_calibration`: [project_state/plans/mpp2_pathway_ridge_calibration_v001_20260717.md](project_state/plans/mpp2_pathway_ridge_calibration_v001_20260717.md)
- `manual_experiment_delivery`: [project_state/governance/独立实验包与手动回传_20260905.md](project_state/governance/独立实验包与手动回传_20260905.md)
- 原 workflow_governance_v3 已退为历史，其工作树/Gitee 规则不再执行。

## Server transport

- 当前通道：`manual_directory`（直接复制文件夹，不自动压缩）；Gitee 暂停，SSH 暂不使用。
- 服务器：PFMval Windows，`D:\AIPatho\qzs\code` 与 `D:\AIPatho\qzs\runs`；未在本次连接或创建目录。
- 新实验模板：`experiments/_template/`；可推导指标与结果登记在本地完成。

## Latest accepted results

- `mpp1_std10val_xzy_ext_uni2h_mlp_20260706` (accepted): PCC=0.7103, raw_MAE=1384.9762, raw_R2=-0.4192
- `mpp2_std10val_xzy_ext_uni2h_mlp_20260706` (accepted): PCC=0.6489, raw_MAE=1209.9316, raw_R2=-0.1554
- `mpp3_std10val_embargo_xzy_ext_uni2h_mlp_20260706` (accepted): PCC=0.6462, raw_MAE=1175.6385, raw_R2=-0.1157
- `mpp4_std10val_xzy_ext_uni2h_mlp_20260706` (accepted): PCC=0.6064, raw_MAE=1083.0882, raw_R2=0.0721
- `mpp5_std10val_embargo_xzy_ext_uni2h_mlp_20260706` (accepted): PCC=0.5759, raw_MAE=1125.1573, raw_R2=-0.0479
- `mpp2_barcode_repair_v003_frozen_baseline_20260711` (accepted): PCC=0.6549, raw_MAE=1176.2114, raw_R2=-0.088
- `mpp1_barcode_repair_v003_frozen_recheck_20260711` (accepted): PCC=0.69, raw_MAE=1360.9619, raw_R2=-0.3719
- `mpp3_barcode_repair_v003_frozen_recheck_20260711` (accepted): PCC=0.6436, raw_MAE=1209.3894, raw_R2=-0.1265
- `mpp4_barcode_repair_v003_frozen_recheck_20260711` (accepted): PCC=0.6151, raw_MAE=1053.1715, raw_R2=0.109
- `mpp5_barcode_repair_v003_frozen_recheck_20260711` (accepted): PCC=0.6072, raw_MAE=1045.6602, raw_R2=0.0811
- `mpp2_huber_loss_paired_v001_20260728` (accepted): metrics recorded in Registry
  - decision_summary: external_xzy_pooled_pcc Δ=-0.002205156130222; next=closed_no_retry
- `mpp2_cpgcr_probe_v001_20260811` (accepted): metrics recorded in Registry
  - decision_summary: four-arm verification accepted on 2026-08-16: single-seed differences are small and mixed; no residual arm is a clear winner. Deep-attribution analysis and interpretation reports remain pending_user_review.; next=review_four_arm_analysis_no_redispatch
- `phase3_dual_baseline_spatial_pathway_transfer_v001_20260804` (accepted): metrics recorded in Registry
  - decision_summary: mean_per_fit_pCR_auc Δ=0.02318181818181819; next=open_new_patient_independent_protocol

## MPP repair evidence

- Verified evidence: `barcode-repair-20260711-d626ad8-v003`.
- Verified data manifest: `barcode-repair-20260711-d626ad8-v003:1204018178a4d355`.
- Active data manifest: `barcode-repair-20260711-d626ad8-v003:1204018178a4d355`.
- Gate status: **active**.

## 当前 Phase 2 会议决策（2026-09-05）

- 当前主线：软对比联合学习，优先针对本数据改进；冻结UNI2-h加两层MLP保留为对照。
- 全部指标暂时待定且继续保留；逐通路平均PCC与展平/整体PCC均记录，最终发表指标后续再定。
- 比较范围：本项目内部前后及方案对照；暂不进行跨论文实验数值优劣比较。
- 排序：先改进软对比指标；六折留一患者暂缓；基因重建、密度、多基础模型及Phase 2/3衔接补充实验后置。
- 用户倾向后续选取有优势的指标发表；两种PCC的绝对高低不能互证优越，需保留完整结果和明确计算定义。
- 本轮只登记决定，未训练、未产生或接纳新结果。详细记录：[project_state/governance/Phase2会议决策_20260905.md](project_state/governance/Phase2会议决策_20260905.md)。
## Hard blocks

- `formal_training_without_user_approval`
- `destructive_or_irreversible_change_without_user_instruction`
- `scientific_data_leakage`
- `promote_pending_result_to_current_conclusion`

## Superseded conclusions

- Gitee is the permanent exclusive server synchronization channel.
- Old three-patient LoRA, Token, frequency and spatial-repair results are tuning references only.
- MPP1, MPP3, MPP4 and MPP5 are not active follow-up training routes.

## Notes and integrity warnings

- Registry is the accepted experiment fact root; raw returned run artifacts take precedence until import completes.
- Automated repair watchers remain deferred to a separate task.
- The conflicting-barcode repair gate was released only for verified staging barcode-repair-20260711-d626ad8-v003; old MPP label assets remain protected and inactive.
- Active MPP repaired-label data manifest: barcode-repair-20260711-d626ad8-v003:1204018178a4d355

## Optional diagnostics

```powershell
python deploy/pfmval_ops.py agent start-check --task general
python deploy/pfmval_ops.py paths validate
```
