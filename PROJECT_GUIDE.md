# PFMval 项目导航

> 导航视图，状态版本 rev266，核对时间 2026-09-28T12:29:36+08:00。完整文档用途与历史关系见[文档注册器](project_state/document_registry.json)。

Phase2 经典空间残差方案保持固定，当前转入论文准备。第二批 v006 九项成功训练及9份内部、9份外部预测已接纳，任务3结果已接纳；任务2首批已接纳但完整同目标匹配比较尚未完成。论文 v2 双语草稿及结果图已就绪。交接材料已在本地形成，实际发送与队友验收尚未确认。

| 要查的事实 | 主入口 |
|---|---|
| 当前状态、材料完成度、待处理事项 | [CURRENT_STATE.md](CURRENT_STATE.md) / [机器源](project_state/current_state.json) |
| 实验结果是否已接纳、结果实际路径 | [实验注册器](experiments/experiment_registry.json) |
| 简洁进度 / 全部实验索引 | [实验进度](experiments/experiment_progress.md) / [实验总览](experiments/experiment_dashboard.md) |
| 最新第二批结果与接纳范围 | [第二批复核记录](experiments/results/phase2_backbone_spatial_ablation_batch2_20260919/20260927_154318_475_2242f53b/analysis/复核接纳记录_20260927.md) |
| 任务3正式结果 | [任务3审核报告](experiments/results/phase2_task3_fullfov_density_ablation_v1/formal_20260919_151714/analysis/结果审核报告.md) |
| 任务2首批结果 | [首批接纳与内部核验](团队项目进度与结论/lzd/task2_gene_pathway_v001/delivery_20260923/analysis/首批结果接纳与内部核验_20260927.md) |
| 当前论文写作包 | [v2 阅读入口](团队项目进度与结论/方案共享/Phase2论文写作包_20260922/00_阅读入口.md) |
| 团队材料交付与评审 | [当前模型包](团队项目进度与结论/qzs/Phase2最终模型_Phase3交接包_20260918/README.md) / [论文 v2 包](团队项目进度与结论/方案共享/Phase2论文写作包_20260922/00_阅读入口.md)；实际发送、队友验收及返还未确认 |
| 9月19日治理记录（历史） | [历史治理记录](maintenance_logs/项目事实与规则治理完成与待决事项_20260919.md) |
| 9月19日交接部署方案（历史） | [历史交接方案](01_指南与解读/部署方案/Phase2实验汇总交接与论文准备_部署方案_20260919.md) |
| 全视野已接纳结果与限定范围 | [接纳记录](experiments/results/phase2_fullfov_hpo_v1/20260916_231813_101_7b1b9a79/analysis/复核接纳记录_20260918.md) |
| 任务3代码包与结果 | [全视野密度对照包](experiments/phase2_task3_fullfov_density_ablation_v1/README.md) / [已接纳结果](experiments/results/phase2_task3_fullfov_density_ablation_v1/formal_20260919_151714/analysis/结果审核报告.md) |
| 第二批病理基础模型代码包与结果 | [v006 空间头替换包](experiments/phase2_backbone_spatial_ablation_batch2_20260919/README.md) / [已接纳结果](experiments/results/phase2_backbone_spatial_ablation_batch2_20260919/20260927_154318_475_2242f53b/analysis/复核接纳记录_20260927.md) |
| 审计材料与适用边界 | [审计入口](03审计报告/README.md) / [审查技能](.agents/skills/pfmval-audit/SKILL.md) |
| 长期边界 / 事实治理方法 | [AGENTS.md](AGENTS.md) / [治理技能](.agents/skills/pfmval-governance/SKILL.md) |
| 用户决定及后续修订 | [指令记录](project_state/directives.jsonl) |
| 实验交付方式 / 服务器路径 | [手动回传约定](project_state/governance/独立实验包与手动回传_20260905.md) / [路径配置](configs/server_paths.yaml) |

文档登记区分 active（当前可用）、historical（历史参考）、superseded（同用途已替代）和 pending_review（待审）。review_due 表示内容时效待核，不是实验拒绝或文档生命周期。包内复制件和历史摘录通过父入口的 covered_documents 定位；它们不因此成为独立事实源。tracked 只表示在 Git 索引中，不表示已提交。文档条目与本地存在状态以文档注册器的逐项记录和实际路径为准，不依赖可能滞后的汇总计数。

旧 MPP 训练合同、Ridge 失败方案、早期 LoRA 建议以及 W###/Gitee/工作树实施流程保留为历史。当前模型包替代了9月12日旧模型包的交付用途，旧包关联实验的科研接纳状态保留。

9月19日治理记录及更早的日期记录只说明各自时点的状态；当前状态以机器源、实验注册器和上方最新结果入口为准。具体文档清单见文档注册器。
