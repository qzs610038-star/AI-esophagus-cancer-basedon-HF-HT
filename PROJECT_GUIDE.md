# PFMval 项目导航

> 导航视图，状态版本 `265`，核对时间 `2026-09-19T18:15:10+08:00`。完整文档用途与历史关系见[文档注册器](project_state/document_registry.json)。

Phase2 后续交接从[首版完整交接包](团队项目进度与结论/方案共享/Phase2完整交接包_20260919_首版/00_从这里开始.md)开始；模型运行接口从[当前模型包](团队项目进度与结论/qzs/Phase2最终模型_Phase3交接包_20260918/README.md)开始。二者均已在本地形成，发送和队友验收尚未确认。

| 要查的事实 | 主入口 |
|---|---|
| 当前状态、材料完成度、待处理事项 | [CURRENT_STATE.md](CURRENT_STATE.md) / [机器源](project_state/current_state.json) |
| 实验结果是否已接纳、结果实际路径 | [实验注册器](experiments/experiment_registry.json) |
| 简洁进度 / 全部实验索引 | [实验进度](experiments/experiment_progress.md) / [实验总览](experiments/experiment_dashboard.md) |
| 本轮修复、未决矛盾及风险 | [治理完成记录](maintenance_logs/项目事实与规则治理完成与待决事项_20260919.md) |
| 交接目的与论文准备安排 | [交接部署方案](01_指南与解读/部署方案/Phase2实验汇总交接与论文准备_部署方案_20260919.md) |
| 全视野已接纳结果与限定范围 | [接纳记录](experiments/results/phase2_fullfov_hpo_v1/20260916_231813_101_7b1b9a79/analysis/复核接纳记录_20260918.md) |
| 任务3当前代码 | [全视野密度对照](experiments/phase2_task3_fullfov_density_ablation_v1/README.md) |
| 第二批病理基础模型当前代码 | [空间头替换批次](experiments/phase2_backbone_spatial_ablation_batch2_20260919/README.md) |
| 审计材料与适用边界 | [审计入口](03审计报告/README.md) / [审查技能](.agents/skills/pfmval-audit/SKILL.md) |
| 长期边界 / 事实治理方法 | [AGENTS.md](AGENTS.md) / [治理技能](.agents/skills/pfmval-governance/SKILL.md) |
| 用户决定及后续修订 | [指令记录](project_state/directives.jsonl) |
| 实验交付方式 / 服务器路径 | [手动回传约定](project_state/governance/独立实验包与手动回传_20260905.md) / [路径配置](configs/server_paths.yaml) |

文档登记区分 `active`（当前可用）、`historical`（历史参考）、`superseded`（同用途已替代）和 `pending_review`（待审）。`review_due` 表示内容时效待核，不是实验拒绝或文档生命周期。包内复制件和历史摘录通过父入口的 `covered_documents` 定位；它们不因此成为独立事实源。`tracked` 只表示在 Git 索引中，不表示已提交。

旧 MPP 训练合同、Ridge 失败方案、早期 LoRA 建议以及 W###/Gitee/工作树实施流程保留为历史。当前模型包替代了9月12日旧模型包的交付用途，旧包关联实验的科研接纳状态保留。

本轮文档条目共 459 项，其中本地存在 387 项。仍保留 72 个早期已标记缺失的记录，主要为退役代理技能与旧 .qoder 入口；它们没有被伪造恢复。具体清单见文档注册器。
