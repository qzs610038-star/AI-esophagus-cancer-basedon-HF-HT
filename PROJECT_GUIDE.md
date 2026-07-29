# PFMval 项目导航

> 面向用户的生成入口；不承载实验事实，也不替代 Registry。
> source_sha256: `87d4fd75f24f352ce9ae36874068acb0fb87d001fc8c85da59c1726bcc02d465`

## 我要找什么

| 目标 | 首选入口 |
|---|---|
| 当前结论与阻塞 | [CURRENT_STATE.md](CURRENT_STATE.md) |
| 简洁实验进度 | [experiments/experiment_progress.md](experiments/experiment_progress.md) |
| 实验机器事实 | [experiments/experiment_registry.json](experiments/experiment_registry.json) |
| 工作树身份与 lease | [project_state/workspace_registry.json](project_state/workspace_registry.json) |
| 资产分类与保护 | [project_state/asset_registry.json](project_state/asset_registry.json) |
| 完整运维视图 | [experiments/experiment_dashboard.md](experiments/experiment_dashboard.md) |
| 服务器路径 | [configs/server_paths.yaml](configs/server_paths.yaml) |
| Agent 专用规则 | [AGENTS.md](AGENTS.md) |

## 目录地图

| 目录 | 用途 | 读者 | 事实角色 | 修改方式 | lifecycle |
|---|---|---|---|---|---|
| `project_state/` | 机器治理状态与 schema | agent | source | CLI-only | active |
| `experiments/` | 实验事实与两类生成视图 | shared | source/generated_view | CLI-only/generated | active |
| `01_指南与解读/` | 方案、指南与学习材料 | user | reference | manual/approval-required | active-by-registry |
| `02_组会汇报/` | 组会材料 | user | reference | manual | active-by-registry |
| `.agents/` | Skill 与 Agent 规则 | agent | reference | approval-required | active |
| `archive/` | 历史材料 | user | historical | approval-required | historical |

## 当前 active 文档

- [doc-921860e03dd1](.agents/skills/compare/SKILL.md)
- [doc-e3142f530aee](.agents/skills/data-guide/SKILL.md)
- [doc-097871797001](.agents/skills/doc-registry/SKILL.md)
- [doc-4fe3126360a9](.agents/skills/experiment-log/SKILL.md)
- [doc-c9fed3a8c1f1](.agents/skills/extract/SKILL.md)
- [doc-9a0e28a6bbc0](.agents/skills/git-rescue/SKILL.md)
- [doc-3d986a3331d2](.agents/skills/health/SKILL.md)
- [doc-ca2c55a62b83](.agents/skills/meeting-report/SKILL.md)
- [doc-64713fd9ae4e](.agents/skills/onboard/SKILL.md)
- [doc-492cd9b1883b](.agents/skills/parallel-extract/SKILL.md)
- [doc-99b982a5eb8c](.agents/skills/pfmval-audit/SKILL.md)
- [doc-caaf0fece4e6](.agents/skills/pfmval-governance/SKILL.md)
- [doc-beb5a9361ddf](.agents/skills/post-train/SKILL.md)
- [doc-bbcfcbdf94e6](.agents/skills/sync-server/SKILL.md)
- [doc-df6a61266d27](.agents/skills/train-guide/SKILL.md)
- [doc-b514d4664439](.agents/skills/train/SKILL.md)
- [doc-814b06f5cfe9](.agents/skills/update-ranking/SKILL.md)
- [doc-52fe7ef7027c](.agents/skills/viz-guide/SKILL.md)
- [doc-db8538d7cd90](.claude/next-steps.md)
- [doc-06ca1fc82608](.claude/session-brief.md)
- [doc-40b8c02101e6](.claude/skills/compare/SKILL.md)
- [doc-0e1a3889f557](.claude/skills/data-guide/SKILL.md)
- [doc-65976615ce59](.claude/skills/doc-registry/SKILL.md)
- [doc-65b117d327e2](.claude/skills/experiment-log/SKILL.md)
- [doc-d5381553a8e6](.claude/skills/extract/SKILL.md)
- [doc-4b40912dde0b](.claude/skills/git-rescue/SKILL.md)
- [doc-f72705822783](.claude/skills/health/SKILL.md)
- [doc-36672b1d149c](.claude/skills/meeting-report/SKILL.md)
- [doc-e0f1ea89a226](.claude/skills/onboard/SKILL.md)
- [doc-b089dd676e8f](.claude/skills/parallel-extract/SKILL.md)
- [doc-31b571198b27](.claude/skills/pfmval-audit/SKILL.md)
- [doc-46398de15079](.claude/skills/pfmval-governance/SKILL.md)
- [doc-1e2cda77a042](.claude/skills/post-train/SKILL.md)
- [doc-298d98c90723](.claude/skills/sync-server/SKILL.md)
- [doc-6ed04f28f7c0](.claude/skills/train-guide/SKILL.md)
- [doc-0f428d649623](.claude/skills/train/SKILL.md)
- [doc-654351a24cf0](.claude/skills/update-ranking/SKILL.md)
- [doc-87b54df9f6ab](.claude/skills/viz-guide/SKILL.md)
- [doc-9328df98eb69](01_指南与解读/分析报告/MPP2后续方案与LoRA新数据实验建议_20260709.md)
- [doc-2a8751c5530f](01_指南与解读/部署方案/服务器路径索引_20260701.md)
- [agent-entry](AGENTS.md)
- [view-current-state](CURRENT_STATE.md)
- [doc-8ec9a00bfd09](README.md)
- [doc-f4e7bea90c0b](automation/README.md)
- [doc-9e8fc4dbabef](experiments/decision_log.md)
- [doc-cab153659acd](experiments/experiment_dashboard.md)
- [doc-97369b0611f5](experiments/experiment_progress.md)
- [doc-2f635595aea6](project_state/plans/mpp2_pathway_ridge_calibration_v001_20260717.md)
- [plan-mpp-training](project_state/plans/mpp_training.md)
- [plan-server-maintenance](project_state/plans/server_maintenance.md)
- [doc-a567a7562613](project_state/plans/workflow_governance_v3_20260726.md)

## 历史与过时资料

- [doc-fe4b048ff34a](.agents/skills/_shared/legacy-compatibility-policy.md)
- [doc-3bbbee463565](.agents/skills/pfmval-audit/assets/audit-report-template.md)
- [doc-eea543c941ec](.agents/skills/pfmval-audit/references/evidence-boundaries.md)
- [doc-a9e91078fa55](01_指南与解读/分析报告/MPP2逐通路Ridge校准_r003实验总结_20260718.md)
- [doc-8d442e98e62d](automation/incidents/20260711_mpp2_baseline_cache_root_mismatch.md)
- [doc-411d4d6be86f](automation/jobs/mpp2-paired-smoke-prediction-supplement-20260712/README.md)
- [doc-6ff3e0b876b4](automation/logs/main_workspace_cleanup_and_report_handoff_20260713.md)
- [doc-b7df29785957](automation/logs/mpp2_paired_smoke_review_20260712.md)
- [doc-cdeafd618631](automation/logs/mpp2_pathway_ridge_calibration_r002_troubleshooting_20260717.md)
- [doc-2cd25761ecf8](automation/logs/pre_maintenance_experiment_preservation_snapshot_20260714.md)
- [doc-97488a7e8511](automation/logs/workflow_governance_v3_safe_cleanup_20260726.md)
- [doc-30d6cd0aae35](automation/report_inputs/gemini_mpp_report_handoff_20260713.md)
- [doc-916d4b92af63](deploy/SYNC_GUIDE.md)
- [doc-ce94fd42b6c4](project_state/plans/mpp2_pathway_ridge_calibration_v001_20260717_goal_prompt.md)
- [doc-92f924374faa](project_state/plans/workflow_governance_v3_g08_local_completion_approval_20260727.md)
- [doc-542d6960864e](project_state/plans/workflow_governance_v3_refactor/REFERENCE_open_source_research_workflow_adaptation_20260726.md)

> historical 不等于 deleted；不得把本节资料作为当前执行依据。
