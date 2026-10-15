# PFMval 文档定期维护方案

> ⚠️ **historical local reference**：本文件的多文件手工同步方案已由受 Git 跟踪的 `project_state/`、`CURRENT_STATE.md` 与 `deploy/pfmval_ops.py` 取代。不得再把本文件中的 `.qoder` 或 SSH/Tunnel 相关条目视为 active 指令。

> **当前入口（2026-09-19）**：下方 P0–P3 表格和命令保留历史维护建议，不构成当前执行清单。项目边界以 [`AGENTS.md`](../AGENTS.md) 为准；实验交付以 [`独立实验包与手动回传`](../project_state/governance/独立实验包与手动回传_20260905.md) 为准；实验事实以 [`experiments/experiment_registry.json`](../experiments/experiment_registry.json) 为准。旧 W###、Gitee、工作树、SSH/Tunnel 和 `.qoder` 同步路线不得据此恢复为当前流程。

> 目标：以最低成本保证 AI 工具每次会话不被过时文档干扰决策。
> 原则：**触发式更新**优先于定期巡检——无事不更新，有事精准更新。

---

## P0 — 即时更新（变更发生后必须同步）

**触发条件**：新增/淘汰模型、切换主力 backbone、环境变更、数据路径重构、核心规则变化

| 文件 | 更新内容 | 预估耗时 |
|------|---------|----------|
| `CLAUDE.md` | 最佳 PCC（仅Top-2）、环境配置、铁律 | 3 min |
| `.qoder/basic_rule.md` | 模型体系、运行环境、数据路径 | 5 min |
| `.qoder/experience.md` | 完整模型性能基线、新踩坑记录 | 5 min |
| `.claude/next-steps.md` | 任务优先级排序 | 2 min |
| `.claude/deployment-gotchas.md` | 新增部署踩坑（按需，非每次） | 2 min |
| `README.md` | 模型演进时间线、当前最优结果、快速启动命令 | 5 min |
| `.qoder/repowiki/zh/content/项目概述.md` | 模型矩阵、最佳 PCC | 3 min |
| `experiments/experiment_registry.json` | 实验状态/指标/时间戳同步 | 2 min |
| `experiments/experiment_dashboard.md` | 由 `finalize_experiment.py` 自动刷新，无需手动编辑 | — |
| `experiments/decision_log.md` | 重大架构/策略决策追加 | 2 min |
| `.claude/doc-registry.json` | 由 `update_doc_registry.py` 自动扫描，手动补充 purpose/tags | 1 min |

**更新检查清单**（每次添加/变更模型时过一遍）：
- [ ] 新模型加入 `.qoder/experience.md` 性能基线表
- [ ] CLAUDE.md Top-2 模型更新（如排名变化）
- [ ] 旧模型标记为淘汰（如适用）
- [ ] `.claude/next-steps.md` 任务状态更新
- [ ] `experiments/experiment_registry.json` 状态同步（实际 best_val_pcc/loss/epoch 以 registry 为准）
- [ ] `experiments/experiment_dashboard.md` 通过 `finalize_experiment.py` 自动刷新
- [ ] 最佳 PCC 更新（所有出现位置）
- [ ] 训练脚本速查表更新（如有新脚本）
- [ ] 环境配置更新（如需要新 Python 环境）

**🆕 实验管理自动化规则（2026-06-10）**：
- [ ] **启动训练前**：`experiments/experiment_registry.json` 必须存在对应 experiment id（`status: planned` 或 `running`）
- [ ] **训练完成后**：先运行 `python scripts/finalize_experiment.py --run-dir <dir> --experiment-id <id>` 更新 registry
- [ ] **新增指南/分析/汇报文档后**：运行 `python scripts/update_doc_registry.py` 同步 `.claude/doc-registry.json`
- [ ] **每次大改后**（环境变更/新脚本/配置变动）：运行 `python scripts/check_project_state.py --mode local` 健康检查
- [ ] **服务器启动训练**：使用 `deploy/run_experiment.ps1` 标准化启动（自动 preflight + 环境变量 + 日志）

---

## P1 — 每周巡检（周五 10min）

**触发条件**：一周内发生了训练实验、特征提取、数据分析等操作

| 文件 | 检查项 | 预估耗时 |
|------|--------|----------|
| `.qoder/repowiki/zh/content/训练系统/训练指南.md` | 训练命令模板是否仍有效 | 2 min |
| `.qoder/repowiki/zh/content/环境与配置.md` | Python 环境、CUDA 路径是否变更 | 2 min |
| `.qoder/repowiki/zh/content/数据系统.md` | 缓存目录、config.yaml 是否新增配置 | 2 min |
| `.claude/skills/*/SKILL.md` | Skill 流程是否与实际操作一致 | 2 min |
| `.claude/agents/code-reviewer.md` | Agent 规则是否仍适用 | 2 min |

**巡检命令**：
```bash
# 快速验证路径配置
python config_utils.py
# 检查最近一周修改的 md 文件
git log --since="1 week ago" --oneline -- "*.md"
# 项目状态健康检查
python scripts/check_project_state.py --mode local
# 文档注册表同步
python scripts/update_doc_registry.py
```

---

## P2 — 月度审计（每月第一个工作日 20min）

**触发条件**：月度例行

| 文件 | 检查项 | 预估耗时 |
|------|--------|----------|
| `.qoder/repowiki/zh/content/故障排除.md` | 是否有新的常见问题需要添加 | 3 min |
| `.qoder/repowiki/zh/content/推理评估/推理与评估.md` | 评估脚本路径、输出格式是否变更 | 2 min |
| `.qoder/skills/*.md` | 经验是否有更新（对照 experience.md） | 5 min |
| `01_指南与解读/01_README.md` | 目录索引是否仍有遗漏 | 2 min |
| 存档文件审计 | 检查 01_指南与解读/ 中是否有文件应标记为"已过时" | 8 min |

**审计要点**：
- 凡是文件名含日期且超过 1 个月的分析报告，在文件顶部添加 `> ⚠️ 本文为存档快照，最后验证于 YYYY-MM-DD，当前状态可能已变化`
- 凡是 repowiki 中已覆盖的内容，01_指南与解读 中对应的解读指南可考虑归档

---

## P3 — 里程碑更新（阶段完成后）

**触发条件**：论文投稿前、项目交接、服务器迁移

| 操作 | 说明 | 预估耗时 |
|------|------|----------|
| 全量文档审计 | 逐文件验证所有 md 文档的准确性 | 60 min |
| 死链检查 | 检查所有文档间的交叉引用是否有效 | 10 min |
| 冗余清理 | 删除确认无用的重复/过时文档 | 10 min |

---

## 成本控制策略

1. **触发式优先**：P0 即时更新比 P1/P2 定期巡检更重要——模型变了立刻更新核心文件，其他文件可等到下次自然触及再改
2. **单文件最小化**：每次只改真正变化的文件。比如新增一个 backbone 不需要改故障排除.md
3. **利用 git diff**：`git diff --stat` 看改了什么代码，只更新受影响的 md 文件
4. **AI 辅助**：变更完成后让 AI 自动检查哪些 md 文件需要同步更新（`git diff` + Grep 相关引用）
5. **存档标记**：给 01_指南与解读 中的分析报告加时间戳标记，6 个月后自动视为"历史快照"，降低维护预期

---

## 更新历史

| 日期 | 操作 | 说明 |
|------|------|------|
| 2026-05-20 | 创建 | 初始维护方案，repowiki 从33文件精简为7文件 |
