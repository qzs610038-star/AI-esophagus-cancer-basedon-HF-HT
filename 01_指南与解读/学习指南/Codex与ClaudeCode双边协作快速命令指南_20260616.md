# Codex 与 Claude Code 双边协作快速命令指南

> 日期：2026-06-16
> 用途：在 VSCode 同时开启 Claude Code 与 Codex 时，快速触发“Codex 主控、Claude 执行、Codex 审查”的双助手协作流程。

## 1. 最短上手版

新任务先发给 Codex：

```text
Codex 主控，启动双助手协作：任务是【在这里写任务】。
请判断由 Codex 直接完成，还是生成 Codex -> Claude Implementation Packet 交给 Claude 执行。
```

如果 Codex 让 Claude 执行，把 Codex 输出的 Implementation Packet 复制给 Claude，并加一句：

```text
按下面的 Codex -> Claude Implementation Packet 执行。只做列出的任务，完成后返回 Claude -> Codex Review Packet，不要启动长训，不要做服务器同步，不要清理或删除文件。
```

Claude 完成后，把它输出的 Review Packet 复制回 Codex，并加一句：

```text
这是 Claude -> Codex Review Packet。请按 PFMval 双助手协作方案审查，重点检查项目铁律、实验事实源、代码风险和是否允许进入下一步。
```

## 2. 什么时候直接找 Codex

以下任务默认 Codex 主控：

- 新实验路线、新模型、新损失函数、新训练矩阵。
- 修改数据加载、路径、样本交集、`predictions.csv`、best epoch 选择。
- 解释训练结果、更新模型排名、判断是否继续投入 GPU。
- 服务器同步、Git 操作、删除/清理文件、覆盖训练结果。
- Claude 给出复杂方案，但没有明确基线、对照组、停止条件。
- 涉及 `histogene/`、`egnv1/`、`egnv2/` 的任何修改建议。

给 Codex 的口令：

```text
Codex 主控：请先审查这个任务是否值得做，给出基线、变量、停止条件、验证命令和交给 Claude 的最小任务包。
任务：【写任务】
```

## 3. 什么时候先找 Claude

以下任务可以先让 Claude 执行：

- 按已有命令模板跑 smoke test、推理、可视化。
- 整理训练日志、结果 CSV、组会初稿。
- 按 `.claude/skills/` 的既有流程做文档或脚本维护。
- 按 Codex 给出的明确任务单改小范围代码。

给 Claude 的口令：

```text
你是执行者，请按 PFMval 项目规则执行下面任务。
任务：【写任务】

要求：
1. 先读 `CLAUDE.md`、`.qoder/basic_rule.md`、`.qoder/experience.md`。
2. 涉及实验状态时先读 `experiments/experiment_registry.json` 和 `experiments/experiment_dashboard.md`。
3. 不修改 `histogene/`、`egnv1/`、`egnv2/`。
4. 不启动长训，不做服务器同步，不删除/清理文件，除非 Codex 审查通过。
5. 完成后输出 Claude -> Codex Review Packet。
```

## 4. Codex -> Claude Implementation Packet 模板

当 Codex 需要把任务交给 Claude 时，使用这个模板：

```markdown
## Codex -> Claude Implementation Packet

Decision:
{执行 / 返工 / 暂停 / 改走方案 B}

Task:
{一句话说明要 Claude 做什么}

Context:
- {关键事实 1}
- {关键事实 2}

Exact tasks:
1. {文件 + 函数/位置 + 改法}
2. {文件 + 函数/位置 + 改法}
3. {验证命令或检查方法}

Guardrails:
- Do not modify `histogene/`, `egnv1/`, `egnv2/`.
- Do not hardcode absolute data paths.
- Do not change `patch_noov_spilt`.
- Best epoch must use `val_loss`, not `val_pcc`.
- Keep old behavior as default unless explicitly requested.
- Do not start long training or server sync before Codex review.

Return packet:
- changed files
- commands run and key outputs
- assumptions
- unresolved questions
```

## 5. Claude -> Codex Review Packet 模板

Claude 完成后必须返回这个模板：

```markdown
## Claude -> Codex Review Packet

Task:
{一句话说明刚完成什么}

User goal:
{用户原始目标}

Files changed:
- {file 1}: {改动意图}
- {file 2}: {改动意图}

Commands run:
- `{command}` -> {关键结果}

Key outputs:
- {指标、样本数、日志结论、生成文件}

Key assumptions:
- {假设 1}
- {假设 2}

Need Codex to check:
- {最担心的风险}
- {需要 Codex 决策的问题}

Do not miss:
- protected dirs: `histogene/`, `egnv1/`, `egnv2/`
- best epoch uses `val_loss`
- registry source: `experiments/experiment_registry.json`
```

## 6. 常见场景口令

### 6.1 新实验设计

发给 Codex：

```text
Codex 主控，设计一个可交给 Claude 执行的新实验方案。
目标：【写目标】
请给出：当前基线、实验变量、最小 smoke test、停止条件、长训前审查点、Codex -> Claude Implementation Packet。
```

### 6.2 Claude 已经改完代码

发给 Codex：

```text
这是 Claude 完成后的 Review Packet。请做代码审查，优先找 bug、项目铁律违规、实验设计风险和缺失验证。必要时给 Claude 返工任务包。
```

### 6.3 准备启动长训

发给 Codex：

```text
Codex 审批长训：Claude 准备启动以下训练。
请检查实验 ID、基线、命令、输出目录、停止条件、是否会覆盖历史结果，以及是否允许启动。
训练说明：【粘贴 Claude 的计划或命令】
```

### 6.4 训练完成验收

发给 Codex：

```text
Codex 结果验收：以下是 Claude 收集的训练结果。
请检查样本数、NaN/Inf、best epoch 是否按 val_loss、PCC/MAE/R2、Train-Val Gap，并判断是否更新 registry/dashboard/decision_log。
结果包：【粘贴 Claude 摘要】
```

### 6.5 服务器同步或 Git 操作

发给 Codex：

```text
Codex 审批服务器/Git 操作：Claude 准备执行以下命令。
请检查是否有 `git clean -fd`、删除/覆盖风险、路径风险、是否会影响 checkpoints 或训练结果。
命令：【粘贴命令】
```

### 6.6 组会汇报

先发给 Claude：

```text
请按 PFMval 组会汇报风格生成初稿。只做材料整理，不新增未经 registry 支撑的结论。完成后输出 Review Packet 给 Codex 复核。
```

再发给 Codex：

```text
请复核 Claude 的组会初稿，重点检查结论是否夸大、是否与 `experiments/experiment_registry.json` 冲突、是否遗漏负结果和停止条件。
```

## 7. 审查通过/返工的标准回复

Codex 审查通过时，可以直接回复：

```text
Codex 审查通过。允许 Claude 进入下一步：【写下一步】。
限制：仍不得删除/清理文件；长训完成后返回 Review Packet。
```

Codex 要求返工时，使用：

```text
Codex 审查未通过。请 Claude 按下面 Implementation Packet 返工，不要自行扩展范围。
```

Codex 要求暂停时，使用：

```text
Codex 决策：暂停。原因是【写原因】。在【补齐数据/确认实验 ID/修复风险】之前，不允许启动长训或服务器同步。
```

## 8. 每日推荐工作节奏

1. 早上：Codex 看 `.claude/next-steps.md`、`experiments/experiment_dashboard.md`，决定当天任务。
2. 白天：Claude 执行低风险任务和日志整理。
3. 每个关键节点：Claude 输出 Review Packet，Codex 审查。
4. 晚上：Codex 汇总决策，必要时更新 `codex_memory/decisions.md` 或要求 Claude 更新 `.claude/session-brief.md`。

## 9. 绝对不要省略的检查

- 是否修改受保护目录：`histogene/`、`egnv1/`、`egnv2/`。
- 是否绕过 `config_utils.py` 硬编码路径。
- 是否把 `patch_noov_spilt` 改成 `split`。
- 是否用 `val_pcc` 选择 best checkpoint。
- 是否覆盖旧结果目录。
- 是否绕过 `experiments/experiment_registry.json` 手动散落实验状态。
- 是否在服务器上执行 `git clean -fd`。

## 10. 迁移到其他项目的复制模板

给 Codex：

```text
Codex 主控，请为这个新项目建立双助手协作骨架。
请先阅读项目结构，创建或建议以下文件：
1. `PROJECT_AGENT_CONTEXT.md` 或 `AGENTS.md`
2. `agent_memory/project_map.md`
3. `agent_memory/review_playbook.md`
4. `agent_memory/decisions.md`
5. 双助手快速命令指南
并给出 Claude 执行层与 Codex 审查层的分工。
```

给 Claude：

```text
请按 Codex 生成的新项目协作骨架执行。你是低成本执行层，负责按任务单修改、运行和整理；所有高风险决策返回 Codex 审查。
```
