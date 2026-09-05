# Codex 与 Claude Code 协作方案

> 日期：2026-06-15
> 适用范围：PFMval 项目先行版；后续可迁移到其他 VSCode 大型项目。

## 1. 背景判断

当前开发环境是 VSCode 中同时开启 Claude Code 与 Codex 对话插件，两个助手之间主要靠用户手动复制粘贴传递信息。历史上 Claude Code 是主实现者，接入 DeepSeek-v4-pro，优点是便宜、上下文量大、适合长时间执行；弱点是复杂架构判断、代码风险控制和实验经济性判断不稳定。Codex 过去因额度不足，主要负责审批、审查和纠偏；现在额度充足，应从“被动审批者”升级为“架构审查 + 高风险实现 + 最终验收”的主控协作者。

本项目已有三套上下文资产：

- `CLAUDE.md`、`.claude/`、`.qoder/`：Claude Code 的主要项目记忆、skills、hooks、任务状态。
- `codex_memory/`：Codex 的轻量审查记忆、项目地图、决策日志、审查清单。
- `experiments/`：实验事实源，其中 `experiment_registry.json` 是实验状态和决策门的唯一事实源。

协作方案的核心目标不是让两个助手同时乱改代码，而是让它们形成“低成本执行者 + 高可靠审查者”的闭环。

## 2. 总体分工

| 角色 | 主责 | 适合任务 | 不适合任务 |
|------|------|----------|------------|
| Claude Code | 低成本长执行、批量修改、项目内既有流程复用 | 根据既定方案改代码、跑训练命令、更新 `.claude/skills`、维护 session brief、批量整理日志 | 独立决定新研究方向、大范围重构、高风险路径/数据逻辑修改、绕过实验 registry |
| Codex | 架构判断、风险控制、复杂实现、最终验收 | 方案评审、代码 diff 审查、实验设计、复杂 bug 修复、跨文件改造、结果解释、报告/dashboard | 长时间陪跑训练、低价值批量日志搬运、已明确模板化的小改动 |
| 用户 | 目标设定、资源授权、最终方向选择 | 确认研究优先级、提供服务器/GPU状态、决定是否启动长训或安装工具 | 在两个助手之间长期手工搬运完整上下文 |

建议默认节奏：

1. Claude 先执行低风险探索或草拟方案。
2. Codex 审查方案，压缩成明确任务单。
3. Claude 按任务单实施小步修改。
4. Codex 看 diff、跑关键验证、给通过/返工结论。
5. 重要结论写入 `experiments/`、`.claude/next-steps.md`、`codex_memory/decisions.md` 中对应位置。

## 3. 任务路由规则

### 3.1 直接交给 Claude 的任务

- 已有命令模板的训练、推理、可视化、文档更新。
- 按 `.claude/skills/*/SKILL.md` 执行的标准流程。
- 批量读取日志、整理训练输出、生成初版组会材料。
- 将 Codex 给出的补丁思路落地为小范围代码修改。

### 3.2 必须交给 Codex 的任务

- 涉及 `histogene/`、`egnv1/`、`egnv2/` 的任何修改建议。
- 涉及 `config_utils.py`、数据交集逻辑、`predictions.csv` 格式、best epoch 选择逻辑的修改。
- 新模型路线、新损失函数、新训练范式、新跨患者实验矩阵。
- Claude 输出“看起来很复杂但没有清晰对照组”的方案。
- 训练结果解释、模型排名更新、是否继续投入 GPU 的决策。
- Git 同步、服务器部署、清理文件、删除文件等可能破坏结果的操作。

### 3.3 可由任一助手处理，但 Codex 最终验收的任务

- 新训练脚本或新参数开关。
- 结果可视化与报告生成逻辑。
- 文档体系重组。
- 长期计划、路线图、组会汇报中的关键结论。

## 4. 双助手交接协议

手动复制粘贴仍可用，但每次不要复制完整长对话。建议统一使用“交接包”，最多 60 行。

### 4.1 Claude 交给 Codex 的模板

```markdown
## Claude -> Codex Review Packet

Task:
{一句话说明 Claude 刚做了什么}

User goal:
{用户原始目标，必要时摘录}

Files changed:
- {file 1}: {改动意图}
- {file 2}: {改动意图}

Commands run:
- `{command}` -> {关键结果}

Key assumptions:
- {假设 1}
- {假设 2}

Need Codex to check:
- {最担心的风险}
- {需要决策的问题}

Do not miss:
- protected dirs: `histogene/`, `egnv1/`, `egnv2/`
- best epoch uses `val_loss`
- registry source: `experiments/experiment_registry.json`
```

### 4.2 Codex 交给 Claude 的模板

```markdown
## Codex -> Claude Implementation Packet

Decision:
{通过 / 返工 / 暂停 / 改走方案 B}

Why:
{1-3 条关键理由}

Exact tasks:
1. {文件 + 函数 + 改法}
2. {文件 + 函数 + 改法}
3. {验证命令}

Guardrails:
- Do not modify `histogene/`, `egnv1/`, `egnv2/`.
- Do not hardcode absolute data paths.
- Keep old behavior as default unless explicitly requested.

Return packet:
- changed files
- command output summary
- unresolved questions
```

## 5. 本项目共享事实源

| 信息类型 | 主文件 | 规则 |
|----------|--------|------|
| Claude 启动上下文 | `CLAUDE.md` | Claude 每次会话优先读 |
| Claude 高频状态 | `.claude/next-steps.md`、`.claude/session-brief.md` | Claude 执行后更新 |
| Claude 技能流程 | `.claude/skills/` | Claude 执行标准工作流 |
| Codex 轻量记忆 | `codex_memory/README.md`、`project_map.md`、`review_playbook.md`、`decisions.md` | Codex 审查前优先读 |
| 实验事实 | `experiments/experiment_registry.json` | 实验状态唯一事实源 |
| 实验摘要 | `experiments/experiment_dashboard.md`、`experiments/decision_log.md` | 用于报告和决策 |
| 指南文档 | `01_指南与解读/` | 新指南、方案、分析报告统一放这里 |

不要让 Claude 和 Codex 分别维护两套互相冲突的实验状态。涉及实验状态时，先更新 `experiments/experiment_registry.json`，再生成 dashboard，再同步摘要文档。

## 6. 推荐工作流

### 6.1 新实验方案

1. 用户向 Codex 描述目标。
2. Codex 读取 `codex_memory/`、`CLAUDE.md`、`experiments/`，判断是否值得做。
3. Codex 输出实验设计：基线、变量、停止条件、验证命令、预期输出。
4. Claude 根据设计实现和运行 smoke test。
5. Claude 返回 Review Packet。
6. Codex 审查 diff 和 smoke 结果。
7. 通过后再启动长训。

### 6.2 Claude 已经改完代码

1. 让 Claude 给出 Review Packet。
2. Codex 先看 `git diff --stat` 和变更文件列表。
3. Codex 按 `codex_memory/review_playbook.md` 审查。
4. Codex 只在必要时改代码；否则给 Claude 精确返工任务。
5. 验证后更新决策日志或实验 registry。

### 6.3 训练完成后的结果闭环

1. Claude 收集 `training_history.csv`、`per_pathway_pcc.csv`、`predictions.csv`、`training_summary.txt`。
2. Codex 判断样本数、NaN、best epoch、PCC/MAE/R2、Train-Val Gap。
3. 如果结果有效，执行或要求执行 `scripts/finalize_experiment.py`。
4. 同步 `experiments/experiment_dashboard.md` 和 `experiments/decision_log.md`。
5. 重要结论写入 `.claude/next-steps.md` 与 `codex_memory/decisions.md`。

## 7. 成本控制策略

Claude 使用 DeepSeek-v4-pro，适合作为“长时间低成本执行层”。Codex 额度充足后，不需要再只做审批，但仍应把高价值任务留给 Codex：

- 让 Claude 做：重复执行、格式整理、按模板更新、长日志初筛。
- 让 Codex 做：新方向判断、实验经济性、复杂 bug、代码审查、最终结论。
- 避免双方都全量读仓库。交接包只传任务、变更文件、关键输出和风险。
- 避免 Claude 长篇解释后再让 Codex 从头读一遍。Claude 应给文件名、行号、命令和摘要。

## 8. VSCode 双插件操作建议

当前不需要立刻做复杂自动化。先把流程标准化：

1. 一个任务只指定一个“当前执行者”。
2. 另一个助手只做审查或给任务单，不同时改同一批文件。
3. 每次切换助手时，用交接包，不复制整段聊天。
4. Codex 审查通过前，不让 Claude 启动长训或服务器同步。
5. 涉及删除、清理、reset、服务器同步、训练结果覆盖，必须 Codex 复核。

后续如果要减少手动复制，可以新增一个本地桥接文件，例如 `.claude/codex-handoff.md` 或 `codex_memory/handoff.md`。但第一阶段建议先不用自动化，避免引入新的同步复杂度。

## 8.1 OpenAI 官方迁移能力

OpenAI 官方 curated skills 中存在 `migrate-to-codex`。它不是 Claude Code 与 Codex 的双向协同 skill，而是 **Claude Code -> Codex** 的迁移工具，目标是把 Claude 侧的 instructions、skills、slash commands、subagents、hooks、MCP config 等转换到 Codex 侧文件。

官方迁移方向：

| Claude Code 来源 | Codex 目标 |
|------------------|------------|
| `CLAUDE.md` / 指令文件 | `AGENTS.md` |
| `.claude/skills/*/SKILL.md` | `.agents/skills/*/SKILL.md` |
| `.claude/commands/*.md` | Codex skills |
| `.claude/agents/*.md` | `.codex/agents/*.toml` |
| `.mcp.json` / `.claude.json` MCP 配置 | `.codex/config.toml` |
| `.claude/settings.json` hooks | `.codex/hooks.json` |

在本项目当前状态下，迁移会主要涉及：

- 项目级 16 个 Claude skills：`.claude/skills/`
- 项目级 1 个 subagent：`.claude/agents/code-reviewer.md`
- 项目级 hooks/settings：`.claude/settings.json`、`.claude/settings.local.json`
- 用户级 6 个 Claude skills：`~/.claude/skills/`
- 用户级 Claude 配置：`~/.claude.json`、`~/.claude/settings.json`

注意事项：

- 官方迁移是单向导入/转换，不会让 Claude 自动调用 Codex。
- 迁移后要人工复核权限、hooks、MCP、subagents 语义，因为 Claude 与 Codex 的运行时并非 1:1。
- 对本项目，建议先做 `--scan-only` / `--plan` / `--dry-run`，不要直接写入 `.codex/`。
- 双向协作仍建议使用本方案的 Review Packet / Implementation Packet 或共享 handoff 文件实现。

## 9. PFMval 当前默认协作模式

对本项目，建议从今天起采用以下默认模式：

| 场景 | 默认主控 | 说明 |
|------|----------|------|
| 9 患者数据到齐状态确认 | Claude 执行，Codex 判断 | Claude 查文件和样本数，Codex 决定后续矩阵 |
| LoRA 9-fold 实验矩阵 | Codex 设计，Claude 执行 | Codex 定义基线、变量、停止条件 |
| 训练脚本小参数扩展 | Claude 初改，Codex 审查 | 保持旧行为默认不变 |
| 新模型方向 | Codex 主控 | 必须先过实验经济性和负向经验检查 |
| 组会汇报 | Claude 初稿，Codex 复核结论 | Codex 审查是否夸大、是否违背 registry |
| 部署/同步 | Codex 审批，Claude 执行 | 禁止 `git clean -fd`，同步命令需复核 |

## 10. 可迁移到其他项目的最小模板

每个新项目只需建立四类文件：

- `PROJECT_AGENT_CONTEXT.md`：项目铁律、环境、入口、禁止事项。
- `agent_memory/project_map.md`：Codex 风格的轻量项目地图。
- `agent_memory/review_playbook.md`：审查清单。
- `agent_memory/decisions.md`：稳定决策日志。

Claude 负责维护长上下文与执行 skills；Codex 负责维护审查记忆与决策质量。这个分工可以迁移到任何 VSCode 大型项目。

## 11. 立即执行清单

- 新任务开始：用户先决定“Claude 执行”还是“Codex 主控”。
- Claude 执行后：必须给 Codex Review Packet。
- Codex 审查后：必须给 Claude Implementation Packet 或明确通过。
- 实验相关：任何结论先对齐 `experiments/experiment_registry.json`。
- 文档相关：新指南放入 `01_指南与解读/` 并更新 `01_README.md`。
- 长期记忆：稳定经验进入 `codex_memory/` 或 `.claude/skills/`，不要只留在聊天里。
