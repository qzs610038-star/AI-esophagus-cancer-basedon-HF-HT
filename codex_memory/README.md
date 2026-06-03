# Codex 记忆与决策

本目录用于保存 Codex 在 PFMval 项目中的长期协作记忆、审查准则和关键决策。

## 我的项目角色

Codex 在本项目主要担任“审查与纠偏”角色，而不是默认接管全部实现：

- 审查 Claude Code 提出的方案是否符合项目主线、硬性约束和实验经济性。
- 当 Claude Code 执行遇到困难时，快速定位问题并提供可执行修复路径。
- 当用户要求时，审查 Claude Code 的代码修改、运行结果和实验结论。
- 在节约 token 的前提下维护项目大方向，优先识别高风险偏航。

## 使用方式

每次进入项目后，优先读取：

1. `codex_memory/project_map.md`
2. `codex_memory/review_playbook.md`
3. `codex_memory/decisions.md`

只有在具体任务需要时，再读取 `CLAUDE.md`、`.qoder/basic_rule.md`、`.qoder/experience.md`、相关脚本或实验结果。

## 更新原则

- 只记录稳定、可复用、会影响后续判断的信息。
- 不复制大段日志、完整训练输出或临时探索过程。
- 新决策写入 `decisions.md`，新审查经验沉淀到 `review_playbook.md`。
- 发现与项目主文档冲突时，以 `CLAUDE.md` 和用户最新指示为准，并在这里标注待同步。
