# P0B：用户仓库导航与实验进度双视图

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `P0B`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: cross-cutting over `R0`, `R5b`, `R6`, `R8`, `R9`
> authorization: `DIR-20260726-002`, `DIR-20260726-004`
> implementation: `local_D/I_accepted; O=PASS; S=N/A`
> boundary: 本文定义用户信息架构与生成规则，不改变实验事实或接纳任何结果。

## 1. 目标

在不复制事实源的前提下，明确区分：

- 给 Agent 阅读和执行约束的机器/治理板块；
- 给用户定位、审阅、学习和维护项目全局的用户板块；
- 两者共同引用的实验与证据层。

用户应能在两次点击内回答：

1. 项目每类文件放在哪里、哪些能改、哪些只读、哪些已过时；
2. 已做过哪些实验、结果如何、当前结论是什么、下一步是什么；
3. 某项结论对应哪个 `experiment_id`、`result_id` 和事实源。

## 2. 一套事实源、三类视图

| 层 | 内容 | 是否允许承载事实 |
|---|---|---|
| 机器事实层 | current state、Experiment/Document/Workspace/Asset Registry、manifest、result envelope | 是，按各自职责 |
| Agent 视图 | `AGENTS.md`、`.agents/skills/`、schema、CLI、完整操作 Dashboard、测试 | 只引用机器事实 |
| 用户视图 | 仓库导航、简洁实验进度、学习卡、方案、分析与汇报 | 只引用，不复制为新事实 |

`.claude/skills/` 只保留指向 `.agents/skills/` 的同名薄适配器，不成为
第四套规则源。

## 3. 用户仓库导航

建议建立 tracked、active 的根入口 `PROJECT_GUIDE.md`，或在实现审查时
批准等价名称。内容至少包括：

### 3.1 “我要找什么”索引

| 用户目标 | 首选入口 |
|---|---|
| 看当前结论与阻塞 | `CURRENT_STATE.md` 的用户摘要链接 |
| 看实验进展 | 生成的用户实验进度表 |
| 看实验事实 | `experiments/experiment_registry.json` |
| 看方案与学习材料 | `01_指南与解读/` 的 active 文档索引 |
| 看组会材料 | `02_组会汇报/` |
| 看服务器路径 | active 路径索引与 `configs/server_paths.yaml` |
| 看 Agent 规则 | `AGENTS.md`，标注为 Agent 专用 |
| 看历史资产 | historical/superseded 索引，不混入当前入口 |

### 3.2 目录地图

每个一级目录显示：

- 用途；
- 主要读者：`user | agent | shared`；
- 事实角色：`source | generated_view | reference | historical`；
- 修改方式：`manual | CLI-only | generated | approval-required`；
- 当前 lifecycle；
- 推荐入口。

不得继续把已登记为 historical 的旧文档索引作为当前用户入口。

## 4. 简洁实验进度表

建议生成 `experiments/experiment_progress.md`，文件名可在实现审查中调整。
它与完整 `experiment_dashboard.md` 并存：

- `experiment_dashboard.md`：Agent/运维全量视图；
- `experiment_progress.md`：用户简洁视图；
- 两者均由 `experiment_registry.json` 生成，禁止手工维护。

用户表最少字段：

| 字段 | 说明 |
|---|---|
| 可读名称 | 面向用户的短名称 |
| experiment ID | 可追溯机器身份 |
| W### | 已分配时显示 |
| 目的/比较 | 本实验回答什么问题 |
| 阶段 | preflight/smoke/formal/posthoc 等 |
| 状态 | planned/running/done/failed/paused |
| 证据等级 | accepted/pending/rejected/historical |
| 关键结果 | 仅显示经允许的核心指标 |
| 当前结论 | 采用、拒绝、仅参考或待审查 |
| 下一步 | 一个明确动作 |
| 更新时间 | Registry 的时间 |

展示规则：

- 当前/待处理实验置顶。
- accepted 结果单列。
- historical、superseded、rejected 放入历史区，不与当前进度混排。
- 不用长英文内部 action 代替用户可读结论。
- 没有 accepted result 时明确写“暂无”，不得从源码或聊天推断。

## 5. 生成与一致性

以下事件必须原子刷新两类实验视图：

- experiment 注册或协议 revision 变化；
- attempt 状态变化；
- result import、accept、reject；
- closeout 或 lifecycle 变化。

生成器必须：

- 从 Registry 读取，不从旧 Dashboard 反向解析；
- 写入 source hash、生成时间和状态 revision；
- 检测表头/行列数；
- 检测摘要中的 experiment/status 是否与 Registry 一致；
- 同一事务失败时恢复 Registry 和两类视图，避免部分更新。

目录导航在 Document/Asset Registry 变化后 regenerate+diff；它不得复制
实验指标。

## 6. 分步实施映射

| 能力 | 主要实施模块 |
|---|---|
| 当前入口与历史文档基线 | [R0](R0_baseline_and_compatibility.md) |
| 文档生命周期和生成门禁 | [R5b](R5b_document_registry_and_state_gates.md) |
| 资产分类与目录角色 | [R6](R6_asset_registry_shadow.md) |
| 分级解禁与 Hook | [R8](R8_lifecycle_migration_and_unprotection.md) |
| closeout 刷新与归档 | [R9](R9_closeout_and_cleanup.md) |

## 7. 主要风险

- 用户视图成为第二事实源并被手工修改。
- historical 文档仍通过 README 被推荐为 current。
- 简洁表省略 evidence tier，使旧结果看起来仍可用于决策。
- Dashboard 摘要与表格状态不一致。
- `01_指南与解读/*` 的 ignore 策略导致用户入口无法跟踪。
- 自动生成时覆盖用户手写学习内容。

## 8. 验证

至少新增：

```powershell
python -m pytest tests/test_user_project_guide.py -q
python -m pytest tests/test_experiment_progress_view.py -q
python -m pytest tests/test_pfmval_state.py -q -k "document or dashboard or source_hash"
```

必须覆盖：

- 用户入口是 tracked、active 且 Document Registry 分类正确。
- 所有用户入口链接存在；current 区不链接 historical/superseded 文档。
- 31 个或更多实验时，当前区仍只显示当前/待处理项目，历史区完整可追溯。
- Registry 中状态改变后，两类视图同时更新。
- 摘要不得建议执行已经完成或关闭的实验。
- Markdown 表头和每行列数一致。
- 生成器连续运行两次字节级稳定，时间戳策略明确。

## 9. P0B 验收条件

- 存在一份可直接面向用户的 tracked 仓库导航。
- 存在一份与 Experiment Registry 挂钩的简洁实验进度表。
- Agent 规则、用户材料和共享事实层的边界在导航中可见。
- current/historical 不混淆，旧 SSH/Tunnel 等非 active 资料不得作为当前入口。
- import/closeout 后视图可自动刷新并通过一致性测试。
- 本文通过审查不等于上述文件已经实现。

## 10. 待用户审查

- [ ] 用户根入口最终命名：`PROJECT_GUIDE.md` 或更新后的 `README.md`。
- [ ] 简洁实验表最终位置与中文名称。
- [ ] 历史实验默认折叠、分文件或保留在同页末尾。
- [ ] 用户表中允许显示哪些服务器资产信息；默认只显示 ID、状态和受控链接。

## 11. 补充记录

- `2026-07-26 / DIR-20260726-004`：首次建立 P0B 横向硬验收模块。
