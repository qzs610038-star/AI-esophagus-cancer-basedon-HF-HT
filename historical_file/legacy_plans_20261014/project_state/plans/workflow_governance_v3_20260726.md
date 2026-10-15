> **历史方案（2026-09-05 退役）**：下文保留原始设计供追溯。工作树/Gitee/固定预检流程不再生效；当前按 [独立实验包与手动回传](../governance/独立实验包与手动回传_20260905.md) 执行。创建工作树仅限用户明确指令。

# PFMval Skill / Workflow / Hook 治理 v3 总方案

> lifecycle: active
> state scope: `workflow_governance_v3`
> effective date: 2026-07-26
> authorization: `DIR-20260726-002`, `DIR-20260726-003`, `DIR-20260727-004`
> execution boundary: R0–R11 与 P0A/P0B/P0C 的本地 D/I 已接受；G08 已批准并收口 S1 canonical Skill、六个本地用户工作流和 manifest 驱动 Catalog 准入。真实 O 运行、高风险迁移、解禁、清理、服务器、训练和结果接纳仍保持原门禁。

## 1. 目标

将项目长期反复出现的方案探索、独立审计、服务器实验、证据门禁、资产维护和协作交接收敛为少量可组合工作流，避免：

- 每次实验生成新的服务器工作树、分支和大段临时命令。
- 旧 Skill 绕过当前 Gitee、job/result、Registry 和用户批准边界。
- 对无关文档、日志和全量历史资产反复做阻断性校验。
- 用户材料、Agent 规则、实验事实和可再生产物混放。
- 旧对话或历史报告重新覆盖当前事实源。

## 2. 当前事实与本轮边界

- 当前事实源顺序仍由 `AGENTS.md` 定义。
- 服务器唯一同步通道仍为 Gitee。
- `CURRENT_STATE.md`、Dashboard、next-steps 和 session-brief 仍是生成视图。
- Explore 仍受 `DIR-20260726-001` 限制：未经后续明确批准不得成为 accepted evidence。
- 本方案不授权训练、服务器操作、结果接纳、受保护资产改写、工作树删除或缓存清理。
- 依据 `DIR-20260809-002`，`histogene/`、`egnv1/`、`egnv2/` 已解除特殊保护并转为 MPP1-5 前历史代码迁移候选；实际迁移、取消 Git 跟踪或删除仍须用户审核精确清单，且必须先处理现有调用方和资产登记。
- P0C/R10/R11 的本地 D/I 已接受；G08 只完成无需真实用户内容或服务器的本地入口、
  fixture、preview 与 Catalog 准入。六个用户 workflow 的真实 O 均为
  `NOT RUN`，维护类科学证据等级均为 `S=N/A`。

## 3. 目标架构

### 3.1 两个正式 Skill

1. `pfmval-governance`
   - 面向方案探索、服务器实验、门禁选择、资产现代化和 closeout。
   - 只负责任务识别、事实源读取、工作流选择、用户决策收敛和结果解释。
   - 不复制 `pfmval_ops.py` 的业务逻辑。

2. `pfmval-audit`
   - 面向“已经完成，请校验”“报告是否可信”“能否 GO”等独立验收请求。
   - 默认只读。
   - 固定输出事实源、声明、证据、PASS/WARN/FAIL、GO/NO-GO 和 Remaining。

低自由度执行继续由 `deploy/pfmval_ops.py` 承担，不再创建第三个重复业务逻辑的操作 Skill。

### 3.1.1 兼容 Skill

- `.agents/skills/` 是 tracked canonical source。
- 16 个旧名称保留为安全 compatibility router；旧实现继续封存在
  `.claude/legacy/skills-20260726/`，不得复制回 active path。
- `.claude/skills/<name>/SKILL.md` 保留同名薄适配器，并直接引用
  `.agents/skills/<name>/SKILL.md`；适配器不得包含业务规则、命令或 legacy
  fallback。
- `train`、`sync-server` 等当前仍标记为 `retired-router`。只有实际 CLI、
  schema 和测试完成后，才能升级为 operational canonical Skill。

### 3.2 五个 Workflow

| Workflow | 入口 | 固定输出 | 状态写入 |
|---|---|---|---|
| `explore-plan` | 用户描述问题、方案或假设 | 完整方案、决策记录、1–2 张学习卡 | 用户确认前无 |
| `gitee-experiment` | 已登记 `experiment_id` | 单实验工作区、attempt 记录、回传和关闭状态 | 仅 CLI |
| `evidence-tiering` | 变更清单或实验协议 | E0–E3 证据等级、必要校验与可降级项 | 仅 CLI |
| `asset-modernization` | 整理、退役、解禁或迁移请求 | 资产清单、血缘、替代关系、迁移候选 | 仅 CLI |
| `closeout` | 实验/维护任务结束 | 未导入结果、未登记文档、清理候选、handoff | 不自动写 |

### 3.3 三层职责

| 层 | 职责 | 禁止事项 |
|---|---|---|
| Skill | 读事实、提问、路由、解释、生成审查材料 | 直接批准、接纳结果、删除资产 |
| CLI | schema 校验、Registry/state/manifest 写入、事务和可审计副作用 | 任意 shell、隐藏式自动决策 |
| Hook | 毫秒级确定性路径/命令防护和提醒 | 理解科研语义、自动写状态、自动清理 |

## 4. 已授权的低风险清理

本轮允许：

1. 建立可跟踪的 `pfmval-audit` Skill canonical source。
2. 将 16 个旧 `.claude/skills` 的原始实现可逆隔离到
   `.claude/legacy/skills-20260726/`；active 同名路径仅恢复直接引用
   `.agents/skills` 的 tracked 薄适配器。
3. 将旧 code-reviewer 隔离，并为 3 份旧 workflow 文档增加 historical
   banner。
4. 修复 `block-git-clean.py` 中错误推荐 `git reset --hard origin/main` 的
   提示；Hook 行为不做结构性改写。
5. 落盘本方案、重构审查包和本轮清理日志。
6. 更新 active plan、document registry 和生成状态。
7. 建立 16 个 `.agents` 安全 compatibility router；不得借兼容入口复活旧
   服务器、直接训练或独立 Registry 写入。
8. 落盘 `gitee_numbered_workspace_protocol_v001_20260726.md`，但不在
   workspace/schema/CLI 验收前宣称自动化已实现。
9. 最小修复文档生命周期：`active_skills` 显式登记 canonical/adapter，
   `pending_plan_reviews` 区分 `approved_design` 与 `pending_review`；不顺带
   实施 R5b 的 scanner 幂等、多写入口收敛等其余重构。

本轮不允许：

- 实现服务器实验级工作树重构。
- 修改 job/result schema。
- 放宽正式训练批准。
- 修改 MPP 数据门禁或结果 import 规则。
- 建立或迁移真实 asset registry 数据。
- 解禁、搬移或删除 protected dirs。
- 清理任何本地或服务器 worktree。

## 5. 方案探索规范

### 5.1 流程

采用 grill-like 的分阶段追问：先分离事实与假设，再挑战会改变方案分叉的
关键前提，最后才收敛执行方案。该流程不依赖某个未安装的同名 Skill。

```text
读取事实源
→ 区分已知事实、可查询事实、用户决策和未知项
→ 一次提出一个会改变方案分叉的决策问题
→ 比较 2–3 个候选方案
→ 明确基线、变量、证据、停止条件和回退路径
→ 输出完整方案
→ 生成 1–2 张决策关键学习卡
→ 用户确认
→ 才允许登记 directive/experiment 或实施
```

### 5.2 方案输出最小字段

- `plan_id`
- `source_state_revision`
- 问题定位
- 已核实事实及来源
- 用户决策与采用假设
- 候选方案比较
- 实施步骤
- 证据与验收
- 停止条件
- 回退路径
- 未决项
- 1–2 个关键知识点

### 5.3 学习卡选择规则

只选择会改变方案判断或实验解释的知识点。每张学习卡包含：

- 为什么本方案必须理解它。
- 一句话心智模型。
- 在 PFMval 中对应的代码、数据或指标。
- 一个正确示例和一个常见误区。
- 3 个自测问题。

## 6. Gitee 实验生命周期目标

### 6.1 身份分层

| 对象 | 约束 |
|---|---|
| `experiment_id` | 一个大实验一个稳定 ID |
| `workspace_id` | `W001` 起三位递增、永久不复用；短名仅作显示 |
| local/server path | 同一逻辑 `W###` 的两个注册路径，物理目录只用稳定编号 |
| Gitee ref | 每个 attempt/revision 一个不可变单写者 ref；不为 ref 建新 worktree |
| server worktree | 一个实验一个持久工作树 |
| `protocol_revision` | 关键实验合同改变时递增 |
| `attempt_id` / `job_id` | 同一实验内可多次 |
| `run_limit` / `run_units` | 用户指定结果性执行总数；job 记录真实独立拟合数 |
| result envelope | 每次 attempt 一个 |

### 6.2 目标状态机

```text
registered
→ workspace_open
→ synced
→ checked
→ ready
→ attempt_running
→ result_returned
→ imported
→ accepted / rejected
→ close_ready
→ workspace_removed
→ archived
```

### 6.3 关闭条件

- 无运行中的训练或诊断进程。
- 无未推送 commit。
- 无未回传或未登记结果。
- 所有 attempt 已进入终态。
- 结果已通过本地 import 或明确 rejected。
- 需要保留的大型资产已有服务器路径、大小和哈希。
- 用户明确批准精确工作树路径的移除。

禁止自动全局 `git worktree prune`、递归删除、顺带删除 branch 或 checkpoint。

### 6.4 对话绑定、单写者和标准往返

- 新实验对话写入前由用户指定 `本对话工作树：W###`；Agent 回显
  experiment、路径、branch、HEAD 和 dirty 状态。
- 新实验的专属代码、配置、最简测试、结构化日志索引和 closeout 必须集中在
  `experiments/workspaces/W###/`；未修改公共代码只写入
  `shared_code_manifest.json`，不复制。原始大日志、checkpoint、预测、缓存和
  中间产物仍写入工作树外的已注册 W### 运行目录。
- 实验结束时先把完整 W### 包作为独立审核提交合入 main，再把可复用代码
  晋升到公共目录。完整包后续可退出 main 当前树，但必须保留对应 Git 提交、
  `CLOSEOUT.md` 和治理索引，以便恢复和总结。
- 绑定后所有修改默认在该 worktree；切换须用户显式指定。禁止用一个全局
  current-workspace 文件服务并发对话。
- 默认由本地工作树维护 canonical code。服务器简单兼容适配经 server-owned
  ref 回传 patch，本地审查提交后再下发新 commit，禁止双端同时写。
- 服务器只 fetch 和 detached checkout 精确 commit；不使用自动 pull、
  merge、rebase 或 force push。
- 日志、lease、PID、checkpoint 与结果必须位于 worktree 外的注册 run root。
- 详细状态机、冲突恢复和关闭条件见
  `project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`。

## 7. 实验门禁 v2 目标

### 7.1 证据等级

| 等级 | 内容 | 默认处理 |
|---|---|---|
| E0 | 超参、loss、正则、关键新功能、数据切分/预处理、实际 checkpoint、结论承载结果 | HARD |
| E1 | CLI、路径配置、包装脚本、环境入口 | 针对性测试；必要时 HARD |
| E2 | Dashboard、说明文档、方案、学习卡 | regenerate/diff WARN |
| E3 | diagnostic、explore、普通日志、plots、scratch | inventory 或 non-evidence |

### 7.2 最小可信合同

1. 完整 40 位 `source_commit`。
2. 规范化 `critical_contract_digest`。
3. active data manifest ID 与根哈希。
4. 本次实际消费的 checkpoint、calibrator 或 base model 哈希。
5. canonical result envelope 与结论承载物哈希。

日志、plots、README、可由 predictions 重建的表格只登记 inventory；如果某文件实际承担 best-epoch 或模型选择证据，则升级为 E0。

### 7.3 拟降级项

- 无关 dirty 文件：HARD → WARN。
- 历史/缺失文档：全局 FAIL → lifecycle WARN。
- Dashboard/next-steps：source hash → regenerate+diff。
- 全 MPP1–5 资产每次运行重验：HARD → 激活时全验、run 时只验本 job 子集。
- 普通 stdout/stderr/plots：SHA HARD → inventory。

即使原则已获用户批准，现有正式批准、Gitee-only、关键合同、实际输入与
结果绑定也必须保持原门禁，直到 v2 实现通过兼容测试和分阶段验收后才可
切换。

### 7.4 次数和简单适配

- 登记或批准前默认询问用户允许的结果性执行次数，不设隐式 one-shot，也不
  使用“一次一批”掩盖多个 seed/拟合。
- preflight、dry-run、路径/编码/解释器检查、纯重传和训练进程启动前的
  E1 适配失败不消耗次数。
- 进入 `EXPERIMENT_STARTED` 并开始独立拟合/正式评估后，按真实
  `run_units` 消耗；此后失败也计数。
- 简单服务器适配只有在 E1 allowlist、关键合同 digest 不变、针对性测试
  通过、新 commit/dispatch revision 和 Gitee 回传全部满足时，才可免重复
  批准。
- loss、超参、正则、关键 feature、数据/split/z-score、checkpoint、
  选择或评估策略改变时必须重新询问。

## 8. 资产双板块目标

### 8.1 Agent / 项目板块

- `AGENTS.md`
- `.agents/skills/`
- `project_state/`
- `configs/`
- `automation/`
- `deploy/`
- `scripts/`
- `tests/`

仅保存固定规则、schema、CLI、Registry、manifest、测试和生成逻辑。

### 8.2 用户板块

- `01_指南与解读/部署方案/`
- `01_指南与解读/学习指南/`
- `01_指南与解读/分析报告/`
- `02_组会汇报/`

保存完整方案、学习卡、审查记录、维护手册和汇报材料。

### 8.3 共享证据层

`experiments/`、`automation/results/`、MPP manifests 和经验证结果包由两类视图引用，但不复制事实。

### 8.4 未来 asset registry 字段

- `asset_id`
- `path`
- `era`
- `custody`
- `audience`
- `asset_class`
- `lifecycle`
- `mutability`
- `evidence_role`
- `tracking`
- `hash_policy`
- `plan_id` / `experiment_id`
- `replacement`
- `retention_policy`

## 9. Protected dirs 解禁前置条件

必须依次满足：

1. 明确 pre-MPP 的血缘范围，而不是只按日期或目录名判断。
2. 生成 tracked 与 ignored 资产 preservation inventory。
3. 建立 asset registry 与 schema，并先以 shadow 模式运行。
4. 抽取仍被 MPP 使用的 `histogene.utils.compute_metrics`。
5. 改写现役 import 并通过针对性测试。
6. 对旧实验和文档执行 append-only lifecycle 转换。
7. 采用 copy → verify → cutover，不做未经验证的原地 move。
8. asset-policy Hook 先 WARN，再 enforce。
9. 最后才修改 `AGENTS.md` 的固定目录保护。

物理删除始终是独立、显式批准、可恢复的 cleanup 任务。

## 10. 实施阶段

| 阶段 | 内容 | 授权状态 |
|---|---|---|
| S0 | 总方案、指令、审计 Skill、legacy 原始实现隔离、安全提示修复 | 已完成 |
| S0b | 16 个安全 router、17 个 `.claude` 同名薄适配器 | 已完成并通过本轮验收 |
| S0c | active Skill / review 文档最小生命周期登记与门禁 | 已完成并通过 2 项 TDD 测试 |
| S1 | canonical `pfmval-governance` Skill、模板、六个用户入口/七条技术路由与初始 Workflow Catalog | G08 本地 D/I 已实现；六项 active、`standardized=false`；fixture O=PASS，真实 O=NOT RUN；S=N/A |
| S1a | P0A 实验全生命周期闭环、P0B 用户导航与实验进度双视图 | 本地 D/I 已接受；P0A O=NOT RUN、P0B O=PASS；S=N/A；真实实验闭环未运行 |
| S2 | workspace registry shadow、W### 对话绑定与外部 run root | 本地 D/I 已接受；O=NOT RUN；S=N/A；真实 workspace/attempt/run root 未运行 |
| S3 | job/result schema v2、次数预算、适配记录与兼容读取 | 本地 D/I 已接受；O=NOT RUN；S=N/A；真实 v2 dispatch/attempt 未运行 |
| S4 | critical contract 与分级结果证据 | 本地 D/I 已接受；O=NOT RUN；S=N/A；真实 contract/result import 未运行 |
| S5 | asset registry、shadow inventory | 本地 D/I 已接受；O=PASS（metadata-only shadow）；S=N/A；资产迁移未运行 |
| S6 | 共享依赖抽取、目录解禁与 lifecycle 迁移 | 本地 D/I 已接受；O=PASS / PASS_SHADOW_ONLY；S=N/A；目录解禁与 lifecycle 迁移未运行 |
| S7 | 经独立批准的 worktree/资产 cleanup | 未授权 |

重构流程、统一框架、阶段依赖和模块索引见：

`project_state/plans/workflow_governance_v3_refactor_review_20260726.md`

每个大环节的影响面、步骤、风险、检验、退出条件和回退分别维护在：

`project_state/plans/workflow_governance_v3_refactor/`

后续需求只补充对应模块或追加新的 R 编号，并回链总控审查包；不得把具体
实施步骤重新堆回总控文件。

## 11. 总体验收

- Skill canonical source 可被格式校验。
- `.claude` 同名 Skill 可跨 agent 发现，但只直接引用 `.agents` 对应 Skill。
- 旧高风险 Skill 原始实现不再给出可执行旧服务器/训练入口。
- 审计 Skill 不把 Dashboard、聊天声明或未 import 结果当作 accepted evidence。
- 状态包与文档 Registry 一致。
- P0A 在最终验收时证明实验注册到结果登记/关闭预览存在无需 Agent 临时
  设计命令的 CLI 闭环。
- P0B 在最终验收时提供 tracked 仓库导航和由 Experiment Registry 生成的
  简洁实验进度表，current 与 historical 明确分区。
- 结构性重构未在未授权情况下发生。
- 无训练、无服务器访问、无 protected 资产改写、无删除。

## 12. 本方案的两张关键学习卡

### K01：实验、attempt 与 worktree 是不同身份

- `experiment_id` 表示科研问题和协议。
- `attempt_id` 表示一次执行或重试。
- worktree 只是执行工作区，不应随 attempt 无限增加。
- 只有关键协议改变，才应创建新的 `protocol_revision` 或实验。

### K02：关键合同哈希不等于全仓库重复哈希

- Git commit 证明 tracked 源码快照。
- 关键合同证明影响科研结论的语义设置。
- manifest/checkpoint 哈希证明实际输入。
- result envelope 证明实际输出。
- 文档、图表和无关文件不应阻断调试，但必须保持可追溯。
