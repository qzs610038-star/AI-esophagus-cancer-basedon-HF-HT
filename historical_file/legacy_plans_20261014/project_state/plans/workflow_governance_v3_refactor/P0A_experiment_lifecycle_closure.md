# P0A：实验全生命周期可执行闭环

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `P0A`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: cross-cutting over `R0`–`R5b` and `R9`
> authorization: `DIR-20260726-002`, `DIR-20260726-003`, `DIR-20260726-004`
> implementation: `local_D/I_accepted; O=NOT RUN; S=N/A`
> boundary: 本文是横向验收合同，不授权服务器操作、训练、结果接纳、工作树删除或受保护资产修改。

## 1. 目标

把“实验注册 → 代码编写 → 校验 → Gitee 同步 → 排障 → 正式训练 →
结果回传 → 本地登记 → 关闭”固化为唯一可执行工作流。正常路径由稳定
CLI 和 schema 驱动，Agent 只负责读取事实、询问必要决策、调用入口并解释
结果，不再为每次实验重新设计整套终端命令、分支或工作树。

## 2. 当前基线

已有 v1 能力：

- strict start-check、job manifest 校验和 dry-run；
- job dispatch/run/pack/import；
- formal approval 绑定 `job_id + source_commit`；
- allowlisted diagnostic request/record；
- result import 联动 Experiment Registry、Dashboard 和 Current State。

尚缺：

- `experiment register` 与永久 `W###` 分配；
- 对话绑定、workspace registry、外部 run root 和 lease；
- 协议级次数预算、`run_units` 和 E1 adaptation record；
- CLI 封装的 Gitee fetch/push/远端 SHA 核验、quarantine 与冲突恢复；
- `close --preview` / `close --execute`。

不得把现有手册中的自由组合命令当成 P0A 已完成。

## 3. 权威身份与状态机

身份层次：

```text
experiment_id
└─ workspace_id = W###
   ├─ protocol_revision
   ├─ attempt_id / job_id
   ├─ adaptation_record_id（可选）
   └─ result_id
```

目标状态机：

```text
registered
→ workspace_open
→ code_ready
→ validated
→ dispatched
→ server_checked
→ diagnostic / ready
→ attempt_running
→ result_returned
→ import_pending
→ imported
→ accepted / rejected
→ close_ready
→ removed
→ archived
```

任何异常必须进入显式状态，例如 `dirty_blocked`、`sync_conflict`、
`fingerprint_mismatch`、`orphaned_lease` 或 `return_conflict`；不得靠重复
pull、覆盖 ref 或重新训练消除。

## 4. 每阶段固定合同

| 阶段 | 必需输入 | 固定输出 | 唯一写者 | 必要门禁 |
|---|---|---|---|---|
| 注册 | 方案、phase、用户给定执行次数 | experiment、W###、protocol revision | 本地 CLI | ID 唯一、次数显式 |
| 代码 | 已绑定 W### | clean commit、critical contract | 本地工作树 | locus、测试、关键合同 |
| 派发 | exact commit、approval、attempt | 不可变 job envelope/ref | 本地 CLI | 远端 SHA、预算余额 |
| 服务器检查 | job、路径 Registry、worktree | preflight 与 lease 候选 | 服务器 CLI | detached exact commit、clean |
| 排障 | diagnostic request | diagnostic record 或 E1 patch | 服务器 return ref | allowlist、非证据性、合同不变 |
| 运行 | validated job、lease | 外部 run root、事件与日志 | 服务器 runner | `EXPERIMENT_STARTED`、run units |
| 回传 | 终态 attempt | immutable result envelope/ref | 服务器 CLI | artifact 分级与哈希 |
| 导入 | quarantine bundle | Registry 事务、两类生成视图 | 本地 CLI | job/result/contract/bundle 绑定 |
| 关闭 | 全部 attempt 终态 | preview、保留清单、精确目标 | 本地 CLI | 无 lease/dirty/unreturned/unimported |

## 5. 目标 CLI 面

名称可在实现 PR 中微调，但职责不得重新分散：

```text
pfmval_ops experiment register/show
pfmval_ops workspace allocate/bind/status/scan
pfmval_ops attempt prepare/dispatch/status
pfmval_ops diagnostic request/record/import
pfmval_ops result fetch/verify/import
pfmval_ops workspace close --preview/--execute
```

要求：

- Registry、manifest、状态和生成视图只经 CLI 写入。
- CLI 只接受结构化参数，不接受任意 shell。
- 正常路径不得要求 Agent 生成未登记的工作树、临时分支或整套 Git 命令。
- 同一请求重试必须幂等；冲突不得通过覆盖或重新训练解决。
- 服务器仍只通过 Gitee 交换代码、状态和小型结果。

## 6. 分步实施映射

| 能力 | 主要实施模块 |
|---|---|
| 基线与 v1 fixture | [R0](R0_baseline_and_compatibility.md) |
| W###、绑定、workspace shadow | [R1](R1_workspace_registry_shadow.md) |
| 单持久工作树、run root、lease | [R2](R2_persistent_experiment_workspace.md) |
| 注册、协议批准、次数预算、E1 | [R3](R3_protocol_approval_and_run_budget.md) |
| critical contract 与门禁 | [R4](R4_critical_contract_and_gate_tiering.md) |
| result pack/import 与 artifact | [R5](R5_result_artifact_tiering.md) |
| Registry/生成视图事务一致性 | [R5b](R5b_document_registry_and_state_gates.md) |
| 关闭与清理 | [R9](R9_closeout_and_cleanup.md) |

## 7. 主要风险

- CLI 只是打印命令而没有真正封装状态转换。
- experiment、workspace、attempt、job 身份混用。
- 双端同时写同一分支或相同 result ID。
- E1 工程适配被用于掩盖关键合同变化。
- Gitee 冲突导致重复消耗实验次数或重复训练。
- result import 与生成视图非原子，出现 Registry/用户视图分叉。

## 8. 最小验证矩阵

必须新增端到端测试，覆盖：

1. 注册一个实验，询问并记录两次结果性执行额度，分配唯一 W 编号。
2. 新对话未绑定时写操作失败；绑定后 cwd/branch/HEAD/dirty 校验通过。
3. 同一实验两个 attempt 复用同一 worktree，输出位于不同外部目录。
4. preflight 失败不消耗次数；跨过 `EXPERIMENT_STARTED` 后失败消耗真实
   `run_units`。
5. E1 适配合同不变时生成新 commit/revision；改变 loss 或数据时批准失效。
6. Gitee 非 fast-forward、tracked dirty、stale lease 和重复 result 的恢复。
7. 相同 result bundle 重复 import 为 no-op；同 ID 不同 SHA 为 HARD FAIL。
8. import 原子更新 Registry、Agent Dashboard 和用户进度表。
9. close preview 无副作用；运行、未回传或未导入状态均阻断 execute。

## 9. P0A 验收条件

- 上述全链路存在一条文档化且被测试覆盖的 CLI happy path。
- Agent 不需要在正常路径自行设计 Git/服务器命令。
- 所有状态转换可从 Registry、manifest、event 或 result envelope 审计。
- v1 job/result 仍可只读验证和 import，不自动覆盖或删除。
- 故障恢复不重训、不覆盖 ref、不绕过次数预算。
- 对应实现需另行分阶段批准；本文通过审查不等于实现完成。

## 10. 待用户/实现 PR 决策

- [ ] `EXPERIMENT_STARTED` 的精确机器事件。
- [ ] E1 文件/参数 allowlist。
- [ ] CLI 子命令最终命名。
- [ ] server writer ref 是 attempt 唯一 ref，还是稳定 W ref 下的 append-only
  attempt 目录；两者均须保持单写者和不可覆盖。

## 11. 补充记录

- `2026-07-26 / DIR-20260726-004`：首次建立 P0A 横向硬验收模块。
