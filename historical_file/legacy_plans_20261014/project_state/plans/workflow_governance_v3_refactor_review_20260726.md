# PFMval 治理 v3 结构性重构审查包

> document role: `refactor-framework-and-index`
> review package revision: `005`
> lifecycle: `approved_design`
> state scope: `workflow_governance_v3_refactor_review`
> authorization: `DIR-20260726-002`, `DIR-20260726-003`, `DIR-20260726-004`, `DIR-20260727-001`
> implementation status: `local_D/I_accepted; O=SEE_MODULE_MATRIX; S=N/A`
> audit baseline: code `main@9d8cc7a4`
> prohibition: 本审查包及其模块不是服务器执行、训练、结果接纳、资产迁移、目录解禁或清理批准。

## 0. 本文件的职责

本文件是结构性重构的唯一总控审查包，只维护：

- 重构目标、架构、固定边界和统一流程；
- P0 横向验收合同；
- 各大环节的依赖、顺序、状态和入口；
- 统一阶段模板、变更协议和总体验收；
- 跨模块已决事项与仍需用户判断的事项。

具体 schema 字段、影响面、实施步骤、风险、测试、退出条件和回退必须维护在
对应独立模块中，不再堆回本文件。

后续用户需求可以：

1. 补充现有模块并提升其 `module revision`；
2. 新增横向验收模块；
3. 新增 R10、R11 等大环节。

已有 P0/R 编号永久保留，不因插入新需求而重排。每次补充都必须更新本文件
的模块索引与依赖关系，并通过 Directive、Document Registry 和状态校验。

## 1. 审查目标

本轮重构需要最终解决：

1. 实验注册、代码、校验、Gitee 同步、排障、训练、回传、登记和关闭使用
   一套可执行闭环，Agent 不再每次自行设计。
2. 一个 experiment 在服务器只保留一个永久 W 工作树；attempt 不新建
   worktree，输出位于工作树外。
3. 正式批准前默认询问实际结果性执行次数；E1 简单工程适配可在关键合同
   不变、留痕和测试条件下免重复批准。
4. 只对关键实验合同、实际输入和结论承载结果严格校验。
5. 建立 Workspace/Asset Registry，逐步完成 pre-MPP 退役、共享依赖抽取和
   分级解禁。
6. Agent 规则、用户审阅学习材料和共享证据层清楚分区，但只使用一套事实源。
7. 用户拥有 tracked 仓库导航和与 Experiment Registry 绑定的简洁实验进度表。

## 2. 固定架构

### 2.1 三层职责

| 层 | 负责 | 不得负责 |
|---|---|---|
| Skill | 识别任务、读取事实、询问用户、路由、解释 | 直接批准、写 Registry、删除资产 |
| CLI | schema、状态机、Registry/manifest 写入、可审计副作用 | 任意 shell、隐式科研决策 |
| Hook | 毫秒级确定性提醒和路径/命令防护 | 接纳结果、写状态、迁移 lifecycle、清理 |

### 2.2 事实与视图

```text
Machine-readable sources
├─ Current State / Directives
├─ Experiment / Document / Workspace / Asset Registry
├─ job / approval / result / manifest
└─ Git + registered external asset evidence
        │
        ├─ Agent full operational views
        └─ User navigation / concise progress / reports
```

派生视图不得反向覆盖事实源。用户视图与 Agent 视图可以使用不同展示粒度，
但必须指向同一 ID、状态和结果。

### 2.3 身份边界

| 身份 | 含义 |
|---|---|
| `experiment_id` | 一个科研问题/大实验的稳定身份 |
| `workspace_id` | `W###`，永久不复用 |
| `protocol_revision` | 关键实验合同 revision |
| `attempt_id` / `job_id` | 协议内的一次具体执行 |
| `run_units` | 一次 job 内真实独立拟合数量 |
| `result_id` | 一次 attempt 的回传与导入身份 |

这些身份不得互相替代。

## 3. P0 横向硬验收

P0 是贯穿多个 R 阶段的验收合同，不是第三套实现入口。

| ID | 审查模块 | 必须解决 | 功能验收 |
|---|---|---|---|
| P0A | [实验全生命周期可执行闭环](workflow_governance_v3_refactor/P0A_experiment_lifecycle_closure.md) | Agent 无需临时拼装实验流程 | 注册到 close preview 的 CLI happy path、异常恢复与幂等测试 |
| P0B | [用户仓库导航与实验进度双视图](workflow_governance_v3_refactor/P0B_user_views_and_navigation.md) | 用户能快速定位文件、看懂实验进展 | tracked 导航、Registry 生成的简洁进度、current/history 分区与一致性测试 |
| P0C | [科研推理与扩展维护工作流横向验收](workflow_governance_v3_refactor/P0C_extended_scientific_workflows_and_acceptance.md) | 探索保持轻量，结论/阴性结果/冲突可追溯 | 证据分层、claim/negative/stop、冲突裁决、离线与模块隔离测试 |

门禁规则：

- 在任何 R 阶段从 `pending_review` 转入实际实现前，P0A/P0B/P0C 中与该阶段有关
  的接口、输出和测试必须已审查。
- 单个 R 阶段通过，不代表 P0A/P0B/P0C 完成。
- 整套结构性重构不得在 P0A/P0B/P0C 相关功能验收未通过时宣称完成。

## 4. 分阶段模块与依赖

```mermaid
flowchart TD
    P0A["P0A 实验全生命周期闭环"]
    P0B["P0B 用户导航与实验进度"]
    P0C["P0C 科研推理与扩展维护验收"]
    R0["R0 冻结基线"]
    R1["R1 Workspace Registry shadow"]
    R2["R2 单持久工作树 / 外部 run root"]
    R3["R3 协议批准 / 次数预算"]
    R4["R4 关键合同 / 门禁分级"]
    R5["R5 Result artifact 分级"]
    R5B["R5b Document Registry / 状态门禁"]
    R6["R6 Asset Registry shadow"]
    R7["R7 共享依赖抽取"]
    R8["R8 Lifecycle / Hook / 解禁"]
    R9["R9 Closeout / Cleanup"]
    R10["R10 项目知识 / 文档生命周期"]
    R11["R11 Workflow 目录 / 演化"]

    R0 --> R1
    R1 --> R2
    R1 --> R3
    R0 --> R4
    R4 --> R3
    R3 --> R5
    R4 --> R5
    R0 --> R5B
    R5B --> R6
    R6 --> R7
    R7 --> R8
    R2 --> R9
    R5 --> R9
    R6 --> R9
    R8 --> R9
    R5B --> R10
    R5B --> R11

    P0A -.约束.-> R1
    P0A -.约束.-> R2
    P0A -.约束.-> R3
    P0A -.约束.-> R4
    P0A -.约束.-> R5
    P0A -.约束.-> R9
    P0B -.约束.-> R5B
    P0B -.约束.-> R6
    P0B -.约束.-> R8
    P0B -.约束.-> R9
    P0C -.约束.-> R10
    P0C -.约束.-> R11
```

### 4.1 模块索引

| 模块 | 独立步骤文件 | 当前状态 | 核心输出 |
|---|---|---|---|
| R0 | [冻结基线与兼容样本](workflow_governance_v3_refactor/R0_baseline_and_compatibility.md) | pending_review | v1 fixture、inventory、已知 WARN |
| R1 | [Workspace Registry 只读 Shadow](workflow_governance_v3_refactor/R1_workspace_registry_shadow.md) | pending_review | W###、scan/status、locus guard |
| R2 | [实验级单持久工作树](workflow_governance_v3_refactor/R2_persistent_experiment_workspace.md) | pending_review | 外部 run root、单 worktree、lease |
| R3 | [协议批准与次数预算](workflow_governance_v3_refactor/R3_protocol_approval_and_run_budget.md) | pending_review | register、approval v2、run units、E1 |
| R4 | [关键合同与门禁分级](workflow_governance_v3_refactor/R4_critical_contract_and_gate_tiering.md) | pending_review | critical digest、E0–E3 |
| R5 | [结果 Artifact 分级](workflow_governance_v3_refactor/R5_result_artifact_tiering.md) | pending_review | envelope v2、quarantine、事务 import |
| R5b | [Document Registry 与状态门禁](workflow_governance_v3_refactor/R5b_document_registry_and_state_gates.md) | pending_review；已观测 verified_at churn | 父子审查关系、稳定生成、多写者收敛 |
| R6 | [Asset Registry 只读 Shadow](workflow_governance_v3_refactor/R6_asset_registry_shadow.md) | pending_review | asset schema、血缘候选 |
| R7 | [共享依赖抽取](workflow_governance_v3_refactor/R7_shared_dependency_extraction.md) | pending_review | 中性核心、兼容 shim |
| R8 | [生命周期迁移与分级解禁](workflow_governance_v3_refactor/R8_lifecycle_migration_and_unprotection.md) | pending_review | asset policy、shadow Hook、用户导航刷新 |
| R9 | [正式关闭与 Cleanup](workflow_governance_v3_refactor/R9_closeout_and_cleanup.md) | pending_review | close preview、精确批准、tombstone |
| R10 | [项目知识与文档生命周期](workflow_governance_v3_refactor/R10_project_knowledge_and_document_lifecycle.md) | pending_review | 项目事实、简述、指南/论文接口、生命周期 |
| R11 | [工作流目录、发现与演化](workflow_governance_v3_refactor/R11_workflow_catalog_and_evolution.md) | pending_review | 本地 manifest、候选准入、版本/合并/弃用 |

## 5. 高层影响面

| 区域 | 目标变化 | 主要模块 | 最高风险 |
|---|---|---|---|
| `deploy/pfmval_ops.py` | 唯一实验生命周期 CLI | P0A、R1–R5、R9 | 多写者、任意命令、状态竞态 |
| job/result schema | v1 双读、新写 v2 | R3–R5 | 兼容破坏、批准放宽 |
| 工作树/launcher | per-job → per-experiment W | R1–R2 | dirty 污染、并发、路径漂移 |
| Gitee 往返 | exact ref/SHA、单写者、quarantine | P0A、R2、R5 | 覆盖 ref、重复训练 |
| 状态/文档生成 | 原子 Registry + 双视图 | P0B、R5、R5b | 用户视图与事实分叉 |
| 资产与路径 | shadow inventory → policy | R6–R8 | 误分类、误解禁 |
| protected dirs | 依赖抽取后分级维护 | R7–R8 | 数值漂移、提前解禁 |
| closeout | preview 与 execute 分离 | R9 | 误删工作树或外部资产 |
| 项目知识/文档 | 共用本地来源、版本和 lifecycle | P0C、R10 | 解释冒充事实、模块膨胀 |
| 工作流目录 | 用户触发的本地候选与准入 | P0C、R11 | 自动启用、采集越界 |

详细符号、字段、步骤和测试只在对应模块维护。

## 6. 按风险分轨的重构流程

### 6.1 高风险执行轨

适用于 P0A、R1–R9 以及 R11 中会生成可执行入口的部分：

```text
pending_review
→ approved_design
→ implementation_authorized
→ tests_red
→ implementation
→ tests_green
→ shadow_or_compatibility
→ independent_audit
→ accepted / rejected / superseded
```

### 6.2 文档/元数据轻量轨

适用于 P0C、R10 以及 R11 的只读目录部分：

```text
pending_review
→ approved_design
→ implementation_authorized
→ implemented
→ verified
→ accepted / rejected / superseded
```

轻量轨仍必须有负向、幂等、冲突和回退验证；只有涉及 schema 迁移、writer、
可执行行为或兼容风险时，才增加 tests-red/green 和 shadow 阶段。不得为了
流程形式要求每次简述、事实口述或接口预留都走完整九阶段门禁。

共同说明：

- `approved_design` 只批准设计，不等于代码实施授权。
- 测试只证明被覆盖的实现行为，不等于服务器运行、实验成功或 scientific
  evidence accepted。
- 涉及服务器、训练、结果接纳、平台接入、资产迁移、目录解禁或删除时，仍需
  精确独立用户批准。

### 6.3 模块最小模板

每个模块至少维护：

1. metadata、依赖、目的与明确边界；
2. 输入、输出和事实源优先级；
3. 本地接口/状态及外部方案只借鉴点；
4. PASS/FAIL/NOT RUN/N/A 验收、证据等级和退出条件；
5. 排除项、风险、回退、用户待决项和追加记录。

高风险模块再补充分步实施、兼容迁移、tests-red/green、shadow 和独立审计。

## 7. 推荐审批与实施顺序

1. 审查 P0A、P0B、P0C 的接口和验收，不实施服务器行为。
2. 实施并审查 R0。
3. 实施 R1 shadow；不得创建/删除 worktree。
4. 先完成 R2 的外部 run root，再启用单持久工作树。
5. R4 extractor 可与 R3 schema 设计并行，但 R3 enforcement 必须等待 R4。
6. 实施 R5 与 R5b，先保证事务和生成视图一致。
7. 实施 R6 shadow；至少观察一轮。
8. 实施 R7，确认现役依赖迁出。
9. 经用户逐项批准后实施 R8。
10. 最后审查 R9；preview 和 execute 分别批准。
11. 原有闭环稳定后实施 R10 的本地知识/文档轻量轨。
12. 最后实施 R11；发现只读部分先验收，任何可执行入口继续走高风险轨。

任何阶段失败或用户未审查，停止在该阶段，不为“完成整套重构”跨越门禁。

## 8. 需求补充协议

后续新增需求按以下方式落盘：

1. 判断是跨阶段验收、现有阶段补充还是新大环节。
2. 用户改变方案、优先级或安全边界时先追加 directive。
3. 跨阶段要求新增/修订 P 模块；现有环节只修订对应 R 文件；新环节追加
   R10+。
4. 在模块“补充记录”追加日期、directive、原因、影响步骤和新验证。
5. 更新本文件的模块索引、依赖图和状态。
6. 将新模块登记为 `pending_plan_reviews`，运行 `docs scan --write` 和
   `state sync`。
7. pending review 不得自动升级为 active/normative。
8. 不通过聊天摘要、README 或 Dashboard 代替上述记录。

## 9. 全局不可变量

- Gitee 是本地与服务器唯一 active 同步通道。
- 禁止 SSH、SCP、HTTP 远程命令和 Tunnel。
- 一个 experiment 对应一个永久 W；W 编号不复用。
- 本地是 canonical code writer；服务器只返回 adaptation/result。
- 不自动 pull、merge、rebase、force push 或全局 worktree prune。
- attempt 输出位于注册的工作树外。
- 正式训练必须有显式批准和剩余次数预算。
- 关键合同变化必须重新批准。
- result 返回不等于 import，import 不等于不同协议的接纳。
- historical 不等于 deleted。
- Hook 不写 Registry、不批准、不接纳、不删除。
- 本审查包不授权真实服务器、训练、结果、迁移、解禁或 cleanup。

## 10. Agent、用户与共享资产框架

### Agent / 项目板块

- `AGENTS.md`
- `.agents/skills/`
- `project_state/`
- `configs/`
- `automation/`
- `deploy/`
- `scripts/`
- `tests/`

保存固定规则、schema、CLI、Registry、manifest、测试和生成逻辑。

### 用户板块

- tracked 仓库导航；
- `01_指南与解读/` 中 active 方案、学习卡、分析与维护手册；
- `02_组会汇报/`。

### 共享证据层

- `experiments/`
- `automation/results/`
- MPP manifests；
- 经验证结果包与受控服务器资产记录。

用户和 Agent 视图只引用共享事实，不复制新的实验事实。

## 11. 已决事项

1. `.agents/skills/` 是 canonical Skill 源。
2. `.claude/skills/` 保留同名 tracked 薄 adapter。
3. W 编号永久不复用；新对话显式绑定。
4. 一个 experiment 一个服务器持久工作树。
5. 默认询问实际结果性执行次数。
6. 严格限定 E1 适配可免重复批准。
7. 关键合同、实际输入和结论结果保持严格指纹。
8. 总控包只维护框架；每个大环节维护独立文件。
9. P0A/P0B/P0C 是整套重构硬验收。

## 12. 暂缓与待用户审查

暂缓：

- workspace/job/result/asset schema 的真实代码改造；
- 现有 worktree 自动编号、收养、迁移或删除；
- asset registry 实际扫描；
- pre-MPP lifecycle 迁移；
- protected dirs 解禁；
- 真实服务器实验、结果接纳和 cleanup。
- 任何外部科研/文档/云端平台、服务、CLI、SDK 或数据库接入；
- 组会字数、图片、表格、模板和渲染格式规范；
- 学习指南内容体系重构与云端备份；
- 论文模板、章节、图表、引用格式、署名、投稿和发布；
- 外部调研缺失项 3–8 的新增字段、门禁与验收。

跨模块待审查：

- [ ] `EXPERIMENT_STARTED` 机器事件和 E1 allowlist。
- [ ] 用户根导航与简洁实验进度文件的最终名称/路径。
- [ ] parent review `module_paths` schema，或继续使用独立 pending review ID。
- [ ] 2026-07-06 pre-repair MPP1–5 legacy accepted 结果的长期分类。
- [ ] ignored checkpoint 的 hash 策略。
- [ ] `histogene/egnv1/egnv2` 依赖抽取后的维护权限。
- [ ] 任何物理归档/删除仍须单独批准。
- [ ] R10 六类 document profile 是否共用一个知识/文档生命周期引擎。
- [ ] R11 首版是否只扫描用户选择的 tracked CLI/Skill/文档清单。

## 13. 总体验收

只有全部满足才能把整套重构提交为用户验收候选：

- P0A 的本地 schema/CLI/fixture 端到端 D/I 验收通过，Agent 正常路径无需临时
  设计命令；真实服务器/训练 O 在无单独授权时保持 NOT RUN。
- P0B tracked 导航与 Registry 生成进度表通过一致性测试。
- P0C 的证据分层、阴性结果、冲突裁决和解释责任测试通过。
- R0–R11 已授权的 D/I 退出条件满足；未授权 O 明确为 NOT RUN，不适用 S 为
  N/A。
- v1 job/result 可只读验证和 import。
- Workspace/Asset/Document/Experiment Registry 不复制职责。
- R10 复用现有 Document Registry，R11 candidate 不会自动成为 active。
- local/server 单写者和 Gitee-only 边界保持。
- 外部平台/工具的安装、调用、联网和运行时依赖均为零。
- 无未授权服务器操作、训练、结果接纳、资产迁移、解禁或删除。
- 每项验收标明 D/I/O/S 证据等级及 PASS/FAIL/NOT RUN/N/A。
- 独立 `pfmval-audit` 给出相应阶段的有界结论。
- 只有全部 Goal 完成后才生成并交付一份最终重构审核包。

## 14. 变更记录

| 日期 | Revision | Directive | 变化 |
|---|---:|---|---|
| 2026-07-26 | 001 | DIR-20260726-002/003 | 初版影响面、R0–R9 步骤与编号工作树协议 |
| 2026-07-26 | 002 | DIR-20260726-004 | 改为总控框架；新增 P0A/P0B；R0–R9 拆为独立可扩展模块 |
| 2026-07-26 | 003 | DIR-20260726-005 | 补充科研灵活性审查、五类后续工作流和“优先适配成熟开源方案”策略 |
| 2026-07-26 | 004 | DIR-20260726-005 | 完成外部开源方案只读调研并关联适配建议与官方资料 |
| 2026-07-27 | 005 | DIR-20260727-001 | 外部方案改为只借鉴；组会格式暂缓；增加论文接口；仅采纳缺失项 1/2/9/10；新增 P0C/R10/R11 与 Goal 验收 |

详细服务器状态机、单写者、冲突恢复和 W 编号规则继续以
`project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`
作为已批准设计基线；实现仍须遵守各模块门禁。

## 15. 科研灵活性审查补充

### 15.1 有界结论

结论为 **CONDITIONAL GO**：现有审查包可以继续供用户审查，但不宜原样进入
代码重构。正式训练、服务器往返、结果接纳和受保护资产必须保持严格治理；
假设形成、本地复算、低成本探索、结果解释和日常项目事实不应套用同等重量
的批准、Registry 和关闭流程，否则会增加科研试错摩擦。

P0A 已覆盖实验执行与证据链的主要骨架，但尚未完整覆盖：

```text
问题/假设
→ 低成本探索或复算
→ 是否晋级为候选实验
→ 正式实验
→ 结果解释
→ 后继假设 / 追加验证 / 停止
```

因此后续设计必须采用“渐进式形式化”：想法与解释只做轻记录；本地非证据
探索留最小可复现记录；候选实验明确晋级理由；只有可产生项目结论的正式实验
进入完整批准、执行、导入和关闭闭环。

### 15.2 扩展工作流与最小模块

扩展工作流包括：

1. 科研问题、假设、探索、正式实验、解释、主张和停止的渐进式闭环；
2. 用户口述的非实验项目事实维护；
3. 组会简述与长期结论/探索路径文档；
4. 学习指南维护接口；
5. 文档与约束 lifecycle/freshness 复核；
6. 用户触发的工作流发现与演化；
7. 论文撰写与产出接口，等待学长提供模板。

为避免过度模块化，落点收敛为：

- [P0C](workflow_governance_v3_refactor/P0C_extended_scientific_workflows_and_acceptance.md)：
  科研推理与扩展工作流横向验收；
- [R10](workflow_governance_v3_refactor/R10_project_knowledge_and_document_lifecycle.md)：
  六类本地知识/文档 profile 共用一个生命周期引擎；
- [R11](workflow_governance_v3_refactor/R11_workflow_catalog_and_evolution.md)：
  因可能影响可执行行为而独立维护的工作流目录与演化。

组会当前只生成简短 Markdown；不再保留 500 字、2–3 张图、固定数据表等硬
要求。`interface_reserved`/`template_pending` 只是学习指南/论文接口状态机的
合法设计终态，不代表整个工作流实现或运行完成。

### 15.3 外部方案只借鉴原则

- 调研结论与官方原始链接见
  [外部开源科研工作流设计借鉴记录](workflow_governance_v3_refactor/REFERENCE_open_source_research_workflow_adaptation_20260726.md)。
- Calkit、eLabFTW、DVC、Snakemake、WorkflowHub、PROV/RO-Crate、Diátaxis、
  The Turing Way 等只提供架构、数据关系、状态机、检查表和测试思想。
- 不安装、不部署、不调用、不试点、不联网，不新增外部 SDK/CLI/数据库，
  不形成第二事实源或未来自动接入代码。
- 后续实施只能“提取设计模式 → 改写为本地最小合同 → 断网 fixture 验证”，
  外部调用次数必须为零。

### 15.4 本轮采纳与暂缓

仅采纳：

1. 主张—证据—不确定性；
2. 阴性结果与停止规则；
9. 事实冲突裁决顺序；
10. 可访问性与解释责任。

外部调研缺失项 3–8 继续保留在参考文档中，本轮不形成字段、门禁、实现或验收
义务。

### 15.5 新对话 Goal 交付方式

用户批准本 revision 后，实际重构必须在新对话创建一个总 Goal，按 P0C 的
G00–G06 `stage_id` 顺序逐模块推进。任何模块未通过退出条件时 Goal 不得标记
完成；未授权服务器、训练、迁移、解禁或清理必须保持 O=NOT RUN。中间回执
只供最终汇总。全部已授权实现、回归与独立审计结束后，只向用户提交一份最终
重构结果审核包。

本 revision 仍只授权设计文档和状态更新，不批准代码重构、外部接入、服务器、
训练、结果接纳、云同步、资产迁移、目录解禁或清理。
