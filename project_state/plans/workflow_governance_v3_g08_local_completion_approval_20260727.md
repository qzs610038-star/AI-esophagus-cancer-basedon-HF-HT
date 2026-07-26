# 治理 v3：G08 本地剩余重构一次性批准包与 Goal

> 状态：`approved_execution_passed`
> 基线：`main@06dac933c8aaafc5cfba565df14d2a86f974d0e5`
> 批准指令：`DIR-20260727-004`
> 执行结果：本地 D/I 与 synthetic fixture O 已通过；真实工作流 O 保持
> `NOT RUN`，S=`N/A`；等待限定本地 commit 与 append-only closeout。
> 用途：供用户一次性批准后，在用户离线期间完成所有安全、确定性的本地非服务器重构。
> 权限边界：本文件本身不构成执行授权；只有用户明确回复批准文本后才能创建并执行 Goal。

## 一、当前结论

- G00–G07 已完成；revision 005 的 R0–R11 本地设计与实现证据已经接受。
- R10 六个 document profile 与 R11 catalog/CLI 已有本地 D/I 实现，但
  Workflow Catalog 当前仍为空，尚未形成用户可直接调用的工作流入口。
- 用户已经明确选择六个用户级工作流；它们不再作为“自动发现后的未知候选”
  等待下一次人工取舍，可在 G08 中直接完成初步实现和本地准入。
- 尚待收口的本地结构项是 S1 canonical `pfmval-governance` Skill、六个用户
  入口及其 Workflow Catalog 初始登记。
- R10 的真实项目事实、组会内容和长期探索文档依赖用户提供真实内容，不能由 Agent 猜测。
- P0A 的真实实验闭环依赖具体 `W###`、实验合同及服务器运行，本 Goal 不执行。
- R8/R9/S7 的物理迁移、解禁和清理必须绑定精确目标，不能使用一揽子批准。

### 已决定、G08 无需再次询问的设计事项

1. 六个用户级工作流均采用本地实现；外部开源方案只借鉴架构、数据模型和
   管理思路，不安装、不接入、不作为事实源。
2. 长期文档对用户保持一个入口，内部按 `meeting_brief` 与 `research_path`
   两个 profile 路由。
3. 组会文档当前只生成简述；500 字、图片、数据表和格式模板均继续暂缓。
4. 学习指南只实现源码/实验关联、版本与生命周期接口，正文体系重构和云备份
   继续暂缓。
5. 论文工作流只实现本项目材料索引和学长模板接入槽位，正确终态为
   `template_pending`；不生成论文正文。
6. 项目事实采用 preview、冲突检查、逐条显式确认、append-only 更正；本次
   对工作流的批准不等于确认任何未来事实。
7. 文档与约束生命周期只生成 `review_due`/影响预览，不自动 supersede、
   historical、关闭 directive 或删除文件。
8. 已知六个 workflow 可在 G08 内准入；未来自动发现的新 workflow 仍只到
   candidate，继续需要人工 review/activation。
9. 初步实现纳入主张—证据—不确定性、阴性结果与停止规则、事实冲突裁决、
   可访问性与解释责任；外部调研缺失项 3–8 继续暂缓。
10. 六个入口本轮均为 `active` 但 `standardized=false`，真实 O 运行保持
    `NOT RUN`，维护类工作流的科学证据等级为 `S=N/A`。

## 二、一次性批准范围

用户批准后，G08 可在无需再次询问的情况下完成以下工作：

1. 追加一条本地 G08 directive，记录本批准包、范围和排除项。
2. 实现 `.agents/skills/pfmval-governance/` canonical Skill：
   - 只负责事实源读取、任务分类、工作流选择、用户决策收敛和结果解释；
   - 为用户已经确定的六个工作流提供稳定入口和到现有 CLI/profile 的路由；
   - 所有状态变更只调用 `deploy/pfmval_ops.py`；
   - 不复制 CLI 业务逻辑，不提供直接服务器、训练、结果接纳或删除入口。
3. 新增 `.claude/skills/pfmval-governance/SKILL.md` 薄适配器，只直接引用 canonical Skill。
4. 新增最小本地模板：
   - 方案/假设探索 intake；
   - project fact intake；
   - meeting brief 与 research path 输入；
   - learning guide 维护接口；
   - document lifecycle review 输入；
   - workflow discovery scope manifest；
   - paper output 材料索引与模板待接入接口；
   - experiment intake 仅保留接口，不生成真实 job。
5. 修订 active `workflow_governance_v3` 方案：
   - 修复已经过时的“结构代码仍待实施”表述；
   - 将 S1 更新为本 Goal 的真实结果；
   - 纳入 P0C/R10/R11 当前本地 D/I 状态；
   - 不改变既有 `O=NOT RUN`、`S=N/A` 或高风险门禁。
6. 新增一个最小、离线、manifest 驱动的
   `workflow register --manifest ... --authorization-ref ...` 入口：
   - 只登记 manifest 中显式列出的 candidate，不扫描其它路径；
   - manifest 必须是 tracked repo-relative 文件并绑定 G08 directive；
   - 相同 ID、相同内容重放为 no-op；相同 ID、不同内容在写入前原子失败；
   - register 不得直接 active、standardized 或执行 workflow。
7. 将用户已经明确提出的六个用户级 workflow 写入 approved manifest，在同一
   Goal 内通过 `register → review → activate` 完成初步准入，不再逐项询问：

   | Catalog ID | 用户级工作流 | 内部实现映射 | 初步实现边界 |
   |---|---|---|---|
   | `WF-PROJECT-FACT` | 非实验项目事实维护 | `project_fact` | intake、preview、冲突检查；每条真实事实仍需单独确认 |
   | `WF-LONGTERM-DOCS` | 长期文档沉淀 | `meeting_brief` + `research_path` | fixture 生成与来源分层；不伪造真实内容 |
   | `WF-LEARNING-GUIDE` | 学习指南编写与管理 | `learning_guide` | `interface_reserved`，不重构正文、不接云端 |
   | `WF-DOCUMENT-LIFECYCLE` | 文档与约束生命周期 | `lifecycle_review` | freshness/impact preview；不自动改 lifecycle |
   | `WF-WORKFLOW-DISCOVERY` | workflow 发现与演化 | R11 catalog/CLI | 用户主动扫描；未来新发现只到 candidate |
   | `WF-PAPER-OUTPUT` | 论文撰写与产出 | `paper_output` | 材料索引并停在 `template_pending` |

   六个条目最终均为 `active`、`standardized=false`、版本 `0.1.0`。这里的
   `active` 只表示本地入口可以使用，不表示发生过真实 O 运行，也不表示
   学习指南正文或论文稿件已经完成。长期文档一个用户入口内部路由到两个
   profile，因此验收时是六个用户入口、七条技术路由。
8. 对六个工作流执行无需用户真实内容的初步实现：
   - 使用 frozen/synthetic fixture 验证全部七条技术路由；
   - 对指定 active 治理文档运行 freshness/影响分析；
   - 生成 `review_due/current` 预览和 intake 模板；
   - 不自动修改文档 lifecycle，不调用真实 `fact-confirm`。
9. R11 首次 discovery：
   - 只扫描本批准包预先限定的 tracked CLI、Skill 与治理文档；
   - 六个已知 workflow 的准入依据是本批准包，不依赖 discovery；
   - 此次扫描发现的其它 workflow 最多记录为 `candidate`；
   - 不对其它 workflow 执行 review、activate、revise、merge 或 deprecate。
10. 更新必要测试、Document/Skill Registry、Workflow Catalog、current state
   与生成视图。
11. 运行本地完整测试、严格门禁、双轮幂等和独立审核。
12. 若全部验收通过，允许：
    - append-only 将 G08 directive 标记为 `completed`；
    - 创建必要的限定本地 commit；
    - 不再等待第二次用户验收。

### 唯一允许修改的范围

- `.agents/skills/pfmval-governance/**`
- `.claude/skills/pfmval-governance/SKILL.md`
- 本批准包及 active `workflow_governance_v3` 方案
- G08 所需的 directive、current state、Document Registry、Workflow Catalog、
  approved workflow manifest、scope manifest 与 governance receipt
- `scripts/pfmval_workflows.py` 与 `deploy/pfmval_ops.py`，仅用于实现上述
  `workflow register` 入口
- `project_state/schemas/workflow_catalog.schema.json`，仅在登记来源、
  能力边界或用户入口映射无法由现有 schema 准确表达时作向后兼容扩展
- 与 G08 直接相关的测试及 fixture
- 只能由既有命令再生的 `CURRENT_STATE.md`、`PROJECT_GUIDE.md`、`README.md`
  和实验派生视图

除上面精确列出的两个实现文件和一个可选 schema 外，若实现必须修改其它
`deploy/**`、`scripts/**`、schema、实验/工作树/资产注册表、审批或结果资产，
则立即按阻塞退出，不得自行扩权。

## 三、R11 精确扫描与准入范围

### 已决定的六个 workflow

- 允许通过 G08 approved manifest 和新增 `workflow register` 把第二部分列出
  的六个固定 Catalog ID 登记为 candidate，再使用现有 `workflow review`、
  `workflow activate` 在 G08 内准入为 active。
- 当前用户指令与 G08 directive 共同作为 review/activation 的批准证据。
- 六个条目不得标记 `standardized=true`；不得据此自动执行真实写入。
- 除这六个精确 ID 外，不得把任何 candidate 自动提升为 reviewed 或 active。

### 未来 workflow discovery

只允许读取以下 tracked 路径：

- `deploy/pfmval_ops.py`
- `scripts/pfmval_*.py`
- `.agents/skills/*/SKILL.md`
- `.claude/skills/*/SKILL.md`
- `project_state/plans/workflow_governance_v3_20260726.md`
- `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
- `project_state/plans/workflow_governance_v3_refactor/*.md`
- `automation/README.md`
- `PROJECT_GUIDE.md`
- `README.md`

禁止读取或扫描：

- 会话历史、终端历史、stdout/stderr；
- 环境变量、credentials、token、SSH 配置；
- ignored 目录、训练输出、缓存、checkpoint、原始数据；
- 服务器、网络、外部平台或云端内容。

## 四、明确排除项

本批准包不授权：

- 服务器测试、服务器访问、Gitee dispatch/fetch/return 或远端 SHA 验证；
- 训练、真实 attempt、结果 import/accept/reject 或科学证据晋级；
- 新建、收养、迁移或删除任何 worktree；
- 资产迁移、目录解禁、Hook enforce、`close --execute`；
- 删除工作树、branch、cache、checkpoint、数据或资产；
- 自动确认项目事实、编写真实组会/长期探索内容；
- 学习指南正文重构、飞书/云备份接入、论文正式写作；
- 除六个固定 bootstrap ID 外的 workflow review/activation，以及任何
  standardized、revise、merge 或 deprecate；
- Git push 或远端变更；
- 修改 `histogene/`、`egnv1/`、`egnv2/`。

## 五、G08 Goal

### Goal 名称

`G08 — Workflow Governance v3 本地非服务器剩余结构收口`

### 完成目标

在不需要服务器和新的用户事实输入的前提下，完成 S1 canonical
`pfmval-governance` Skill、最小模板、active plan 语义收敛，并把用户已经
确定的六个工作流完成初步实现、Catalog 准入和七条技术路由 fixture 验证；
同时保留 R11 对未来新 workflow 的限定范围 candidate discovery，使治理 v3
的本地结构重构达到 `local_complete`，并输出一份最终本地审核包。

### 阶段

| stage_id | 内容 | 退出条件 |
|---|---|---|
| G08-0 | 基线、directive、允许范围冻结 | clean `main@06dac933...` 或只含本批准包 |
| G08-1 | S1 canonical Skill、适配器与模板 | 格式、引用、负向安全测试通过 |
| G08-2 | active plan 与 Registry/状态收敛 | 事实源与生成视图一致 |
| G08-3 | approved manifest、register 及六个 workflow 准入 | register 幂等/冲突原子失败；6 个 active、均非 standardized |
| G08-4 | R10 fixture/freshness 与 R11 future discovery | 零真实事实/文档写入；新发现仅 candidate |
| G08-5 | 全量回归、幂等、独立审核与提交 | 所有必需项 PASS，排除项均 NOT RUN |

## 六、验收测试

1. canonical Skill 存在、可发现、格式合法。
2. `.claude` 适配器只引用 canonical Skill，不复制业务规则。
3. Skill 与模板中不存在直接服务器、训练、结果接纳、删除、外部平台入口。
4. S1 可把六个用户入口确定性路由到七条现有技术 profile/CLI，不产生第二
   Registry 或独立业务逻辑。
5. active plan 不再把已接受的 R0–R11 本地结构写成“代码待实施”。
6. `workflow register` 只接受 tracked approved manifest 和 authorization
   reference；重放为 no-op，同 ID 异内容不产生半写，且 register 自身只生成
   candidate。
7. Workflow Catalog 恰好新增上述六个固定 ID；最终全部为 `active`、
   `standardized=false`、`version=0.1.0`，且批准依据可追溯到 G08。
8. 七条技术路由 fixture 全部通过：
   - `project_fact` 停在 confirmation preview；
   - `meeting_brief` 与 `research_path` 保留 accepted/explore/failed、
     阴性结果、停止规则和不确定性；
   - `learning_guide` 停在 `interface_reserved`；
   - `lifecycle_review` 只产生 `review_due` preview；
   - R11 future discovery 只产生 candidate；
   - `paper_output` 停在 `template_pending`。
9. R10 真实写入计数为零：无新增 confirmed project fact，无正式组会/研究
   路径文档，无 lifecycle 自动转换。
10. R11 future discovery 仅读取批准清单；六个固定 ID 之外的 active 增量为零。
   没有真实重复模式时，以“零候选”通过，不得为凑数生成 candidate。
11. Experiment、Workspace、Asset Registry 的真实条目不增加。
12. `histogene/`、`egnv1/`、`egnv2/` 差异为空。
13. docs scan、views refresh、state sync 连续两轮字节稳定。
14. 完整 `tests/` 测试集零失败；严格门禁 `FAIL=0` 且不新增 WARN。
15. `git diff --check` 通过，提交后工作区干净，未 push。

## 七、阻塞退出机制

- 基线存在未知 dirty 文件：写入前停止，保留用户内容并报告。
- S1 必须复制 CLI 业务逻辑、扩展独立状态写入器或放宽门禁才能实现：停止。
- register 无法做到 tracked manifest、授权绑定、幂等和冲突前原子失败：停止，
  不得手改 Catalog 或伪造重复 workflow marker。
- 六个 workflow 缺真实事实或真实文档内容：以 fixture/preview/interface 完成
  初步实现，真实 O 标记 `NOT RUN`，不得伪造输入。
- R11 发现六个固定 ID 之外的候选：保留 candidate，不自动 review/activate。
- 测试失败、状态不幂等或出现新 WARN：Goal 保持 active，不标记完成。
- 任一步需要服务器、网络、训练、迁移、解禁、删除或 push：标记 `NOT RUN`，
  继续完成仍可安全完成的本地阶段；若关键目标因此无法完成则报告 blocked。
- 同一阻塞连续三个 Goal 回合仍无法解决时，标记 blocked，并交付最小失败证据。

## 八、用户一次性批准文本

用户如同意，可只回复：

> 批准修订后的 G08 本地非服务器剩余结构收口整包。允许按本文第二、三部分
> 执行；允许新增 canonical pfmval-governance Skill、薄适配器、模板和测试，
> 并在限定的 `scripts/pfmval_workflows.py`、`deploy/pfmval_ops.py` 及必要的
> Workflow Catalog schema 中实现 manifest 驱动、幂等的 workflow register；
> 将 `WF-PROJECT-FACT`、`WF-LONGTERM-DOCS`、`WF-LEARNING-GUIDE`、
> `WF-DOCUMENT-LIFECYCLE`、`WF-WORKFLOW-DISCOVERY`、`WF-PAPER-OUTPUT`
> 六个用户已确定 workflow 在本 Goal 内完成初步实现并从 candidate 准入到
> active，但均不得 standardized；允许执行七条技术路由 fixture、R10
> freshness 预览、未来 workflow 的限定 candidate discovery、状态同步和
> 本地提交。若全部验收通过，允许自动 append-only 标记 G08 completed，无需
> 二次确认。明确不批准服务器测试/访问、训练、结果接纳、真实事实确认、
> worktree/资产迁移、解禁、Hook enforce、close execute、任何删除、其它
> workflow activation/standardization、外部平台、云同步或 push。
