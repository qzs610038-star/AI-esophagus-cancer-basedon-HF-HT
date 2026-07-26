# PFMval 外部开源科研工作流调研与适配建议

> document role: `external-reference-research`
> research date: `2026-07-26`
> authority: `DIR-20260726-005`
> review status: `research_complete_design_pending`
> execution authority: `none`
> source policy: 只采用官方文档、官方规范或上游仓库作为主要依据。
> prohibition: 本文不授权安装依赖、迁移事实源、服务器操作、训练、结果接纳、
> 云端同步、资产迁移、目录解禁或清理。

## 1. 结论

不建议把 PFMval 整体迁移到 Renku、eLabFTW、MLflow 或 Calkit 等单一平台。
PFMval 已有实验批准、W### 工作树、Gitee 往返、结果接纳、受保护资产和
Document Registry 等项目特有边界；整体替换会形成第二套事实源，或丢失已有
安全约束。

建议采用“**本地唯一事实源 + 可组合的成熟标准与工具**”：

1. **直接适配**：Quarto 负责报告派生渲染；Diátaxis 负责学习指南分类；
   RO-Crate / Workflow Run RO-Crate 负责归档或发布边界的元数据封装；
   Snakemake Workflow Catalog 与 WorkflowHub 的目录规则负责工作流发现治理。
2. **局部借鉴或隔离试点**：借鉴 Calkit 的研究问题—计算—产物对象模型、
   eLabFTW 的记录修订与软归档模型、DVC 的依赖—产物哈希、W3C PROV 的
   `Entity–Activity–Agent` 来源关系；Snakemake 或 noWorkflow 只在独立试点
   中验证，不得直接成为权威状态。
3. **当前不采用**：不整体部署 Renku/eLabFTW，不让 MLflow、DVC Experiments
   或 Calkit 接管 Experiment Registry、Git/Gitee 或云端传输；不同时引入
   Quarto 与 MyST/Jupyter Book 两套文档构建栈。

这不是“工具越少越好”，而是让每个外部组件只承担它已经成熟、且不会制造
双重事实源的职责。

## 2. 方案筛选矩阵

| 方案 | 可复用能力 | PFMval 判断 | 当前动作 |
|---|---|---|---|
| [Quarto](https://quarto.org/docs/projects/quarto-projects.html) | 项目级模板、Markdown/HTML/DOCX/PDF、图表与表格交叉引用 | 与组会短报、周度探索文档高度匹配；只能是渲染层 | **优先做一个冻结输入的报告原型** |
| [RO-Crate 1.3](https://www.researchobject.org/ro-crate/specification.html) / [Workflow Run RO-Crate](https://www.researchobject.org/workflow-run-crate/profiles/) | 文件、人员、软件、输入、输出和运行来源的机器可读封装 | 可补足图片、表格、报告、源码与 accepted result 的跨对象关系 | **直接复用最小 Profile，但只在导出边界生成** |
| [Diátaxis](https://diataxis.fr/) | Tutorial、How-to、Reference、Explanation 四类文档 | 可解决现有学习指南类型混杂、导航不稳定 | **直接采用为 `guide_kind` 分类** |
| [Snakemake Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/docs/catalog.html) | 通用目录与标准化工作流分层、结构门槛、测试与版本 | 很适合 R13 的 candidate 与 standardized 分层 | **直接借鉴目录规则；执行引擎另行试点** |
| [WorkflowHub](https://about.workflowhub.eu/) | 工作流元数据、版本、快照、作者、许可证、集合 | 适合公开或稳定工作流目录，不适合挖掘原始命令历史 | **借鉴 manifest；暂不自建服务** |
| [Calkit](https://docs.calkit.org/) | questions、datasets、figures、pipeline、publications 的研究对象模型 | 最接近完整科研闭环，但其治理文档明确仍处早期阶段 | **借对象模型，不安装、不接管 Git** |
| [Snakemake](https://snakemake.github.io/) | 确定性 DAG、环境、缓存、报告、集群执行 | 可在 P0A 稳定后承载确定性数据处理；不能覆盖科研判断 | **后续影子试点候选** |
| [DVC](https://dvc.org/doc/start/data-pipelines/data-pipelines) | Git 旁路的大文件哈希、DAG、依赖与输出 | 会与现有 Registry、Gitee 边界和受保护资产规则重叠 | **只借鉴 lock/hash 设计，暂不引入 remote/exp refs** |
| [DVCLive](https://dvc.org/doc/dvclive) | 把参数、逐步指标、图片和图表写成普通文件 | 文件日志适合 result envelope 前的运行目录，但 hidden commits/Studio 会越界 | **可窄适配文件格式；附带写 Git/联网能力必须关闭** |
| [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) | 参数、指标、产物、父子 run 与 UI | 若双向写入会成为第二套实验事实源 | **仅考虑从 accepted Registry 单向投影的只读 UI** |
| [eLabFTW](https://doc.elabftw.net/docs/usage/intro/) | 模板、团队、修订、锁定、附件、审计、归档与 API | 数据模型值得借鉴；部署会增加数据库、账号、存储与备份运维 | **不部署，只借记录生命周期** |
| [noWorkflow](https://gems-uff.github.io/noworkflow/) | 无需改脚本的 Python 运行、文件、环境和值依赖溯源 | 能回答“实际运行了什么”，不能回答“为何得出结论”；可能采集敏感内容 | **仅限显式开启的本地探索试点** |
| [Renku](https://github.com/SwissDataScienceCenter/renku) | Git、Notebook、容器、工作流和知识图谱平台化整合 | 是平台级替换，和本地/Gitee/Windows 优先边界不匹配 | **只作架构参考** |
| [OSF Registrations](https://help.osf.io/article/330-welcome-to-registrations) | 时间戳快照、归档、协作与公开发布 | 适合预注册或最终归档，不适合运行期事实维护 | **出现公开发表需求后再接入** |
| [CWL](https://www.commonwl.org/specification/) | 跨引擎命令行工具与计算 DAG 标准 | 只适合已经稳定、容器化、需要跨平台复用的流程 | **本轮不改写现有工作流** |

## 3. 对现有设计的补强

### 3.1 P0C 候选：科研推理闭环

现有 P0A 继续负责正式实验执行和证据门禁；新增轻量推理层只维护：

```text
research_question
→ hypothesis
→ discriminating_test
→ observation
→ interpretation
→ claim / rejected_option / uncertainty
→ successor_hypothesis / more_validation / stop
```

其中 `discriminating_test` 是“能区分候选解释的验证方式”。想法和本地探索
只要求最小记录；晋级正式实验时才绑定 experiment id、批准、次数预算和关键
合同。Calkit 的 questions/outputs 对象模型与
[The Turing Way 研究纲要](https://book.the-turing-way.org/reproducible-research/compendia/)
可作为字段和验收检查表，但不得成为新事实源。

必须额外记录阴性结果、无显著差异、失败尝试和不确定性。否则周报会只沉淀
“成功路线”，下一轮智能体会重复已否决方案。

### 3.2 R10 候选：非实验项目事实

建议使用 append-only（只追加）事实事件，当前视图由事件派生。最小字段：

- `fact_id`：事实的永久编号；
- `fact_type`：目标、分工、任务、队友成果、沟通决议等类别；
- `subject`、`value`、`source`、`observed_at`；
- `effective_from`、`effective_to`：事实适用时间；
- `supersedes`：本事件替代的旧事实；
- `confirmation_status`：待用户确认或已确认；
- `privacy_class`：是否允许进入 Git、飞书或公开归档。

用户口述必须先生成变更预览，进行冲突与隐私检查，再由用户确认追加；不得
直接覆盖历史值。eLabFTW 的修订、锁定、归档与软删除模型值得借鉴，但无需
部署其服务。

### 3.3 R11 候选：长期报告与学习指南

报告的最小不可变产物包为：

```text
report.md
figures/*.png
tables/*.csv
manifest.json
```

`manifest.json` 至少关联 `doc_id`、`experiment_refs`、`result_refs`、
`source_commit`、`state_revision`、`asset_refs + sha256`、`supersedes` 和
`generated_at`。报告只读取已接纳结果快照，默认不得因渲染而重新运行实验。
Quarto 的 [GFM 输出](https://quarto.org/docs/output-formats/gfm.html)、
[交叉引用](https://quarto.org/docs/authoring/cross-references)和
[项目执行控制](https://quarto.org/docs/projects/code-execution.html)可直接复用。

飞书 CLI 应是单向发布器：

```text
本地权威事实
→ 不可变报告包
→ 飞书
→ 只回写 remote_doc_id、远端版本、内容哈希、上传时间
```

远端编辑不得自动反写 Registry；需要回传时只能进入 inbox，再人工接纳。

学习指南本轮只增加中立元数据接口，至少包含 `guide_kind`、目标读者、先决知识、
源码符号/文件引用、实验/结果引用、最后验证提交和替代关系。按 Diátaxis 分为：

- tutorial：带领首次完成一条学习路径；
- how-to：解决一个具体任务；
- reference：准确查询接口、字段或命令；
- explanation：解释机制、设计理由与取舍。

MyST/Jupyter Book 适合未来多章节知识库，但当前应避免和 Quarto 同时成为构建
权威；仅保留可导出 Markdown、交叉引用和稳定 ID 的接口。

### 3.4 R12 候选：文档与约束生命周期

必须分开两条轴：

- `lifecycle`：draft、active、superseded、historical、discarded；
- `freshness`：fresh、review_due、stale、unknown。

文件没有变化不代表仍适用；依赖的实验、结果、源码符号、指令或约束变化时，
只生成 `review_due` 候选，不自动改写或删除。图片、表格、附件、报告渲染物和
学习指南也要进入同一依赖图，避免正文更新而图表仍来自旧结果。

关闭实验时只触发影响分析：哪些文档需要复核、哪些约束可弃置、哪些报告仍是
历史证据。真实生命周期转换继续需要显式确认。

### 3.5 R13 候选：工作流发现与演化

参考 Snakemake Catalog 将目录分为：

1. `candidate`：用户主动扫描后发现的重复操作模式；
2. `review`：已完成脱敏、用途和风险审查；
3. `active`：已有稳定输入/输出、实现位置和验证；
4. `standardized`：有 schema、示例、自动测试、版本和替代关系；
5. `deprecated`：保留历史入口和迁移说明。

权威 manifest 至少包含 `workflow_id`、用途、触发方式、输入/输出、实现路径、
风险级别、审批要求、版本、生命周期、替代关系和最近验证时间。Task/just 等
命令入口只能由 manifest 派生，不能再手工维护第二份目录。

发现过程只存脱敏后的命令模板、使用次数和成功率；不得采集原始参数、stdout、
环境变量、完整绝对路径、数据标识或凭证。候选不得自动激活，高风险工作流
不得因高频使用而放宽批准。

## 4. 当前方案仍缺少的方面

1. **主张—证据—不确定性关系**：不能只登记“实验完成”；要知道哪个结果
   支持或反驳哪个 claim，以及结论强度和替代解释。
2. **阴性结果与停止规则**：记录否决路线、无效超参数区间、继续验证门槛和
   何时停止，避免重复探索与无上限追加实验。
3. **人工步骤与运行环境**：数据筛选、图表挑选、临时转换等手工步骤也需最小
   来源记录；环境锁定应覆盖解释器、关键依赖、CUDA/硬件和随机性。
4. **隐私、伦理与备份分级**：在进入 Git、飞书、OSF 或公开包前，按数据、
   路径、队友信息和未发表结论确定可发布范围。
5. **贡献、许可与署名**：项目分工、队友成果与复用工作流应带作者、贡献角色、
   来源许可证和引用信息，不能只保留任务分配。
6. **独立复算与恢复演练**：定期从干净环境复算一个代表性结果，并验证本地
   报告包、飞书备份和未来归档能否恢复；“上传成功”不等于可恢复。
7. **schema 演进与退出计划**：每个外部工具必须有版本锁定、迁移器、回退路径
   和可导出中立格式，避免以后被某个工具的数据模型锁住。
8. **存储层级与保留预算**：区分源码/元数据、可再生产物、大型不可替代数据、
   临时缓存和公开归档；为保留周期、校验频率与清理批准设规则。
9. **事实冲突的裁决顺序**：用户确认指令、机器 Registry、accepted result、
   派生 Markdown 与云端副本必须有明确优先级，不能按“最新文件”覆盖。
10. **可访问性和解释责任**：图表需来源、单位、统计口径和替代文本；自动摘要
    要区分观察、解释和建议，不能把模型推断写成项目事实。

[The Turing Way 项目设计清单](https://book.the-turing-way.org/project-design/pd-overview/pd-checklist)
可作为上述范围的人工验收清单，不作为强制运行时依赖。

## 5. 推荐适配顺序

### 阶段 0：本轮设计层

- 不安装外部依赖；
- 固化 P0C、R10–R13 的职责边界和中立 schema；
- 明确本地 Registry 是唯一事实源，所有外部工具均为渲染、导出或受控执行层；
- 把阴性结果、隐私分级、贡献信息、附件生命周期和恢复验证纳入验收。

### 阶段 1：低风险直接适配

1. 用 Quarto 对一份冻结输入的组会报告做原型，不触发实验代码；
2. 为学习指南添加 Diátaxis 类型与源码/实验引用接口；
3. 对一个已关闭且已接纳结果的固定 fixture 生成最小 RO-Crate；
4. 建立 workflow manifest，并借鉴 Snakemake Catalog 的 candidate /
   standardized 两级准入。

### 阶段 2：隔离影子试点

- P0A 稳定后，只选 Snakemake 或 DVC 中一个做单流程试点；当前更推荐先评估
  Snakemake，因为它可以只承担确定性计算 DAG，而不需要新增数据远端。
- 若训练过程缺少结构化曲线日志，可先验证 DVCLive 的纯文件输出，或实现兼容
  其 JSON/TSV/图片布局的轻量 logger；不得生成 hidden commit 或连接 Studio。
- noWorkflow 只在 `scripts/explorations/` 对非敏感脚本显式启用，先评估存储、
  性能和泄露风险。
- 如确需可视化查询，先从 accepted Experiment Registry 单向导出到临时
  MLflow UI；严禁从 MLflow 反写。

### 阶段 3：出现真实需求后再接入

- 团队需要 ELN 权限、签名或审计时再评估 eLabFTW；
- 论文预注册、永久时间戳或公开材料包出现时再接入 OSF；
- 稳定、容器化、跨机构复用的计算流程出现时再考虑 CWL/WorkflowHub 发布。

## 6. 明确不采用或不自动化的事项

- 不执行 Calkit 的自动 Git/DVC 保存或云端推送；
- 不启用 DVC remote、DVC experiment refs 作为项目实验身份；
- 不允许 DVCLive 自动提交隐藏 Git ref、向 Studio 发送数据或绕开 result import；
- 不让 MLflow、Quarto、飞书、OSF 或 eLabFTW 成为项目事实源；
- 不让报告渲染自动重跑训练或生成新的可比较实验结论；
- 不从原始命令历史自动生成 active workflow；
- 不自动把云端编辑、noWorkflow 记录或外部工具数据库写回 Registry；
- 不同时维护 Quarto 和 MyST/Jupyter Book 两条生产文档流水线；
- 不因外部方案“功能完整”而绕开现有审批、Gitee、受保护资产或结果接纳边界。

## 7. 官方原始资料

### 科研项目与可复现工作流

- Calkit：[文档](https://docs.calkit.org/)、
  [`calkit.yaml` 对象模型](https://docs.calkit.org/calkit-yaml/)、
  [Pipeline](https://docs.calkit.org/pipeline/)、
  [治理与成熟度说明](https://docs.calkit.org/governance/)、
  [上游仓库](https://github.com/calkit/calkit)
- Snakemake：[文档](https://snakemake.github.io/)、
  [Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/docs/catalog.html)、
  [添加工作流](https://snakemake.github.io/snakemake-workflow-catalog/docs/about/adding_workflows.html)、
  [上游仓库](https://github.com/snakemake/snakemake)
- DVC：[数据 Pipeline](https://dvc.org/doc/start/data-pipelines/data-pipelines)、
  [Experiments](https://dvc.org/doc/start/experiments)、
  [上游仓库](https://github.com/iterative/dvc)
- DVCLive：[文档](https://dvc.org/doc/dvclive)、
  [上游仓库](https://github.com/iterative/dvclive)
- MLflow：[Tracking](https://mlflow.org/docs/latest/tracking/)、
  [Tracking Server 架构](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)、
  [上游仓库](https://github.com/mlflow/mlflow)
- noWorkflow：[文档](https://gems-uff.github.io/noworkflow/)、
  [上游仓库](https://github.com/gems-uff/noworkflow)
- CWL：[规范](https://www.commonwl.org/specification/)、
  [上游组织](https://github.com/common-workflow-language)

### 文档、来源链与归档

- Quarto：[项目](https://quarto.org/docs/projects/quarto-projects.html)、
  [格式](https://quarto.org/docs/reference/formats/)、
  [GFM](https://quarto.org/docs/output-formats/gfm.html)、
  [交叉引用](https://quarto.org/docs/authoring/cross-references)、
  [执行与 freeze](https://quarto.org/docs/projects/code-execution.html)、
  [上游仓库](https://github.com/quarto-dev/quarto-cli)
- MyST/Jupyter Book：[Jupyter Book](https://jupyterbook.org/latest/)、
  [MyST 文档](https://mystmd.org/guide/)、
  [上游仓库](https://github.com/jupyter-book/mystmd)
- RO-Crate：[规范](https://www.researchobject.org/ro-crate/specification.html)、
  [Profiles](https://www.researchobject.org/ro-crate/specification/1.3/profiles.html)、
  [Workflow Run RO-Crate](https://www.researchobject.org/workflow-run-crate/)、
  [上游仓库](https://github.com/ResearchObject/ro-crate)
- W3C PROV：[PROV-O 规范](https://www.w3.org/TR/prov-o/)
- OSF：[Projects](https://help.osf.io/article/353-welcome-to-projects)、
  [Registrations](https://help.osf.io/article/330-welcome-to-registrations)、
  [API](https://developer.osf.io/)、
  [上游仓库](https://github.com/CenterForOpenScience/osf.io)

### 治理、知识组织与团队记录

- Diátaxis：[方法首页](https://diataxis.fr/)、
  [入门说明](https://diataxis.fr/start-here/)
- The Turing Way：[Research Compendia](https://book.the-turing-way.org/reproducible-research/compendia/)、
  [项目设计清单](https://book.the-turing-way.org/project-design/pd-overview/pd-checklist)、
  [高级项目仓库](https://book.the-turing-way.org/project-design/pd-overview/project-repo/project-repo-advanced/)
- WorkflowHub：[项目说明](https://about.workflowhub.eu/)、
  [Workflow RO-Crate](https://about.workflowhub.eu/Workflow-RO-Crate/)、
  [文档](https://about.workflowhub.eu/docs/)
- eLabFTW：[文档](https://doc.elabftw.net/)、
  [实验条目](https://doc.elabftw.net/docs/usage/user-guide/experiments/)、
  [追溯与审计](https://doc.elabftw.net/docs/edge/usage/traceability-and-auditability/)、
  [API](https://doc.elabftw.net/docs/usage/api/)、
  [上游仓库](https://github.com/elabftw/elabftw)
- Renku：[文档](https://docs.renkulab.io/)、
  [上游仓库](https://github.com/SwissDataScienceCenter/renku)

## 8. 调研边界

本轮只读检索了截至 2026-07-26 可访问的官方资料，没有安装、运行或基准测试
任何外部工具。上表中的“直接适配”表示设计接口适配，不等于依赖已经批准或
兼容性已经实测。外部项目活跃度、许可证、Windows 行为和 API 仍需在各自
原型开始前重新核验并固定版本。
