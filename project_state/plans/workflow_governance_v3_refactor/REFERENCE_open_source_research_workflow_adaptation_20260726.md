# PFMval 外部开源科研工作流设计借鉴记录

> document role: `external-reference-research`
> research date: `2026-07-26`
> user selection revision: `2026-07-27`
> authority: `DIR-20260727-001`（supersedes `DIR-20260726-005`）
> review status: `user_selection_recorded`
> execution authority: `none`
> source policy: 只采用官方文档、官方规范或上游仓库作为主要依据。
> prohibition: 所有外部方案只作为架构、数据模型、状态机、维护模式和测试思想
> 的参考；不安装、不部署、不调用、不试点、不联网、不迁移事实源，也不形成
> 运行时或云端依赖。

## 1. 用户最终选择

PFMval 不直接接入 Calkit、DVC、DVCLive、MLflow、Snakemake、WorkflowHub、
eLabFTW、Renku、OSF、Quarto、MyST/Jupyter Book、RO-Crate 服务/库或其它
平台。后续只把经过用户审核的设计思想翻译为 PFMval 本地 schema、状态机、
检查表和 fixture 测试。

当前采用：

- 本地 Registry、directive 和 accepted result 继续是唯一事实源；
- 组会文档当前只写简述，不设格式、字数、图片、表格或模板门槛；
- 学习指南只保留本地中立维护接口；
- 论文撰写与产出只保留“本项目材料 → 学长后续模板”的接入接口；
- 飞书、公开归档、ELN、云端备份和协作平台均不接入。

## 2. 可借鉴但不得接入的设计模式

| 来源 | 只借鉴什么 | PFMval 本地落点 | 明确不做 |
|---|---|---|---|
| [Calkit](https://docs.calkit.org/) | research question、主要产物与派生产物分离 | P0C 推理链、R10 派生文档 | 不安装，不运行 `calkit`，不接管 Git/DVC/cloud |
| [eLabFTW](https://doc.elabftw.net/docs/usage/intro/) | 追加修订、锁定、替代、软归档 | R10 事实与文档生命周期 | 不部署 ELN/数据库/账号系统 |
| [DVC](https://dvc.org/doc/start/data-pipelines/data-pipelines) | 依赖—输出、内容指纹、stale 判断 | R5/R10 的本地哈希和影响分析思想 | 不使用 remote、experiment refs 或 DVC cache |
| [Snakemake](https://snakemake.github.io/) | DAG、dry-run、受影响节点 | 原有 CLI/测试的依赖图思想 | 不引入执行引擎、Snakefile 或 `.snakemake` 状态 |
| [Snakemake Catalog](https://snakemake.github.io/snakemake-workflow-catalog/docs/catalog.html) | 候选目录、标准化准入、版本 | R11 本地 workflow manifest | 不连接或发布到目录服务 |
| [WorkflowHub](https://about.workflowhub.eu/) | 版本、快照、作者/替代关系 | R11 工作流演化元数据 | 不自建 SEEK/WorkflowHub |
| [W3C PROV](https://www.w3.org/TR/prov-o/) / [RO-Crate](https://www.researchobject.org/ro-crate/specification.html) | Entity–Activity–Agent、used/generated/derivedFrom | 本地 source refs 与 claim/evidence 关系 | 不生成标准包，不安装库，不上传 |
| [Diátaxis](https://diataxis.fr/) | tutorial/how-to/reference/explanation 的用途区分 | R10 学习指南可选语义 | 不强制迁移目录或重写现有指南 |
| [The Turing Way](https://book.the-turing-way.org/project-design/pd-overview/pd-checklist) | 人工项目检查表 | 最终审核包参考项 | 不作为运行时门禁或依赖 |
| [MLflow](https://mlflow.org/docs/latest/tracking/) | 参数/指标/产物分层与比较视图 | 仅供本地 Registry 视图设计参考 | 不部署 server/SQLite，不导入导出 MLflow |
| [noWorkflow](https://gems-uff.github.io/noworkflow/) | 执行事实与科研解释分离 | P0C 区分 run observation 与 interpretation | 不运行采集器，不建立第二数据库 |
| [Renku](https://github.com/SwissDataScienceCenter/renku) | Plan/Run 和代码—数据—结果关系 | P0C/P0A 身份关系参考 | 不迁移平台、Git 或项目结构 |
| [OSF](https://help.osf.io/article/330-welcome-to-registrations) | 工作状态与不可变归档状态分离 | lifecycle 语义参考 | 不归档、不上传、不调用 API |
| [Quarto](https://quarto.org/docs/projects/quarto-projects.html) / [MyST](https://mystmd.org/guide/) | 源文档与派生渲染物分离 | R10 文档边界参考 | 不建立构建链，不做组会格式原型 |

## 3. 当前采纳的四个缺失方面

### 3.1 主张—证据—不确定性

正式主张至少需要：

- `claim_id` 和明确陈述；
- evidence ref 及支持/反驳/未决方向；
- 观察与解释分离；
- uncertainty 与替代解释；
- 证据等级，避免 explore 或测试结果冒充 accepted science。

### 3.2 阴性结果与停止规则

保留失败路线、无显著差异、已否决假设、继续验证条件、停止条件和后继假设。
后续摘要、学习指南或论文材料不得只抽取“成功路线”。

### 3.3 事实冲突裁决顺序

先按事实域决定权威来源：

- 当前显式用户指令：意图、授权和路线；
- current state/Registry：身份、生命周期和当前状态；
- accepted result：其协议范围内的科学测量；
- active plan：未来拟执行步骤，不得改写 accepted measurement；
- generated view/摘要/交接/云端副本：只读派生。

用户可以改变下一步，但新指令不会改写历史测量值。同域冲突不能按文件修改
时间覆盖；必须使用 supersedes/accepted 状态，仍无法裁决时保留冲突并请求
用户确认。

### 3.4 可访问性与解释责任

- 内容标记 observation、interpretation、recommendation；
- 图表存在时提供来源、单位、统计口径和文字说明；
- 自动摘要不得把模型推断写成项目事实；
- 找不到证据时写 GAP/待验证，不补造结论。

## 4. 暂缓的六个缺失方面

下列内容只保留为未来候选，本轮不进入字段、实现、门禁或验收：

3. 人工步骤与运行环境；
4. 隐私、伦理与备份分级；
5. 贡献、许可与署名；
6. 独立复算与恢复演练；
7. schema 演进与外部工具退出计划；
8. 存储层级与保留预算。

已有项目固定安全边界继续有效，但不得把这些候选扩写为本轮新增治理要求。

## 5. 对扩展工作流的设计贡献

### P0C 科研推理

借鉴 question/plan/run、source/provenance 与 primary/derived 分离：

```text
question → hypothesis → explore/candidate → formal experiment
→ observation → interpretation → claim/negative/uncertainty
→ successor / more validation / stop
```

### R10 项目知识与文档生命周期

借鉴追加修订、source refs、stale 候选和派生文档思想，以一个本地模块承载：

- 非实验项目事实；
- 组会简述；
- 长期结论和探索路径；
- 学习指南接口；
- 论文模板待接入接口；
- 文档与约束生命周期。

组会只生成简短 Markdown；论文模板缺失时停在 `template_pending`。

### R11 工作流目录与演化

借鉴 catalog manifest：

```text
candidate → reviewed → active → revise / merge / deprecated
```

`standardized` 是 active 的属性；用户不主动调用时零扫描、零写入。

## 6. 本地化实施方法

后续只允许以下三步：

1. 从外部官方资料提取一个设计模式，并记录来源；
2. 把模式改写为 PFMval 本地最小字段、状态或检查表；
3. 用仓库内 fixture 在断网环境验证，确认外部调用次数为零。

禁止“先安装试用再决定”、平台 PoC、外部 SDK stub、临时 SQLite 服务、云端
只读 UI、标准包生成或任何未来自动接入预留代码。所谓“接口保留”只指本地
中立字段和状态槽位。

## 7. 官方原始资料

- Calkit：[文档](https://docs.calkit.org/)、
  [`calkit.yaml`](https://docs.calkit.org/calkit-yaml/)、
  [Pipeline](https://docs.calkit.org/pipeline/)、
  [治理说明](https://docs.calkit.org/governance/)
- Snakemake：[文档](https://snakemake.github.io/)、
  [Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/docs/catalog.html)、
  [添加工作流](https://snakemake.github.io/snakemake-workflow-catalog/docs/about/adding_workflows.html)
- DVC：[Pipeline](https://dvc.org/doc/start/data-pipelines/data-pipelines)、
  [Experiments](https://dvc.org/doc/start/experiments)
- MLflow：[Tracking](https://mlflow.org/docs/latest/tracking/)、
  [Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- noWorkflow：[文档](https://gems-uff.github.io/noworkflow/)
- Quarto：[项目](https://quarto.org/docs/projects/quarto-projects.html)、
  [执行与 freeze](https://quarto.org/docs/projects/code-execution.html)
- MyST/Jupyter Book：[MyST](https://mystmd.org/guide/)、
  [Jupyter Book](https://jupyterbook.org/latest/)
- RO-Crate：[规范](https://www.researchobject.org/ro-crate/specification.html)、
  [Workflow Run](https://www.researchobject.org/workflow-run-crate/)
- W3C PROV：[PROV-O](https://www.w3.org/TR/prov-o/)
- Diátaxis：[首页](https://diataxis.fr/)、
  [入门](https://diataxis.fr/start-here/)
- The Turing Way：[Research Compendia](https://book.the-turing-way.org/reproducible-research/compendia/)、
  [项目设计清单](https://book.the-turing-way.org/project-design/pd-overview/pd-checklist)
- WorkflowHub：[项目说明](https://about.workflowhub.eu/)、
  [Workflow RO-Crate](https://about.workflowhub.eu/Workflow-RO-Crate/)
- eLabFTW：[文档](https://doc.elabftw.net/)、
  [追溯与审计](https://doc.elabftw.net/docs/edge/usage/traceability-and-auditability/)
- Renku：[文档](https://docs.renkulab.io/)、
  [仓库](https://github.com/SwissDataScienceCenter/renku)
- OSF：[Registrations](https://help.osf.io/article/330-welcome-to-registrations)、
  [API](https://developer.osf.io/)

## 8. 调研边界

外部资料说明其原始设计，不证明 PFMval 已实现相同能力。本文为 historical
reference 和用户选择记录，不是 active 执行方案；真正实现与验收只以总控包、
P0C、R10 和 R11 的用户批准版本为准。
