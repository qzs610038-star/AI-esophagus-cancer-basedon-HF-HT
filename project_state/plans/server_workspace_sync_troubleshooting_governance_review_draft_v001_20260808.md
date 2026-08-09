# PFMval 服务器 W### 同步、关键合同纠正与排障知识治理修改方案

> plan_id: `server_workspace_sync_troubleshooting_governance_v001_20260808`
> plan_revision: `003`
> lifecycle: `pending_review`
> execution_status: `NOT RUN`
> authority: `review_draft`
> source_state_revision: `182`
> source_commit: `b07eb9376ba6b34678c266da346dcfaaa7deffff`
> working_tree_snapshot: `dirty; pre-existing and concurrent user changes preserved`
> audit_snapshot: `PASS=6 WARN=6 FAIL=1; workspace branch binding drift`
> created_at: `2026-08-08`
> updated_at: `2026-08-08; align with user execution-flow requirements, add comparison coverage, and refine W004 temporary-exception boundary`
> boundary: 本文件只提出修改方案；不授权服务器操作、目录迁移、训练、job/attempt 创建、结果导入或接纳、Registry/current state 修改、受保护资产修改、清理、提交或推送。

补充说明：上述前提只约束后续治理/同步/目录切换工作的启动顺序，不要求在当前 W004 本轮实验中先把历史复制残留全部清干净；当前轮次只要不新增无关脏文件，并保持实验相关改动可追溯即可。

## 0. 服务器前提工作（适用于后续所有方案）

本方案及后续所有涉及服务器同步、回传、治理、清理、目录切换与知识晋升的方案，均以一项先行前提为基础：

- 在当前 W004 本轮实验推进完成前，不因历史复制残留、未跟踪文件或工作区不干净而强行中止正在推进的实验。
- 在本轮实验推进完成后，再联合 W001～W005 以及服务器现有文件做一次统一工作区治理。
- 统一治理的目标是把旧工作树复制残留、弃用文件、未跟踪残留和服务器侧历史产物，按“归档 / 迁移 / 明确忽略规则 / 保留可审计痕迹”收敛，而不是在单个实验期内临时清理。
- 统一治理完成后，再评估各工作区是否恢复到可复用、可持续同步的状态，并据此推进后续方案、目录切换和知识晋升。
- 该前提不改变现有训练批准、Gitee-only 传输、结果导入/接纳和受保护资产边界。

## 1. 目的

本方案面向 W001～W005 暴露的服务器同步、环境兼容、关键合同校验、结果回传、
Git 分支分散和排障经验无法复用问题，建立一套兼顾速度与可审计性的标准流程。

目标包括：

1. 每个正式实验继续绑定永久、不复用的 `W###`，且在绑定前先完成主分支成果/残留文件的审核、迁移范围确认和工作树建设准备。
2. 工作树内代码开发、本地测试、服务器同步、环境调试、正式训练、结果回传和 closeout 构成一条连续闭环，而不是彼此脱节的零散治理动作。
3. 日常服务器排障默认快速、完整留痕、非结果性且不消耗 `run_units`。
4. 正式运行前只对关键实验语义、关键代码身份和关键输入执行严格冻结。
5. 关键冻结失败时存在简洁的标准纠正流程，不通过覆盖旧合同或重建实验逃避审计。
6. 排障原始记录归属当前 W###；可复用规则和平台修复经审查后进入 `main`。
7. 实验可绑定宏观部署方案，并按步骤更新执行状态；短期排障不反复改写部署方案。
8. 服务器结果通过 job 预绑定的统一回传清单返回；关键结果严格校验，原始训练日志完整回传但不制造逐文件 SHA 阻塞。
9. 训练产物保存在对应 W### 的受管运行/回传目录，而不是回填到主分支或散落在源码工作树中，以同时保持代码可追溯和工作树可维护。

## 2. 现状判断

### 2.1 已存在但尚未统一的能力

- 编号工作树协议已经规定一个 experiment 对应一个持久 W###，attempt 不新建工作树。
- G15 已实现 `execution_bundle_v1`、聚合 preflight 和可恢复执行状态机。
- 当前存在 `diagnostic request` / `diagnostic record` 非证据诊断通道。
- 当前存在 `critical_contract_sha256`、结果包、quarantine/import 和远端 SHA 校验。
- Asset shadow inventory 默认只读取 metadata，不计算大文件内容哈希。

### 2.2 仍需修改的差距

- `ensure_job_worktree` 仍按 `job_id` 创建服务器工作树，与持久 W### 目标不一致。
- `critical_contract` 同时混入源码身份与科学语义，导致工程修复难以保持协议语义不变。
- 当前结果校验器实际上要求 supporting/diagnostic artifact 也携带 SHA256，范围过宽。
- Experiment Registry 没有统一的部署方案 ID/revision/step 绑定结构。
- 排障记录分散在各工作树、聊天和历史文档中，没有明确的 main 晋升流程。
- `pfmval_deploy_git` 同时承载代码、配置、split、cache 和 checkpoint，容易形成脏工作树。
- 当前方案尚未把“主分支成果清点 → 迁移范围审核 → 工作树建设 → Agent 在工作树内开发 → 本地逐步试运行 → 统一提交同步 → 服务器调试 → 正式训练 → 结果回传 → 结果/经验/模块沉淀”串成一条显式执行闭环。
- 当前 `server_job_v2` 明确把诊断日志设为 `include_only_when_needed_for_agent_analysis`，
  即由 Agent 在事后决定是否回传；它没有定义成功训练、训练中失败、训练前失败分别必须返回哪些原始日志。
- 当前 `result_envelope_v2` 能登记 artifact 和校验结果包闭包，但没有 job 预绑定的
  `expected / returned / missing / not_applicable` 清单，也没有面向用户的“本包到底包含什么”摘要。
- 2026-08-01 曾因正式回传未覆盖全部原始预测，需从 Gitee 引用及用户手工服务器副本补档；
  这说明“结果包有效”不等于“研究所需原始产物完整”。
- W004 A001/R002 的已登记 `runner.log` 曾被仓库 `*.log` 忽略规则漏出远端结果分支；
  旧发布器使用普通 `git add`，直到远端 bundle SHA 不一致才发现。该案例证明 manifest 声明、
  本地打包、Git 实际纳入和远端内容之间尚缺少前置闭环检查。
- 当前主工作区存在未提交并发修改，严格门禁因 workspace branch 漂移失败；本方案不得被解释为现行能力已经完成。

## 3. 固定原则

### 3.1 两态分离

服务器流程拆为两种状态：

| 状态 | 用途 | 默认校验 | 是否可训练 | 是否消耗 run unit |
|---|---|---|---|---|
| `DIAGNOSTIC` | Python、依赖、路径、import、编码、打包和回传排障 | 轻校验、全记录、非 SHA 阻断 | 否 | 否 |
| `FROZEN_EXECUTION` | smoke/formal 正式运行 | 关键语义、代码身份、输入与结果 HARD | 需批准 | 是，从真实拟合开始计 |

诊断态不因普通 SHA256 不一致阻断下一轮排障，但以下边界始终 HARD：

- W### 归属和路径边界；
- Gitee-only 传输；
- 受保护资产不可修改；
- 不得使用任意 shell 或未登记 command id；
- 不得把诊断伪装成训练或结果证据；
- 本地/服务器不得同时修改同一 canonical source branch。

### 3.2 科学语义与代码身份拆分

后续 v3 合同不再让一个 digest 同时承担所有职责：

```text
experiment approval
├─ protocol_digest             # 科学语义：数据、split、模型、loss、超参、选择策略
├─ code_snapshot
│  ├─ source_commit            # Git 跟踪源码快照
│  ├─ critical_code_paths      # 关键代码闭包
│  └─ code_closure_digest      # 关键代码闭包内容指纹
├─ input_binding_digest        # 已冻结输入身份
└─ artifact_policy             # 结果回传要求
```

`protocol_digest` 是关键实验参数不可变性的核心。工程兼容修复可以产生新的
`source_commit` 和 `code_closure_digest`，但不得静默改变 `protocol_digest`。

历史 `critical_contract_sha256` 保持只读兼容；新协议采用双读、新写 v3，不原地改写
旧合同或旧结果。

## 3A. 与用户执行流方案的对比整理

### 3A.1 高度重叠 / 已覆盖部分

| 用户方案环节 | 本方案覆盖情况 | 对应位置 |
|---|---|---|
| W### 绑定、分支规范、单写者 | 已覆盖 | 前提工作、2.2、4、10 |
| 本地测试通过后再同步服务器 | 已覆盖但仍需补细节 | 6、10 |
| 服务器调试与正式训练分离 | 已覆盖 | 3.1、6、10 |
| 标准结果回传包先冻结、再严格校验 | 已覆盖 | 11.1-11.5 |
| 回传后沉淀结果、排障经验与可复用规则 | 已覆盖 | 7、8、12 |

### 3A.2 用户方案提出、原方案此前未写清的部分

| 用户方案新增关注点 | 现状 | 本次修订动作 |
|---|---|---|
| 主分支成果文件与旧成果文件先做审核，再决定迁移到工作树的范围 | 原方案未单列 | 纳入 1、2.2、3A、3B |
| Agent 在工作树内按部署方案完成代码开发 | 原方案偏治理，未写开发流程 | 在 3B 补充执行闭环 |
| 代码模块化方向与可复用模块后续沉淀 | 原方案只写 main 晋升，未写代码沉淀节奏 | 在 3B、12 补充 |
| 本地测试需支持每轮次、每步骤独立执行 | 原方案只有 preflight/targeted verify | 在 3B.3 补充 |
| 调试阶段仅简单记录，不做严格哈希拖慢节奏 | 原方案已有方向，但不够直白 | 在 3.1、3B.5 明确 |
| 训练进度需每个 epoch 输出可见 | 原方案只要求日志回传，未要求运行中可见性 | 在 3B.6 补充 |

### 3A.3 原方案已覆盖、但用户方案中未显式写出的关键约束

1. Gitee 是唯一日常服务器同步通道；GitHub/SSH/SCP/HTTP remote command/Tunnel 都不是执行通道。
2. 正式训练前必须冻结 protocol/code/input，不得把工程适配伪装成协议不变的正式运行。
3. 回传包必须区分 `required / returned / missing / not_applicable`，不能只看 bundle SHA。
4. 结果导入、accepted 证据、main 晋升与 closeout 必须解耦，不能把“回传成功”直接等同于“结论成立”。
5. 工作树内允许保留当前轮次必需的实验改动，但不允许服务器与本地并行修改同一 canonical source branch。

### 3A.4 用户方案与原方案的关键分歧及统一结论

| 议题 | 用户原始想法 | 本方案统一口径 |
|---|---|---|
| 训练产物存放位置 | 全部放在对应工作树内 | 代码留在源码工作树；训练产物放在对应 W### 的受管 run/return/diagnostic 目录 |
| 调试记录管理 | 先简单留存，后续统一管理 | 同意；诊断期只做轻记录与回执，closeout 时再统一分类沉淀 |
| 当前 W004 脏工作区 | 希望不中断实验 | 同意；本轮先不强行清干净，待统一治理时再系统处理 |

## 3B. 面向完整实验执行闭环的补充规范

### 3B.1 主分支成果审核与工作树建设前置

正式绑定新实验工作树前，应先完成：

1. 主分支中“最新且有价值成果文件”与“旧成果/弃用残留/不应纳入 Git 的文件”初步分类；
2. 明确本次实验真正需要迁入工作树的文件范围，并形成可审核清单；
3. 对涉及历史结果、旧日志、缓存、checkpoint、临时分析文件的迁移作单独考量，不默认整批复制；
4. 只有迁移范围经过审视后，才创建或继续扩充对应 W### 工作树。

### 3B.2 工作树内代码开发要求

- Agent 在已绑定的 W### 工作树内单独开展代码开发，不回到主分支直接堆叠实验代码。
- 当前优先级是实现本次实验所需新增功能和关键目标，不强制在当轮把所有历史代码一次性模块化重构。
- 但新增代码应按功能边界分区，避免把训练入口、数据装配、结果回传和临时诊断混写在同一处。
- 识别到具有复用潜力的模块时，只需在本轮保留清晰接口和沉淀候选；正式抽象和主分支收敛可放到 closeout 后统一进行。

### 3B.3 本地测试与独立执行要求

本地测试至少分三层：

1. 静态/基础层：import、路径、参数、配置解析、入口脚本可启动；
2. 流程层：按部署方案中的每个实验轮次、每个步骤可独立运行，不依赖隐藏的全局手工状态；
3. 轻量运行层：最小样本或最小轮次 dry-run 能打通训练入口、日志输出、结果打包或预回传逻辑。

本地未通过上述最小闭环前，不进入服务器同步。

### 3B.4 统一提交与服务器同步要求

- 本地验证通过后，再统一提交当前 W### 的代码与必要配置。
- 服务器同步仍严格经 Gitee-only 闭环，不新增旁路。
- 同步说明应包含：本次 source commit、目标 W###、拟执行阶段、是否仅诊断、以及预期回传 profile。

### 3B.5 服务器环境调试要求

- 服务器调试阶段默认属于 `DIAGNOSTIC`，以 2-3 轮快速适配为目标。
- 该阶段不启用会显著拖慢节奏的全量严格哈希阻断，只简单记录变更参数、适配点、命令 ID、回执和结果。
- 调试遵循“只改问题点、无问题部分不动”的最小修改原则。
- 所有调试记录先保存在当前 W### 的诊断记录与服务器回执中，待实验闭环完成后再统一整理进 closeout / main 候选。

### 3B.6 正式训练与训练可见性要求

- 正式训练前仍需经过冻结和聚合 preflight。
- 训练运行中必须保证 epoch 级进度对用户可见：至少每完成一个 epoch 输出一次关键训练/验证结果，避免出现“终端长时间无反馈而无法判断是否仍在运行”的黑箱体验。
- GPU 训练加速、加速策略比较和更深的性能优化只保留后续扩展接口，不作为本轮治理落地前提。

### 3B.7 回传后的三类沉淀

确认回传结果可用后，再统一做三类沉淀：

1. 结果登记与注册：将本次实验结果按既有 import/accept 流程登记到主线事实体系；
2. 排障知识分类：区分本轮实验专属排障记录与可跨实验复用的通用经验；
3. 代码复用候选整理：识别工作树内可模块化拆分、可通用复用的代码块，经过复核后再沉淀回主分支。

以上三类沉淀都发生在回传与核验之后，而不是调试期间并行大规模整理。

## 4. 服务器目录目标结构

以下结构是统一治理完成后的目标态；在当前 W004 本轮实验未结束前，不以此要求立刻迁移或清空现有目录。

```text
D:\AIPatho\qzs\
├─ pfmval_governance\
│  └─ current\                 # 干净的治理 checkout
├─ pfmval_automation\
│  ├─ W001\                    # W001 持久源码 worktree
│  ├─ W004\                    # W004 持久源码 worktree
│  └─ W005\
├─ pfmval_runs\
│  └─ W004\A001\               # 日志、预测、checkpoint 和中间输出
├─ pfmval_returns\
│  └─ W004\A001\R001\          # 结果包 staging
├─ pfmval_diagnostics\
│  └─ W004\D001\               # 诊断回执、patch 和环境记录
└─ shared_assets\              # 公共只读数据、cache、模型和受保护资产
```

规则：

1. 所有实验专属目录必须含对应 W###；所有 job/result/diagnostic 必须反向绑定该 W###。
2. 源码 worktree 只放 Git 跟踪源码，不存放日志、checkpoint、预测或结果包。
3. 公共大资产不复制进各 W###，只通过稳定 `path_id` 引用。
4. `pfmval_deploy` 可作为历史参考，不承接新实验。
5. `pfmval_deploy_git` 先标记 `legacy_mixed` 并停止新增作业；完成 inventory、copy-verify、path-id 切换和验收后才可转为只读历史目录。
6. 未经单独批准不得移动、删除或清理任何旧目录、cache、checkpoint 或受保护资产。

## 5. 哈希校验优化

### 5.1 分级策略

| 等级 | 对象 | 诊断阶段 | 冻结/导入阶段 |
|---|---|---|---|
| H0 | `protocol_digest`、关键代码、关键输入、关键结果 | 记录差异 | HARD |
| H1 | Gitee commit/ref、bundle manifest | 快速核对或 WARN | HARD |
| H2 | 原始 stdout/stderr、训练历史、resolved config/argv、开始/终态回执 | 按作业状态强制回传；size/inventory | 回传完整性 HARD；默认不做逐文件 SHA |
| H3 | 额外 debug dump、GPU 采样、历史日志、可重建 plots、无关文档 | 按 return profile 或故障条件回传 | 不参与科学结果门禁 |

### 5.2 大型输入哈希缓存

- manifest、split、z-score、checkpoint 和 calibrator 首次激活或迁移时计算完整 SHA256。
- 后续 job 引用已登记的 `asset_id + sha256`，不在每轮排障重新读取大文件。
- size、mtime、路径或资产身份发生变化时才重新计算内容哈希。
- 独立审计保留显式 full verification 命令，但不放入日常快速路径。

### 5.3 结果包校验范围

- `critical`：指标来源、selection proof、每个实际评估 split 的逐样本原始预测，必须 SHA256。
- `supporting`：原始训练日志、结构化训练历史、实际配置、命令和运行回执按作业状态必须回传；
  默认只要求路径、大小、类型、retention 和 bundle/Git 闭包一致，不要求逐文件 SHA256。
- `diagnostic`：额外 debug dump 按 return profile 或失败条件回传，不得作为 accepted metrics 来源，
  不要求阻断式 SHA256。
- 大型 checkpoint/embedding/cache：记录服务器 path、size、缓存 SHA 和 retention/recompute policy，不通过结果包复制。

## 6. 正式冻结失败后的快速纠正流程

### 6.1 纠正状态机

```text
FREEZE_FAILED
→ CORRECTION_OPEN
→ CLASSIFIED
→ PATCHED_IN_BOUND_W
→ TARGETED_VERIFIED
→ REFROZEN
→ READY
```

如果检测到科学语义变化，则转为：

```text
CLASSIFIED
→ PROTOCOL_CHANGE_REQUIRED
→ AWAITING_USER_REAPPROVAL
```

### 6.2 纠正记录

每次冻结纠正分配 `correction_id`，建议格式 `W004-FC001`。最小字段：

- `workspace_id`、`experiment_id`、`attempt_id`；
- 失败的 preflight check id 和错误签名；
- 原 `source_commit`、`protocol_digest`、`code_closure_digest`；
- 分类、影响范围和允许修改文件；
- 修改 diff、目标测试和诊断回执；
- 新 `source_commit`、新 `code_closure_digest`；
- protocol invariants 对比结果；
- 是否需要新 protocol revision / 用户重新批准；
- refreeze 和 Gitee remote SHA 结果。

### 6.3 四类纠正

| 类别 | 示例 | 标准处理 | 重新批准 |
|---|---|---|---|
| FC0 环境 | Python 路径、缺包、GPU/CUDA 探测 | 更新环境记录或 allowlisted command；针对性 preflight | 否 |
| FC1 工程适配 | import bootstrap、路径分隔符、CRLF、PowerShell、日志、打包 | 在当前 W### 本地 canonical 分支修复；新 commit；保持 protocol digest | 否 |
| FC2 合同生成缺陷 | 默认值未解析、canonicalization 不稳定、字段序列化错误 | 修合同生成器并比较 resolved semantics；生成新 dispatch revision | 语义完全一致时否 |
| FC3 科学协议变化 | 数据、split、模型、loss、超参、checkpoint、选择策略 | 停止；提升 protocol revision；重新生成合同和批准 | 是 |

### 6.4 简洁执行步骤

1. preflight 一次性返回全部错误，不 fail-fast。
2. 训练不启动、lease 不获取、run unit 不消耗。
3. 在当前绑定 W### 创建 FC 记录，不修改旧 job/result/bundle。
4. 自动执行 protocol invariant comparator：比较旧/新 resolved scientific fields。
5. 只运行受影响检查，例如 import、包依赖、entrypoint、路径、结果打包或回传测试。
6. 生成新的完整 source commit；服务器补丁只能经 Gitee 返回并由本地接纳。
7. 在同一 attempt 下递增 `dispatch_revision`，保留旧 dispatch revision；不得覆盖。
8. 重新冻结关键代码和受影响输入，核验 Gitee 远端 SHA 后再进入 READY。

### 6.5 监督规则

- FC0～FC2 必须证明 `protocol_digest_before == protocol_digest_after`。
- 修改范围必须落在 adapter allowlist 或本次 correction 显式文件清单。
- 旧 source commit、旧 bundle 和失败回执全部保留，不 reset/clean 掩盖失败。
- 如果 comparator 无法证明科学语义未变，自动升级为 FC3，等待用户裁决。
- correction 不得把新的模型拟合包装成 dry-run；真实拟合仍需要 job/approval/run unit。

## 7. 排障记录归属与 main 晋升

### 7.1 当前 W### 保存原始事实

每个诊断轮次只保存与该轮诊断/实验直接相关、应进入版本管理的整理记录：

```text
experiments/<experiment_id>/server_diagnostics/
├─ D001/diagnostic_receipt.json
├─ D001/patch.diff              # 仅在需要时
├─ D002/diagnostic_receipt.json
└─ server_troubleshooting_closeout.md
```

大型 raw log 留在注册的 `pfmval_diagnostics/W###/D###` 或返回包中；Git 默认只保存
结构化回执和人工整理摘要，避免将机器噪声永久写入仓库历史。

### 7.2 main 只保存可复用结论

W### 的每条排障记录在关闭时分类：

| 分类 | 保存位置 | 是否自动被新工作树继承 |
|---|---|---|
| `experiment_specific` | 当前 W### closeout | 否 |
| `reusable_rule_candidate` | W### 候选记录，等待治理复核 | 否 |
| `reusable_rule_accepted` | main 中央服务器配置与排障知识库 | 是 |
| `platform_bug_candidate` | W### 修复 commit/patch | 否 |
| `platform_fix_integrated` | main 公共代码、测试和知识库 | 是 |

原始 W### 记录不直接写 main。main 的中央条目必须保存：

- 故障签名和适用条件；
- 标准快速检查与永久修复；
- 原始 W###、branch、source commit、closeout path 和内容 SHA；
- 平台修复 commit 与回归测试；
- 首次生效版本和失效/替代关系。

### 7.3 晋升流程

```text
W### D-records
→ W### closeout
→ promotion candidate
→ main governance review
→ reusable rule / platform fix
→ regression preflight
→ next W### inherits from main
```

只有进入 main 的规则和代码才视为跨工作树可复用。仅存在于某个 W###、聊天、
Agent memory 或历史文档中的经验不得假定已被新工作树继承。

## 8. 中央服务器配置与排障知识库

建议升级现有 active 文件 `project_state/plans/server_maintenance.md`，不再新建多个并列
规范。该文件作为唯一中央入口，实际路径值仍以 `configs/server_paths.yaml` 为机器事实源。

建议增加：

1. 当前目录拓扑与 path-id 索引。
2. W### / A### / D### / R### 身份和路径规则。
3. 五分钟快速环境预检。
4. Python、依赖、GPU、Git、Gitee、PowerShell、CRLF/LF、打包和结果返回错误索引。
5. 错误签名 → 根因 → 快速检查 → 标准修复 → 生效 commit。
6. W### closeout 索引。
7. 平台修复晋升记录和回归检查。
8. historical/superseded 文档入口及禁止继续使用的旧命令。

首批应固化的预检包括：

- Python 可执行文件、版本和模块搜索路径；
- 使用 `python -m` 还是直接执行文件的入口合同；
- 关键包 import 与版本；
- PyTorch/CUDA/GPU 可用性；
- cwd、repo root、source worktree 和 governance root；
- Gitee remote/ref 可达性与远端 SHA；
- PowerShell `${Branch}` refspec 语法；
- UTF-8、CRLF/LF 和 Git autocrlf 行为；
- 输出根是否位于源码 worktree 外；
- `.gitignore` 是否会漏掉 manifest 声明的 required artifact；
- Git LFS pointer/lock 检查的超时和无新增 pointer 快速路径。

## 9. 宏观部署方案绑定与动态调整

### 9.1 新绑定字段

Experiment Registry 后续增加：

```text
deployment_plan_binding
├─ deployment_plan_id
├─ adopted_plan_revision
├─ plan_digest
├─ current_step_id
├─ binding_status
└─ adopted_at
```

部署方案本身采用不可变 revision；实验推进状态通过 append-only event 记录，不直接改写
方案正文。

### 9.2 调整分类

| 变化 | 记录方式 | 是否改变 plan revision | 是否改变 protocol revision |
|---|---|---|---|
| 步骤完成、状态更新 | deployment event | 否 | 否 |
| 临时服务器排障 | diagnostic/adaptation record | 否 | 否 |
| 稳定部署流程调整 | 新 plan revision，声明 effective step | 是 | 通常否 |
| 关键实验设置变化 | 新 plan + protocol revision | 是 | 是 |

每次 plan revision 必须声明：

- `effective_from_step`；
- `affected_steps`；
- `recompute_scope`；
- `invalidated_artifacts`；
- `unchanged_protocol_fields`。

只重新验证受影响节点，不因说明文字或短期排障变化重新计算全部服务器数值。

## 10. 标准同步流程

### 10.1 诊断往返

```text
绑定 W###
→ 本地提交本轮诊断源码（不包含历史复制残留治理）
→ 生成 D### request
→ Gitee fast-forward push + remote SHA
→ 服务器精确 fetch
→ 轻量环境/入口检查
→ 记录错误、环境、patch 和输出
→ Gitee 返回 D### receipt/adapter
→ 本地 W### 接纳或拒绝
→ 新 source commit 或继续下一 D###
```

### 10.2 正式运行

```text
诊断完成
→ 冻结 protocol/code/input
→ 聚合 preflight
→ 如失败进入 FC 流程
→ READY
→ 获取 lease
→ 真实 FIT_STARTED 时消耗 run unit
→ 外部 run root
→ 按 return profile 构建 result bundle
→ 生成 expected/returned/missing 摘要
→ Gitee return ref
→ quarantine
→ inspect/validate
→ 完整时 import；不完整时只登记 return incident
→ 用户审核后 accept
```

## 11. 服务器结果回传标准

### 11.1 回传内容必须在运行前冻结

每个正式 job、recovery 和 diagnostic request 均绑定：

```text
return_profile_id
return_profile_revision
expected_artifacts[]
├─ artifact_id / kind / path_pattern
├─ evidence_role
├─ required_when
├─ producer
├─ compression_policy
└─ retention_policy
```

`required_when` 至少支持 `always`、`success`、`fit_started`、`failed_before_fit`、
`failed_after_fit` 和 `when_produced`。Agent 不得在作业结束后自主缩减回传范围；确需调整时必须产生
新的 return profile revision，并在下一次 dispatch 前冻结。

### 11.2 标准回传清单

所有状态共同必传：

- `return_summary.json` 和便于人工审核的 `RETURN_README.md`；
- result envelope、job/attempt/recovery 身份、source commit、protocol/approval/contract ID；
- 实际 `resolved_config`、`resolved_argv` 和 entrypoint；
- started/terminal event、status、exit code、开始/结束时间；
- 原始控制台 stdout/stderr。新 job 优先分别保存；若 runner 只能生成合并日志，
  `runner.log` 仅在 manifest 声明 `stream_mode=merged` 且保留原始字节时可替代；
- 环境/preflight 回执和服务器路径 ID 解析结果。

成功训练额外必传：

- `metrics.json`、selection/best-epoch proof；
- 每个 fit/epoch 的结构化 `training_history`；
- 每个实际评估 split 的逐样本原始预测；
- checkpoint inventory，包括 best/latest 的服务器路径、大小、SHA256 和保留策略；
- 实际完成的 seed/fold/task/run-unit 清单。

失败作业按状态回传：

| 终态 | 强制回传 | 可标记不适用 |
|---|---|---|
| preflight 或 FIT_STARTED 前失败 | config/argv、环境回执、stdout/stderr、完整 traceback、terminal event | training history、预测、checkpoint |
| FIT_STARTED 后失败 | 上述全部、部分 training history、已生成 checkpoint inventory、已生成预测清单 | 尚未执行 split 的预测 |
| success | 共同项和成功训练全部项目 | 仅经 profile 明确证明未产生的条件项 |
| diagnostic | request/receipt、命令 ID、环境差异、stdout/stderr、patch/adapter 清单、terminal event | 科学指标、预测 |

“文件未产生”和“漏回传”必须分开：只有 producer 状态证明该 artifact 本就不适用时才允许
`not_applicable`；其余缺失一律为 `missing`。

### 11.3 用户可见的回传摘要

每个 bundle 根目录必须同时提供机器清单和中文摘要，至少展示：

- W### / experiment / attempt / recovery / job / result / source commit；
- 运行状态、run units、return profile revision；
- `required / returned / missing / not_applicable` 数量和逐项说明；
- 每个 artifact 的用途、路径、大小、证据角色、校验方式和本地落点；
- 未随包复制的大资产的服务器路径、SHA、保留期限和复算策略；
- 当前结论：`RETURN_COMPLETE`、`RETURN_INCOMPLETE` 或 `RETURN_INVALID`；
- 建议下一步：可导入、需补传、需重打包或需排障。

本地接收后，Agent 必须先向用户展示该摘要，再讨论结果分析或接纳；不得只报告 bundle SHA
而不说明实际收到哪些内容。

### 11.4 打包、Git 和远端闭环

1. 训练前运行 `return preview`，确认所有必传 producer 和 path pattern 有明确来源。
2. 训练后按 manifest 精确收集，不使用“目录里有什么就打什么”的扫描逻辑。
3. manifest 声明的 required 文件即使命中 `.gitignore`，发布器也必须显式 `git add --force`；
   但只能 force-add 已验证的 revision path，禁止扩大范围。
4. 打包器在 push 前比较 expected 与 staged Git tree；任何 required 漏项立即
   `RETURN_INCOMPLETE`，不得发布成“完整结果”。
5. push 后从远端 ref 重建 tree，验证 revision 路径、artifact closure、bundle digest 和 remote SHA。
6. 大日志允许无损压缩，但不得截断、只留截图或只返回 Agent 摘要；压缩前后大小和编码写入清单。
7. 关键预测、指标和 selection proof 继续逐文件 SHA；普通原始日志依赖大小、Git object/bundle closure，
   除非被提升为 critical，不做逐文件 SHA。

### 11.5 本地接收、补传与导入边界

```text
fetch immutable return ref
→ quarantine
→ render return summary
→ validate identity + completeness + critical hashes
├─ COMPLETE → result import pending_review
├─ INCOMPLETE → record return incident → 新 revision 补传/重打包
└─ INVALID → 隔离并停止导入
→ 用户独立决定 accept/reject
```

- 不完整回传允许先登记“已收到但不完整”的事件，以免丢失线索，但不得伪装为完整 result import。
- 补传必须使用新的不可变 revision，保留旧 revision 及其缺失清单；禁止覆盖原回传目录或 ref 历史。
- 若只缺回传文件且服务器原始 run 未变，走 packaging-only 补传，不重训、不消耗 run unit。
- W004 `runner.log` 漏包是强制回归样例：测试必须覆盖 `*.log` 被忽略、manifest required、
  force-add、远端 tree 再验证和新 revision 补传全过程。

## 12. W### 完成后的强制 closeout

每个大实验关闭前必须生成 `server_troubleshooting_closeout.md`，至少包含：

- 最终 W### / experiment / source commit / plan revision；
- D### 和 FC### 索引；
- 最终有效服务器环境和目录；
- return profile revision、最终回传摘要和所有补传 revision；
- 原始训练日志、training history、预测和 terminal receipt 是否齐全；
- 错误签名、根因、有效修复、无效尝试；
- 新增或缺失依赖；
- 可复用规则候选；
- 平台代码修复候选；
- 是否已经进入 main；
- 未解决问题和建议保留期限。

W### 不因实验完成自动删除；关闭只先执行 preview。原始记录和 main 晋升结论均需可追溯。

## 13. 拟实施文件范围

用户批准后，预计涉及但不限于：

- `project_state/plans/server_maintenance.md`
- 主分支成果审核/迁移清单模板（可并入 `server_maintenance.md` 或新增附属 schema / checklist）
- `project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`
- `configs/server_paths.yaml`
- `project_state/schemas/critical_contract.schema.json` 或新增 v3 schema
- `project_state/schemas/server_job_v2.schema.json` 的兼容升级
- `project_state/schemas/result_envelope_v2.schema.json` 的分级修正
- 新增 `return_profile_v1.schema.json`、deployment plan binding / correction / diagnostic closeout schema
- 新增 success / failed-before-fit / failed-after-fit / diagnostic 四类 return profile 模板
- `scripts/pfmval_governance.py`
- `scripts/pfmval_execution_roundtrip.py`
- `scripts/pfmval_result_bundle.py`
- `scripts/pfmval_assets.py`
- `deploy/pfmval_ops.py`
- 对应 tests 和 fixtures

不得直接修改 `CURRENT_STATE.md`、Dashboard 或 Registry 事实；需要状态变化时只能通过既有 CLI 和新的受测命令完成。

## 14. 实施阶段

### P0：审查与字段冻结

- 用户确认本方案、执行闭环补充项和待裁决项。
- 冻结 identity、状态机、目录、回传清单和主分支迁移审核字段。
- 不操作服务器、不迁移目录。

### P1：v3 身份与双读兼容

- 拆分 protocol/code/input digest。
- 旧 critical contract、job 和 result 保持只读兼容。
- 新写入采用 v3；禁止批量重写历史记录。

### P2：轻诊断与 FC 纠正流

- 扩展 D### 记录。
- 实现 FC 状态机、invariant comparator 和 targeted preflight。
- 证明 FC0～FC2 不改变协议语义、不消耗 run unit。

### P3：哈希范围收缩

- supporting/diagnostic 取消阻断式 SHA。
- 启用大型输入 hash cache。
- 保留关键结果、关键输入和显式 full audit。

### P4：标准回传合同

- 为 job/recovery/diagnostic 增加 return profile 绑定和状态条件矩阵。
- 实现 preview、pack、inspect、validate 和 incomplete-return incident。
- 生成用户可读回传摘要；补齐 `.gitignore`、远端 tree 和 packaging-only 补传回归测试。
- 先只使用 fixture 和历史 W004 漏包样例验证，不操作服务器、不重训。

### P5：W### 持久服务器目录

- 先 shadow inventory。
- 新建独立 governance/worktree/run/return/diagnostic path ids。
- copy → verify → cutover；不执行原地 move。
- 停止按 job 创建工作树。

### P6：部署方案与知识晋升

- 增加 deployment plan binding 和 step event。
- 升级中央 server maintenance 知识库。
- 实现 W closeout → main promotion candidate → integrated 的闭环。

### P7：无训练演练与试点

- 使用本地 fixture 和 bare Gitee remote 演练。
- 至少覆盖两个不同实验形态。
- 真实服务器只先运行非证据 preflight/compatibility 诊断。
- 通过独立审核后才决定是否作为新实验默认入口。

## 15. 验收标准

### 15.1 效率

- 普通诊断不读取无关大文件内容。
- 同一环境指纹未过期时直接复用。
- 一次聚合 preflight 返回全部 blocker。
- FC0～FC2 只重跑受影响检查。
- 不因 README、日志、Dashboard 或可重建图表变化阻断服务器排障。

### 15.2 不可变性

- FC0～FC2 的 protocol digest 前后完全一致。
- FC3 必须提升 protocol revision 并重新批准。
- 每个 dispatch revision、result ref 和 correction record 不可覆盖。
- 关键预测、指标来源和 selection proof 被篡改时 import 必须失败。

### 15.3 工作树与目录

- 一个实验在服务器只有一个持久 W### worktree。
- 三个 attempt 仍只有一个源码 worktree。
- 所有实验输出都在外部、带 W### 的注册路径。
- path escape、错 W、writer 冲突、以及将未登记的临时修改带入错误工作树仍为 HARD；本轮 W004 允许存在的历史复制残留不因其“脏”本身立即阻断实验推进，但不得再继续扩散到新的无关文件。

### 15.4 知识复用

- 每个关闭的 W### 有 closeout。
- main 中每条通用规则可追溯到原 W###、commit 和记录。
- 新工作树只继承 main 已接纳规则，不读取聊天作为权威依据。
- 已晋升平台修复具有回归测试和首次生效 commit。

### 15.5 回传完整性与可见性

- 每个新 job 在 dispatch 前可生成确定的 expected artifact 清单。
- success、fit 前失败、fit 后失败和 diagnostic 四种 fixture 均能得到正确的必传/不适用判定。
- 漏掉 raw console log、training history、terminal receipt 或成功作业任一评估 split 预测时，
  状态必须为 `RETURN_INCOMPLETE`，不能进入完整结果导入。
- 用户在 import 前能看到 required/returned/missing 表和每个文件用途。
- `.gitignore=*.log` 时 required log 仍被精确纳入；远端 tree 与本地 bundle 闭包一致。
- 超过用户最终确认阈值的大日志测试样例可无损压缩回传，不截断且无需逐文件 SHA 门禁。

## 16. 回退与停止条件

- 任一 schema 双读失败：停止新写 v3，保持旧记录只读。
- invariant comparator 无法证明协议不变：升级 FC3，等待用户批准。
- 目录 copy/verify 不一致：保持旧 path id active，不 cutover、不删除。
- 新诊断通道可能启动训练：立即停止并修正 command allowlist。
- main promotion 与 W### 原始记录不一致：不晋升，保留 candidate。
- 结果分级导致关键预测或指标来源漏校验：回退到严格结果验证。
- return profile 不能根据终态唯一判断必传项：停止发布完整结果，只允许隔离和人工复核。
- required log 被 `.gitignore`、LFS、换行归一化或压缩流程改变/漏掉：停止该 revision，保留旧包并新建补传 revision。

## 17. 待用户裁决

1. **中央知识库载体**：是否批准直接升级现有 `project_state/plans/server_maintenance.md`，而不是新建并列规范？推荐：批准升级现有文件。
2. **旧目录切换时点**：`pfmval_deploy_git` 是在 W004 当前工作完成后冻结，还是等下一新 W### 试点成功后再冻结？推荐：先停止新增作业并保持读取，下一 W### 试点成功后正式 cutover。
3. **历史合同兼容**：是否采用“旧 v1/v2 只读、新写 v3”的双读策略？推荐：是，不批量重算或改写历史结果。
4. **原始日志保留期**：推荐正式训练原始日志随不可变结果 revision 长期保留；纯 diagnostic raw logs 在实验关闭后保留 90 天，Git 永久保存 curated closeout。
5. **main 晋升批准**：通用知识条目和平台修复是否都需用户逐批审核后进入 main？推荐：平台代码修复必须审核；纯知识条目可随 W closeout 审核一并批准。
6. **首个试点工作树**：是否避开正在推进的 W004，改用下一个新 W### 做无训练试点？推荐：是。
7. **日志缺失的阻断点**：推荐允许把不完整包写入 quarantine 并登记 return incident，但禁止完整 `result import` 和 accept，补传完成后再导入；是否确认？
8. **大日志压缩阈值**：推荐单文件超过 20 MB 使用无损 gzip，并同时返回可读的头尾索引；是否采用 20 MB？
9. **stdout/stderr 规范**：推荐新 job 分别保存 `console_stdout.log`、`console_stderr.log`；兼容旧 job 的 merged `runner.log`，但必须声明 `stream_mode=merged`；是否确认？

## 18. 当前结论边界

本方案可以进入用户审核，但当前仍为 `pending_review / NOT RUN`。文件落盘不表示：

- 用户已经批准这些规则；
- schema、CLI、目录或知识库已经实现；
- 服务器已经迁移；
- W004 或其他实验获得新的训练、attempt、job、同步或结果接纳授权。
