# AGENTS.md

本文件只维护跨 agent 的固定安全边界和事实源读取顺序。可变项目状态不得复制到本文件。

## 必读顺序

1. `CURRENT_STATE.md`：当前用户指令、活跃方案、最新已验收结果与阻塞项。
2. `project_state/current_state.json`：机器可读的可变状态单一事实源。
3. 涉及实验状态、性能或下一步决策时，读取 `experiments/experiment_registry.json`；`experiments/experiment_dashboard.md` 仅为派生视图。
4. 涉及服务器训练、缓存、路径或同步时，读取 `configs/server_paths.yaml` 和 `01_指南与解读/部署方案/服务器路径索引_20260701.md`。
5. 只使用 `project_state/document_registry.json` 中 `lifecycle=active` 的方案作为执行依据；`superseded`、`historical`、`missing` 文档不得作为当前结论。
6. 涉及方案、完成声明、实验结果或 GO/NO-GO 独立校验时，读取 `.agents/skills/pfmval-audit/SKILL.md`；`.agents/skills/` 是项目 Skill 权威源，`.claude/skills/<name>/SKILL.md` 只能作为直接引用同名权威 Skill 的薄适配器，不得复制或覆盖规则。
7. 涉及 `团队项目进度与结论/` 的成员资料、汇总索引或术语表维护时，读取 `.agents/skills/team-progress-maintainer/SKILL.md`；涉及 `qzs/` 的本机稳定实验结论、双语伪代码或配套图表时，同时读取 `.agents/skills/qzs-stable-conclusion-writer/SKILL.md`。

开始训练、修改服务器路径或生成项目结论前，运行：

```powershell
python deploy/pfmval_ops.py agent start-check --strict
```

只读服务器故障排查可使用：

```powershell
python deploy/pfmval_ops.py agent start-check --task diagnostic
```

该例外只适用于 allowlisted、非证据性诊断；不得训练、写 Registry/current state、写受保护资产或使用任意 shell。需要回传时先创建 `diagnostic request`，仍只经 Gitee 同步。

## 固定安全边界

- `histogene/`、`egnv1/`、`egnv2/` 为受保护目录，未经用户明确授权不得修改。
- 禁止 `git clean -fd`；不得自动删除 checkpoints、MPP 数据、缓存或未跟踪训练结果。
- 服务器与本地当前只允许通过已配置的 Gitee Git remote 同步代码、状态和小型结果；SSH、SCP、HTTP 远程命令、Tunnel 均不是 active 通道。
- GitHub `origin` 仅是用户逐次明确触发的备份与网页端只读镜像：Agent 不得自动推送、周期同步或将其用作服务器代码、状态、诊断、作业或结果传输通道；该用途不改变 Gitee 作为日常服务器同步唯一通道的规则。
- 正式训练必须存在绑定 `job_id` 与 `source_commit` 的显式用户批准文件。
- `diagnostic` 只能使用 allowlisted command id 并记录 source commit、分支、时间和输出校验值；它不是 experiment/job/result import 的替代通道。
- 本地 `explore` 只允许位于 `scripts/explorations/` 与 `experiments/explorations/`，不得访问服务器、训练数据或生成可比较实验结论。
- MPP 原始 ssGSEA、标准划分、z-score 参数、manifest 和 group 3/5 embargo 审计为受保护资产；重新生成必须另开任务并比较输入、参数和校验值。
- 监督学习预处理必须在训练集上拟合，再应用到验证集和外部测试集；不得用 external XZY 拟合 z-score 或选择 checkpoint。
- `CLAUDE.md`、`.claude/` 等本地适配文件只能补充工具特定说明，不得覆盖受跟踪状态包。

## 对话与实验工作树绑定

- 每个实验工作树使用永久、不复用的 `W###` 编号；可附短名帮助记忆，但权威身份始终是编号。
- 新对话开始实验代码写入前，用户需指定 `本对话工作树：W###`。Agent 必须核对其 experiment、路径、branch、HEAD 和 dirty 状态并回执；未绑定时只允许只读定位，不得默认写 `main`。
- 本地诊断/探索例外：若任务不涉及服务器对接或同步、模型训练、正式实验调度、结果 import/accept，也不改写受保护资产，可无需指定 `W###`，直接在本地 `main` 开展。开始任何实验代码写入或结果性运行前，必须先将当前非忽略工作区完整提交到本地 Git；该基线提交不得自动推送到任何 remote。此例外不改变服务器实验、训练、证据晋级和 Gitee 往返门禁。
- 绑定后，本对话后续修改、测试、提交和 job 生成均默认限于该工作树。切换工作树必须由用户显式指定并重新核验，不得使用全局“当前工作树”文件代替逐对话绑定。
- 一个 experiment 在服务器只对应一个持久工作树；protocol revision、job 和 attempt 不新建工作树。attempt 输出必须写到注册的工作树外运行目录。
- 服务器与本地代码默认由本地单写者维护；服务器简单兼容适配通过 Gitee 返回 patch/记录，由本地工作树接纳后再下发新 commit，禁止两端同时修改同一代码分支。
- 编号、次数预算、传输、lease 和关闭规范见 `project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`。该规范在相应 CLI/schema 完成前不得被解读为已具备自动执行能力。

## 状态更新规则

- 用户明确改变方案、优先级、路径、训练协议或安全边界时，通过 `state record-directive` 追加规范化指令。
- 指令状态仅可显式 append-only 转为 `completed`、`superseded` 或 `cancelled`；生命周期检查只提供人工复核候选，不得自动关闭指令。
- 新训练先登记 experiment id；服务器结果先进入 inbox，经 `result import` 验证后才可成为 accepted 证据。
- `CURRENT_STATE.md`、Dashboard、next-steps 和 session-brief 均为生成文件，禁止手工维护事实。

## 团队进度维护固定边界

- `团队项目进度与结论/` 内的正式维护内容必须先经过用户显式审核与批准；未批准内容只能作为对话中的待审概要，不得提前写入。
- 禁止覆写、删除、替换、重排或移动维护文档内任何既有内容。新增进展必须追加为带日期的新内容。
- 后期需要修正既有表述时，必须保留原文，只能在需修正位置紧邻追加独立的“补充说明（YYYY-MM-DD，已获用户审核）”块，写明新证据、修正理解和适用边界。
- 写入前后必须使用对应 Skill 的 append-only 校验脚本比较修改前快照和修改后文件；出现旧内容删除或改写时不得交付。
