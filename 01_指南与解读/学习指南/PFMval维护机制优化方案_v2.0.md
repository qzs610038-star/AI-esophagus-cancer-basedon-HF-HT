# PFMval 更新维护机制优化方案 v2.1

> 状态：已批准实施；本文件是维护流程设计稿，不替代 `project_state/document_registry.json` 中的 active normative plan。  
> 修订：2026-07-14（Codex 收敛版）  
> 基线：`main` 已纳入 `mpp2-lora-r8-dropout10-smoke-20260714-r001-result-20260714182311-7971c360`；dropout=0.10 路线已关闭。实验数据、检查点、标准划分与原始资产不在本方案的修改范围内。

## 一、目标与决策边界

本方案的目标是让“服务器故障排查与本地快速探索”不再被正式训练的全部流程拖慢，同时保持可复现实验的证据链完整。

必须保留的边界：

- 正式训练必须有绑定 `job_id + source_commit` 的显式用户批准；smoke/formal 必须固定提交、使用干净 detached worktree，并经 Gitee 回传与本地导入验证。
- MPP 原始 ssGSEA、标准划分、train-only z-score 参数、manifest、group 3/5 embargo 审计，以及 `histogene/`、`egnv1/`、`egnv2/` 均受保护。
- external XZY 不得用于 z-score 拟合、checkpoint 选择或调参；train/val overlap 是硬停止条件。
- 服务器仅使用已配置的 Gitee Git remote 同步；不得恢复 SSH、SCP、HTTP 远程执行或 tunnel。
- 禁止 `git clean -fd`，也不得自动删除 checkpoint、缓存、MPP 数据或未跟踪训练结果。清理功能只能列出候选项，删除必须由用户在独立操作中确认。
- `CURRENT_STATE.md`、dashboard、next-steps、session-brief 都是派生视图，只能通过 `state sync` 生成。

## 二、四条执行通道

| 通道 | 适用范围 | 最低门禁 | 禁止事项 | 证据地位 |
|---|---|---|---|---|
| `diagnostic` | 服务器环境、依赖、路径、缓存和 dry-run 故障排查 | 状态包可读、无未完成事务、Gitee-only、诊断命令白名单 | 训练、任意 shell、写受保护资产、写 registry/current state、形成实验结论 | 非实验性审计日志 |
| `explore` | 本地代码原型、静态检查、小型非训练试验 | 物理隔离目录、显式探索标注、追加审计 | 服务器执行、训练数据、可比较指标、实验结论 | 不进入 registry |
| `smoke` | 小规模、可比较的训练/评估 | 已登记实验、显式 directive、固定提交、manifest、Gitee 回传 | 超出批准范围的参数/资源搜索，绕过导入 | 导入后方可成为审查证据 |
| `formal` | 正式训练 | 完整严格门禁、`job_id + source_commit` 批准、完整结果包 | 未批准启动或绕过导入 | 导入后方可成为 accepted evidence |

`explore` 绝不是“免登记训练”。只要涉及服务器、训练数据或可比较指标，就必须进入 `smoke`；已关闭路线（当前为 dropout=0.10）不得借任何通道重试、正式训练、多 seed、rank 或超参数 sweep。

## 三、诊断通道（Phase 1–2）

### 3.1 可放宽的约束

- `agent start-check --task diagnostic` 不以文档新鲜度、无关 MPP 警告或一般性方案哈希为阻断项；这些项目记录为 WARN。
- 诊断不创建 experiment、job envelope 或 result bundle，也不进入 `experiment_registry.json`。
- 诊断可复用一个已推送的 source branch/commit 和受锁保护的 diagnostics 回传分支；不必为每次排障新开结果分支。
- 非 MPP 诊断不扫描无关 MPP 训练脚本与路径。

### 3.2 不可放宽的诊断门禁

`diagnostic` 必须阻断：状态包/指令日志不可读或 schema 失效、未完成 state transaction、遗留 state lock、非 Gitee-only 配置、非法 source commit、越出 allowlist 的命令，以及任何受保护写入。

诊断请求须记录诊断 ID、source commit、允许命令 ID、开始/结束时间、输出清单 SHA-256、是否发生写入。命令参数采用结构化 allowlist，不接受任意 shell 字符串；输出仅能写入诊断输出根。

### 3.3 分支与结果约定

- 每次诊断使用已推送提交创建 detached worktree，保证可复现；本地未提交代码不得上服务器验证。
- diagnostics 回传分支仅接收小型诊断审计包；它不是实验结果分支，不能被 `result import` 导入。
- 诊断发现的修复可以在常规代码分支完成；只有需要训练验证时，才新建/登记 smoke 实验。

## 四、本地 explore 沙盒（Phase 3）

新增目录：

```text
scripts/explorations/          # 探索脚本；必须带 PFMVAL_EXPLORE 标注
experiments/explorations/      # 本地输出；默认忽略但保留 .gitkeep
project_state/exploration_log.jsonl  # 追加式探索审计
```

探索脚本不能修改正式训练脚本或受保护资产，不得连接服务器、读取训练数据或输出可比较指标。清理命令只报告候选项和大小，不删除任何文件。

探索脚本可以升级，但升级仅产生“候选迁移”：需用户显式 directive，复制/迁移到非探索路径并保留起源审计；之后仍必须单独提交、登记 experiment、创建 smoke/formal job 并通过结果导入。升级命令不得预先伪造 `job_id`、`source_commit` 或 accepted evidence。

## 五、指令生命周期（Phase 4）

指令状态为 `active`、`completed`、`superseded`、`cancelled`。所有变更都是 append-only `status_update`；`completed` 表示已完成而非被替代。

新指令可选结构化字段：

- `related_experiment_ids`：本指令关联的实验 ID；
- `review_after`：建议人工复核的日期；
- `completion_evidence`：完成证据的路径或 ID。

`state directives --check-stale` 只输出“人工复核候选”（到达 `review_after`、超过阈值、关联实验均终态）；它绝不自动完成、取消、归档或覆盖指令。状态变更使用显式的 `state directive transition`，并要求理由；`superseded` 还必须给出替代 directive ID。

## 六、CLI 契约与兼容性

实现前不把尚不存在的命令称为“现有行为”。现有 `--strict` 与 `--task {general,server,training}` 保持可用；新增参数不得改变 smoke/formal 的严格语义。

目标命令：

```powershell
# Phase 1：只读诊断门禁
python deploy/pfmval_ops.py agent start-check --task diagnostic
python deploy/pfmval_ops.py diagnostic request --diagnostic-id diagnostic-YYYYMMDD-name --source-commit <commit> --source-branch <branch> --command-id cache_probe
python deploy/pfmval_ops.py diagnostic record --diagnostic-id diagnostic-YYYYMMDD-name --output automation/diagnostics/diagnostic-YYYYMMDD-name/stdout.txt

# Phase 4：指令人工复核与显式转换
python deploy/pfmval_ops.py state directives --check-stale --older-than-days 14
python deploy/pfmval_ops.py state directive transition --id DIR-... --status completed --reason "..." --evidence-ref <path-or-id>

# Phase 3：本地探索（将在该 Phase 实现）
python deploy/pfmval_ops.py explore new --purpose "..."
python deploy/pfmval_ops.py explore list
python deploy/pfmval_ops.py explore candidates --older-than-days 30
python deploy/pfmval_ops.py explore promote --script <path> --target <path> --directive DIR-...
```

`explore clean` 不在契约中：方案不提供自动删除命令。

## 七、实施计划与验收

| Phase | 工作 | 验收 |
|---|---|---|
| 0（完成） | 结果导入、main 合入、状态同步、实验保护快照 | strict gate 通过；dropout=0.10 关闭结论可追溯 |
| 1 | 实现并测试 `diagnostic` start-check profile | 非关键文档漂移为 WARN；状态/事务/transport 硬错误仍 FAIL；不改变 smoke/formal |
| 2 | 诊断请求 allowlist、审计包与 source/回传分支复用 | 无任意 shell；审计字段齐全；不进入 experiment registry |
| 3 | 本地 explore 目录、审计、候选升级与无删除候选报告 | 物理隔离；无服务器/训练权限；升级不伪造正式证据 |
| 4 | 指令 `completed`、结构化字段、提醒与 active 文档更新 | 仅提示不自动关闭；显式转换 append-only；生成视图同步 |

每个 Phase 都需单元测试、`agent start-check --strict` 回归（在适用时）及对受保护路径的变更审计。若 Phase 的实现需要改变当前用户指令、训练协议、服务器路径或安全边界，先通过 `state record-directive` 记录。

## 八、风险与缓解

| 风险 | 缓解 |
|---|---|
| 诊断通道被伪装成训练 | 命令 allowlist、无任意 shell、无 registry/result import、受保护写入硬阻断 |
| explore 污染正式流程 | 目录隔离、探索标注、升级仅为候选且要求 directive |
| 指令提醒误关闭工作 | 只给候选；状态转换必须显式执行并写理由/证据 |
| 快速清理误删结果 | 永不自动删除、永不 `git clean -fd`，只输出候选 |
| 新机制破坏既有训练 | 保留原 smoke/formal 门禁与参数兼容，并以回归测试保护 |

## 九、Codex 审查结论

当前项目过严的部分是“将只读诊断与可复现实验混为同一门禁”，不是 detached worktree、固定提交、Gitee 回传或训练审批本身。v2.1 通过独立 diagnostic 与本地 explore 减少排障摩擦，同时不改变实验可靠性所必需的安全边界。
