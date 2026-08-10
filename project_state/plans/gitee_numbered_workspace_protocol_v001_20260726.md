# PFMval 编号工作树与 Gitee 实验往返协议 v001

> lifecycle: approved-protocol
> implementation: code-pending
> authorization: `DIR-20260726-003`
> effective date: 2026-07-26
> transport: Gitee only
> boundary: 本文批准身份、状态机和验收规则；不授权真实服务器操作、训练、结果接纳、工作树删除或受保护资产修改。

## 1. 目标与当前差距

目标是让一个实验在本地和服务器各有一个稳定、可识别、可复用的工作树，
所有 attempt（一次结果性执行）在同一逻辑工作区内完成，避免每次排障都
创建工作树、手写整套命令或靠反复 pull/push 解决冲突。

当前实现尚未具备该能力：

- `deploy/pfmval_ops.py::ensure_job_worktree` 仍按 `job_id` 建立服务器
  worktree。
- 现有正式批准仍绑定单个 `job_id + source_commit`。
- `deploy/run_experiment.ps1` 仍可能把日志写入 worktree 内部，使持久
  worktree 变 dirty。
- 当前工作区清单不得被自动编号、收养、迁移或删除；须先以 shadow
  inventory 复核每个 worktree 的实验归属。

因此，本协议先作为后续实现的唯一设计基线；在 workspace registry、外部
run root、lease 和 schema v2 通过测试前，现有 v1 流程不得假装已经升级。

## 2. 身份模型

### 2.1 编号

- 稳定编号：`W001`、`W002`、……，三位递增，永久不复用。
- 显示名：`W017-mpp2-ridge`。短名可以修改，编号不可修改。
- 一个 `experiment_id` 只能绑定一个 `W###`。
- protocol revision、job、attempt、服务器适配或结果重传均不分配新
  worktree。
- 只有新的独立实验问题才分配新编号。

一个 `W###` 是跨主机的逻辑工作区，不是共享目录：

```text
W017-mpp2-ridge
├─ local_path:  本地实验 worktree
└─ server_path: <server_automation_worktrees>/W017
```

物理目录只使用稳定编号，避免短名修改造成路径漂移；短名仅存在 Registry、
对话回执和人类可读报告中。

### 2.2 attempt 与次数

- `attempt_id` 使用工作区内递增编号，例如 `A001`。
- `job_id` 是一次不可变 dispatch envelope，不等于 worktree。
- `run_units` 是本次 job 实际包含的独立拟合/校准数量。
- 一个脚本执行三个 seed 时，默认 `run_units=3`，不能把它包装为“一批
  一次”。

### 2.3 最小 registry 字段

workspace registry 至少记录：

- `workspace_id`
- `display_name`
- `experiment_id`
- `local_path_id` / `server_path_id`
- `local_branch` / `server_dispatch_ref` / `server_return_ref`
- `current_source_commit`
- `protocol_revision`
- `status`
- `active_attempt_id`
- `lease`
- `run_limit` / `run_consumed`
- `opened_at` / `last_verified_at`
- `close_eligibility`
- `retention_policy`

工作区绑定不得保存在一个全局 `current_workspace` 文件中；并发对话会因此
串工作树。绑定属于每个对话自己的上下文。

## 3. 新对话绑定规范

用户在会话开头指定：

```text
本对话工作树：W017
目标：修正……
```

Agent 在任何写操作前返回一次绑定回执：

```text
workspace: W017-mpp2-ridge
experiment: <experiment_id>
local path: <resolved path>
server path id: <registered path id>
branch / HEAD / dirty: <verified values>
binding: active
```

规则：

1. 绑定成功后，本对话的修改、测试、提交、job 生成和结果核验默认都在
   `W017` 内进行。
2. 未指定 `W###` 的实验对话只允许只读定位；需要写入时只询问一次工作树
   编号，不得默认写 `main`。
3. 非实验治理任务若确需使用 `main`，也应明确说
   `本对话工作树：main`。
4. 切换时用户明确说 `切换工作树：W018`；Agent 重新核验并回执，不继承
   前一个工作树状态。
5. 一个对话若需要修改两个实验，默认拆分对话；确需同时处理时分别绑定并
   对每次操作显式给出 locus。

### 3.1 本地诊断/探索的 main 例外

依据 `DIR-20260801-008`，同时满足以下条件的任务可不分配或绑定 `W###`：

- 不涉及服务器对接、服务器同步或真实远程操作；
- 不涉及模型训练、正式实验 dispatch、结果 import/accept；
- 不修改受保护资产，不把输出晋级为 accepted 证据；
- 代码与结果保持本地诊断或 explore 边界。

这类任务可直接在本地 `main` 开展，但在任何实验代码写入或结果性运行前，
必须先把当前非忽略工作区完整提交到本地 Git，形成可恢复的基线提交；不得
自动推送该提交。若任务范围随后触及服务器、训练或证据晋级，main 例外立即
失效，必须停止并重新按本节前述规则绑定 `W###`。

## 4. 服务器工作树生命周期

```text
reserved
→ provisioned
→ idle
→ synced
→ checked
→ leased
→ running
→ packaging
→ returned
→ import_pending
→ imported / rejected
→ idle
→ close_ready
→ removal_approved
→ removed
→ archived
```

异常状态：

- `dirty_blocked`
- `sync_conflict`
- `orphaned_lease`
- `return_conflict`
- `fingerprint_mismatch`

管理规则：

1. 登记实验并分配 `W###` 后，服务器 worktree 只创建一次。
2. attempt 开始前只 `fetch`；在无 lease 且 worktree 干净时，切到 manifest
   指定的完整 `source_commit`。自动化不得用 `pull`、merge 或 rebase。
3. worktree 保持 detached 执行视图；代码的 canonical writer 默认在本地
   对应工作树。
4. 日志、checkpoint、predictions 和结果包写入工作树外的注册目录：

   ```text
   <server_experiment_runs>/W017/A003/
   ```

5. 一个 `W###` 同时只允许一个写 lease 或运行 lease。
6. stale lease 不按时间自动释放；必须核验 PID、heartbeat 与实际进程，
   确认不存在后记录 orphan evidence，再显式恢复。
7. 实验结束只生成 `close --preview`。只有用户批准精确 `W###` 及解析后的
   路径后才能移除。
8. 禁止全局 `git worktree prune`、递归删除注册根、顺带删除 branch、
   checkpoint、缓存或外部 run root。

## 5. 实验次数协议

实验登记或正式批准前，Agent 默认询问：

> 这个协议允许产生几次结果性执行？请按独立模型拟合、seed 或校准拟合的
> 实际数量填写。

不提供隐式默认值，也不把多次执行表述为“一次一批”。

### 5.1 消耗边界

不消耗次数：

- preflight、dry-run 和参数解析；
- 路径、编码、解释器、import bootstrap 检查；
- 未启动结果性进程的兼容适配失败；
- 同一结果包的重新打包、重传或幂等 import；
- Gitee ref 冲突恢复。

消耗次数：

- 数据与关键合同验证完成后，进入预定义 `EXPERIMENT_STARTED` 边界并开始
  一个独立模型/seed/校准器的拟合或正式评估。
- 进入该边界后即使运行失败也消耗对应 `run_units`。

预算耗尽时必须重新询问用户；不得通过新建 job、attempt、worktree 或换
分支绕过。

### 5.2 批准绑定

批准 v2 目标绑定：

- `experiment_id`
- `protocol_revision`
- `phase`
- `run_limit`
- `critical_contract_sha256`
- `adaptation_policy`

每个 job 继续记录完整 `source_commit`、`attempt_id` 和本次
`run_units`。

## 6. 简单服务器适配

允许在不重复询问批准、且不消耗实验次数的情况下处理 E1 工程兼容问题：

- 注册路径映射、Windows 分隔符或大小写；
- UTF-8、CRLF/LF 和日志编码；
- launcher 参数透传与语义不变的显式 alias；
- Python/import bootstrap、解释器或环境入口；
- stdout/stderr、退出码和错误传播；
- packaging、Gitee 回传和幂等 import。

必须同时满足：

1. 生成新的完整 `source_commit` 或不可变 adaptation bundle。
2. `critical_contract_sha256` 不变。
3. diff 只落在明确 allowlist 的 adapter 范围。
4. 针对性测试与 dry-run 通过。
5. 生成 `adaptation_record_id`，job/result 引用该记录。
6. 服务器补丁经 Gitee 返回，本地绑定工作树审查并成为 canonical
   commit；服务器不得与本地同时写同一 source branch。

以下变化不属于简单适配，必须停止并重新询问：

- loss、超参数、正则化或关键 feature flag；
- 数据、split、train-only z-score、checkpoint、base model、calibrator；
- 模型结构或训练/校准逻辑；
- best epoch、通路/模型选择或 evaluation policy；
- smoke/formal phase；
- 任何改变 critical contract digest 的修改。

## 7. 不可篡改指纹

E0 严格指纹最少包括：

- 完整 40 位 `source_commit`；
- 规范化 `critical_contract_sha256`；
- active data manifest 根哈希；
- split 和 train-only z-score；
- 实际 checkpoint、base model 或 calibrator；
- resolved critical argv；
- selection / external evaluation policy；
- 结论承载的 metrics、selection proof 和必要 predictions。

普通 README、说明文档、stdout/stderr、可重建 plots 和纯诊断文件只登记为
supporting/diagnostic；若它们承担模型选择或结论证据，则升级为 E0。

## 8. 标准 Gitee 往返

```text
本地绑定 W017
→ 验证 clean、Registry、次数余额和关键合同
→ 生成不可变 A003 job manifest
→ 提交到 local-owned dispatch ref
→ fast-forward push 并核验 Gitee 远端 SHA
→ 服务器 fetch，不 pull
→ 核验 W017 映射、source_commit 与全部 E0 指纹
→ detached checkout 精确 source_commit
→ preflight / dry-run
→ 原子获得 lease
→ 记录 EXPERIMENT_STARTED 并消耗 run_units
→ 输出到外部 run root
→ 生成不可变 result envelope
→ fast-forward push server-owned return ref
→ 本地 fetch 到 quarantine inbox
→ 校验 job/result/fingerprint/bundle SHA
→ 幂等 result import
→ 释放 lease，W017 回到 idle
```

每个 attempt/revision 使用不可变、单写者 ref：

```text
automation/local/W017/A003/D001
automation/server/W017/A003/result-R001
automation/server/W017/A003/adapter-R001
```

- local ref 只允许本地写，承载不可变 dispatch revision。
- result/adapter ref 只允许服务器写；adapter ref 只返回补丁和诊断记录，
  不成为 canonical source branch。
- 两端均禁止 force push。
- 本地获取 server ref 后只进入 quarantine/import，不 merge 到实验分支或
  `main`。
- ref 可以按 retention policy 归档，但不得为 ref 新建结果 worktree。

## 9. 冲突恢复

| 情况 | 标准处理 | 禁止处理 |
|---|---|---|
| local ref 非 fast-forward | 停止，fetch 并核验未知 commit | force push |
| server ref 非 fast-forward | 同 bundle SHA 视为幂等；否则在已知 tip 上重放同一 bundle | 重新训练 |
| 同 result ID 不同 SHA | HARD FAIL，记录 writer violation | 覆盖旧结果 |
| worktree 有关键 tracked diff | 阻断，回传 diff/hash | reset/clean |
| 输出误入 worktree | 隔离、登记、修正输出根 | 自动删除 |
| 服务器产生代码适配 | 返回 patch，由本地工作树接纳后下发新 commit | 双端同时写 |
| source commit 缺失 | 核验 Gitee 远端 SHA 后 fetch exact ref | 反复 pull/push 猜测 |
| stale lease | 查 PID/heartbeat，记录 orphan 后显式恢复 | 超时静默解锁 |
| 重复 import | 相同 result ID + bundle SHA 为 no-op | 再跑实验 |

## 10. 清理条件

只有全部满足才可进入 `close_ready`：

- 无运行进程和有效 lease。
- worktree 干净，无未审查 server adaptation。
- 无未推送 source commit。
- 无未回传、未登记或未进入终态的 attempt。
- 结果均已本地 import 或明确 rejected。
- 需保留的大型资产已有注册路径、size 和 SHA。
- `close --preview` 给出唯一解析路径和保留清单。
- 用户明确批准该 `W###` 的精确移除目标。

移除后保留永久 tombstone、编号、experiment 绑定、branch/tag 和结果记录；
`W###` 永不复用。

## 11. 分步实现与验收

### P0：只读 shadow

- 建立 workspace schema/registry。
- inventory 现有 worktree；不自动编号、收养、创建或删除。
- 检验路径规范化、越界、重复 experiment、悬空记录。

### P1：对话绑定与只读状态

- 增加 `workspace status/bind` 的无副作用解析。
- 所有写命令接收显式 `workspace_id`，并执行 locus guard。
- 禁止全局 current-workspace 文件。

### P2：外部 run root 与持久 worktree

- 先让 launcher 的所有输出离开 worktree。
- 再把 `ensure_job_worktree` 改为 experiment workspace。
- 同一 experiment 三个 attempt 的 worktree 数量必须仍为一个。

### P3：次数、合同和适配记录

- schema 双读：v1 继续只读验证/import，新 dispatch 写 v2。
- 覆盖多 seed 的 `run_units`、预算耗尽、启动边界和 digest 漂移。
- E1 适配必须以 allowlist 与 digest equality 共同判定。

### P4：传输与幂等冲突恢复

- 实现 local/server 单写者 ref、远端 SHA 复核和 quarantine import。
- 相同 bundle 重传不得训练；同 ID 不同 SHA 必须阻断。

### P5：关闭

- `close --preview` 无副作用。
- running、dirty、unpushed、unreturned、unimported 均阻断。
- `close --execute` 仅在独立用户批准后作用于一个解析后的工作树。

每阶段失败即停止，不跨阶段追求“整体完成”。现有 v1 job/result 和现有
worktree 保持可读；不得自动迁移或删除。

## 12. 新实验部署方案到工作树实施方案的前置门

本节为用户后续补充的流程约束，优先约束新实验入口，不改写历史方案或既有工作树。

1. 新实验部署方案讨论稿统一放在 `01_指南与解读/部署方案/`；`project_state/plans/` 等旧 Agent plan 目录不得承接新的实验部署讨论稿。
2. 用户明确批准最终部署方案并确认准备绑定实验后，才能依据该文件创建 `W###-<工作树名>_实施方案.md`。
3. 实施方案统一放在 `project_state/implementation_plans/`，并在文件中记录部署方案 revision、experiment、W###、代码目录、修改前配置、任务、验证和证据。
4. 实施方案必须再次经用户审核和明确批准；批准前不得创建/绑定工作树、建立改动目录、写代码或写执行配置。
5. 实施方案批准后才创建/绑定工作树并完成代码目录与前置配置；用户后续显式切换到该 W### 后，Agent 才能按实施方案编辑代码。
6. 每个实施条目完成后，必须在同一实施方案追加进度并标记 `completed_pending_user_review`，同步记录 commit、测试和文件证据。
