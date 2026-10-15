# PFMval 服务器治理分步执行方案 1：传输调试、回传包与脏工作区收口

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先把服务器实验传输调试、回传包规范化、本地与服务器脏工作区收口三件最急的事做稳，为后续完整实验闭环打底。

**Architecture:** 采用“三段式收口”：先把本轮实验的同步/调试通道打通，再把回传包合同固定下来，最后用统一清单处理 W001～W005 与服务器侧的脏工作区和历史残留。当前 W004 允许继续推进，但所有新增治理动作都要落在受管目录、可审计清单和可回退边界内。

**Tech Stack:** Git/Gitee、`deploy/pfmval_ops.py`、`project_state/*`、`scripts/*`、`tests/*`、现有 result envelope / state schema。

## Global Constraints

- 服务器日常同步只允许 `gitee_only`。
- 禁止 SSH、SCP、HTTP 远程命令与 Tunnel 作为同步通道。
- 当前 W004 本轮实验先继续推进，不因历史残留立即中止。
- 禁止 `git clean -fd`，不得删除 checkpoints、MPP 数据、缓存或未跟踪训练结果。
- 训练产物保存在对应 W### 的受管运行/回传目录，不回填主分支源码树。
- 正式结果导入/接纳前，回传包必须先完成完整性核验。
- 受保护资产与历史 accepted 结果保持只读，不做隐式重写。

---

### Task 1: 服务器传输调试闭环

**Files:**

- Modify: `project_state/plans/server_maintenance.md`
- Modify: `deploy/pfmval_ops.py`
- Modify: `scripts/pfmval_state.py`
- Test: `tests/test_pfmval_state.py`

**Interfaces:**

- Consumes: 当前 `gitee_only` 传输规则、`diagnostic request` / `diagnostic record` 约定、W### 绑定信息。
- Produces: 一条可重复的“本地准备 → Gitee 推送 → 服务器精确 fetch → 轻量诊断 → 回执返回”调试链路。

- [ ] **Step 1: 固定调试入口与请求字段**

```text
request_fields = [
  "workspace_id",
  "source_commit",
  "source_branch",
  "remote_name",
  "command_id",
  "expected_action",
  "allowed_artifact_kinds",
]
```

- [ ] **Step 2: 把调试流程写成单向闭环**
  - 本地只做准备和提交；
  - 服务器只做精确 fetch、环境检查、最小回执；
  - 所有修改通过 Gitee 回传，不走旁路。

- [ ] **Step 3: 明确 2-3 轮适配边界**
  - 每轮只改当前问题点；
  - 未相关文件保持不动；
  - 诊断态不启用会拖慢节奏的强阻断哈希校验。

- [ ] **Step 4: 补测试与 fixture**
  - 增加一个成功调试回执 fixture；
  - 增加一个“命令 ID 缺失 / 非 allowlist”失败 fixture；
  - 验证调试记录只落在允许的诊断字段里。

- [ ] **Step 5: 通过回归测试确认闭环**

Run: `pytest tests/test_pfmval_state.py -q`
Expected: 调试请求字段、allowlist 检查、回执结构检查全部通过。

---

### Task 2: 回传包规范化

**Files:**

- Modify: `project_state/schemas/result_envelope_v2.schema.json`
- Add: `project_state/schemas/return_profile_v1.schema.json`
- Modify: `project_state/plans/server_maintenance.md`
- Modify: `scripts/pfmval_views.py`
- Test: `tests/test_pfmval_knowledge_documents.py`
- Test: `tests/test_pfmval_state.py`

**Interfaces:**

- Consumes: result envelope、job/attempt/recovery 身份、原始 stdout/stderr、预测 CSV、training history、best epoch proof。
- Produces: `return_profile`、`expected / returned / missing / not_applicable` 清单、用户可读回传摘要、`RETURN_COMPLETE / RETURN_INCOMPLETE / RETURN_INVALID` 判定。

- [ ] **Step 1: 定义回传分级**
  - `always`：每次必回；
  - `success`：成功训练必回；
  - `fit_started`：一旦开始拟合即必回；
  - `failed_before_fit` / `failed_after_fit`：按终态分流。
- [ ] **Step 2: 把关键产物写入标准矩阵**
  - 必含：`return_summary.json`、`RETURN_README.md`、stdout/stderr、resolved_config、resolved_argv、terminal event；
  - 成功训练再加：metrics、training_history、逐 split 原始预测、best epoch proof、checkpoint inventory；
  - 不适用与缺失严格分开，不允许混写。
- [ ] **Step 3: 固定“漏包强回归样例”**
  - 以 W004 的 `runner.log` 漏包问题作为主样例；
  - 验证 `.gitignore`、`git add --force`、远端 tree 再验证与补传 revision 机制。
- [ ] **Step 4: 增加用户可见摘要**
  - 必须先展示 `required / returned / missing / not_applicable`；
  - 再讨论导入、补传或排障；
  - 不能只报 bundle SHA。
- [ ] **Step 5: 通过回归测试确认回传合同**

Run: `pytest tests/test_pfmval_knowledge_documents.py tests/test_pfmval_state.py -q`
Expected: 回传包矩阵、摘要字段、漏包判定、补传边界全部通过。

---

### Task 3: 本地与服务器脏工作区收口

**Files:**

- Modify: `project_state/plans/server_workspace_sync_troubleshooting_governance_review_draft_v001_20260808.md`
- Modify: `project_state/plans/server_maintenance.md`
- Modify: `project_state/plans/gitee_numbered_workspace_protocol_v001_20260726.md`
- Modify: `project_state/workspace_registry.json`
- Test: `tests/test_user_project_guide.py`

**Interfaces:**

- Consumes: W001～W005 工作树清单、服务器现有文件、主分支成果文件、历史复制残留。
- Produces: 一份可执行的分类清单：keep / migrate / ignore / retire / protect。

- [ ] **Step 1: 做一次统一 inventory**
  - 扫描 W001～W005；
  - 统计 tracked / untracked / ignored / legacy copy / server-only five 类文件；
  - 标出当前 W004 允许暂存的历史残留与必须立即阻断的新增垃圾。
- [ ] **Step 2: 先定规则，再定动作**
  - 可留在工作区的，只能是当前轮次确实需要的文件；
  - 可忽略的，必须进入明确规则；
  - 不再参与回传的弃用文件，统一标记为 retire，而不是靠临时手工判断。
- [ ] **Step 3: 把“当前轮次豁免”写清楚**
  - W004 当前实验可继续推进；
  - 但不得继续扩散新无关文件；
  - 本轮结束后再统一收口。
- [ ] **Step 4: 输出治理落点**
  - 形成 cleanup / ignore / migrate / protect 四类清单；
  - 为后续正式治理准备审核入口；
  - 不直接删除受保护资产。
- [ ] **Step 5: 用测试或审查清单验证**

Run: `pytest tests/test_user_project_guide.py -q`
Expected: 工作区规则、迁移边界、临时豁免和保护边界与现有项目口径一致。

---

## Out of Scope for Step 1

- 主分支全面重构。
- 完整模块化拆分与长期复用封装。
- 训练协议变更与正式候选门控重设。
- 大规模目录迁移或自动删除旧资产。

## Exit Criteria

- 服务器调试有一条稳定、可复现、可回执的最短闭环；
- 回传包能明确告诉用户“收到了什么、缺了什么、为什么缺”；
- W001～W005 与服务器侧的脏工作区能先分层归类，再进入下一轮统一治理。

## 2026-08-10 实施登记

### PR1：路径登记（Path Registration 1）— COMPLETED_LOCAL

- `configs/server_paths.yaml` 已登记 governance checkout、持久 `W###` 源码、外置 run、return、diagnostic、runtime bundle 六类 path id。
- `pfmval_deploy_git` 已降为 legacy/no-new-output；`server_automation_worktrees` 仅作 deprecated 兼容。
- active `server_maintenance.md` 已写入路径职责与首次服务器使用前的现场核验停止点。

### WB2：工作树绑定（Workspace Binding 2）— COMPLETED_LOCAL

- `ensure_job_worktree()` 已从 job-id 目录改为 `server_experiment_worktrees/W###` 持久目录；attempt 输出继续位于外置 `server_experiment_runs/W###/A###`。
- Workspace Registry schema 已增加 `abandoned_reserved`；W005 已登记为 experiment id 空、lease/attempt 空、编号/branch/HEAD/path 保留，任何 job/lease 都必须拒绝。
- 新实验必须使用 `next_workspace_number=6` 对应的 W006 或更高编号。

### RR3：结果回传（Result Return 3）— COMPLETED_LOCAL

- 已新增 `return_profile_v1.schema.json`；新 job 自动要求原始 `raw_training_csv` 与 `raw_training_txt`。
- 成功/失败 attempt 必须回 terminal JSON + 原始训练 CSV/TXT；incomplete 回 terminal JSON + 可取得的原始 CSV/TXT；产生预测的成功 attempt 仍需逐 split 原始预测表。
- publisher 已对精确 revision path 使用 force-add，并在提交前比较 expected/staged tree，防止 `.gitignore` 漏掉 CSV/TXT/日志。

### HT4：哈希分层（Hash Tiering 4）— COMPLETED_LOCAL

- critical 继续 SHA-256 HARD；supporting/diagnostic 改为必传清单、精确大小和 Git tree 闭包。
- diagnostic record 不再逐文件计算 SHA-256；supporting 文件不会被写入 artifact SHA。
- bundle 身份只覆盖规范化 result manifest 与 critical artifacts；supporting/diagnostic 不得作为 accepted metric 来源。

### LP5：现场试点（Live Pilot 5）— PENDING_SERVER

- 本地实现与目标测试已完成，但尚未连接服务器或训练。
- 新实验正式训练前仍需一次无训练 Gitee 往返：exact fetch、W006 或后续 `W###` 绑定核验、diagnostic、含被忽略 CSV/TXT 的最小 return、remote tree 对账。
- 只有现场试点通过，才能把本计划的服务器闭环从 `COMPLETED_LOCAL` 提升为 `VERIFIED_SERVER`。
