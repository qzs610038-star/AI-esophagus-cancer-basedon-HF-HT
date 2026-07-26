# R2：实验级单持久工作树与外部 Run Root

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R2`
> module revision: `001`
> lifecycle: `pending_review`
> dependencies: `R0`, `R1`
> implementation: `not-started`
> boundary: 未经独立实施批准，不创建、迁移或删除本地/服务器工作树。

## 1. 目的

把当前按 `job_id` 创建工作树的模式升级为“一个 experiment 对应一个永久
W 工作树，多个 attempt 复用”，并先把日志、PID、lease、checkpoint 和结果
迁移到工作树外的注册 run root。

## 2. 目标布局

```text
<workspace_root>/W017/
<run_root>/W017/A001/
<run_root>/W017/A002/
```

- local/server 是同一逻辑 W 的两个注册路径。
- protocol revision、job、attempt 和 Gitee ref 不创建新 worktree。
- worktree 只放可复现代码；运行产物全部进入外部目录。

## 3. 修改影响面

- `deploy/pfmval_ops.py::ensure_job_worktree`
- `deploy/run_experiment.ps1`
- `configs/server_paths.yaml`
- workspace/job/result schema
- launcher 日志、PID、lease 与输出路径
- Gitee local/server 单写者 ref

## 4. 分步步骤

### 4.1 外部 Run Root 先行

1. 注册 local/server experiment run root。
2. runner 只从 manifest 解析 `<run_root>/W###/A###`。
3. 日志、PID、lease、checkpoint、predictions 和 result staging 全部迁出
   worktree。
4. 增加路径越界、重用非空 attempt 目录和跨 W 写入阻断。

在 4.1 通过前不得修改工作树复用逻辑。

### 4.2 工作树按 W 创建

1. 将 `ensure_job_worktree` 拆为 `ensure_experiment_workspace`。
2. 只从 Workspace Registry 解析物理路径。
3. 新 attempt 前 fetch 并 detached checkout exact source commit。
4. 错误 HEAD、tracked dirty 或未登记输出均阻断。

### 4.3 Lease 与单写者

1. 一个 W 同时只允许一个运行 lease。
2. lease 记录 W、experiment、attempt、commit、critical digest、PID、
   holder、heartbeat 和 run root。
3. stale lease 必须核验进程并记录 orphan evidence 后显式恢复。
4. 本地只写 local ref，服务器只写 server ref；禁止 force push。

## 5. 风险

- attempt A 的 dirty 状态污染 attempt B。
- 输出遗漏迁移后使 worktree 永久 dirty。
- server/local 同时写同一 ref。
- stale lease 被超时静默释放。
- v1 job 查找路径被破坏。

## 6. 检验

```powershell
python -m pytest tests/test_pfmval_workspaces.py -q
python -m pytest tests/test_mpp_manifest_routing.py -q
python -m pytest tests/test_pfmval_state.py -q
```

必须覆盖：

- 同一 experiment 三个 attempt 只创建一个 worktree。
- 不同 experiment 不得复用。
- source commit 更新必须经过 fetch/sync/check。
- dry-run 与模拟真实运行后 worktree clean。
- 所有运行输出位于外部 run root。
- dirty critical path 阻断；误入输出隔离登记，不自动删除。
- lease 未释放时第二 attempt 阻断。
- stale lease 不因时间自动解锁。
- v1 job 继续按 legacy 路径只读验证或给出明确迁移提示。

## 7. 退出条件

- 外部 run root 先通过路径和隔离测试。
- 单实验多 attempt 复用测试通过。
- local/server writer ownership 可审计。
- 没有自动 pull、merge、rebase、force push 或 worktree prune。
- 没有迁移或删除旧 worktree。

## 8. 回退

- feature flag 切回 `workspace_mode=job_v1`。
- v2 worktree 只列为 cleanup candidate，不自动删除。
- 外部 run root 保留只读 inventory；不得搬回工作树覆盖已有内容。

## 9. 待审查

- [ ] local/server workspace root 的最终 path ID。
- [ ] lease heartbeat 周期与 orphan 判定证据。
- [ ] attempt ref 的最终命名。

## 10. 补充记录

- `2026-07-26 / DIR-20260726-004`：首次拆分为独立阶段文件。
