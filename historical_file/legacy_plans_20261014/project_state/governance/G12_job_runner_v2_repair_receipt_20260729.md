# G12 前置治理修复回执：job run-v2

## 结论

**LOCAL IMPLEMENTATION PASS；G12 训练仍为 NOT RUN。**

本修复只补齐 job v2 的服务器执行入口，不替换 G12 的 `source_commit`、不改写 protocol/approval/manifests、不启动 A001/A002，也不执行 result import、accept 或 G13。

## 已实现

- 新增 `python deploy/pfmval_ops.py job run-v2`。
- 不再读取旧版 `state_revision`，也不调用旧 `ensure_job_worktree()`；不会创建 `W001-A001`/`W001-A002` 工作树。
- 只接受 `schema_version=2.0` 且具有 `execution_binding` 的新 job；绑定模式固定为 `source_plus_governance_bundle`，包含不可变 `governance_commit`。
- 运行前同时核验治理包 HEAD/clean、显式 source worktree 的 HEAD/clean、训练入口路径及其 SHA-256。
- attempt 运行目录固定为外部 `<run-root>/W001/A001`（或相应 attempt），首次真实执行前写出可回传 `attempt_started.json`；已存在目录直接拒绝，禁止隐式重试。

## 验证

- `tests/test_job_runner_v2.py tests/test_pfmval_governance_v3.py`：`14 passed`。
- 旧 G10 A001/A002 manifest：`job validate` 仍 PASS；但 `job run-v2 --dry-run` 因没有 `execution_binding` 在创建 run root 前拒绝，符合必须重新批准/重建的边界。
- 严格门禁：`PASS=7 WARN=6 FAIL=0`；WARN 均为既有 legacy result envelope/MPP label 债务。
- 全量 pytest：`171 passed, 1 failed, 2 warnings`；唯一失败为既有 `PROJECT_GUIDE.md` 指向未随治理工作树复制的 `CLAUDE.md`，与本修复无关。

## 未完成边界

1. 此提交尚未提交或推送。
2. 现有 G12 `a04319...`、`APR-...-r001`、A001/A002 仍不可用于 `run-v2`；需要后续独立生成新的 protocol revision、approval、critical contract 与带 `execution_binding` 的 manifests。
3. 本次没有创建服务器 run root、没有 `EXPERIMENT_STARTED` 实际事件、没有 result ref，`run_consumed` 仍为 0。

