# R1：Workspace Registry 只读 Shadow

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R1`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `R0`
> implementation: `local_D/I_accepted; O=NOT RUN; S=N/A`
> boundary: 本阶段只建立 schema、Registry 和只读 scan/status；不得创建、删除、收养或切换真实 worktree。

## 1. 目的

建立永久、不复用的 `W###` 身份和本地/服务器逻辑工作区映射，在不改变
现有 job v1 流程的前提下，先以 shadow 模式发现路径、身份和状态漂移。

## 2. Workspace Schema

至少包含：

- `workspace_id`
- `display_name`
- `host_scope`
- `experiment_id`
- `branch`
- `path_id`
- `relative_path`
- `current_source_commit`
- `protocol_revision`
- `status`
- `active_attempt_id`
- `lease`
- `opened_at`
- `last_verified_at`
- `close_eligibility`
- `retention_policy`

约束：

- `W001` 起三位单调递增，tombstone 后也不复用。
- 一个 experiment 只能绑定一个 W。
- 同一 W 可有 local/server 两个 host record，但属于一个逻辑工作区。
- 短名只作显示；物理路径只使用 `W###`。

## 3. 步骤

1. 新增 workspace schema 和空 Registry。
2. 实现 `workspace scan --host local|server` 的只读对比。
3. 实现 `workspace status --json`。
4. 归一化 Windows 盘符、大小写、分隔符和 Junction/real path。
5. 将未登记 worktree、悬空 Registry 和越界路径分级为 WARN/FAIL。
6. 为后续写命令定义显式 `--workspace-id` locus guard。
7. 对话绑定只存在于各对话上下文，不创建全局 current-workspace 文件。
8. shadow 期间不得把扫描结果自动写成 accepted server truth。

## 4. 风险

- 路径归一化把两个目录误认为同一工作区。
- 本地 Registry 推断服务器目录存在。
- W 编号并发分配或复用。
- 未绑定对话误写 main。

## 5. 检验

```powershell
python deploy/pfmval_ops.py workspace scan --host local
python deploy/pfmval_ops.py workspace status --json
python -m pytest tests/test_pfmval_workspaces.py -q
```

必须覆盖：

- 同一路径不同大小写归一化。
- W 编号唯一、单调、不复用。
- 同一 experiment 不得绑定两个 W。
- 未绑定 W 或 cwd 不一致时，模拟写操作阻断。
- 未登记 worktree 与悬空记录仅 WARN。
- 路径越出注册根时 FAIL。
- scan/status 不修改 Git 或文件系统。

## 6. 退出条件

- schema 与空 Registry 通过验证。
- shadow scan 可稳定重复运行。
- 无真实 worktree 副作用。
- W 分配规则和对话 locus guard 测试通过。
- 用户能从状态输出看懂 W、experiment、local/server path 和 dirty 状态。

## 7. 回退

删除 shadow Registry 和命令；原 job v1 流程保持原样。已经分配并对外使用的
W 编号即使回退也不得复用。

## 8. 待审查

- [ ] W 编号分配是否使用 append-only counter 还是 tombstone 最大值加一。
- [ ] server shadow inventory 的 Gitee envelope 格式。

## 9. 补充记录

- `2026-07-26 / DIR-20260726-004`：首次拆分为独立阶段文件。
