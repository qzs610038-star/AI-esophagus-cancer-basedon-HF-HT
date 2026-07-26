# R9：正式关闭与 Cleanup

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R9`
> module revision: `001`
> lifecycle: `pending_review`
> dependencies: preview requires `R1`, `R2`, `R5`, `R6`; execute also requires `R8` and separate approval
> implementation: `not-started`
> boundary: 本文不批准删除任何现有 worktree、branch、checkpoint、缓存或数据。

## 1. 目的

在实验正式完成后提供可审计的关闭预览，并仅在用户批准一个精确解析路径时
移除单个工作树；永久保留 W 编号、Registry tombstone、result import、
branch/tag 和 retained asset 记录。

## 2. Close Ready 条件

全部满足才可进入 `close_ready`：

- 所有 attempt 已终态；
- 无运行或诊断 lease/PID；
- worktree tracked clean；
- 无未推送 commit；
- 无未回传 result；
- result 已 import 或明确 rejected；
- 大型保留资产已有 path、size、SHA 和 retention；
- 用户已审阅关闭摘要。

## 3. 步骤

### 3.1 Preview

`close --preview` 只生成：

- W/experiment/路径解析结果；
- branch、HEAD、dirty、lease、attempt 状态；
- 未完成阻断项；
- 保留资产与清理候选；
- 将刷新的 Registry/用户视图；
- 需要用户批准的精确目标。

preview 无副作用。

### 3.2 Execute

1. 用户明确批准该 W 和解析后的单一物理路径。
2. CLI 重跑全部 close-ready 检查。
3. 创建 preservation tag 或等价不可变记录。
4. 只移除单个注册 worktree；不执行全局 prune。
5. 不连带删除 branch、checkpoint、run root 或数据。
6. append removal tombstone，并刷新 Workspace Registry、用户进度和导航。

## 4. 风险

- 尚有进程或未回传结果时误删。
- worktree 外大型结果被误当 scratch。
- 全局 prune 影响其他实验。
- 路径解析逃逸。
- close 后用户进度仍显示 running。

## 5. 检验

```powershell
python -m pytest tests/test_workspace_close.py -q
python -m pytest tests/test_experiment_progress_view.py -q
```

必须覆盖：

- running lease、dirty worktree、unpushed commit、unreturned result、
  unimported result 均阻断。
- preview 字节可审计且无副作用。
- execute 只作用于一个解析后的注册路径。
- checkpoint、run root 和数据不随 worktree 删除。
- W 编号不复用。
- close 后 Registry 与两类用户/Agent 视图一致。

## 6. 退出条件

- preview/execute 权限和状态转换分离。
- 路径、W、experiment 三者绑定。
- 删除目标可从批准文本唯一解析。
- 无全局 prune、递归清理或隐式资产删除。

## 7. 回退

物理删除前建立 preservation tag 和独立恢复策略。若无法证明可恢复，
`close --execute` 保持 NO-GO。append-only tombstone 不倒写。

## 8. 待用户确认

- [ ] preservation tag 命名。
- [ ] worktree 外 run root 的长期 retention。
- [ ] 是否永远只允许 CLI 执行 removal。

## 9. 补充记录

- `2026-07-26 / DIR-20260726-004`：closeout 增加用户进度和导航原子刷新。
