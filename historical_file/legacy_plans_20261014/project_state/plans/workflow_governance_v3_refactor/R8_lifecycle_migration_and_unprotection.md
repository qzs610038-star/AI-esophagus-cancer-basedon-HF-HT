# R8：生命周期迁移、策略 Hook 与分级解禁

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R8`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `R6`, `R7`, at least one shadow observation cycle
> implementation: `local_D/I_accepted; O=PASS_SHADOW_ONLY; S=N/A`
> boundary: 目录解禁、资产 lifecycle 迁移和 Hook enforce 均需后续独立批准。

## 1. 目的

在资产血缘清晰、共享依赖已抽取并经过至少一轮 shadow 后，将保护从粗粒度
目录名迁移为受跟踪的 asset policy；append-only 标记过时资产，但不把
historical 解释为 deleted。

## 2. 迁移原则

- pre-MPP 非现役资产可候选转 `historical/superseded`。
- shared dependency 不转 historical。
- MPP raw、split、z-score、manifest、group 3/5 embargo 继续 immutable /
  approval-required。
- mixed roots 未拆分前不得 fully managed。
- lifecycle 事件 append-only，不倒写历史。
- 物理移动、归档或删除始终另行批准。

## 3. Hook 最终职责

| Hook | 最终职责 | 失败策略 |
|---|---|---|
| `session-locus` | 报告 branch/HEAD/state/worktree/ref freshness | WARN-only |
| `asset-policy` | 阻止 immutable/append-only/workflow-managed 路径直写 | enforce 前 shadow |
| `critical-contract-drift` | 编辑关键 selector 后提醒批准可能失效 | CLI 最终判定 |
| `workspace-command-guard` | 阻止绕过 CLI 直接 add/remove/prune | 精确命令解析 |
| `closeout-advisor` | 提醒未导入、未关闭、未登记项 | 不自动写 |

Hook 永远不得：

- 接纳结果或批准实验；
- 迁移 lifecycle 或修改 current state；
- 删除工作树、checkpoint 或资产；
- 决定 GO/NO-GO；
- 在 agentmemory 不可用时阻断主任务。

## 4. 步骤

1. 编译只读 asset policy，并以 shadow 日志运行至少一轮。
2. 审查误报、漏报、路径大小写和越界。
3. 为 lifecycle 迁移生成 preview 和逐项证据。
4. 用户按精确资产集合批准后 append 状态事件。
5. Hook 从固定目录名切换到编译 policy。
6. 最后才修改 `AGENTS.md` 的固定目录保护。
7. regenerate 用户仓库导航，显示 current/historical/approval-required。

## 5. 风险

- accepted 历史结果被误标无效。
- local ignored checkpoint 被误写或误删。
- Hook 误拦正常读操作或 fail-open 漏写。
- 目录保护提前移除。
- 用户导航把“可维护”误写为“可随意修改”。

## 6. 检验

```powershell
python deploy/pfmval_ops.py asset validate
python deploy/pfmval_ops.py state validate
python -m pytest tests/test_pfmval_asset_policy.py -q
python -m pytest tests/test_pfmval_state.py -q
python -m pytest tests/test_user_project_guide.py -q
```

必须覆盖：

- historical 不等于 deleted。
- 受保护 MPP 资产继续 immutable。
- mixed roots 不 fully managed。
- Hook 支持绝对/相对路径、大小写和路径逃逸。
- Hook 不写 Registry、不删除文件。
- shadow 不阻断，enforce 只拦批准规则中的写操作。
- 用户导航准确显示 lifecycle 与修改方式。

## 7. 退出条件

- 至少一轮 shadow 观察完成并审查。
- 每项 lifecycle 迁移都有用户批准和 append-only 记录。
- shared dependency 已迁出。
- 固定目录保护最后才调整。

## 8. 回退

恢复目录级保护和旧 Hook；不回滚 append-only lifecycle 事件。新 policy
退回 shadow，不删除 Registry。

## 9. 待用户确认

- [ ] `histogene/egnv1/egnv2` 在依赖抽取后允许何种常规维护。
- [ ] pre-MPP 资产的首批迁移清单。
- [ ] Hook shadow 观察周期。

## 10. 补充记录

- `2026-07-26 / DIR-20260726-004`：加入用户导航同步作为解禁验收。
