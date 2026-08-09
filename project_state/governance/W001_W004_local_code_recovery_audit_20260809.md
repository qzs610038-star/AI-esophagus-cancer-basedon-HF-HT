# W001–W004 本地代码回收审计

> 实现提交：`66c06a0`

## 审计结论

- W001：注册分支和结题分支均已进入 main，独有提交为 0；实验为 `done/accepted/closed_no_retry`。但 main 的 attempt event 仅保留四条 `ATTEMPT_PREPARED`，关闭预检仍会报告非终态阻塞。
- W002：实际 HEAD `05df5de` 已进入 main，独有提交为 0，工作树干净；Registry HEAD 已修正。
- W003：实际 HEAD `28e3aef` 已进入 main，独有提交为 0，工作树干净；Registry HEAD 已修正。
- W004：source `b898c9d` 和 posttrain `c4143e6` 均未进入 main；posttrain 分支有 44 个独有提交，禁止整体合并。

## W004 结果边界

W004 分支记录为 `running/pending`，active attempt 为 A004：

| attempt | returned | validated | imported |
|---|---|---|---|
| A001 | 旧链路未记 RESULT_RETURNED | 未单列 | 是 |
| A002 | 是 | 是 | 否 |
| A003 | 是 | 是 | 否 |
| A004 | 是 | 是 | 否 |

因此本轮只回收代码和必要小型配置，不导入或接纳 A002–A004，不把 W004 标为完成，也不移除其工作树。

## W004 回收范围

- 回收：14 个 Phase 3 核心 Python 模块、5 个对应测试、4 个小型配置/合同文件、结构化日志索引和 closeout 骨架。
- 暂不回收：2 个 split 生成脚本、247 个 CSV、server operation card、结果包、原始日志、checkpoint 和缓存；完整内容继续由 W004 分支 Git 历史保存。
- 公共目录改动：测试证明 W004 仅缺少 `build_result_v2` 的可选
  `prediction_return` 透传，已最小晋升到 `scripts/pfmval_governance.py`；
  `deploy/` 与 `scripts/pfmval_result_bundle.py` 的其余旧分支改动未合入。

## 操作边界

本轮不操作服务器、不推送远端、不训练、不导入结果、不删除分支或 worktree。实际移除 W001–W003 前，必须再次回显精确绝对路径并获得用户批准。

## 关闭预检结果

- W001：`blocked`，唯一 blocker 为 `nonterminal_attempt`。
- W002：`close_ready`，blockers 为空，仍未获得精确路径移除批准。
- W003：`close_ready`，blockers 为空，仍未获得精确路径移除批准。

## 最小验证

- W004 五组 Phase 3 测试、公共结果接口回归及 workspace close-preview 测试：`68 passed`。
- 严格门禁：PASS 7、WARN 7、FAIL 0；WARN 均为既有项目事项。
- W006 示例代码路径已放行，`logs/raw/` 仍被 Git 忽略。

## 用户批准后的物理关闭记录（2026-08-09）

- 用户明确批准移除精确路径 `D:\AI空间转录病理研究\PFMval_new_governed_workspaces\W002` 与 `D:\AI空间转录病理研究\PFMval_new_governed_workspaces\W003`。
- 移除前复核两者均为已登记 Git worktree、普通目录且工作区干净；父目录和 W 编号与批准目标一致。
- 已使用 `git worktree remove` 仅移除上述两个工作树；未使用 `--force`、全局 prune 或递归清理。
- 移除后两个物理路径均不存在，Git worktree 清单中均无残留；对应分支继续保留。
- Workspace Registry 将 W002、W003 标记为 `tombstoned`，永久保留编号、分支及 Git 历史。
