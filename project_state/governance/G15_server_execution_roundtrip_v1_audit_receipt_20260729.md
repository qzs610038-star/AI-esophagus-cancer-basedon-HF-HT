# G15 项目级服务器执行往返 v1 审计回执

> 审计对象：本地实现、测试、状态门禁与任务分支发布准备
> 实现工作树：`W002`
> RED：`089232b`（5 failed / 1 passed）
> GREEN：`1d33977`（定向与关联回归 37 passed）

## 1. 事实边界

| 证据 | 分类 | 说明 |
|---|---|---|
| 源码、Schema、CLI 与 fixture | current / static | 证明实现存在，不证明真实服务器运行 |
| 204 个本地测试 | current / local runtime | 含 bare remote Git 发布、幂等与冲突模拟 |
| strict start-check | current / local governance | 要求最终 `FAIL=0` |
| G12 review | historical regression evidence | 只用于 14 blocker fixture，不是实现归属 |
| 真实服务器/Gitee 执行回执 | NOT RUN | 本任务明确禁止 |
| 训练、run unit、result import/accept | NOT RUN | 本任务明确禁止 |

## 2. 验收

| 检查 | 结论 | 证据 |
|---|---|---|
| W002 独立、永久、不复用 | PASS | registry 分配 W002，`next_workspace_number=3` |
| TDD 先 RED 后 GREEN | PASS | `089232b` → `1d33977` |
| 最小 execution bundle | PASS | closure、排除项、双构建同 bundle ID |
| fingerprint 可复用/可过期 | PASS | 59 秒复用，61 秒刷新 fixture |
| preflight 不 fail-fast | PASS | 一次返回 17 个独立 blocker；tamper 与主机缺失同时出现 |
| 单入口 | PASS | `execute-v1 --bundle` / `run_server.ps1` |
| 重放幂等 | PASS | executor/package/publish 均只调用一次，run_units 恒为 1 |
| terminal 恢复不重训 | PASS | terminal fixture 下 executor 调用即使测试失败，但实际未调用 |
| 同 ID 不同 SHA | PASS | HARD FAIL |
| result 协议复用 | PASS | 调用既有 build/validate/publish `result_bundle_v1` |
| bare remote FF/幂等/冲突 | PASS | 首次 publish、重复 already_published、错误 parent 阻断 |
| legacy v1/v2 只读兼容 | PASS | 关联回归 37/37；全量 204/204 |
| protected dirs | PASS | Git diff 未包含 `histogene/`、`egnv1/`、`egnv2/` |
| 真实服务器与训练边界 | PASS | 均 NOT RUN；未生成新 attempt/result ref |

## 3. 新鲜命令证据

```text
python -m pytest -q
204 passed, 2 warnings in 62.06s

python deploy/pfmval_ops.py agent start-check --strict
PASS=7 WARN=6 FAIL=0
```

最终 strict gate、W002 clean 状态和 Gitee 任务分支远端 SHA 在最终文档提交后重新执行并补入最终回执。

## 4. 判断

总体结论：**CONDITIONAL GO**。

- 对“W002 本地实现完成、进入最终 strict gate 与任务专用 Gitee 分支发布”无代码或测试 blocker。
- 对“真实服务器启用或训练”仍为 **NOT RUN / 未授权**；必须按迁移边界另行批准。
- 当前保留的 condition：最终文档提交后需重新取得全量测试、strict `FAIL=0`、clean worktree 与远端 SHA 四项新鲜证据。
