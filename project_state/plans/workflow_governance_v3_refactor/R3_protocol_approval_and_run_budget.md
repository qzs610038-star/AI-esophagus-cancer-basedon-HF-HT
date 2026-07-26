# R3：实验注册、协议级批准与次数预算

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R3`
> module revision: `001`
> lifecycle: `pending_review`
> dependencies: `R0`, `R1`; enforcement depends on `R4`
> implementation: `not-started`
> boundary: 本阶段设计不构成任何具体实验的训练批准。

## 1. 目的

将批准从单个 job 提升为受控的实验协议 revision，同时保持每个 attempt 的
完整 commit/job 追踪；在注册或批准前默认询问用户允许的结果性执行次数，
不以“一次一批”隐藏多个拟合。

## 2. Server Job v2

新增字段：

- `workspace_id`
- `attempt_id`
- `protocol_revision`
- `approval_id`
- `critical_contract_sha256`
- `workspace_branch`
- `input_binding`
- `run_units`
- `adaptation_record_id`（可选）

兼容规则：

- v1 继续只读验证和 result import。
- v1 不自动原地升级。
- 新 dispatch 只生成 v2。
- 迁移生成新文件并保留旧 manifest SHA。

## 3. 批准合同

用户批准至少绑定：

```text
experiment_id
protocol_revision
phase
run_limit
critical_contract_sha256
adaptation_policy
```

每个 attempt 继续绑定：

```text
attempt_id
job_id
source_commit
resolved argv
input binding
run_units
```

`run_units` 表示本 job 实际包含的独立模型/校准拟合数量。

## 4. 次数消费

- preflight、dry-run、参数解析、路径/编码/解释器适配和纯结果重传不消耗。
- 数据与关键合同验证通过并进入 `EXPERIMENT_STARTED` 后开始消耗。
- 一个 seed、一个模型拟合或一次校准拟合通常各计 1。
- 脚本内三个 seed 必须 `run_units=3`。
- 开始后即使失败也计数。
- Gitee 冲突、重打包、重复 import 不得消耗，也不得以重新训练解决。
- 预算耗尽后必须重新询问用户。

## 5. E1 工程适配

可候选免重复批准：

- 注册路径、分隔符、大小写；
- UTF-8、CRLF/LF、日志编码；
- launcher 参数传递和显式兼容 alias；
- Python/import bootstrap 与环境入口；
- stdout/stderr、退出码；
- packaging、Gitee 回传和幂等 import。

同时必须：

- 新 source commit、dispatch revision 和 adaptation record 完整记录；
- critical contract digest 不变；
- diff 只落在实现 PR 批准的 E1 allowlist；
- 针对性测试和 dry-run 通过；
- patch 经 server-owned ref 回传并由本地 W 接纳。

loss、超参、正则、关键 feature、数据/split/z-score、checkpoint/calibrator、
模型结构、训练逻辑、选择或评估策略、phase 任一改变，批准立即失效。

## 6. 风险

- 关键合同提取遗漏，导致旧批准误复用。
- 一个 job 隐藏多个拟合。
- E1 范围过宽。
- attempt budget 在异常恢复中被重复扣减或绕过。
- R3 先于 R4 enforcement，出现无可靠 digest 的协议批准。

## 7. 检验

```powershell
python -m pytest tests/test_experiment_approval_v2.py -q
python -m pytest tests/test_server_job_v2.py -q
```

必须覆盖：

- 注册时缺 run limit 阻断并要求用户输入。
- 文档/注释变化不改变合同。
- loss/default hyperparameter/regularization/feature flag 改变使批准失效。
- data manifest/checkpoint 改变使批准失效。
- smoke → formal 必须新批准。
- 超预算阻断。
- 三 seed 单脚本消耗三个 run units。
- E1 合同不变可在原预算继续，但不覆盖旧 manifest。
- `EXPERIMENT_STARTED` 前后失败的计数差异。

## 8. 退出条件

- experiment register 与 approval schema 有唯一写入口。
- 预算消费 append-only、可审计且幂等。
- R4 critical contract extractor 已通过后，才允许开启 v2 enforcement。
- v1 approval 默认路径可回退。

## 9. 回退

保留 per-job approval 为默认；v2 仅在 feature flag 下启用。已经消费的次数
记录不得倒写。

## 10. 已决与待审查

- [x] 默认询问实际结果性执行次数。
- [x] 同意协议级批准与不可篡改关键合同。
- [x] 严格限定的 E1 适配可免重复批准。
- [ ] `EXPERIMENT_STARTED` 精确事件。
- [ ] E1 文件和参数 allowlist。

## 11. 补充记录

- `2026-07-26 / DIR-20260726-004`：把注册和次数询问纳入 P0A 闭环。
