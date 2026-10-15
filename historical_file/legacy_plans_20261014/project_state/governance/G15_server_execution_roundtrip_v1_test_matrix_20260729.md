# G15 服务器执行往返 v1 测试矩阵

> fixture：`tests/fixtures/g12_server_roundtrip_blockers_v1.json`
> 定向测试：`tests/test_execution_roundtrip_v1.py`
> 既有结果协议回归：`tests/test_result_bundle_v1.py`

## G12 14-blocker 映射

| Fixture | 原阻塞 | 固化的回归边界 | 结果 |
|---|---|---|---|
| G12-B01 | runner SHA mismatch | runner 进入内容寻址 bundle，不依赖 dirty main HEAD | PASS |
| G12-B02 | dirty main checkout | bundle build/validate 不使用共享 main 作为执行 locus | PASS |
| G12-B03 | dirty governance worktree |普通不可变 bundle directory 为公共验证对象 | PASS |
| G12-B04 | archive 被当作 Git worktree | bundle 与 source Git 检查分组；缺失主机资产仍生成聚合报告 | PASS |
| G12-B05 | source object 在错误对象库查询 | `source_object` 仅在登记的 source repository 检查 | PASS |
| G12-B06 | 无关历史结果 CRLF 阻断 | bundle closure 排除 `automation/results/**`，逐文件校验实际字节 | PASS |
| G12-R1 | completed/success 混淆 | 复用 `result_bundle_v1` 的 terminal→result 状态映射 | PASS |
| G12-R2 | metrics basename 重名 | 复用 artifact path 唯一性与 closure 校验 | PASS |
| G12-R3 | 单元素数组退化 | 复用 `artifacts` / `large_artifacts` 数组门禁 | PASS |
| G12-R4 | raw CRLF 与 Git LF SHA 不一致 | 复用 raw + Git-normalized 双完整性记录 | PASS |
| G12-RR1 | implementation SHA 被要求等于 tip | bundle 固定实际 runtime 内容；治理分支允许 clean 前进 | PASS |
| G12-RR2 | bare Python 隐藏 stderr | 单入口保留子进程 stdout/stderr/exit code，用户不填解释器与 staging | PASS |
| G12-RR3 | 固定 venv 无 jsonschema | fingerprint 记录 capability-tested backend 并可过期刷新 | PASS |
| G12-RR4 | 无 Python Schema backend | 继续复用 PowerShell `Test-Json -SchemaFile` 严格后端 | PASS |

## 行为矩阵

| 行为 | 证据 |
|---|---|
| 相同 job 重建 bundle 字节身份稳定 | `test_execution_bundle_is_minimal_closed_content_addressed_and_reproducible` |
| 操作员只提供 bundle 路径 | `test_execution_bundle_cli_generates_one_path_operation_card` |
| 两种实验类型无训练构建/验证 | `test_two_distinct_experiment_types_build_and_verify_without_execution` |
| fingerprint 未过期复用、过期刷新 | `test_server_fingerprint_is_reused_until_expiry_and_then_refreshed` |
| 一次返回 17 个模拟 blocker | `test_preflight_v2_aggregates_all_independent_blockers_without_fail_fast` |
| bundle tamper 与主机缺失同时报告 | `test_preflight_v2_reports_bundle_tamper_and_missing_host_assets_together` |
| 重放不新增 run unit / publish | `test_state_machine_replay_and_terminal_recovery_never_rerun` |
| terminal 已存在只恢复封装和推送 | 同上，executor 被设置为调用即失败 |
| 同 job 不同 bundle SHA 阻断 | `test_state_machine_rejects_same_job_identity_with_different_bundle_sha` |
| 治理维护分支 clean 前进 | `test_governance_maintenance_workspace_can_advance_clean_branch_head` |
| bare remote 首次发布、重复幂等、错误 parent 冲突 | `test_publish_is_fast_forward_idempotent_and_verifies_remote_sha` |
| legacy job/result v1/v2 回归 | `tests/test_job_runner_v2.py`、`tests/test_result_bundle_v1.py`、`tests/test_pfmval_governance_v3.py` |

## 新鲜执行结果

```text
python -m pytest -q
204 passed, 2 warnings in 62.06s
```

两条 warning 均为 `sklearn` 在少于两个样本时 R² 未定义的既有兼容性测试提示；无 test failure。
