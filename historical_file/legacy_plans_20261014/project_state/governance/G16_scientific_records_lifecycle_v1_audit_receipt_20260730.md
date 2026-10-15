# G16 科学记录与长期维护 v1 审计回执

> 执行授权：`DIR-20260730-002`；工作树：`W003`；基线：`a0a6dd69dc409c8b3f366ffa41bb980aa2d47b4b`。
> 边界：仅本地治理维护；未训练、未 dispatch、未操作真实服务器、未消耗 run unit、未 import/accept 新结果、未改写 G14 结论。

## 实现说明

- `science decision --input` 接受 `science_decision_v1` 单一输入包，进行一次 JSON Schema 校验后，以一次原子替换投影 `claim`、`negative_result`、`explanation` 三条 append-only 记录，并返回含投影 SHA-256 的回执。
- `result_binding.kind` 明确区分 `single_result` 与 `paired_result`；paired 包仅接受 Registry 中已接纳的 `pair_id`。重放返回 `already_recorded`，同一 decision 的内容冲突或部分写入均为 HARD FAIL。
- `prediction_artifact_contract_v1` 固定五列：`sample_id`、`spatial_cluster_id`、`pathway_id`、`y_true`、`y_pred`。G15 `aggregate_preflight_v2` 在传入该扩展观察值时增加 `prediction_columns`，一次报告所有缺列与重复列。
- 科学日志中的 `decision_summary` 被确定性投影到 Experiment Registry；Progress、Dashboard 与 Current State 由 `views refresh --experiments` 刷新。停止规则 `do_not_retry_unchanged_protocol` 投影为 `closed_no_retry`，其他路线为 `open_new_hypothesis`。
- `views refresh --experiments` 不再扫描文档或刷新 `PROJECT_GUIDE.md`；文档路径改为独立 `docs scan` 和 `docs guide`。

## 测试矩阵

| 检查 | 结果 |
|---|---|
| RED：未实现 `record_science_decision` 时测试收集失败 | PASS，首个 RED 提交 `f5dc0cf` |
| 单结果：单包、三记录、单回执、重放 byte-stable | PASS |
| paired 绑定、冲突拒绝与回滚 | PASS |
| 列合同缺列与重复列一次性 preflight 报告 | PASS |
| decision_summary Registry 投影与二次投影 byte-stable | PASS |
| experiment-only refresh 不改文档 Registry/PROJECT_GUIDE | PASS |
| docs-only scan 不改 Experiment Registry/科学记录 | PASS |
| 全量 pytest | PASS：`209 passed, 2 warnings` |
| strict start-check | PASS：`PASS=7 WARN=6 FAIL=0` |

## 迁移与长期债务边界

- 不自动迁移五个 legacy accepted result；它们持续为 `legacy result envelope` WARN。
- 不伪造历史 `sample_id` 或 `spatial_cluster_id`；历史预测缺失仍是 GAP，只有新训练/新产物在开始前使用列合同。
- 重复/冲突 barcode 仍为受保护资产债务；本任务没有重建、删除或变更标签。
- 文档导航在 W003 的独立 `docs guide` 刷新中排除了本工作树不存在的 ignored 文档链接；未执行 docs scan，因此未改写 Document Registry，也未影响实验或科学记录。
