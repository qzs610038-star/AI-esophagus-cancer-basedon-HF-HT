# G14 最终收口回执：W001 MSE / Huber 配对科学结论

> 日期：2026-07-29  
> experiment：`mpp2_huber_loss_paired_v001_20260728`  
> implementation commit：`9baf0e5`  
> evidence boundary：本回执登记既有 A003/A004 结果的配对验收，不授权训练、重试或服务器操作。

## 结论

- 配对结果 ID：`PAIR-W001-A003-A004`。
- control：A003 / MSE / `W001-A003-result-R004`。
- treatment：A004 / Huber(delta=1) / `W001-A004-result-R002`。
- external XZY pooled PCC：
  - MSE：`0.6548958191047396`
  - Huber：`0.6526906629745176`
  - Huber−MSE：`-0.002205156130222`
- 主门控结论：Huber(delta=1) 未提升 external XZY pooled PCC。
- 17/30 通路提升仅为描述性统计，不参与门控。
- 相同合同不得重试；剩余 2 个 run units 不得因本结果不理想而消耗。

## 治理收口

- 两条 `RESULT_IMPORT_VERIFIED` 均保留。
- 新增一条 `PAIR_RESULT_ACCEPTED`：
  `accept-PAIR-W001-A003-A004`。
- Registry 同时保存两个 member result；不再把 A003 标为被 A004 supersede。
- `supersedes_results=[]`。
- `next_action=closed_no_retry`。
- acceptance event 绑定两个 bundle SHA、共同 approval、critical contract、三条科学记录及明确用户确认。
- 重复执行 `result-pair finalize` 返回 `already_finalized`，不新增事件。

## 验证

- 配对回归测试：`4 passed`。
- 相关治理/科学/视图/result-bundle 测试：`24 passed`。
- 全量测试：`174 passed, 2 warnings`。
- 严格门禁：`PASS=7 WARN=6 FAIL=0`。
- WARN 均为既有 legacy result envelope 与受保护重复 barcode 债务。
- 未启动训练、未创建 A005、未修改远端 result refs、`run_consumed=2/4`。

## 后续边界

- 服务器 execution bundle、聚合 preflight、fingerprint 和单入口状态机另开任务。
- `science_decision_v1`、预测产物列合同、views/docs 解耦和 legacy envelope 维护另开任务。

