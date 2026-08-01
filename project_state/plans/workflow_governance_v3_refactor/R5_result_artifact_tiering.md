# R5：结果 Artifact 分级与事务导入

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R5`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `R3`, `R4`
> implementation: `local_D/I_accepted; O=NOT RUN; S=N/A`
> boundary: pack 或返回不等于 import，import 不等于跨协议接纳。

## 1. 目的

只严格校验支撑实验选择和最终结论的结果；日志、环境探测和可重建图表不再
造成无关阻断，同时保持 result/job/attempt/contract 的不可篡改绑定和本地
幂等事务导入。

## 2. Result Envelope v2

artifact 增加：

- `evidence_role`: `critical | supporting | diagnostic`
- `retention`
- `source_attempt_id`

分类：

- critical：metrics、selection proof、逐样本原始 predictions、calibrator、最终
  checkpoint 或其受控 path/size/SHA 记录。
- supporting：training history、可重建表格和图表。
- diagnostic：stdout/stderr、环境探测和排障输出。

规则：

- critical 必须 SHA-256。
- supporting 默认 size+inventory；若用于选择或报告结论则升级 critical。
- diagnostic 不进入 accepted metrics。
- 会产生预测结果的成功作业必须为每个实际评估 split 内联回传
  `raw_prediction_table`；不得将 prediction 放入 `large_artifacts`。
- 特征缓存、embedding/中间特征、checkpoint/权重等大文件可只登记
  server path、size、SHA-256 与 retention/recompute policy。
- stdout/stderr 等日志仅按 Agent 分析需要回传，保持 diagnostic、非证据。

## 3. 步骤

1. 保留 v1 importer，增加 v2 双读。
2. pack 阶段验证 artifact 分类、大小上限、路径和 attempt 绑定。
3. 服务器通过 Gitee 返回 envelope、指标、selection proof 及每个已评估
   split 的逐样本原始 prediction；大型非 prediction 资产记录受控 path、
   size、SHA 和复算/保留策略。
4. 本地 fetch 到 quarantine，不 merge 到实验分支或 main。
5. verify 后执行事务化 import。
6. import 原子更新 Experiment Registry、完整 Dashboard、用户进度表和
   Current State source hashes。
7. 相同 result ID + bundle SHA 重复导入为 no-op；同 ID 不同 SHA 硬阻断。

## 4. 风险

- best epoch 只在 history 中却被降级。
- predictions 被误归入 large artifact，导致本地无法复核逐样本结果。
- supporting 后来用于结论但未升级。
- 事务只更新 Registry，用户视图仍旧。
- Gitee 冲突触发重新训练。

## 5. 检验

```powershell
python -m pytest tests/test_result_envelope_v2.py -q
python -m pytest tests/test_result_import.py -q
python -m pytest tests/test_experiment_progress_view.py -q
```

必须覆盖：

- critical 缺 SHA 或被篡改时 FAIL。
- diagnostic 不进入 accepted metrics。
- formal selection proof 缺失时 import FAIL。
- prediction 缺失、split 绑定缺失或被登记为 large artifact 时 HARD FAIL。
- 特征缓存、checkpoint 等大型非 prediction 资产允许只登记 path/size/SHA。
- 相同 bundle 幂等；同 ID 不同 SHA HARD FAIL。
- 导入中断后 Registry 与两类视图全部恢复或全部完成。
- result 重传不消耗 run units，不重新训练。

## 6. 退出条件

- v1/v2 双读测试通过。
- critical/supporting/diagnostic 分类可审计。
- quarantine、verify、import 之间没有 merge 或直接接纳。
- 用户/Agent 两类视图与 Registry 原子一致。

## 7. 回退

importer 切回 v1 全 artifact SHA 模式；保留 v2 bundle 只读，不覆盖或转换
原文件。

## 8. 待审查

- [x] 所有实际评估 split 的逐样本原始 predictions 必须随结果包回传，
  不允许仅登记服务器资产（2026-08-01 用户指令）。
- [ ] 用户进度表允许显示的核心指标集合。

## 9. 补充记录

- `2026-07-26 / DIR-20260726-004`：把用户进度表纳入 import 原子事务。
