# 科学记录与长期维护 v1 升级计划

> 状态：`proposed_for_new_goal`  
> 范围：科学决策、产物合同、派生视图和文档维护接口。  
> 禁止：重写已接纳科学结论、重新训练、自动关闭 directive、自动迁移或删除资产。

## 目标

把 G13/G14 暴露的多次输入、产物缺列、派生视图缺失和文档扫描耦合收敛为可复用公共接口。

## 实施顺序

### P0：`science_decision_v1`

- 用一个决策包取代 claim、negative result、explanation 三份临时 JSON。
- 一次 Schema 校验、一次事务、一个 receipt，内部仍投影为三类 append-only 记录。
- 必须绑定 experiment、单结果或 paired result、import/acceptance event、用户确认和不确定性。
- 重放返回 `already_recorded`；同 ID 不同内容 HARD FAIL。

### P1：科学产物列合同

声明需要配对/分层/bootstrap 的实验，预测表必须预先登记：

- `sample_id`
- `spatial_cluster_id`
- `pathway_id`
- `y_true`
- `y_pred`

preflight 在训练前验证列合同；缺列时一次报告，不得等结果回传后才发现。
历史结果保持 GAP，不追溯伪造标识。

### P2：Registry 与派生视图投影

- 统一 `decision_summary`：主指标、差值、方向、停止规则、不确定性、开放问题。
- paired result 在 Progress/Dashboard 中显示 pair ID、成员、主要差值和明确 next action。
- 终态只允许 `closed_no_retry`、`open_new_hypothesis` 等明确值，不保留
  `validate_attempt` 一类陈旧动作。
- 科学日志是事实源，视图仅作确定性投影。

### P3：views/docs 解耦

- `views refresh --experiments` 只刷新 Registry、Dashboard、Progress、Current State。
- `docs scan` 与 `PROJECT_GUIDE refresh` 为独立命令和独立提交。
- 提交前检查事务预期文件集合；超出范围时 HARD FAIL 或显式提示。
- 验收一个实验不得产生无关的大规模 PROJECT_GUIDE diff。

### P4：长期债务

- 为五个 legacy accepted result 建立明确的 `legacy_envelope_gap`，不得伪造完整 envelope。
- 重复/conflicting barcode 保持受保护资产与独立批准边界。
- 两项债务不得阻塞与其无关的新实验，但必须持续显示 WARN。

## 验收矩阵

- science decision 三类记录的原子写入、回滚、幂等和冲突。
- 单结果与 paired result 两种绑定。
- prediction contract 缺列、重复 sample、cluster 缺失、pathway 顺序漂移。
- Registry/Progress/Dashboard/Current State 二次刷新 byte-stable。
- experiment-only refresh 不修改 document registry 或 PROJECT_GUIDE。
- docs-only refresh 不修改 experiment Registry 或科学记录。

## 完成定义

- G14 同类科学结论只需一个输入包和一个命令。
- 科学结论、阴性路线、停止规则和开放问题在所有视图一致可见。
- 缺失 bootstrap 输入在训练前被发现。
- 实验验收与全项目文档维护可以独立提交、独立回滚。

