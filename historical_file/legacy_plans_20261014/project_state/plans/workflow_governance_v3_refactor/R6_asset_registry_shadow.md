# R6：Asset Registry 只读 Shadow

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R6`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `R0`, `R5b`
> implementation: `local_D/I_accepted; O=PASS; S=N/A`
> boundary: 只做 inventory 和候选分类，不移动、删除、解禁或改变保护。

## 1. 目的

将实验资产按 custody、lifecycle、mutability 和 evidence role 管理；基于
真实引用血缘区分 pre-MPP、MPP active、shared dependency 和 mixed roots，
为后续依赖抽取、退役和分级解禁提供只读事实。

## 2. Asset Schema

至少包含：

- `asset_id`
- `path`
- `era`: `pre_mpp | mpp_active | shared_dependency | uncertain | legacy_mixed`
- `custody`: `user | project | agent | external`
- `asset_class`
- `lifecycle`
- `mutability`
- `evidence_role`
- `tracking`
- `hash_policy`
- `owner_plan_id`
- `owner_experiment_id`
- `replacement`
- `retention_policy`

Document、Experiment、Workspace、Asset Registry 各守职责，不复制彼此状态。

## 3. 分类原则

- `pre_mpp`：未被当前 MPP 主线引用的旧模型、三患者方案、入口和结果。
- `shared_dependency`：创建时间虽早，但仍被现役 MPP 引用。
- `mpp_historical`：MPP 时代 rejected/superseded 的证据，保留但不支撑当前
  结论。
- `mpp_active`：active manifest、运行代码、accepted evidence 和 active
  plans。
- 不确定或混合目录先标 `uncertain/legacy_mixed`，不得强行二分。

## 4. 步骤

1. 新增 schema 和空 Registry。
2. 只读扫描 tracked、ignored、本地 inventory 和经 Gitee 回传的服务器
   inventory。
3. 默认只读取 metadata/path/size，不读取大文件内容。
4. 只有 E0 或用户明确指定的关键资产才计算内容 SHA。
5. 基于 import、manifest、job、result 和运行代码引用建立血缘。
6. 首轮 mixed roots 全部保持 `legacy_mixed`。
7. 输出分类候选、冲突和需要用户确认的资产，不自动迁移 lifecycle。

## 5. 风险

- ignored 大目录扫描耗时或误触敏感文件。
- 按日期而非引用血缘误判 pre-MPP。
- shared dependency 被错误退役。
- 把服务器快照当实时事实。
- Registry 复制 Experiment/Document 状态形成分叉。

## 6. 检验

```powershell
python deploy/pfmval_ops.py asset inventory --mode shadow
python deploy/pfmval_ops.py asset validate
python -m pytest tests/test_asset_registry.py -q
```

必须覆盖：

- shadow 默认不读取大文件内容。
- pre-MPP 分类基于引用血缘。
- `histogene.utils` 等现役依赖被识别为 shared dependency。
- mixed roots 不进入 fully managed。
- 敏感 token/cache 内容不进入日志或 Registry。
- 各 Registry 不复制状态。

## 7. 退出条件

- schema、空 Registry 和 shadow inventory 通过验证。
- 所有不确定资产保持保护。
- 用户可审查 pre-MPP/shared/mpp-active 候选。
- 没有移动、删除、改写或解禁。

## 8. 回退

删除 shadow Registry 和扫描命令；文件位置、权限和 Git 状态不变。

## 9. 待用户确认

- [ ] 2026-07-06 pre-repair MPP1–5 accepted legacy 结果的长期分类。
- [ ] ignored checkpoint 是否只 inventory，不做全量 SHA。
- [ ] 物理归档/删除继续保持单独批准。

## 10. 补充记录

- `2026-07-26 / DIR-20260726-004`：纳入用户仓库导航所需的目录角色字段。
