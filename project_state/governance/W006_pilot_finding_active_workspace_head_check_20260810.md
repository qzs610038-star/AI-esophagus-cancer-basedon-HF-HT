# W006 试点重要发现登记：active 工作树 HEAD 校验与登记后持续提交流程不兼容

> 登记日期：2026-08-10
> 来源试点：gitee_roundtrip_pilot_w006_v001_20260810（零训练 Gitee 往返试点，W006）
> 状态：`registered_for_unified_governance`（供 main 工作分支回收后统一治理；本登记不改变任何校验行为）
> 关联实施方案：`project_state/implementation_plans/W006-gitee_roundtrip_pilot_实施方案.md` T8 notes ⑦

## 1. 发现摘要

`agent start-check --strict` 的 workflow governance v3 状态校验（`scripts/pfmval_state.py` `validate_state`，约 line 2515-2528）对非 `governance_maintenance` 工作树强制要求：

```
git 工作树实际 HEAD == workspace_registry.json 中该工作树 hosts.local.current_source_commit（严格相等）
```

而工作树登记时（`scripts/pfmval_governance.py` `allocate_workspace`，约 line 292）将 `current_source_commit` 写死为绑定 `source_commit`，全仓无任何命令更新该字段。

**冲突**：W006 按绑定指令"所有步骤在 W006 分支提交（本地提交即可）"在登记后持续提交（12 个 commit，HEAD 从 `42b2643` 前移至 `f06a087`），导致 HEAD ≠ registry 记录 → strict start-check 报 `absolute workspace HEAD does not match registry`（FAIL）。

## 2. 为什么不可自行闭合（SHA 自引用环）

- 尝试"更新 registry 为当前 HEAD 再提交"会产生**新提交**，HEAD 又前移 → 再次不等；
- 要严格相等，必须 registry 记录 = 包含该记录的提交自身的 SHA，而 SHA 由内容（含记录）决定 → 自引用，数学上不可收敛；
- 既有工作树 W001/W004/W005 的 `current_source_commit` 恰好等于各自分支 HEAD，是因为它们**登记后从未再提交**——校验器隐含假设"登记后工作树 HEAD 冻结不再提交"，W006 试点首次触发该假设失效。

## 3. 影响面

- 影响：W006 工作树内 strict start-check 含 1 个该原因 FAIL（其余 3 个 FAIL 为部署方案正文不同步/legacy 本地路径的工作树环境差异，已按用户处置 A 批准）。
- 不影响：主仓 main 同 HEAD 基线始终 PASS=7 WARN=7 FAIL=0；本试点 final_verdict=CONDITIONAL GO（用户已裁决采用方案 A）；不涉及训练、结果导入、证据 accept。
- 设计侧注：`validate_state` 中 `task == "knowledge"` 路径已将同一检查降级为 WARN（line 2524-2525），说明设计者已知该检查对活跃工作树过严，但 general 路径仍 raise。

## 4. 建议治理方向（供 main 回收后统一决策，不在此试点执行）

1. 将相等检查放宽为"工作树 HEAD 必须是 `current_source_commit` 的祖先链后代"（`git merge-base --is-ancestor <current_source_commit> <HEAD>`），或对 active/shadow 工作树降级为 WARN（对齐 knowledge 路径既有行为）；
2. 或为工作树登记引入"HEAD 重新绑定/确认"命令（如 `workspace rebind --head <sha>`），使登记后合法提交可显式同步 registry；
3. 或调整工作树生命周期语义：登记后工作树应切换为"不可再提交"或显式 close 后校验跳过。
4. 任选其一均需：独立治理任务、用户批准、回归测试覆盖（tests/test_pfmval_state.py 相关用例）。

## 5. 试点内已采取的处置（不修改治理代码）

- 已在实施方案 T8 notes ⑦ 与本节登记该发现；
- W006 内 strict start-check 的该 FAIL 记录为"已知校验冲突"，不作为试点新增 FAIL 判据；
- workspace_registry 中 W006 `current_source_commit` 已按当前 HEAD 快照更新（语义见实施方案 T8 证据回填），后续治理定型后统一校正。
