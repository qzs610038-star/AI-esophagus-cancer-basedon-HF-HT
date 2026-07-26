# R0：冻结基线与兼容样本

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R0`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: none
> implementation: `local_D/I_accepted; O=PASS; S=N/A`
> boundary: 只增加只读 fixture、inventory 和基线报告，不创建、迁移或删除工作树。

## 1. 目的

在任何 schema、工作树、门禁或 Registry 重构前，冻结可重复的 v1 行为与
现有异常，使后续每一步都能区分“预期变化”和“意外回归”。

## 2. 前置条件

- 当前 HEAD、branch、dirty/untracked 状态已记录。
- strict gate、paths validate、docs scan 输出已保存。
- v1 job/result 和现有 worktree 只读可访问。
- 不把服务器快照当成实时事实；需要时只登记“待服务器实时核验”。

已知基线锚点必须同时保留：

- revision 88 的历史重现为 `PASS=5 WARN=6 FAIL=1`，唯一 FAIL 是 Document
  Registry 落后；
- 低风险清理后以及本轮 revision 101 的新鲜 strict gate 均为
  `PASS=6 WARN=6 FAIL=0`。

R0 不得用较新的全绿结果覆盖历史差异，也不得把历史 FAIL 当作当前仍存在。

## 3. 影响面

- `tests/fixtures/` 或等价只读 fixture 目录；
- v1 job/result schema 兼容测试；
- 本地 `git worktree list --porcelain` inventory；
- 当前 Document/Experiment Registry 与生成视图基线。

不修改运行代码、服务器路径、工作树或实验结果。

## 4. 步骤

1. 保存当前 Git locus、state revision、source hashes 和 active manifests。
2. 清点本地 worktree、automation job/result/attempt 数量，不自动收养。
3. 选取 v1 fixture，至少覆盖：
   - preflight；
   - smoke；
   - formal；
   - success；
   - failed；
   - incomplete；
   - legacy accepted result 缺完整 envelope。
4. 每个 fixture 记录来源路径、原始 SHA、source commit、schema version 和
   evidence boundary。
5. 保存当前用户入口、Dashboard 和 historical 文档索引问题，作为 P0B
   回归样本。
6. 建立“允许保留的既有 WARN”清单，不通过改写历史伪造全绿。

## 5. 风险

- fixture 未覆盖历史特殊 result。
- 将 ignored 或服务器资产误登记为 Git 事实。
- 把当前 strict 结果覆盖为历史 revision 的结果。
- fixture 含敏感凭据或大文件。

## 6. 检验

```powershell
git status --short --branch
git rev-parse HEAD
python deploy/pfmval_ops.py agent start-check --strict
python deploy/pfmval_ops.py paths validate
python deploy/pfmval_ops.py docs scan
python deploy/pfmval_ops.py job validate --manifest automation/jobs/mpp2-pathway-ridge-calibration-20260717-r003/job.json
python -m pytest tests/test_pfmval_state.py -q
```

附加测试：

- fixture 只读；
- fixture 中无 token、密码或大模型权重；
- v1 validator 对冻结样本输出稳定；
- 当前 README/Dashboard 的已知不一致能被测试捕获，而非静默接受。

## 7. 退出条件

- 样本覆盖上述七类状态。
- 每个样本有来源和 SHA。
- 基线报告区分 current 与 historical。
- strict gate 的既有 WARN 得到准确归因。
- 未创建、删除、收养或迁移任何 worktree。

## 8. 回退

删除本阶段新增 fixture/inventory 即可；不得改写原 job/result、Registry 或
历史事件。

## 9. 待审查

- [ ] fixture 的最终存放目录。
- [ ] 是否将本地现有 worktree 数量只记录摘要，避免路径成为跨机器事实。

## 10. 补充记录

- `2026-07-26 / DIR-20260726-004`：首次拆分为独立阶段文件，并纳入 P0B
  现有用户入口回归样本。
