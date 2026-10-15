# R5b：Document Registry、用户视图与非实验状态门禁

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R5b`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `R0`
> implementation: `local_D/I_accepted; O=PASS; S=N/A`
> boundary: 已完成的 canonical/adapter/review 分类不代表本模块全部完成。

## 1. 目的

稳定 Document Registry 和状态生成链，消除多写者、重复扫描与无意义
revision 漂移；为总控审查包的子模块、用户仓库导航和简洁实验进度提供
明确 lifecycle 与确定性生成。

## 2. 当前已完成最小切片

- `active_skills` 登记 canonical/adapter。
- scanner 将 `.agents` 标为 active/normative，将 `.claude` 同名薄适配器
  标为 active/reference。
- `pending_plan_reviews` 支持 `approved_design` 与 `pending_review`。
- state validation 阻断缺失或分类错误的 canonical/adapter/review path。

其余内容仍待审查。

### 2.1 本轮新鲜观测到的既有缺口

2026-07-26 对当前实现连续执行 `docs scan --write` 时，200 条文档记录的
`verified_at` 均被重写，Document Registry 文件 SHA 随之变化；lifecycle、
authority 和内容哈希未发生变化。这说明“连续扫描字节级稳定”目前仍是
真实未完成项，不得因为模块登记成功而宣称 R5b 稳定化已经通过。

## 3. 本轮模块化审查包的临时登记方式

当前 scanner 只按 `pending_plan_reviews` 的精确 path 识别审查文件，因此：

- 总控包和每个 P0/R 模块分别登记为 `pending_review`；
- 未登记模块不得由 scanner 回落为 historical；
- 后续可设计 parent review 的 `module_paths`，但迁移前保持当前精确路径
  方案，不同时维护两套关系。

## 4. 步骤

1. 保留人工维护的 `supersedes`、`superseded_by`、`truth_sources`。
2. 内容和分类不变时不重写 `verified_at/state_revision`。
3. pending review 不自动激活。
4. 新增父审查包—模块关系的 schema/validator，或明确继续使用独立 review ID。
5. inventory 漂移默认 regenerate WARN；active plan 丢失、重复或指向禁止通道
   仍 HARD FAIL。
6. `finalize_experiment.py`、`update_doc_registry.py`、
   `check_project_state.py` 收敛到同一库/API/validation policy。
7. 建立 `PROJECT_GUIDE.md` 与用户进度表的 lifecycle、source hash 和生成门禁。
8. 连续两次 scan/sync 必须稳定。

## 5. 风险

- 降级过宽使 active plan 漂移未被发现。
- scanner 把新模块归为 historical。
- 父子关系与独立 review ID 双写。
- 多写入口产生不同 Registry。
- 生成用户入口时误覆盖手写学习内容。

## 6. 检验

```powershell
python -m pytest tests/test_pfmval_state.py -q -k "document or lifecycle or source_hash"
python deploy/pfmval_ops.py docs scan
python deploy/pfmval_ops.py docs scan
git diff --exit-code -- project_state/document_registry.json
```

必须覆盖：

- 连续两次无变化扫描字节级稳定。
- 每个 P0/R 模块均为 pending_review/reference。
- pending review 不被当作 active 执行依据。
- active plan 缺失/重复继续 FAIL。
- 只增加 historical 文档不阻断训练。
- 旧 writer 调同一 API 或给出退役提示。
- 用户入口 active/tracked，且 current 区不链接 historical 文档。

## 7. 退出条件

- 总控—模块关系可机器验证。
- Document Registry 扫描稳定。
- 状态、Registry、README/用户导航和实验视图无多写者。
- 新模块增补流程有测试。

## 8. 回退

保留 v1 scanner 只读入口；恢复旧门禁前先比较 lifecycle，禁止覆盖人工关系。
用户派生视图可重新生成，不影响事实 Registry。

## 9. 待审查

- [ ] parent review `module_paths` 是否进入 schema，还是长期保留独立 review ID。
- [ ] `PROJECT_GUIDE.md` 最终路径。
- [ ] 哪些用户文档从 `.gitignore` 精确放行。

## 10. 补充记录

- `2026-07-26 / DIR-20260726-004`：新增模块化审查包和用户双视图登记要求。
