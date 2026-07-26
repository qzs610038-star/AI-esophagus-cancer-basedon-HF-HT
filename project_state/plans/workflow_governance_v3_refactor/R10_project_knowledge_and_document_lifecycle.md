# R10：项目知识与文档生命周期

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R10`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `P0C`, `R5b`
> authorization: `DIR-20260727-001`
> implementation: `local_D/I_accepted; O=NOT RUN; S=N/A`
> boundary: 复用现有本地状态与 Document Registry；不新增第二文档 Registry，
> 不接入团队、文档、云备份、论文、文献或科研管理平台。

## 1. 为什么合并

非实验项目事实、组会简述、长期结论、学习指南、论文材料和文档生命周期都
依赖同一组能力：来源引用、变更预览、用户确认、版本替代、freshness 与
lifecycle。它们只通过 `document_profile`（文档类型配置）区分，不各建 Skill、
Registry 或治理引擎。

本模块包含六个子工作流：

1. `project_fact`：项目目标、分工、任务、队友成果和沟通决议；
2. `meeting_brief`：当前只生成组会简述 Markdown；
3. `research_path`：长期重要结论、阴性结果、探索路径和开放问题；
4. `learning_guide`：本轮只保留学习指南维护接口；
5. `paper_output`：本轮只保留等待学长模板的论文材料接入接口；
6. `lifecycle_review`：文档与约束的 freshness/lifecycle 复核。

## 2. 共享本地合同

共享字段按 profile 取最小子集：

- 永久 ID、profile、标题/主体和当前状态；
- source refs：directive、fact、experiment、accepted result、源码或上版文档；
- `supersedes/superseded_by` 与用户确认状态；
- `lifecycle`：draft、active、superseded、historical；
- `freshness`：fresh、review_due、stale、unknown；
- observation / interpretation / recommendation 的内容类别。

本轮不把外部调研缺失项 3–8 转成字段或门禁。

## 3. 子工作流

### 3.1 非实验项目事实

```text
用户口述
→ 结构化解析
→ 歧义与冲突检查
→ 变更预览
→ 用户确认
→ append-only 追加
→ current view
```

最小事实字段：`fact_id`、`kind`、`statement`、`source`、`observed_at`、
`status`、`supersedes`。Agent 推断只能成为 candidate；旧事实不得覆盖。

权威事实事件拟写入本地 append-only `project_state/project_facts.jsonl`；
Document Registry 只登记文档，不得承载项目事实。最终文件名可在实现前审查，
但这两类职责不得合并。

### 3.2 组会简述

输入为用户指定时间范围、当前指令、accepted result 及明确标注的 explore/
failed 材料。输出只是一份简短 Markdown，说明近期工作、主要结论、难点和
下一步，并保留最小来源。

不设 500 字、图片数量、数据表、模板、排版或渲染工具门槛；不自动制图、
重跑实验、云同步或定时生成。

### 3.3 长期结论与探索路径

由用户或项目事件触发，输出版本化 Markdown，区分：

- 已确认结论；
- 初步解释和不确定性；
- 被否定路线、阴性结果及停止原因；
- 开放假设和下一候选。

建议周期不构成自动定时或实验关闭硬门禁。新版本通过替代关系保留旧版。

### 3.4 学习指南接口

本轮状态为 `interface_reserved`，只保证未来接口能表达：主题、目标读者、
先决知识、源码 commit/file/symbol、实验/结果引用、最后核验提交和替代关系。

Diátaxis 只作为文档用途分类参考；不批量重构、迁移、修复或生成现有指南，
不接入 Quarto/MyST/Jupyter Book 或云端知识库。

### 3.5 论文撰写与产出接口

本轮接口状态只到：

```text
interface_reserved
→ materials_indexed
→ template_pending
```

收到学长模板并经新设计修订批准后，未来才允许进入：

```text
template_received
→ mapping_review
→ draft
→ team_review
→ final_candidate
```

模板由项目团队学长后续提供。当前只保留 `template_ref`、项目材料索引、
主张—证据—不确定性引用、章节映射槽位、输出位置和 review status。
论文材料只允许引用本仓库 directive、fact、experiment、accepted result、
明确标记的本项目 explore 和本项目文件。外部材料必须拒绝或进入待用户审查，
不得自动并入。不得虚构模板/章节、推断作者排序、自动追加实验、投稿或发布。

### 3.6 文档与约束生命周期

依赖的实验、accepted result、源码、directive、事实或论文模板变化时，只生成
`review_due` 预览。freshness 变化不得自动改变 lifecycle；supersede 或转为
historical 必须保留理由、历史和替代指向，不自动删除或关闭 directive。

## 4. 只借鉴的设计思路

- eLabFTW：追加式修订、锁定、替代和软归档；
- Calkit：主要事实/产物与派生文档分离；
- PROV/RO-Crate：来源与派生关系；
- DVC：依赖变化产生 stale 候选；
- Diátaxis：学习文档用途分类。

只实现本地字段、检查表和 fixture；外部工具调用次数必须为零。

## 5. 简要验收矩阵

| ID | Fixture | 通过标准 |
|---|---|---|
| R10-T01 | 口述事实新增/更正/冲突 | 未确认不写；更正追加；冲突按 P0C 裁决 |
| R10-T02 | 组会简述 | accepted/explore/failed 不混淆；不强制格式 |
| R10-T03 | 长期探索路径 | 阴性结果、停止规则和不确定性未丢失 |
| R10-T04 | 学习指南接口 | 能表达源码/实验引用；正文和旧目录字节不变 |
| R10-T05 | 论文接口无模板 | 正确停在 template_pending，不伪造稿件 |
| R10-T06 | 论文来源门禁 | 只接受本项目来源；外部材料拒绝或 pending review |
| R10-T07 | 生命周期影响分析 | 只生成 review_due；无自动删除/关闭/改写 |
| R10-T08 | 幂等与隔离 | 重放 no-op；不写 Experiment Registry |
| R10-T09 | 离线边界 | 断网且未安装调研工具时全部测试通过 |

## 6. 暂缓与不做

- 组会格式规范、固定字数/图片/表格和模板；
- 学习指南内容体系重构、批量迁移和云端备份；
- 飞书 CLI、Quarto/MyST、OSF、eLabFTW 或其它平台接入；
- 论文具体模板、章节、图表、引用格式、署名和投稿；
- 自动周期生成、自动发布或根据论文缺口自动训练；
- 外部调研缺失项 3–8。

## 7. 退出与回退

R10-T01–T09 全部通过，R5b 扫描稳定性已达到退出条件，Document Registry
保持唯一文档登记源，project facts 与文档职责分离，生成视图可从权威输入
重建。失败时关闭新 writer/profile，保留追加历史，恢复旧只读视图；不得
删除现有文档。

## 8. 待用户审查

- [ ] 是否批准六类子工作流共用一个本地知识/文档生命周期模块。
- [ ] `learning_guide` 是否暂不启用可选的 Diátaxis 类型字段。
- [ ] 是否同意论文模板到达前以 `template_pending` 作为接口完成状态。

## 9. 补充记录

- `2026-07-27 / DIR-20260727-001`：按最新用户边界合并文档类能力，删除平台
  接入与组会格式硬要求，新增论文模板待接入接口。
