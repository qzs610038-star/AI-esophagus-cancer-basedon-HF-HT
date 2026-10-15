# P0C：科研推理与扩展维护工作流横向验收

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `P0C`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `P0A`, `P0B`, `R5b`
> authorization: `DIR-20260727-001`
> implementation: `local_D/I_accepted; O=NOT RUN; S=N/A`
> boundary: 只借鉴外部方案的架构、数据模型和维护思路；不安装、部署、调用或
> 接入任何外部平台、服务、CLI、SDK、数据库或云端接口。

## 1. 目标

在不加重日常科研试错的前提下，补齐：

```text
问题/假设
→ 轻量探索
→ 候选晋级
→ P0A 正式实验
→ 观察与解释
→ 主张 / 否定结果 / 不确定性
→ 后继假设 / 追加验证 / 停止
```

P0C 是 R10–R11 的横向验收合同，不另建一套实验 Registry，也不替代 P0A。

## 2. 渐进式形式化

| 层级 | 最小记录 | 不允许宣称 |
|---|---|---|
| 想法/解释 | 问题、假设、来源、下一动作 | 项目结论、实验通过 |
| 本地探索 | Explore id、输入范围、观察、限制 | accepted evidence、跨实验比较 |
| 候选方案 | 晋级理由、判别条件、预计成本 | 已获正式执行批准 |
| 正式实验 | 进入 P0A 全生命周期 | 未 import 即 accepted |
| 结论/停止 | accepted evidence、解释、不确定性、停止或后继条件 | 把解释写成机器事实 |

轻量层不得被正式门禁拖慢；一旦结论会影响项目路线、论文或正式实验，必须
绑定相应 experiment/result/directive。

## 3. 只借鉴的设计思路

- Calkit：问题、主要研究产物和派生产物分离。
- eLabFTW：追加式修订、锁定、替代和软归档。
- DVC/Snakemake：依赖—输出关系、内容指纹、失效检测、dry-run。
- W3C PROV/RO-Crate：`Entity–Activity–Agent` 与
  `used/generated/derivedFrom` 关系。
- Snakemake Catalog/WorkflowHub：candidate、review、active、
  standardized、deprecated 的目录治理。
- Diátaxis：按教程、操作指南、参考、解释区分学习文档目的。
- The Turing Way：用检查表做人工复核。

以上全部只转译为 PFMval 本地 schema、状态机、检查表或测试思想；不得新增
外部运行时依赖、联网步骤或第二事实源。

## 4. 本轮采纳与暂缓

本轮只采纳外部调研缺失项：

1. **主张—证据—不确定性**：区分支持、反驳、未决和替代解释。
2. **阴性结果与停止规则**：记录失败路线、无差异结果、继续条件和停止条件。
9. **事实冲突裁决**：先按事实域裁决，禁止跨域覆盖：
   - 当前显式用户指令管意图、授权和路线选择；
   - current state/Registry 管身份、生命周期和当前状态；
   - accepted result 管其协议范围内的科学测量与结果；
   - active plan 管未来拟执行步骤，不能改写 accepted measurement；
   - 生成视图、摘要和云端副本只能派生，不能反向覆盖。
   用户可改变下一步路线，但不能用新指令改写历史测量值。同域冲突无法由
   supersedes/accepted 状态确定时，保留冲突并请求用户裁决。
10. **可访问性与解释责任**：区分观察、解释、建议；图表存在时记录来源、单位、
    统计口径和文字说明。

缺失项 3–8（人工步骤/环境、隐私伦理分级、贡献许可、独立复算与恢复演练、
schema 退出计划、存储保留预算）保留在参考文档中，本轮不形成实现要求或
阻断门禁。

## 5. 扩展模块

| 模块 | 工作流 | 当前设计边界 |
|---|---|---|
| R10 | 项目知识与文档生命周期 | 六个 document profile，共用本地事实、来源和生命周期 |
| R11 | 工作流目录、发现与演化 | 用户主动触发，候选不得自动启用 |

## 6. 验收证据分层

每项验收都必须写 `PASS`、`FAIL`、`NOT RUN` 或 `N/A`，并注明证据等级：

- `D`（design）：职责、事实源、状态、输入输出、冲突和回退已定义；
- `I`（implementation）：schema/CLI/API、单元、负向、幂等、fixture 端到端和
  一致性测试通过；
- `O`（operation）：用户真实触发、使用真实项目输入的一次本地/受控运行可审计；
- `S`（science）：只用于评估科学主张；来源必须是已批准、import 并 accepted
  的实验结果。

冻结 fixture 属于 `I`，不是 `O`。维护/文档/接口工作流通常为 `S=N/A`；
消费 accepted result 只证明引用边界正确，不会让维护工作流获得 `S-PASS`。
`I-PASS` 不等于 `O-PASS`，二者都不等于科学结论成立。接口预留模块允许
`O=NOT RUN`，但不得伪装为运行完成。

## 7. 横向验收测试

| ID | 必须证明 | 通过标准 |
|---|---|---|
| P0C-T01 | 分层不串级 | idea/explore 不能成为 accepted；formal 仍走 P0A |
| P0C-T02 | 主张可追溯 | 每个正式 claim 能定位 evidence、方向和 uncertainty |
| P0C-T03 | 阴性结果保留 | failed/null 路线可查询，停止条件不会被下一次生成覆盖 |
| P0C-T04 | 冲突裁决确定 | 同一 fixture 重放得到相同权威值并保留冲突记录 |
| P0C-T05 | 解释责任 | 输出能机器区分 observation/interpretation/recommendation |
| P0C-T06 | 无外部接入 | 测试在断网且未安装调研工具时通过，无外部服务调用 |
| P0C-T07 | 幂等与回退 | 重放为 no-op；失败不留下半写状态；旧读路径仍可用 |
| P0C-T08 | 模块隔离 | R10–R11 不写 Experiment Registry、不放宽实验批准 |

单元测试通过只证明实现行为，不等于实验成功、结果 accepted 或论文结论成立。

## 8. 新对话 Goal 推进协议

用户批准本审查包后，在**新对话**显式创建一个总 Goal：

```text
按已批准的 workflow governance v3 审查包完成本地结构性重构，
逐模块验证并交付最终审核包；不含服务器、训练、平台接入和资产清理。
```

G00–G06 是该总 Goal 下的 `stage_id`，不是七个独立 Goal：

| Stage | 范围 | 必须达到 | 无新增授权时允许/必须标记 |
|---|---|---|---|
| G00 | R0、R5b、HEAD/dirty、Registry revision | D/I/O-PASS；S=N/A | 只做本地一致性 |
| G01 | P0A、R1–R5 | D/I-PASS | 真实服务器/训练 O=NOT RUN；S=N/A |
| G02 | P0B、R6–R9 | D/I-PASS | 迁移、解禁、close execute、清理 O=NOT RUN；S=N/A |
| G03 | P0C | D/I-PASS | 未发生真实推理记录时 O=NOT RUN；S=N/A |
| G04 | R10 | D/I-PASS | 未经用户触发的事实/文档运行 O=NOT RUN；S=N/A |
| G05 | R11 | D/I-PASS | 未经用户主动扫描时 O=NOT RUN；S=N/A |
| G06 | 集成兼容、回退、独立审计 | D/I-PASS，O/S 矩阵完整 | 不得把 NOT RUN/N/A 改写为完成 |

执行规则：

1. 同一时间只有一个模块处于 `in_progress`。
2. 高风险模块依次完成：基线证据 → tests red → 最小实现 → tests green →
   兼容/影子验证 → 独立审计；文档/元数据轻量模块采用：设计基线 → 最小本地
   接口 → 负向/幂等/冲突验证 → 独立审计。
3. 模块未达到退出条件时，Goal 保持 active，不用“部分完成”替代验收。
4. 新增权限、服务器、训练、外部接入、迁移或删除需求时暂停并另请批准。
5. 每个阶段保存内部 Goal 回执，至少包含基线、允许路径、实际变更、验证命令、
   原始结果、证据等级、NOT RUN、风险和回退；回执不冒充最终审核。
6. 全部必需 D/I 项通过、允许的 O=NOT RUN 与 S=N/A 如实记录后，才标记总
   Goal complete，并只向用户提交一份最终重构审核包。Goal complete 只表示
   “已授权的本地结构性重构完成”，不表示服务器、训练、迁移、清理或科学结论
   已运行/通过。

## 9. 最终审核包最小内容

- 基线 commit、最终 commit、变更文件与模块状态；
- 每个 P0/R 模块的 PASS/WARN/FAIL、测试命令和新鲜输出；
- D/I/O/S 全矩阵，明确 O=NOT RUN、S=N/A 及原因；
- schema/CLI/生成视图的兼容性与回退证据；
- 未完成、暂缓和明确不适用项；
- 未执行服务器、训练、平台接入、结果接纳和清理的边界声明；
- 独立 `pfmval-audit` 总结论：GO、CONDITIONAL GO 或 NO-GO。

唯一总体 verdict 必须回答“是否接受已授权的本地重构实现”，不得笼统宣称
全部工作流已运行完成。

## 10. 退出与回退

退出条件为 P0C-T01–T08 全部通过，R10–R11 各自达到退出条件，并且 P0A/P0B
未回归。任何失败均恢复到模块开始前的本地 checkpoint；不得通过删除历史、
放宽门禁或引入外部服务规避。

## 11. 待用户审查

- [ ] 是否批准 P0C 作为 R10–R11 的横向硬验收。
- [ ] 是否批准上述新对话 Goal 推进顺序与最终单包审核方式。

## 12. 补充记录

- `2026-07-27 / DIR-20260727-001`：按用户最新选择建立架构借鉴边界、采纳项和
  Goal 模式验收合同。
