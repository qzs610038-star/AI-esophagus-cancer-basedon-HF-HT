# R11：工作流目录、发现与演化

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R11`
> module revision: `001`
> lifecycle: `approved_design`
> dependencies: `P0C`, `R5b`
> authorization: `DIR-20260727-001`
> implementation: `local_D/I_accepted; O=NOT RUN; S=N/A`
> boundary: 只能由用户主动触发；候选不能自动启用、执行或放宽批准。

## 1. 为什么单独保留

工作流目录最终可能产生可执行入口，风险高于文档 profile，因此不并入 R10。
借鉴 Snakemake Catalog/WorkflowHub 的目录、版本、快照和准入思想，但不接入
其平台、API 或执行引擎。

## 2. 目标流程

```text
用户指定本地扫描范围
→ 发现重复结构
→ candidate
→ reviewed
→ active
→ revise / merge / deprecated
```

`standardized` 是 active 工作流的属性，表示有 schema、示例、测试和稳定版本，
不是另一套生命周期状态。

## 3. 最小 manifest

`workflow_id`、用途、触发方式、输入/输出、实现位置、风险级别、批准要求、
版本、生命周期、替代关系、是否 standardized、最近验证时间。

首版只扫描用户选择的 tracked CLI/Skill/文档清单，不读取全部会话或终端历史，
不采集 stdout、环境变量、凭证和任意原始参数。运行入口只能从权威 manifest
派生。

## 4. 简要验收

| ID | 必须证明 | 通过标准 |
|---|---|---|
| R11-T01 | 用户触发 | 未调用 discovery 时零扫描、零写入 |
| R11-T02 | 候选不执行 | 重复 fixture 只产生 candidate |
| R11-T03 | 准入门禁 | 高风险、无 schema/测试的候选不能 active/standardized |
| R11-T04 | 版本演化 | revise/merge/deprecate 保留旧 ID 和替代关系 |
| R11-T05 | 幂等与范围 | 相同扫描 no-op，范围外零读取/零写入 |
| R11-T06 | 审批隔离 | 高频使用不改变实验批准或安全边界 |
| R11-T07 | 离线边界 | 无 Snakemake/WorkflowHub/网络依赖 |

## 5. 暂缓与不做

- 后台自动监控或读取全部命令历史；
- 自动生成、修改或执行 Skill/Hook/Workflow；
- 自动合并、删除或根据使用频率降级门禁；
- 向 WorkflowHub 或其它公共目录发布；
- 外部调研缺失项 3–8。

## 6. 退出与回退

R11-T01–T07 全部通过；停用 discovery 不影响现有 CLI/Skill。派生 candidate
可由 tracked manifest 重建，active 条目不因发现器回退而丢失。

## 7. 待用户审查

- [ ] 是否批准首版只扫描 tracked CLI/Skill/文档清单。

## 8. 补充记录

- `2026-07-27 / DIR-20260727-001`：将工作流发现作为唯一独立扩展执行模块，
  外部目录方案仅作架构参考。
