# 03 审计报告

本目录是 PFMval 项目审计文件的统一存放处。方案审计、代码审计、超参数审查、划分/z-score 诊断报告，以及总管下达的查找提示词，均写入此处。

这些文件默认只是待审审计产物，不是实验结论，也不是训练或改代码授权。

## 目录约定

```text
03审计报告/
├── README.md
├── GIT_INDEX.md
└── YYYYMMDD_<主题>/
    ├── 00_总管任务包.md
    ├── 提示词_*.md
    ├── 回传模板_审批报告.md
    ├── CA2代码审计报告_YYYYMMDD.md
    ├── 超参数审查报告_YYYYMMDD.md
    └── 划分与zscore诊断报告_YYYYMMDD.md
```

## 命名

- 主题目录：`YYYYMMDD_<短主题>`，例如 `20260821_Phase2-MPP2-PCC瓶颈`
- 提示词：`提示词_<任务名>.md`
- 审计正文：`<审计类型>报告_YYYYMMDD.md`
- 给总管/用户的短回传：`审批回传_<任务名>_YYYYMMDD.md`

## 生命周期

| 状态 | 含义 | 可否作为当前结论 |
|---|---|---|
| 已写入本目录 | 查找对话已落盘，待用户审核 | 否 |
| 用户在对话中接受 | 可作为该次审计的已审记录 | 仅覆盖该次审计范围 |
| 已登记 `document_registry` 且 `lifecycle=active` | 后续任务可引用 | 是，且不得超出登记范围 |

未完成后两步的报告，一律按 `pending_review` 处理。不得用本目录文件覆盖 `project_state/current_state.json`、`experiments/experiment_registry.json` 或受保护资产。

## 写入边界

- 只写审计、审查、诊断和总管提示词。
- 新实验部署讨论稿仍只写 `01_指南与解读/部署方案/`。
- 实施方案仍只写 `project_state/implementation_plans/`。
- 禁止在本目录存放 checkpoints、原始训练数据、密钥、服务器凭据。
- 查找对话必须只读；修复代码、重训、重生成划分/z-score 必须另开对话并获得用户明确批准。

## 权威模板

独立审计正文遵循 `.agents/skills/pfmval-audit/SKILL.md` 与 `assets/audit-report-template.md`。
