# Main 工作区整理与报告交接记录（2026-07-13）

## 1. 用户授权与范围

- 用户要求整理脏工作区、更新重要约束与实验记录，并准备回到 `main`。
- 后续报告正文由 Gemini 生成；本轮不生成报告正文，也不启动任何新训练。
- 生效指令：`DIR-20260713-001`。

## 2. 整理前状态

- 原分支：`automation/local/mpp5-repair-v003-recheck-20260711`。
- 权威 LoRA 状态分支：`automation/local/mpp2-paired-smoke-prediction-supplement-20260712`，提交 `8cec114`，状态 revision 54。
- 脏工作区包含三类内容：
  1. 已被 revision 54 权威分支覆盖的状态快照和 LoRA 源码；
  2. 已验收但未纳入主线的 MPP1/3/4/5 repaired recheck job/result envelope；
  3. 本机配置、编辑器目录、临时盘点文件和大型 checkpoint/cache。

## 3. 保全与整合动作

- 建立安全分支：`codex/pre-main-cleanup-20260713`。
- 将 MPP1/3/4 job 和 MPP1/3/4/5 accepted fixed-eol 结果包独立归档为提交 `c41b7b5`；该提交已被 cherry-pick 到本地 `main`，对应提交 `9d5d2b9`。
- 初次 cherry-pick 后执行 artifact byte-level 复核，发现早期提交上下文未应用 `automation/results/** -text`，导致 MPP1/3/4/5 文本 artifact 在工作树中被规范化为 LF。已从各 Gitee fixed-eol 权威分支按原始 blob 恢复 CRLF 字节，并逐项复核 `size_bytes` 与 SHA-256；MPP1-5 accepted bundle 的全部 artifact 均通过。
- 将旧状态快照、重复 LoRA 源码和未纳入本轮结论的辅助脚本保存在：
  `stash@{0}: pre-main-cleanup-20260713 stale state snapshots and duplicated LoRA sources`。
- 建立回退分支：`codex/main-before-sync-20260713`，保留同步前 `main`。
- 本机资产仅写入 `.git/info/exclude`，未移动、未删除、未提交：`.vs`、本机 MCP 配置、`tmp`、盘点文本、`checkpoints/online_tokens` 及伙伴 checkpoint 汇总目录。
- 受保护 MPP 数据、manifest、split、z-score、embargo 审计和 checkpoint 均未修改。

## 4. 实验记录修正

- `mpp2_barcode_repair_v003_frozen_baseline_20260711` 已从待运行改为闭环，绑定 accepted `attempt-003` 结果路径。
- `mpp1/3/4/5_barcode_repair_v003_frozen_recheck_20260711` 已从错误残留的 `dispatch_formal_parallel_recheck` 改为 `closed_use_accepted_repaired_result_for_reporting`，并绑定各自 accepted fixed-eol 结果路径。
- MPP2 LoRA r=8 结论保持不变：`engineering_pass / candidate_effectiveness_fail / no_go_formal`。
- 不允许 Gemini 将 smoke 结果写成正式有效性结论，也不允许依据 external XZY 选择新超参数或建议自动训练。

## 5. 后续入口

- Gemini 报告输入清单：`automation/report_inputs/gemini_mpp_report_handoff_20260713.md`。
- 机器事实源：`experiments/experiment_registry.json`。
- LoRA 诊断记录：`automation/logs/mpp2_paired_smoke_review_20260712.md`。
- 本记录只描述工作区整理和交接，不替代实验注册表。
