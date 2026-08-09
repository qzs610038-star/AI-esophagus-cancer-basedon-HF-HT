# W004 原始训练回传数据归档

> 本地收集时间：2026-08-08 18:27:25（Asia/Shanghai）  
> 实验：`phase3_dual_baseline_spatial_pathway_transfer_v001_20260804`  
> 工作树：`W004`  
> 共同训练源码：`b898c9d80c2702a32a51773c9cfb8ca75312731a`  
> 共同协议版本：2  
> 共同审批：`APR-W004-phase3-stage2-v002-20260807`

## 归档说明

本目录保存从 Gitee 固定回传提交原样导出的 W004 Stage 2 原始训练回传包。导出后已完成：

1. 远端 ref、远端 commit 与预期 commit 一致性复核；
2. Git blob 与本地 524 个文件逐文件一致性复核；
3. 使用 W004 当前治理工作树中的 `result-bundle-v1 validate` 逐包校验。

本 README 位于四个不可变回传包之外，不属于任何结果包，也不改变包内校验值。它只是本地归档索引，不替代 Experiment Registry，不代表 result import 或 scientific accept。

## 实验与来源

| Attempt | 训练卡 / 实验内容 | 服务器训练终止记录 | 本地包路径 | Gitee 回传 ref | 固定回传 commit | bundle SHA-256 |
|---|---|---|---|---|---|---|
| `A001` | `S2-ABS-001`：绝对通路表示锚点 | 2026-08-07 20:11:22 +08:00 | `A001/R002-repack-R001` | `automation/server/W004/A001` | `c689f8bc5a3ce36c8bfe3d6a81e0b901eaf282c5` | `3a49d4c005ae40c6d761f2f84aef442f19f2a50139cc1cbb6407d331b17fa1ff` |
| `A002` | `S2-ORD-SLIDE-001`：切片内相对通路百分位 | 2026-08-08 01:45:52 +08:00 | `A002/R001-repack` | `automation/server/W004/A002` | `f9889b32a8ed2da308b24e6c3f750d417cbc1d31` | `930a842213e852a3bc5e177f0b0fae04c44ba8b7a9c2f43ac62b58db62bf720d` |
| `A003` | `S2-SPATIAL-IDENTITY-001`：真实坐标固定空间平滑 | 2026-08-08 06:23:51 +08:00 | `A003/R001-repack` | `automation/server/W004/A003` | `0245e8b04c25d10bf8db6f06c3246413d3abb382` | `d2a74a04f33b7522fdfff934007b890df6e917c378a9763e5e2381983a26b068` |
| `A004` | `S2-SPATIAL-SHUFFLE-001`：坐标打乱空间反事实 | 2026-08-08 12:13:29 +08:00 | `A004/R001-repack` | `automation/server/W004/A004` | `6d8bde5d4d5042b9f606725678a285dcc2b21556` | `ba534f66f4cd58ec57bb449c9318baac2495941838632bbefd4f2e45178f6cb0` |

时间来自各包的 `artifacts/attempt_terminal.json` 中 `recorded_at`，上表已换算为 Asia/Shanghai。

## 每个包包含什么

每个目录包含 131 个文件：`result.json` 加 130 个登记 artifact，主要包括：

- 原始合并运行日志：`artifacts/runner.log`；
- 汇总与逐次指标：`artifacts/metrics.json`；
- 80 份逐切片原始预测 CSV：2 个任务 × 4 seeds × 5 folds × validation/test；
- 40 条 checkpoint 元数据与 selection proof；
- 实际 job、resolved config、启动/终止事件和完整性记录。

回传包**不包含**模型特征缓存，也**不包含** `.pt` checkpoint 二进制文件；这些不能从本目录恢复。

## 快速查看 pCR 数据

- A001 指标：`A001/R002-repack-R001/artifacts/metrics.json`
- A002 指标：`A002/R001-repack/artifacts/metrics.json`
- A003 指标：`A003/R001-repack/artifacts/metrics.json`
- A004 指标：`A004/R001-repack/artifacts/metrics.json`
- 单份预测示例：`A002/R001-repack/artifacts/predictions/pCR/seed_42/fold_0/test.csv`

## 证据边界

- 四包均为 `status=success`，各 40 run units，已通过本地包校验。
- 当前仍按 `COMPATIBILITY_ONLY / pending_review` 使用；归档动作不提升科学证据等级。
- 输入为 `legacy_zscore_compatibility`，不能称为 raw MPP2 结果。
- 本次仅执行 Gitee 拉取、本地原样归档和校验；没有训练、重打包、result import、accept、Registry 或 current state 修改。
