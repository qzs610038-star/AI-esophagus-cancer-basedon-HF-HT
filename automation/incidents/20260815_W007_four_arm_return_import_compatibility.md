# W007 四臂回传导入兼容问题（2026-08-15）

## 现象

W007 的 FBR、RCC、HCR、CPGCR 已完成并形成四个独立 return commit，但通用 `result import` 无法登记。

## 拦截约束

通用导入器强制要求：

1. 回传根目录必须有单个 `result.json`；
2. 结果必须绑定单个 `automation/jobs/<job_id>/job.json`；
3. 同一实验正式结果的 `source_commit` 必须与 Registry 中单一来源一致。

W007 历史回传实际是四个 `return_manifest.json`，分别对应 FBR、A019、A022、A023，并包含三条不同 source commit，因此不能套用上述单结果合同。

## 最小处置

新增只识别 W007 已知四臂合同的 `result import-w007-four-arm`：

- 不修改通用结果 schema；
- 不 merge/cherry-pick 四个 return commits；
- 不启动训练；
- 从原始预测复算指标，并核对 artifact 清单、大小和 Gitee return commit；
- 通过治理 CLI 一次登记聚合结果及四个成员结果，保留各自 attempt/source/return provenance（来源链）。

## 结果

- 聚合结果：`W007-four-arm-seed42-result-R001`
- 四个成员结果：`W007-FBR-result-R001`、`W007-A019-result-R001`、`W007-A022-result-R001`、`W007-A023-result-R001`
- 证据状态：`accepted`
- 科学结论状态：`pending_user_review`
- 用户指令：`DIR-20260815-001`

本问题已解除“结果无法登记”的阻塞。结论是否进入团队稳定文档，仍等待用户审核；不会据 XZY 追加训练或选择配置。
