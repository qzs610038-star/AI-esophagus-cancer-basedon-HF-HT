# 用户强制指定的本地 Explore 实验

本目录中的实际输出默认被 Git 忽略，仅保存本说明和目录哨兵文件。

关联 directive：`DIR-20260726-001`。

## 证据边界

- `authority`: `explicit_user_requested`（用户强制指定实验）
- `adoption_status`: `pending_user_approval`
- `evidence_status`: `non_evidence`
- 外部 XZY 真值仅可用于本地探索、oracle 诊断或 anchor/held-out 分析；不得据此直接更新训练、部署或 accepted evidence。
- 任一结果只有在用户后续明确批准后，才能经正式 Registry/result import 流程登记采纳。

相关源码位于 `scripts/explorations/`，每个可执行探索脚本均以 `# PFMVAL_EXPLORE` 标记。
