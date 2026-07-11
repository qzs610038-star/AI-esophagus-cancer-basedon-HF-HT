# AGENTS.md

本文件只维护跨 agent 的固定安全边界和事实源读取顺序。可变项目状态不得复制到本文件。

## 必读顺序

1. `CURRENT_STATE.md`：当前用户指令、活跃方案、最新已验收结果与阻塞项。
2. `project_state/current_state.json`：机器可读的可变状态单一事实源。
3. 涉及实验状态、性能或下一步决策时，读取 `experiments/experiment_registry.json`；`experiments/experiment_dashboard.md` 仅为派生视图。
4. 涉及服务器训练、缓存、路径或同步时，读取 `configs/server_paths.yaml` 和 `01_指南与解读/部署方案/服务器路径索引_20260701.md`。
5. 只使用 `project_state/document_registry.json` 中 `lifecycle=active` 的方案作为执行依据；`superseded`、`historical`、`missing` 文档不得作为当前结论。

开始训练、修改服务器路径或生成项目结论前，运行：

```powershell
python deploy/pfmval_ops.py agent start-check --strict
```

## 固定安全边界

- `histogene/`、`egnv1/`、`egnv2/` 为受保护目录，未经用户明确授权不得修改。
- 禁止 `git clean -fd`；不得自动删除 checkpoints、MPP 数据、缓存或未跟踪训练结果。
- 服务器与本地当前只允许通过已配置的 Gitee Git remote 同步代码、状态和小型结果；SSH、SCP、HTTP 远程命令、Tunnel 均不是 active 通道。
- 正式训练必须存在绑定 `job_id` 与 `source_commit` 的显式用户批准文件。
- MPP 原始 ssGSEA、标准划分、z-score 参数、manifest 和 group 3/5 embargo 审计为受保护资产；重新生成必须另开任务并比较输入、参数和校验值。
- 监督学习预处理必须在训练集上拟合，再应用到验证集和外部测试集；不得用 external XZY 拟合 z-score 或选择 checkpoint。
- `CLAUDE.md`、`.claude/` 等本地适配文件只能补充工具特定说明，不得覆盖受跟踪状态包。

## 状态更新规则

- 用户明确改变方案、优先级、路径、训练协议或安全边界时，通过 `state record-directive` 追加规范化指令。
- 新训练先登记 experiment id；服务器结果先进入 inbox，经 `result import` 验证后才可成为 accepted 证据。
- `CURRENT_STATE.md`、Dashboard、next-steps 和 session-brief 均为生成文件，禁止手工维护事实。
