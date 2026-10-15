# G10 独立审核回执：MPP2 paired MSE vs Huber delta1

## 唯一结论

**PASS（仅限 G10 本地预派发治理闭环）**。

## 身份绑定

- experiment：`mpp2_huber_loss_paired_v001_20260728`
- workspace：`W001`
- experiment worktree：`D:/AI空间转录病理研究/PFMval_new_governed_workspaces/W001`
- branch：`codex/w001-mpp2-huber-loss-20260728-bound`
- source/HEAD：`a04319a1d6aeafcd35d6adc7e2893f1d7b683bda`
- experiment worktree dirty：`false`
- 绑定核验：`workspace check-locus --workspace-id W001`，双轮 PASS。

## 治理身份

- approval：`APR-mpp2-huber-loss-paired-v001-20260728-r001`
- protocol revision：`1`
- critical contract SHA-256：`569bb6d83dcc45ef77f900d92f58645211d05702aad6cb189e00131ae08d9962`
- prepared attempts：`A001`（A0/MSE，1 unit）、`A002`（A1/Huber delta=1，1 unit）
- prepared jobs：`W001-A001`、`W001-A002`
- run_limit：`4`；run_consumed：`0`；reserved：`2`；未预留：`2`。

## 合同与接口

- A0：`nn.MSELoss()`；A1：`nn.HuberLoss(delta=1.0, reduction="mean")`。
- 无 pathway weights、无复合损失；其他数据、split、UNI2-h frozen features、两层 MLP 和训练参数均冻结。
- 训练入口新增显式 `--loss mse|huber` 与 `--huber-delta`；默认 MSE 保持兼容，并写出 loss type、delta、reduction 元数据。
- 预注册已覆盖 pooled/mean-per-pathway/逐通路 delta PCC、改善数/比例（描述性）、raw MAE/R2、患者聚类比较和 XZY 空间聚类 bootstrap；XZY 不可改变门控。

## 本地验证

- CPU 单元与参数校验、治理定向测试：17 passed。
- 全量：169 passed，2 个既有单样本 R2 警告。
- 训练 CLI parser-only dry-run：`--help` 显示新增 loss 参数，未进入 state/training 分支。
- 两份 job v2 manifest：双轮 `validate` PASS。
- 严格 start-check：双轮 PASS（`PASS=7 WARN=6 FAIL=0`）；WARN 均为既有 legacy result envelope 不完整及既有 MPP label 冲突提醒。

## 明确未运行

- G11：NOT RUN
- 服务器、SSH/SCP/HTTP remote command/Tunnel：NOT RUN
- Gitee dispatch：NOT RUN
- 训练或模型拟合：NOT RUN
- `EXPERIMENT_STARTED`：0
- result import/accept：NOT RUN

## 反向核验入口

- `project_state/governance/g10_mpp2_huber_loss_paired_v001_20260728_critical_contract.json`
- `project_state/experiment_approvals.jsonl`
- `project_state/attempt_events.jsonl`
- `project_state/governance/g10_A0_attempt_prepared.json`
- `project_state/governance/g10_A1_attempt_prepared.json`
- `project_state/governance/g10_A0_job_prepared.json`
- `project_state/governance/g10_A1_job_prepared.json`
