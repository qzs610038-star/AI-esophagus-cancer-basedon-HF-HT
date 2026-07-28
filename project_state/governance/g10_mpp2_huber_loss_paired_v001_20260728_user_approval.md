# G10 用户批准记录：MPP2 paired MSE vs Huber delta1

- experiment_id：`mpp2_huber_loss_paired_v001_20260728`
- display_name：`MPP2 paired MSE vs Huber delta1`
- phase：`formal`
- 批准来源：本对话中的明确用户指令，2026-07-28。

## 已批准合同

- A0：`nn.MSELoss()`；A1：`nn.HuberLoss(delta=1.0, reduction="mean")`。
- 除 loss_type 外冻结 repaired manifest、split、UNI2-h frozen features、两层 MLP、hidden_dim=1024、dropout=0.3、Adam、lr=1e-4、batch_size=32、seed=42、max_epochs=50、patience=10、min_delta=1e-4。
- 每臂由自身 internal validation loss 最小 checkpoint 选择；XZY 仅在 checkpoint 冻结后每臂一次评估，不参与选择或调参。
- `run_limit=4`：A0 首次拟合 1、A1 首次拟合 1；余下 2 仅可用于合同完全不变且已启动臂因工程/运行故障失败后的显式重试。无隐式重试。
- 通路改善数量与比例仅为描述性分析，不是放行条件。
- 预注册输出：legacy pooled PCC、mean-per-pathway PCC、逐通路 delta PCC、改善数量/比例、raw MAE、raw R2、内部患者聚类比较、外部 XZY 空间聚类 bootstrap 设计。

## 本阶段边界

本记录仅授权本地预派发治理准备；不授权 G11、服务器访问、Gitee dispatch、训练/拟合、run-unit 消耗或 result import/accept。
