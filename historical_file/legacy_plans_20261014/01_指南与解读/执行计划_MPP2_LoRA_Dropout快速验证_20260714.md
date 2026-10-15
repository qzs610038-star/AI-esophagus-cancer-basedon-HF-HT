# MPP2 LoRA Dropout 快速验证方案（已获用户批准）

> 状态：approved exploratory smoke；用户已明确批准执行一次本文件定义的 smoke。它不构成正式训练批准，也不授权任何重试、扩展或超参数搜索。
>
> 事实边界：当前全 24 block、`qkv+proj`、rank 8、`lora_dropout=0.0` 的 S1 已关闭为 `engineering_pass / candidate_effectiveness_fail / no_go_formal`。不得将本方案解释为该配置的续跑、正式训练或多 seed 扩展。

## 一、目标与单因素假设

在不改动数据、模型结构、路径配置或训练脚本的前提下，仅将 LoRA dropout 从 `0.0` 改为 `0.10`，检验轻量正则化能否缓解当前 smoke 中“epoch 1 后 internal validation 退化”的现象。

选择该变量的原因：`train_mpp_uni2h_lora.py` 已原生接受 `--lora_dropout`，`lora_utils.LoRALinear` 已实际将该值应用到 LoRA 支路。因此无需新增服务器依赖、无需修改模型代码、无需加入新的路径或缓存配置。

本方案不改变：

- repaired manifest：`barcode-repair-20260711-d626ad8-v003:1204018178a4d355`；
- MPP2、固定空间 block split、训练集拟合的 z-score；
- accepted MPP2 head checkpoint 与 SHA-256；
- rank `8`、alpha `16`、`qkv+proj`、全部 24 个 block；
- `head_lr=1e-4`、`lora_lr=1e-4`、`weight_decay=1e-4`、seed `42`、batch `1`、grad accumulation `8`、最多 3 epoch；
- checkpoint 仅由 internal validation 的最小 val loss 选择；XZY 不参与任何选择。

## 二、命名与配置铁律

| 概念 | 唯一值 | 规则 |
|---|---|---|
| experiment id | `mpp2_lora_r8_dropout10_smoke_20260714` | 仅下划线；供 registry 与 Python 使用 |
| job id | `mpp2-lora-r8-dropout10-smoke-20260714-r001` | 仅连字符；供 job 目录、Gitee 分支与结果包使用 |
| dataset name | `mpp2_lora_r8_dropout10_smoke_20260714_r001` | 仅下划线；供训练输出目录使用 |
| 唯一变量 | `lora_dropout=0.10` | 只在 job manifest 的 `parameters` 中出现一次 |

严禁在服务器手工拼接 Python 长参数串。所有参数必须先写入 job manifest，再由 `python deploy/pfmval_ops.py job run` 生成 argv；该工具会将 manifest 的下划线参数名原样绑定到训练脚本，并自动注入受注册表管理的路径。

## 三、任务分解

| 步骤 | 内容 | 依赖 | 预计时间 | 风险与缓解 |
|---|---|---|---|---|
| A | 将用户批准登记为 directive，并注册 experiment id | — | 5 分钟 | 只授权此处的单次 smoke |
| B | 本地生成、验证 job manifest；执行 `job run --dry-run` | A、批准文件、已推送 source commit | 10 分钟 | manifest 是唯一参数源，禁止手输替代参数 |
| C | 服务器执行单个 S1 dropout smoke | B | 约 1 小时 | 无自动重试；失败只回传日志和失败 result bundle |
| D | 导入结果并以 internal validation 先判定 | C | 15 分钟 | XZY 仅在已固定 checkpoint 后读取，不用于改参 |

依赖图：`A → B → C → D`。

## 四、与旧 S0/S1 的可比性

本方案只运行新的 S1，不重复运行 S0：S0 已是来自相同 repaired manifest、相同 head checkpoint、相同 seed、相同样本顺序和相同 checkpoint 规则的已验收对照。新结果必须在报告中同时列出该 S0 result id：

`mpp2-paired-s0-frozen-continue-smoke-20260712-r002-result-20260712173215-1393b29c`。

如任一固定条件无法从 manifest 或训练摘要证明一致，则本方案失去快速验证前提，必须停止并重新登记一个完整 S0/S1 配对 smoke；不得临时补参数或沿用未验证目录。

## 五、预注册判定

### Internal-first 选择

新 S1 的候选 checkpoint 由 internal val loss 最小值确定。它只有同时满足下列条件，才可保留为“值得报告的探索信号”：

1. best internal val loss 不高于历史 S0 的 `0.3789220998`；
2. internal val PCC 不低于历史 S0 的 `0.7947915972`；
3. 六位内部验证患者不再出现 `6/6` 标准化 MAE 全部增加；
4. 训练 history、参数统计、internal/external predictions、per-pathway PCC、raw MAE/R2 和 result envelope 均完整回传。

未满足任一项即关闭该 dropout 路线；不得再以 XZY 结果为依据追加 dropout、rank、学习率或 epoch 搜索。

### External final readout

只有 internal-first 条件满足后，才将训练脚本在固定 checkpoint 上给出的 XZY 结果写入结论。即使外部结果改善，也仍需满足既有候选门（PCC `>=0.6749`、raw MAE 与 raw R2 安全边界）；否则只记探索性结果，不升级为正式训练依据。

## 六、唯一允许的派发方式

以下命令是已获用户批准、experiment 已登记、包含 job manifest 的 source commit 已推送 Gitee 后的唯一派发路径。它是本轮唯一允许的执行路径。

```powershell
# 本地：生成后只检查 manifest；不直接运行 train_mpp_uni2h_lora.py
python deploy/pfmval_ops.py job dispatch `
  --job-id mpp2-lora-r8-dropout10-smoke-20260714-r001 `
  --experiment-id mpp2_lora_r8_dropout10_smoke_20260714 `
  --phase smoke --command-id standard_training `
  --path-id mpp_data_root --path-id mpp_standard_splits --path-id server_mpp_results `
  --path-id server_mpp2_frozen_baseline_checkpoint `
  --data-manifest-id barcode-repair-20260711-d626ad8-v003:1204018178a4d355 `
  --param mode=lora `
  --param train_mpp_id=2 --param external_mpp_id=2 --param external_patient=XZY `
  --param dataset_name=mpp2_lora_r8_dropout10_smoke_20260714_r001 `
  --param num_epochs=3 --param batch_size=1 --param grad_accum_steps=8 `
  --param head_lr=1e-4 --param lora_lr=1e-4 --param weight_decay=1e-4 `
  --param patience=2 --param min_delta=1e-4 --param gradient_clip=1.0 `
  --param seed=42 --param num_workers=0 --param amp=true `
  --param lora_rank=8 --param lora_alpha=16.0 --param lora_dropout=0.10 `
  --param hidden_dim=1024 --param dropout=0.3 `
  --param head_checkpoint_sha256=c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98

python deploy/pfmval_ops.py job validate `
  --manifest automation/jobs/mpp2-lora-r8-dropout10-smoke-20260714-r001/job.json `
  --require-head
```

服务器只执行 manifest，不直接调用 PowerShell launcher：

```powershell
python deploy/pfmval_ops.py job run `
  --manifest automation/jobs/mpp2-lora-r8-dropout10-smoke-20260714-r001/job.json `
  --dry-run

# 仅在 dry-run 全部 PASS 后，去掉 --dry-run 执行一次；失败不得自动重试。
python deploy/pfmval_ops.py job run `
  --manifest automation/jobs/mpp2-lora-r8-dropout10-smoke-20260714-r001/job.json
```

## 七、明确禁止项

- 不启动当前 `lora_dropout=0.0` 配置的正式训练、多 seed、rank sweep 或大规模超参搜索；
- 不改动 MPP 原始 ssGSEA、split、z-score、manifest、embargo、cache 或 checkpoint；
- 不把 `experiment_id` 写成连字符，或把 `job_id` 写成下划线；
- 不在服务器手工执行 `train_mpp_uni2h_lora.py`、不手工复制参数、不中途修改 job.json；
- 不根据 XZY 选择 dropout、epoch、rank、学习率或 checkpoint。

## 八、当前结论

这是一个低配置成本的、单变量正则化探索计划，而非对失败 r=8 方案的放大。用户已授权单次 smoke；只有在 registry 登记、job manifest 本地验证与服务器 dry-run 都通过后，才可实际启动它。
