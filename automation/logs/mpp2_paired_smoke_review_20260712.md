# MPP2 paired S0/S1 smoke 复核与后续操作记录

> 日期：2026-07-12
> 性质：historical audit record；不替代 `CURRENT_STATE.md` 或 active plan
> 基线提交：`997c1fb`
> 复核状态：工程门通过；候选有效性门未通过；正式训练未批准

## 1. 已验收结果

- S0 result：`mpp2-paired-s0-frozen-continue-smoke-20260712-r002-result-20260712173215-1393b29c`
- S1 result：`mpp2-paired-s1-lora-r8-smoke-20260712-r002-result-20260712173316-163abad4`
- 状态修订：revision `48`
- 数据 manifest：`barcode-repair-20260711-d626ad8-v003:1204018178a4d355`
- 两组 head checkpoint SHA-256：`c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98`

## 2. 配对复核

| 指标 | S0 frozen | S1 LoRA r=8 | S1-S0 |
|---|---:|---:|---:|
| External XZY PCC | 0.6445460401 | 0.6464390475 | +0.0018930074 |
| Raw MAE | 1215.2838610 | 1206.8262988 | -8.4575621 (-0.6959%) |
| Raw R2 | -0.1449134899 | -0.1358910786 | +0.0090224113 |
| Internal val loss | 0.3789220998 | 0.3823199877 | +0.0033978879（S1 较差） |
| Internal val PCC | 0.7947915972 | 0.7929694771 | -0.0018221201（S1 较差） |

S1 的 LoRA 参数数为 `1,769,472`，总可训练参数为 `3,374,110`，峰值 CUDA 显存约 `4.24 GiB`，证明 LoRA 路径实际参与训练且工程运行稳定。

S1 在 epoch 1 达到最优 internal val；随后 train loss 持续下降，而 val loss 从 `0.382320` 上升到 `0.384387`、`0.390371`，显示早期过拟合。逐通路 external PCC 仅 `4/30` 改善、`26/30` 下降，中位变化 `-0.0140`；raw MAE 为 `14/30` 改善、`16/30` 变差，中位变化约 `+9.12`。整体微小增益不是广泛一致的改善。

## 3. 判定

- 工程可行性：PASS。
- 安全暂停线：未触发。
- 预注册候选有效性：FAIL；S1 external PCC `0.646439 < 0.6749`。
- 相同配置正式 seed 42：NO-GO。相同 seed、初始化和 early-stop 规则预计再次选择 epoch 1，新增信息有限。
- 多 seed、rank、dropout 或大规模超参搜索：禁止自动启动，必须另行获得显式批准。

## 4. 本轮门禁修复

初始 strict gate 有两个 FAIL：document registry 落后 revision 48，以及导入工作树缺少 `legacy_partner_labels_local`。

已执行：

1. 将导入工作树的 legacy 路径 junction 到主工作区现有只读标签目录；未复制、删除或重建受保护资产。该机器本地路径已加入 `.git/info/exclude`，避免误提交受保护标签。
2. 执行 `python deploy/pfmval_ops.py docs scan --write`，将 document registry 对齐 revision 48。
3. 复跑 `python deploy/pfmval_ops.py agent start-check --strict`，得到 `PASS=6 / WARN=6 / FAIL=0`。
4. 从不可变导入提交 `997c1fb` 创建独立补充证据分支，未合并当前脏主工作区。

## 5. 尚待补充的证据

已验收 S0/S1 result envelope 未包含：

- `predictions_internal_val.csv`
- `predictions_external_xzy.csv`

这些文件是原方案要求的诊断交付物。不得修改已经验收的 result bundle；应通过独立 Gitee supplement branch 回传，并在本地验证 SHA-256 后用于 bootstrap、患者切片和误差分析。服务器操作说明见：

`automation/jobs/mpp2-paired-smoke-prediction-supplement-20260712/README.md`

## 6. 后续门禁

补充预测文件只用于诊断，不会自动改变当前 NO-GO。若诊断不能证明收益具有患者/通路一致性，则关闭当前 LoRA r=8 配置。若仍要继续，只允许在新显式批准下设计一个依据 internal val 的单因素 exploratory smoke，不得利用 external XZY 选择超参数。

状态指令：`DIR-20260712-002`。
