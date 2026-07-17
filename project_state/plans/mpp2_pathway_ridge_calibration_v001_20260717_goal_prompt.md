# 新对话执行用强 Goal 提示词

以下内容可直接复制到新对话。它只授权执行本方案，不授权 Huber、depth-aware 或其他模型搜索。

```text
你现在负责在 D:\AI空间转录病理研究\PFMval_new 执行一次正式的 MPP2 后校准实验。

唯一目标：

将 accepted 的 MPP2 epoch-15 基线包装为一个可交付的 CalibratedMPP2 模型，使模型直接输出30个校准后的 raw ssGSEA 通路分数，并用冻结后的校准器验证 XZY 的绝对误差和 mean-per-pathway raw R2 是否改善。Phase 3 应直接消费最终30列，不得再做 inverse z-score、重新校准、患者内 z-score、rank normalization 或 quantile normalization。

实验ID：mpp2_pathway_ridge_calibration_v001_20260717

开始前必须完成：

1. 读取并核对 CURRENT_STATE.md、project_state/current_state.json、experiments/experiment_registry.json 和 active project_state/plans/mpp_training.md。
2. 运行：python deploy/pfmval_ops.py agent start-check --strict
3. 确认 active repaired manifest 为 barcode-repair-20260711-d626ad8-v003:1204018178a4d355。
4. 确认 accepted checkpoint SHA-256 为 c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98，epoch=15。
5. 确认 `CURRENT_STATE.md` 已将 `mpp2_pathway_ridge_calibration` 列为 active plan，Registry 中实验 `mpp2_pathway_ridge_calibration_v001_20260717` 为 `planned`，且 `execution_approved=false`。
6. 确认方案登记 directive 为 `DIR-20260717-001`，但它只授权方案修订与登记，不授权执行；必须另有显式执行批准，才可创建绑定 job_id、source_commit、批准文件和 Gitee 回传路径的 job manifest。

执行方案：

1. 只使用 accepted epoch-15 checkpoint 对六名训练患者的 internal-val 生成逐patch预测；此阶段禁止读取或分析 XZY。
2. 输出并校验 1078 行的 patient_id、barcode/patch_id、坐标、30个 true_z 和30个 pred_z；校验通路顺序、有限值、重复键和 baseline internal-val 指标。
3. 在 train-z 空间对每个通路拟合：
   z_cal = k_j * z_pred + b_j
   目标函数使用患者平衡误差和 lambda*((k_j-1)^2+b_j^2)，要求 k_j>0。
4. 预注册共享 lambda 网格收窄为 0.1、1、10。内层按患者 OOF 的 patient-balanced z-MSE 选择；一个标准误范围内选择更大的 lambda，并记录每个外层折的选择。若选择在三个候选值之间高度分散，或最弱正则 `0.1` 缺乏跨折一致性，internal stability gate 失败。
5. 使用六名患者的嵌套患者留一验证：外层留一患者评估，内层在其余五名患者中选择 lambda。不得以 patch 数代替患者分组；通过后在全部6人上重新选lambda和refit属于标准最终拟合，不得用最终in-sample指标替代外层OOF证据。
6. 只有满足 `Var(pred_z)>=1e-6`、k>0、至少4/6患者 z-MSE 不恶化、患者中位数 raw MAE 下降、该通路外层 raw R2 改善且 `std(k)/(abs(mean(k))+1e-8)<=0.5` 的通路才启用；其余回退到 k=1,b=0。
7. 通过 internal stability gate 后，冻结最终 lambda、30组 k/b、pathway mask 和校准器 SHA。冻结之前不得加载或使用 XZY。
8. 冻结后对 XZY 做一次性 baseline-vs-calibrated 对照。XZY 不得参与任何参数、通路、checkpoint 或阈值选择。
9. 指标契约必须分开：`legacy_pooled_pcc` 用于复现 accepted 0.6549、允许校准后变化；`mean_per_pathway_pcc` 及逐通路最大delta仅在最终单一冻结校准器上用于正斜率实现不变量检查，最大绝对变化必须小于 `1e-10`。嵌套 LOPO 的跨折 OOF 拼接使用折特异映射，应报告 PCC 差异但不得作为不变量或放行门；主要有效性指标是 mean-per-pathway raw R2、mean raw MAE、CCC。不得改成 pooled R2，必须报告逐通路 delta R2/MAE 的中位数、IQR和最差3条。
10. 只有外部最低门通过才可交付：mean raw R2 > -0.088、raw MAE <1176.2114、mean-per-pathway PCC保持不变、至少18/30通路 raw R2 不恶化。legacy pooled PCC只报告、不设不变门。建议交付门为 delta mean raw R2 >=0.03 且 raw MAE 下降。
11. 通过后将校准和 train-only inverse z-score 集成到 CalibratedMPP2 推理包装，直接输出 raw calibrated 30列；交付 checkpoint、calibrator.json、zscore_params_from_train.json、pathway_order.json、provenance 和 SHA manifest。

必须交付的证据：

- experiment/job/result 的完整 provenance；
- internal-val baseline predictions；
- nested_lopo_metrics.json；
- calibrator.json；
- pathway_decisions.csv；
- frozen calibrator hash；
- XZY baseline-vs-calibrated per-pathway metrics；
- 最终模型包和 README_PHASE3_INPUT_CONTRACT.md；
- 结果摘要，明确 Phase 3 输入列已经是最终 raw calibrated 分数。
- 指标口径说明：0.6549 是 legacy pooled PCC，不是 mean-per-pathway PCC；R2/MAE headline 使用逐通路等权均值。
- 每折 lambda 选择日志、每患者patch数与per-patient指标、每通路delta R2/MAE分布和fallback原因。
- 空间敏感性：按 `(floor(x/g),floor(y/g))` 聚类，g=1024/2048/4096像素网格边长，cluster bootstrap B=5000、seed=42；只有至少两个网格的 delta mean raw R2 95% CI 下界大于0，才可称为“空间稳健强证据”，否则只能报告为单患者弱/中等证据。

硬边界：

- 不得训练 Huber；
- 不得加入测序深度变量、除以 depth 或拟合 log(depth)；
- 不得在 XZY 上拟合任何参数；
- 不得复用 paired S0 prediction supplement 作为 accepted 基线，因为 S0 在 epoch-15 后继续训练了3轮；
- 不得修改或重生成 protected ssGSEA、split、z-score、manifest、checkpoint 或缓存资产；
- 不得使用 Phase 3 结局选择通路或调参；
- 服务器只能通过已配置的 Gitee Git remote 同步；禁止 SSH、SCP、HTTP remote command、Tunnel；
- 未有新的显式批准文件时不得启动服务器训练；本实验只允许冻结模型推理和轻量校准；
- 不得将 proposed 方案文件当成 execution approval。
- 不得声称“只对预测做 kx+b 时 R2 不变”。R2(y,k*pred+b) 会随 k,b 改变；只有 true/pred 同时做相同仿射变换时才不变。单通路 z/raw R2 因 true/pred 同时 inverse z 而相等。
- 不得声称 z-MSE 与 raw R2 根本错配；对同一通路，train-only inverse z 同时作用于 true 与 prediction，非退化 std 下两者 R2 完全等价。
- CalibratedMPP2 必须把 k/b/mean/std 作为不可训练buffer，并在加载时硬校验 base checkpoint、data manifest、z-score参数和pathway order的SHA；任何不一致直接raise。

阻塞退出条件：

任一条件成立就停止，不得自行修复后继续：

- start-check 出现 FAIL；
- checkpoint、manifest、z-score 参数、pathway order 或 active document 与预期不一致；
- internal-val 预测无法复现 baseline 指标；
- 患者+barcode 键冲突、重复或缺失；
- 任何 XZY 数据在校准器冻结前被读取；
- lambda、k、b 或 pathway mask 需要依赖 XZY 选择；
- internal nested-LOPO gate 失败；
- lambda跨外层折高度不稳定、`0.1` 缺乏跨折一致性，或病态通路未按 `Var(pred_z)<1e-6` 回退；
- 最终单一冻结校准器的 `mean_per_pathway_pcc` 最大绝对变化不小于 `1e-10`；不得把嵌套 LOPO 跨折 OOF 拼接的差异误作该不变量；
- 空间 bootstrap 的强证据门被误报为跨患者泛化证明；
- 需要修改 protected 资产或新增未批准权限；
- 缺少 directive、experiment、job_id、source_commit、批准文件或 Gitee 回传路径；
- 结果包无法通过 provenance、SHA、列结构或 result import 校验。

阻塞退出时只输出：阻塞原因、已验证证据、差异、所需用户决策；不要启动替代实验，不要自动切换到 Huber 或其他模型方案。
```
