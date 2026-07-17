# MPP2 逐通路 Ridge 校准第一次实验方案

> experiment_id: `mpp2_pathway_ridge_calibration_v001_20260717`
> status: `execution_approved_pending_bound_replacement_job`
> lifecycle: `active`
> created_at: `2026-07-17`
> owner: `PFMval MPP2`
> directive_id: `DIR-20260717-001`（方案登记） + `DIR-20260717-003`（r002 门控语义修正后的一次替代正式执行；取代 `DIR-20260717-002`）
> execution_authority: `state directive + experiment registry + approved job manifest`

## 1. 目标与交付定义

本实验只尝试一个方向：在冻结的 MPP2 神经网络后增加“逐通路、正斜率、带恒等先验的 Ridge 仿射校准”。目标不是输出一份供 Phase 3 再处理的中间 z-score，而是交付一个完整的 `CalibratedMPP2` 推理包：

```text
UNI2-h feature -> frozen MPP2 head -> pathway-wise calibration -> raw ssGSEA score
```

Phase 3 直接消费最终 30 个 raw ssGSEA 通路分数。除既有的 ID 对齐和预先固定的患者聚合外，Phase 3 不得再次 inverse z-score、重新拟合校准器、患者内 z-score、rank normalization 或 quantile normalization。

本文件是可追溯的实验方案，不在未登记 directive、experiment、job approval 和 source_commit 前授权训练或服务器执行。

## 2. 当前事实源与固定资产

执行前必须重新读取 `CURRENT_STATE.md`、`project_state/current_state.json`、`experiments/experiment_registry.json` 及 active `project_state/plans/mpp_training.md`。本方案采用的基线快照如下；若 live 文件冲突，以 live 文件为准并停止执行。

| 项目 | 固定值 |
|---|---|
| accepted 基线 | `mpp2_barcode_repair_v003_frozen_baseline_20260711` |
| checkpoint | epoch 15 |
| checkpoint SHA-256 | `c69191d4a67939724988bc3656c3cd2e0b173d9c871456a5fb900ab446e86a98` |
| repaired manifest | `barcode-repair-20260711-d626ad8-v003:1204018178a4d355` |
| calibration 数据 | 六名训练患者的 internal-val，共 1078 patches |
| external | XZY，共 1039 patches |
| accepted XZY legacy pooled-z PCC | 0.6549 |
| accepted XZY raw MAE | 1176.2114 |
| accepted XZY mean per-pathway raw R2 | -0.088 |
| z-score 资产 | active repaired staging 的 group 2 参数和 pathway 顺序 |

不得使用旧标准化标签资产。现有 paired S0 prediction supplement 不可替代本实验的基线预测，因为 S0 从 epoch-15 head 继续训练了 3 轮，checkpoint 已发生变化。

### 2.1 指标口径（执行时不得混用）

- `legacy_pooled_pcc`：当前 accepted `0.6549` 的兼容口径。`histogene.utils.pearson_corrcoef` 会把 `N×30` 拉平后计算 PCC；逐通路仿射校准可能改变不同通路之间的相对尺度，所以该 pooled PCC 不要求保持不变。
- `mean_per_pathway_pcc`：分别计算 30 个通路 PCC 后等权平均。对 `k_j>0` 的单通路仿射校准，它应保持不变；其前后最大绝对差只作为实现不变量检查，不作为有效性门。
- `mean_per_pathway_raw_r2`：分别计算 30 个通路 raw R2 后等权平均，是本实验的主要目标指标。当前基线为 `-0.088`。
- `mean_per_pathway_raw_mae`：分别计算 30 个通路 raw MAE 后等权平均；因为各通路样本键完全一致，它与对完整 `N×30` 矩阵求平均绝对误差等价。
- 同时报告 CCC、逐通路 `delta R2/MAE` 的中位数、IQR 和最差 3 条；禁止只报告 headline mean。

## 3. 算法定义

项目模型输出为训练 z-score 空间：

\[
z_j=(s_j-\mu_j^{train})/\sigma_j^{train}
\]

对每个通路单独拟合：

\[
\tilde z_{ij}=k_j\hat z_{ij}+b_j,\qquad k_j>0
\]

目标函数为患者平衡的 Ridge：

\[
\min_{k_j,b_j}
\frac{1}{6}\sum_p\frac{1}{n_p}\sum_{i\in p}
\left[z_{ij}-(k_j\hat z_{ij}+b_j)\right]^2
+\lambda\left[(k_j-1)^2+b_j^2\right]
\]

所有通路共享一个预注册的 `lambda`。考虑只有 6 名患者，候选收窄为：

```text
0.1, 1, 10
```

内层按患者 OOF 的 patient-balanced z-MSE 选择 `lambda`；若多个候选处于最优值一个标准误范围内，选择更大的 `lambda`（更强恒等收缩）。必须保存每个外层折的 `lambda` 选择；若选择落在三个不同值、或最弱正则 `0.1` 的选择缺乏跨折一致性，则内部稳定性门失败。

最终 raw 输出为：

\[
\tilde s_j=\tilde z_j\sigma_j^{train}+\mu_j^{train}
\]

校准器默认接近恒等映射：`k=1, b=0`。若通路斜率非正、跨折不稳定或未通过内部稳定性门，则该通路回退到 `k=1, b=0`。

### 3.1 R2 与 z/raw 空间的数学边界

Claude 审查中“只对预测做线性校准时 R2 不变”的判断不成立。实际为：

\[
R^2(y,k\hat y+b)=1-\frac{\sum_i(y_i-k\hat y_i-b)^2}{\sum_i(y_i-\bar y)^2}
\]

分母固定，`k,b` 会改变分子，因此校准可以改善或恶化 R2。R2 只在真实值与预测值**同时**做相同非零仿射变换时保持不变。

对单个通路，`raw = z*std + mean` 同时作用于 true 与 calibrated prediction，因此 raw SSE 和 TSS 都乘以同一 `std²`：

\[
R^2_{raw}(y,\tilde y)=R^2_z(y,\tilde y)
\]

所以逐通路 z 空间平方误差与逐通路 raw R2 在数学上同向；不存在“z-MSE 无法改善 raw R2”的根本错配。需要额外防范的是患者平衡拟合权重与 headline 非加权指标之间的差异，因此两套指标都必须报告并同时过门。

## 4. 执行步骤

### Step 0：登记与预检

1. 运行 `python deploy/pfmval_ops.py agent start-check --strict`。
2. 追加用户 directive，明确“Phase 3 直接消费 calibrated raw 30 列，不做二次数值处理”。
3. 登记 experiment，绑定 repaired `data_manifest_id`、source commit、checkpoint SHA 和 z-score 参数 SHA。
4. 创建只读推理/校准 job manifest；禁止修改 protected MPP 资产。

### Step 1：基线冻结推理

只使用 accepted epoch-15 checkpoint 生成 internal-val 预测，不读取或分析 XZY：

```text
predictions_internal_val_base.csv
```

至少包含：

```text
patient_id, barcode/patch_id, x, y,
true_<30 pathways>, pred_<30 pathways>
```

必须验证：

- 1078 行且患者+barcode 无冲突重复；
- 30 个通路顺序与 active repaired z-score 参数一致；
- true/pred 全部有限；
- internal-val loss、legacy pooled PCC 与 accepted 结果在预设容差内一致；
- 另外重算 mean-per-pathway PCC、raw R2、raw MAE 与 CCC，后续不得拿 pooled PCC 代替逐通路均值；
- 输出写入 checkpoint、manifest、z-score 参数和 pathway order 的 SHA。

### Step 2：嵌套患者留一校准

外层按六名患者留一，内层在剩余五名患者中按患者留一选择共享 `lambda={0.1,1,10}`。每个外层折：

1. 内层选择 `lambda`；
2. 用剩余五名患者拟合 30 组 `k,b`；
3. 在外层患者上计算 patient-balanced z-MSE、raw MAE、mean-per-pathway raw R2、CCC、legacy pooled PCC 和 mean-per-pathway PCC；
4. 六名患者轮换，形成 calibration OOF 结果。

嵌套 LOPO 的 OOF 结果是校准层的内部泛化估计；通过后在全部 6 名患者上重新选择 `lambda` 并拟合最终参数，是标准的 model-selection-then-refit 流程，不把最终 6 人 in-sample 指标当作泛化证据。

逐通路启用条件：

- 该通路 `Var(pred_z) >= 1e-6`，否则直接 identity fallback；
- `k>0`；
- 外层六折至少 4/6 名患者的 z-MSE 不恶化；
- 患者中位数 MAE 下降；
- 该通路外层 OOF raw R2 改善且 raw MAE 不恶化；
- 六折 `k` 的 `std/(abs(mean)+1e-8) <= 0.5`，否则判为失稳；
- 保存每个通路的 baseline/calibrated raw R2、MAE、CCC、`k/b` 折间分布和 fallback 原因。

未满足的通路使用 identity fallback：`k=1,b=0`。

### Step 3：冻结校准器

仅在内部 nested-LOPO 通过后：

1. 用六名 internal-val 患者重新选择最终共享 `lambda`；
2. 用全部 1078 行拟合最终 30 组 `k,b`；
3. 应用 pathway mask；
4. 写出并哈希 `calibrator.json`；
5. 校准器冻结后才允许评估 XZY。

### Step 4：XZY 一次性对照

同一批 XZY 推理同时输出 accepted baseline 和 calibrated prediction，但所有 `lambda`、mask、`k,b` 均已冻结。主要指标必须先逐通路计算 raw R2，再对 30 个通路求算术平均；不得改成 pooled R2。兼容性报告另保留 `legacy_pooled_pcc`，但不得把它与 mean-per-pathway PCC 混写。

XZY 只能用于最终一次性评估，不得参与任何参数或通路选择。由于 XZY 只有一名患者，空间 bootstrap 只能作为局部稳定性检查，不能替代跨患者验证。

空间敏感性分析定义为：按坐标 `(floor(x/g), floor(y/g))` 聚类，`g` 为 `1024/2048/4096` **像素网格边长**，不是 bin 数或重采样次数；分别进行 cluster bootstrap `B=5000`、`seed=42`，报告 `delta mean raw R2` 与 `delta raw MAE` 的 95% CI。该分析用于将证据分成强/弱，不使用“局部 R2 不得超过全局 2 倍”这类受局部 TSS 影响、不可稳定解释的门槛。

## 5. 验收门

### A. 技术门

- checkpoint、manifest、z-score 参数和 pathway order SHA 匹配；
- 30 个通路列顺序和名称完全一致；
- 无 NaN/Inf、无患者+barcode冲突；
- baseline internal-val 复现通过；
- 校准器哈希在 XZY 评估前已经冻结。

### B. 内部校准门

- nested-LOPO 患者平衡 MAE 不恶化并优先下降；
- mean-per-pathway raw R2 改善；
- 每个启用通路满足 4/6 患者稳定性条件；
- 对最终单一冻结的 `k,b,mask` 应用时，mean-per-pathway PCC 的逐通路最大绝对变化小于 `1e-10`（仅作实现不变量检查）；嵌套 LOPO 的跨折 OOF 拼接使用折特异映射，必须报告其 PCC 差异但不得将其作为不变量或放行门；
- 报告外层各折选择的 `lambda`；若选择高度不一致，内部稳定性门失败。

### C. 外部交付门

最低有效标准：

- calibrated mean raw R2 > -0.088；
- raw MAE < 1176.2114；
- mean-per-pathway PCC 保持不变；legacy pooled PCC 只报告、不设“不变”门；
- 至少 18/30 通路 raw R2 不恶化。

建议的可交付标准：

```text
Δ mean raw R2 >= 0.03
raw MAE 下降
至少18/30通路不恶化
空间敏感性分析未显示收益来自单一区域
```

若只由 `-0.088 -> -0.083`，记录为弱信号，不替换交付模型；若 mean raw R2 达到 0 以上，记录为强成功。若三个空间网格中至少两个 `delta R2` 的 95% CI 下界大于 0，记为“空间稳健强证据”；否则保留为单患者弱/中等证据，不宣传为跨患者证明。

## 6. 最终模型包与 Phase 3 接口

通过外部门后，将校准和 train-only inverse z-score 集成到推理包装，而不是交给 Phase 3 另行处理：

```python
class CalibratedMPP2(nn.Module):
    def forward(self, features):
        pred_z = self.head(features)
        pred_z_cal = pred_z * self.k + self.b
        pred_raw_cal = pred_z_cal * self.std + self.mean
        return pred_raw_cal
```

`k/b/mean/std` 必须作为不可训练 buffer 与 provenance 一起封装。初始化时必须计算并断言：

- base checkpoint SHA 等于 `calibrator_provenance.json` 绑定值；
- `data_manifest_id`、z-score 参数 SHA、pathway-order SHA 完全一致；
- 30 个通路名称和顺序完全一致。

任一不一致必须 `raise`，不得警告后继续，也不得把旧校准器静默应用到未来 repaired v004 或其他 checkpoint。

模型包建议为：

```text
mpp2_calibrated_raw_v001/
├── best_checkpoint.pth
├── calibrated_mpp2.py
├── calibrator.json
├── zscore_params_from_train.json
├── pathway_order.json
├── model_provenance.json
├── sha256_manifest.json
└── README_PHASE3_INPUT_CONTRACT.md
```

对 Phase 3 暴露的预测表只提供 ID 和最终 30 个 raw calibrated pathway 列。禁止混用旧 `pred_z_*`、旧 raw 输出或未经校准的 baseline 输出。

## 7. 明确排除项

本实验不包含：

- Huber 重训；
- depth-aware 校准或 `/depth`；
- 在 XZY 上拟合 `lambda`、`k`、`b` 或 pathway mask；
- 全局共享一个 `k,b`；
- 修改 accepted 神经网络权重；
- 多 seed、rank、学习率或其他超参数搜索；
- 重新生成 protected ssGSEA、split、z-score 或 manifest 资产。

## 8. 阻塞退出条件

出现以下任一情况，立即停止，不得自行绕过：

- start-check 出现 FAIL；
- active manifest、checkpoint、z-score 参数或 pathway order 与快照不一致；
- 发现训练、校准或通路选择读取了 XZY；
- baseline internal-val 无法复现；
- 患者+barcode 键冲突、缺失或重复；
- `k<=0` 且无法 identity fallback；
- 任一通路 `Var(pred_z)<1e-6` 却仍被启用，或 `k` 稳定性阈值不满足；
- 内部稳定性门失败；
- 需要修改 protected MPP 资产、服务器路径或新增训练权限；
- 缺少显式 directive、experiment 登记、job approval、source commit 或 Gitee 回传路径。

阻塞时只提交证据、差异和下一步所需的用户决策，不继续执行。

## 9. 相关事实源

- `CURRENT_STATE.md`
- `project_state/current_state.json`
- `experiments/experiment_registry.json`
- `project_state/plans/mpp_training.md`
- `mpp_standard_splits/group_2/split_info.json`
- `mpp_standard_splits/group_2/zscore_manifest.json`
- `automation/results/mpp2-repair-v003-frozen-20260711-r002/attempt-003/result.json`
- `train_mpp_uni2h_mlp.py`
- `train_mpp_uni2h_lora.py`

## 10. Claude 对抗性审查处理记录

| 审查项 | 判定 | 处理 |
|---|---|---|
| z-MSE 与 raw R2 根本错配 | 驳回核心结论，部分采纳报告建议 | 单通路 inverse z 同时作用于 true/pred，R2 完全等价；新增逐通路 delta 分布、MAE/CCC 与双口径报告。 |
| 6 人 nested-LOPO 必然过拟合 | 部分采纳 | 低患者数确实限制稳定性；网格收窄为 `{0.1,1,10}`、强正则优先并新增 lambda 跨折稳定门。nested-LOPO 与全6人最终 refit 是标准流程，不要求患者集相同。 |
| pooled PCC 与逐通路 PCC 混淆 | 采纳 | 明确 `0.6549` 是 legacy pooled PCC；新增 mean-per-pathway PCC，pooled PCC 不设不变门。 |
| 线性校准不能改变 R2 | 驳回 | 只变换预测会改变 SSE，因此 R2 可以改变；只有 true/pred 同时仿射变换时 R2 才不变。 |
| wrapper 与 inverse z 冲突 | 部分采纳 | wrapper 内 inverse z 正是接口设计；采纳 manifest/SHA 绑定与加载时 hard fail。 |
| seed/病态矩阵未定义 | 采纳 | bootstrap 固定 `seed=42`、`B=5000`；`Var(pred_z)<1e-6` 回退，量化 k 稳定阈值。 |
| provenance 链不足 | 采纳 | 增加 checkpoint/manifest/z-score/pathway SHA、患者计数和逐折 lambda 日志，并在加载时断言。 |
| 空间网格含义不清 | 采纳澄清，驳回具体替代门槛 | 1024/2048/4096 是像素网格边长；不采用“局部 R2 不超过全局2倍”的不稳定标准。 |
| 等患者权重反而违背初衷 | 驳回结论，采纳透明度建议 | `1/n_p` 正是让每名患者总权重相同；当前每人 internal-val 为 114–337 patches。继续报告患者计数与 per-patient 指标。 |
